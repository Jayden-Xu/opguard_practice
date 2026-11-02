"""One FSDP2 update on a fixed batch: 2 ranks, 6 samples, micro-batch size 1.

Based on verl tests/models/test_engine.py at 8df5baeed0f5ecdd7f870d83bc3d79194816e2a9.
Fill in run_one_step before running. The target environment must provide verl.
"""

import argparse
import json
import math
import os
import sys
import uuid
from collections.abc import Mapping
from functools import partial

os.environ["NCCL_DEBUG"] = "WARN"

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDictBase
from transformers import AutoModelForCausalLM, Qwen3Config

from verl import DataProto
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_nccl_backend, get_torch_device
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding

WORLD_SIZE = 2
GLOBAL_MINI_BATCH = 6  # samples over all DP ranks
PER_RANK_BATCH = GLOBAL_MINI_BATCH // WORLD_SIZE  # 3
MICRO_BATCH_SIZE_PER_GPU = 1
SEQ_LEN = 32
RESPONSE_LEN = SEQ_LEN // 2
VOCAB_SIZE = 128
LR = 1e-3
STRATEGY = "fsdp2"

# Same tiny model as test_fsdp_engine_split_forward_backward_and_optimizer_step.
MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
)


def create_checkpoint(path: str, seed: int) -> None:
    """Single initial checkpoint, written once by the parent process (cf. create_actor_model)."""
    torch.manual_seed(seed)
    config = Qwen3Config(**MODEL_CONFIG)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(path)
    config.save_pretrained(path)


def build_rank_batch(rank: int, seed: int, engine_config: FSDPEngineConfig, model_config: HFModelConfig):
    """Slice the fixed global batch before converting it to the no-padding representation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_ids = torch.arange(GLOBAL_MINI_BATCH * SEQ_LEN, dtype=torch.long)
    input_ids = input_ids.reshape(GLOBAL_MINI_BATCH, SEQ_LEN).remainder(VOCAB_SIZE)
    attention_mask = torch.ones_like(input_ids)
    position_ids = compute_position_id_with_mask(attention_mask)
    responses = input_ids[:, RESPONSE_LEN:]
    response_mask = attention_mask[:, RESPONSE_LEN:]

    rows = slice(rank * PER_RANK_BATCH, (rank + 1) * PER_RANK_BATCH)
    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids[rows],
            "prompts": input_ids[rows, :RESPONSE_LEN],
            "attention_mask": attention_mask[rows],
            "position_ids": position_ids[rows],
            "responses": responses[rows],
            "response_mask": response_mask[rows],
            "old_log_probs": torch.zeros_like(responses[rows], dtype=torch.float32),
            "advantages": torch.ones_like(responses[rows], dtype=torch.float32),
        },
        meta_info={"temperature": 1.0},
    )
    data_td = left_right_2_no_padding(data.to_tensordict())
    # Same non-tensor keys that _make_split_step_batch injects (TrainingWorker.train_batch injects them
    # for the worker path; calling the engine directly does not).
    tu.assign_non_tensor(
        data_td,
        global_batch_size=GLOBAL_MINI_BATCH,
        use_remove_padding=model_config.get("use_remove_padding", False),
        use_dynamic_bsz=engine_config.use_dynamic_bsz,
        max_token_len_per_gpu=engine_config.max_token_len_per_gpu,
        micro_batch_size_per_gpu=engine_config.micro_batch_size_per_gpu,
        use_fused_kernels=engine_config.use_fused_kernels,
    )
    return data_td, int(response_mask[rows].sum().item())


def run_one_step(engine, batch, loss_fn):
    # perform one standard engine training update and return its output.
    out = engine.train_batch(batch, loss_fn)
    return out


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (Mapping, TensorDictBase)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "aggregate") and hasattr(value, "aggregation"):
        return {"aggregation": str(value.aggregation), "aggregated": _plain(value.aggregate())}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def extract_result(output):
    if not isinstance(output, (Mapping, TensorDictBase)):
        raise TypeError(f"Unexpected engine output: {type(output).__name__}")
    if "loss" not in output or "metrics" not in output:
        raise ValueError(f"Missing loss or metrics; output keys: {list(output.keys())}")
    metrics = output["metrics"]
    if not isinstance(metrics, (Mapping, TensorDictBase)) or "grad_norm" not in metrics:
        raise ValueError("Engine output is missing metrics.grad_norm")
    grad_norm = _plain(metrics["grad_norm"])
    if isinstance(grad_norm, bool) or not isinstance(grad_norm, (int, float)):
        raise TypeError(f"Expected scalar grad_norm, got {grad_norm!r}")
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise ValueError(f"Expected finite positive grad_norm, got {grad_norm!r}")
    return {
        "loss": _plain(output["loss"]),
        "metrics": _plain(metrics),
        "grad_norm": grad_norm,
        "raw_keys": sorted(output.keys()),
    }


def worker(rank: int, rendezvous_file: str, ckpt_dir: str, output_dir: str, seed: int):
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(), init_method=f"file://{rendezvous_file}", rank=rank, world_size=WORLD_SIZE
    )
    try:
        from verl.workers.engine import EngineRegistry  # imported after dist init, as in the test

        model_config = HFModelConfig(
            path=ckpt_dir,
            load_tokenizer=False,
            enable_gradient_checkpointing=False,
            override_config={"attn_implementation": "sdpa"},
            use_remove_padding=True,
        )
        engine_config = FSDPEngineConfig(
            forward_only=False,
            fsdp_size=WORLD_SIZE,
            strategy=STRATEGY,
            ulysses_sequence_parallel_size=1,
            use_remove_padding=True,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            micro_batch_size_per_gpu=MICRO_BATCH_SIZE_PER_GPU,
            max_token_len_per_gpu=128,  # unused when use_dynamic_bsz=False; kept as in the test
        )
        engine = EngineRegistry.new(
            model_type="language_model",
            backend=engine_config.strategy,
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=FSDPOptimizerConfig(lr=LR, lr_scheduler_type="constant"),
            checkpoint_config=CheckpointConfig(),
        )
        engine.initialize()

        # token-mean, no entropy / KL terms (values spelled out although they equal the ActorConfig defaults)
        actor_config = ActorConfig(
            strategy=STRATEGY,
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=-1,
            loss_agg_mode="token-mean",
            entropy_coeff=0,
            use_kl_loss=False,
        )
        loss_fn = partial(ppo_loss, config=actor_config)

        batch, n_response_tokens = build_rank_batch(rank, seed, engine_config, model_config)

        info = {
            "rank": rank,
            "dist_backend": dist.get_backend(),
            "engine_class": type(engine).__name__,
            "strategy": engine_config.strategy,
            "module_class": type(engine.module).__name__,
            "param_dtype": str(next(engine.module.parameters()).dtype),
            "engine_config.dtype": engine_config.dtype,
            "engine_config.model_dtype": engine_config.model_dtype,
            "autocast_dtype": str(getattr(engine, "_autocast_dtype", None)),
            "dp_size": engine.get_data_parallel_size(),
            "dp_rank": engine.get_data_parallel_rank(),
            "samples_this_rank": batch.shape[0],
            "response_tokens_this_rank": n_response_tokens,
            "micro_batch_size_per_gpu": engine_config.micro_batch_size_per_gpu,
            "global_mini_batch": GLOBAL_MINI_BATCH,
            "use_dynamic_bsz": engine_config.use_dynamic_bsz,
            "loss_agg_mode": actor_config.loss_agg_mode,
            "entropy_coeff": actor_config.entropy_coeff,
            "use_kl_loss": actor_config.use_kl_loss,
            "lr": LR,
            "use_torch_compile": engine_config.use_torch_compile,
            "enable_gradient_checkpointing": model_config.enable_gradient_checkpointing,
            "seed": seed,
            "checkpoint": ckpt_dir,
        }
        assert info["dp_size"] == WORLD_SIZE and info["samples_this_rank"] == PER_RANK_BATCH, info

        assert n_response_tokens == PER_RANK_BATCH * RESPONSE_LEN, info
        print(f"[rank {rank}] config " + json.dumps(info, indent=2), flush=True)

        with engine.train_mode(zero_grad_on_exit=True):
            output = run_one_step(engine, batch, loss_fn)

        result = extract_result(output)

        report = {"info": info, "result": result}
        print(f"[rank {rank}] " + json.dumps(report, indent=2, default=repr), flush=True)
        with open(os.path.join(output_dir, f"rank{rank}.json"), "w") as f:
            json.dump(report, f, indent=2, default=repr)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="checkpoint, rendezvous file and rank*.json go here")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reuse-checkpoint", action="store_true", help="use <output-dir>/init_ckpt if it exists")
    args = parser.parse_args()

    if get_torch_device().device_count() < WORLD_SIZE:
        sys.exit(f"need >= {WORLD_SIZE} accelerator devices, found {get_torch_device().device_count()}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "init_ckpt")
    if os.path.exists(ckpt_dir) and not args.reuse_checkpoint:
        sys.exit(f"{ckpt_dir} exists; pass --reuse-checkpoint to load it instead of regenerating")
    if not os.path.exists(ckpt_dir):
        create_checkpoint(ckpt_dir, args.seed)

    rendezvous_file = os.path.join(output_dir, f"rdzv_{uuid.uuid4().hex}")
    try:
        mp.spawn(fn=worker, args=(rendezvous_file, ckpt_dir, output_dir, args.seed), nprocs=WORLD_SIZE, join=True)
    finally:
        if os.path.exists(rendezvous_file):
            os.remove(rendezvous_file)


if __name__ == "__main__":
    main()
