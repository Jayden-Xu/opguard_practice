"""练习 5：用 OpGuard harness 的只读查询工具，亲手查出 DeepSpeed demo 的第一个差异。
对应《OpGuard导读》第 1 阶段 1.3、第 10 阶段 10.2、10.5、10.8。
数据是仓库自带的真实导出文件，不需要 GPU，也不需要装 OpGuard 或 PyTorch。
这个脚本只是帮你依次执行下面这种命令，并把结果里关键的字段挑出来：
    cd ~/Desktop/OpGuard/harness
    python3 -m opguard_harness --root <临时目录> call first_diff --arguments args.json
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

HARNESS = os.path.expanduser("~/Desktop/OpGuard/harness")
DEMO = os.path.join(HARNESS, "examples/demo_case5/paired-capture-20261019/opguard")
_TMP = tempfile.TemporaryDirectory(prefix="opguard_ex5_")   # 脚本结束时自动删除
WORK = _TMP.name


def call(tool, arguments):
    """执行一个 harness 工具，返回它输出的 JSON 里的 result 部分。"""
    path = os.path.join(WORK, "args.json")
    with open(path, "w") as f:
        json.dump(arguments, f)
    out = subprocess.run([sys.executable, "-m", "opguard_harness", "--root", os.path.join(WORK, "root"),
                          "call", tool, "--arguments", path],
                         cwd=HARNESS, capture_output=True, text=True,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})   # 不往仓库里写 __pycache__
    reply = json.loads(out.stdout)
    if not reply.get("ok"):
        raise SystemExit(f"{tool} 出错了：{reply.get('error')}")
    return reply["result"]


def as_float32(sig):
    """float32 标量只有 4 个字节，第一段指纹就是它的全部字节。"""
    return struct.unpack('<f', struct.pack('<I', sig[0]))[0]


def as_float64(sig):
    """demo 里这个输入是 float64 标量：把前两段指纹拼回 8 个字节，就能还原数值。"""
    return struct.unpack('<d', struct.pack('<II', sig[0], sig[1]))[0]


for rank in (0, 1):
    artifact = os.path.join(DEMO, f"perfetto_gpu_{rank:03d}.json.gz")
    print(f"################ rank {rank}：{os.path.basename(artifact)} ################")

    summary = call("trace_summary", {"artifact": artifact})
    print(f"【trace_summary】一共 {summary['operation_count']} 个被记录的操作，状态统计：")
    for item in summary["status_counts"]:
        print(f"    {item['run']} rank {item['rank']}  {item['status']:9s} {item['count']}")

    boundaries = call("first_diff", {"artifact": artifact})["boundaries"]
    print("【first_diff】每个 (run, rank) 的第一个差异：")
    for b in boundaries:
        d = b["first_observed_diff"]
        first_un = (b.get("first_unaligned") or {}).get("name")
        print(f"    {b['run']}：{d['name']}  位置 {os.path.basename(d['file'])}:{d['line']}  "
              f"之前有 {b['unaligned_before']} 个 unaligned" + (f"（第一个是 {first_un}）" if first_un else ""))

    ref = boundaries[0]["first_observed_diff"]["event_ref"]
    event = call("inspect_event", {"artifact": artifact, "event_ref": ref})
    peer_ref = event["peer_evidence"][0]["event"]["event_ref"]
    peer = call("inspect_event", {"artifact": artifact, "event_ref": peer_ref})
    print(f"【inspect_event】这次调用（记录里的原始行号 {event['event']['line']}）的输入、输出指纹，两边配上的那一对：")
    for name, ev in (("run1", event), ("run2", peer)):
        for kind in ("inputs", "outputs"):
            for t in ev["io"][kind]:
                line = f"    {name} {kind}：dtype {t['dtype']}  shape {t['shape']}  指纹 {t['xor_signature']}"
                if t["numel"] == 1 and t["dtype"] == "torch.float64":
                    line += f"  → 还原成数值 {as_float64(t['xor_signature'])}"
                if t["numel"] == 1 and t["dtype"] == "torch.float32":
                    line += f"  → 还原成数值 {as_float32(t['xor_signature'])}"
                print(line)
    print(f"    差异说明：{event['diff']['summary']}")

    print("【source_context】当时的调用栈（最内层 4 帧；被插桩的文件，调用栈记的是改写后的行号，所以两个数一样）：")
    frames = event["source_context"]
    for fr in frames[-4:]:
        print(f"    {fr['function']:28s} {os.path.basename(fr['source'])}  调用栈里的行号 {fr['source_line']}  快照行号 {fr['target_line']}")

    last = frames[-1]
    ctx = call("code_context", {"artifact": artifact, "source": last["source"], "line": last["target_line"]})
    print(f"【code_context】{ctx['function']}，快照里第 {ctx['target_line']} 行附近：")
    for item in ctx["lines"]:
        if abs(item["line"] - ctx["target_line"]) <= 3:
            mark = "→" if item["line"] == ctx["target_line"] else " "
            print(f"    {mark} {item['line']}: {item['text']}")
    print()
