"""练习 6：自己验证"谁设置了 NCCL_ALGO 和确定性开关"。
对应《DeepSpeed导读》第 4 阶段 4.1、第 11 阶段 11.1，《Megatron导读》第 2 阶段 2.5、第 3 阶段 3.6。
导读里说：DeepSpeed 和 OpGuard 都不设 NCCL_ALGO，Megatron 的 determinism.py 会设。
这个脚本在三个仓库的 Python 源码里逐行搜索关键词，让你亲眼确认。只读文件，不改任何东西。
"""
import os

REPOS = {
    "DeepSpeed": os.path.expanduser("~/Desktop/DeepSpeed/deepspeed"),
    "OpGuard": os.path.expanduser("~/Desktop/OpGuard/opguard"),
    "Megatron": os.path.expanduser("~/Desktop/Megatron-LM/megatron"),
}
KEYWORDS = ["NCCL_ALGO", "use_deterministic_algorithms", "CUBLAS_WORKSPACE_CONFIG", "NVTE_ALLOW_NONDETERMINISTIC_ALGO"]


def search(root, keyword):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8", errors="ignore") as f:
                for lineno, text in enumerate(f, 1):
                    if keyword in text:
                        hits.append((os.path.relpath(path, os.path.dirname(root)), lineno, text.strip()))
    return hits


for keyword in KEYWORDS:
    print(f"======== 关键词：{keyword} ========")
    for repo, root in REPOS.items():
        if not os.path.isdir(root):
            print(f"  {repo}：没找到目录 {root}")
            continue
        hits = search(root, keyword)
        print(f"  {repo}：{len(hits)} 处")
        for path, lineno, text in hits[:3]:
            print(f"      {path}:{lineno}  {text[:90]}")
        if len(hits) > 3:
            print(f"      ……（还有 {len(hits) - 3} 处没列出）")
    print()
