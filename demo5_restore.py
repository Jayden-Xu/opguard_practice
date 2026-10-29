"""把 demo_case5 的 trace 打包文件还原成 compare 能读的 exec_trace 目录。
用法: python3 demo5_restore.py <raw-captures 解压目录> <输出目录>
trace 阶段把每个 rank 的记录打成一个 tar（.opguard-bundles/），tar 里 manifest.json 写着
每个成员（data/N）原本的路径（exec_trace/gpu_000/ops/...jsonl）。这里按 manifest 放回原位。
"""
import glob, json, os, sys, tarfile

raw, out = sys.argv[1], sys.argv[2]
for side, run in (("baseline", "run1"), ("candidate", "run2")):
    for tar_path in sorted(glob.glob(f"{raw}/experiment/{side}-trace-1/output/.opguard-bundles/*/*.tar")):
        with tarfile.open(tar_path) as t:
            manifest = json.load(t.extractfile("manifest.json"))
            for e in manifest["entries"]:
                if ".." in e["path"] or e["path"].startswith("/"):
                    raise SystemExit(f"bad path {e['path']}")
                dst = os.path.join(out, run, e["path"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(t.extractfile(e["member"]).read())
        print(f"{run} <- {os.path.relpath(tar_path, raw)}  ({len(manifest['entries'])} 个文件)")
