"""练习 2：浮点精度和加法顺序。
对应《OpGuard导读》第 0 阶段 0.8~0.9、第 8 阶段 8.1，《DeepSpeed导读》第 3 阶段 3.3。
Python 自己的小数是 64 位的。这里用标准库 struct 把每一步结果舍入成 fp32 / fp16 / bf16，
模拟 GPU 上用这些格式计算的效果。不需要 GPU，也不用装任何东西。
"""
import random
import struct


def fp32(x):
    """把 x 舍入成最接近的 fp32。"""
    return struct.unpack('<f', struct.pack('<f', x))[0]


def fp16(x):
    """把 x 舍入成最接近的 fp16。"""
    return struct.unpack('<e', struct.pack('<e', x))[0]


def bf16(x):
    """把 x 舍入成最接近的 bf16（bf16 就是 fp32 的高 16 位，按"最近、平局取偶"舍入）。"""
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


print("== 1. 0.1 在三种格式里实际存成多少（0.8）==")
print(f"fp32: {fp32(0.1):.17g}")
print(f"bf16: {bf16(0.1):.10g}")
print(f"fp16: {fp16(0.1):.10g}")

print("\n== 2. 从 1 开始加 1000 次 0.001（0.8）==")
s32, s16 = 1.0, 1.0
for _ in range(1000):
    s32 = fp32(s32 + fp32(0.001))
    s16 = bf16(s16 + bf16(0.001))
print(f"fp32 累加：{s32:.6f}    bf16 累加：{s16}    （正确答案是 2.0）")

print("\n== 3. 浮点加法不满足结合律（0.9）==")
big = fp32(1e8)
print(f"fp32：(1e8 + 1) - 1e8 = {fp32(fp32(big + 1) - big)}")
print(f"fp32：(1e8 - 1e8) + 1 = {fp32(fp32(big - big) + 1)}")

print("\n== 4. 四张卡求和：环形顺序 vs 树形顺序（fp16，《OpGuard导读》8.1）==")
a, b, c, d = 1024.0, 0.5, 0.5, 0.5
ring = fp16(fp16(fp16(a + b) + c) + d)
tree = fp16(fp16(a + b) + fp16(c + d))
print(f"环形 ((a+b)+c)+d = {ring}    树形 (a+b)+(c+d) = {tree}    精确的和 = {a + b + c + d}")

print("\n== 5. ZeRO-1 和 ZeRO-2 的加法顺序（fp16，《DeepSpeed导读》3.3）==")
g0_mb1, g0_mb2, g1_mb1, g1_mb2 = 1024.0, 1024.0, 0.5, 1.0
zero1 = fp16(fp16(g0_mb1 + g0_mb2) + fp16(g1_mb1 + g1_mb2))   # 每张卡先本地累加，再跨卡求和
zero2 = fp16(fp16(g0_mb1 + g1_mb1) + fp16(g0_mb2 + g1_mb2))   # 每个 micro batch 先跨卡求和，再累加
print(f"ZeRO-1 = {zero1}    ZeRO-2 = {zero2}    精确的和 = {g0_mb1 + g0_mb2 + g1_mb1 + g1_mb2}")

print("\n== 6. 同样 10000 个数，打乱顺序求和 5 次（fp32）==")
rng = random.Random(0)                           # 固定种子，每次运行生成的数都一样
nums = [fp32(rng.uniform(-1, 1)) for _ in range(10000)]
results = []
for trial in range(5):
    order = nums[:]
    random.Random(trial).shuffle(order)          # 第 trial 种顺序
    s = 0.0
    for v in order:
        s = fp32(s + v)
    results.append(s)
    print(f"第 {trial + 1} 种顺序：{s!r}")
print(f"5 种顺序里，一共出现了 {len(set(results))} 种不同的结果")
