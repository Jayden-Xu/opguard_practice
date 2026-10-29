"""练习 3：OpGuard 的 XOR 指纹。
对应《OpGuard导读》第 1 阶段 1.2~1.3、1.6，第 6 阶段 6.3~6.4。
分段规则照抄 opguard/tracer/xor_signature.py 里的 flat_word_layout（第 64-80 行）。
"""
import struct
from functools import reduce


def to_bytes(values, fmt):
    """把一组数按指定格式（'f' = float32，'d' = float64）变成内存里的字节（小端序）。"""
    return struct.pack('<' + fmt * len(values), *values)


def flat_word_layout(nbytes, regions=4):
    """和 OpGuard 一样：字节按 4 个一组拼成 uint32（word），word 平均分成 regions 段，余数分给前面的段。"""
    base, rem = divmod((nbytes + 3) // 4, regions)
    ranges, start = [], 0
    for r in range(regions):
        end = start + base + (r < rem)
        ranges.append((min(4 * start, nbytes), min(4 * end, nbytes)))
        start = end
    return ranges


def fingerprint(data, regions=4):
    """每一段里的 uint32 全部异或起来，得到这一段的指纹；空的段记为 0。"""
    nbytes = len(data)                              # 分段按原始字节数算
    data = data + b'\0' * ((-nbytes) % 4)          # 最后不满 4 字节的部分补 0
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    out = []
    for lo, hi in flat_word_layout(nbytes, regions):
        part = words[lo // 4:(hi + 3) // 4]
        out.append(reduce(lambda a, b: a ^ b, part, 0))
    return out


print("== 1. [1.0, 2.0] 的指纹（只分 1 段）==")
for v in (1.0, 2.0):
    w = struct.unpack('<I', struct.pack('<f', v))[0]
    print(f"{v} 的 4 个字节看成 uint32 = {w} = {hex(w)}")
fp = fingerprint(to_bytes([1.0, 2.0], 'f'), regions=1)[0]
print(f"指纹 = 0x3f800000 XOR 0x40000000 = {fp} = {hex(fp)}")

print("\n== 2. 第二个数只差最后一个比特 ==")
next_after_2 = struct.unpack('<f', struct.pack('<I', 0x40000001))[0]
fp2 = fingerprint(to_bytes([1.0, next_after_2], 'f'), regions=1)[0]
print(f"2.0 之后紧挨着的 float32 是 {next_after_2!r}")
print(f"[1.0, {next_after_2:.7f}] 的指纹 = {fp2} = {hex(fp2)}    （和上面只差最后一位）")

print("\n== 3. 指纹相同 ≠ 张量相同（1.6 的第 1 个\"不等于\"）==")
print(f"[1.0, 2.0] 分 1 段：{fingerprint(to_bytes([1.0, 2.0], 'f'), 1)}")
print(f"[2.0, 1.0] 分 1 段：{fingerprint(to_bytes([2.0, 1.0], 'f'), 1)}    ← 顺序换了，指纹却一样（异或和顺序无关）")
print(f"[1.0, 2.0] 分 2 段：{fingerprint(to_bytes([1.0, 2.0], 'f'), 2)}")
print(f"[2.0, 1.0] 分 2 段：{fingerprint(to_bytes([2.0, 1.0], 'f'), 2)}    ← 分段以后就能区分了")

print("\n== 4. OpGuard 默认分 4 段：不同大小的张量怎么分 ==")
for n, fmt, name in [(1, 'd', '1 个 float64'), (3, 'f', '3 个 float32'), (10, 'f', '10 个 float32')]:
    nbytes = n * struct.calcsize(fmt)
    print(f"{name}（{nbytes} 字节）：byte_ranges = {flat_word_layout(nbytes, 4)}")

print("\n== 5. demo 里的真实数据（第 1 阶段 1.3）==")
for run, value in (("run1", 32.28125), ("run2", 29.984375)):
    print(f"{run}：float64 标量 {value} 的 4 段指纹 = {fingerprint(to_bytes([value], 'd'), 4)}")
print("对照 demo：run1 是 [0, 1077945344, 0, 0]，run2 是 [0, 1077804032, 0, 0]")
lo, hi = 0, 1077945344
print(f"反过来，把 [{lo}, {hi}] 拼回 float64：{struct.unpack('<d', struct.pack('<II', lo, hi))[0]}")
