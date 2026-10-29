"""练习 4：用两个线程模拟两个 CUDA 流之间的竞态。
对应《OpGuard导读》第 1 阶段 1.4 的小例子、《DeepSpeed导读》第 6 阶段 6.2 的双缓冲小例子。
- 线程"通信流"：把 bucket 里的两个梯度一个一个读出来，准备发给别的卡；
- 线程"默认流"：要往同一个 bucket 写下一批梯度 [5, 6]。
两个线程什么时候执行，由操作系统调度决定，每次都可能不同，就像 GPU 上两个流的先后顺序。
这只是一个类比：真实的 CUDA 流是在 GPU 上执行的，不是 Python 线程。
"""
import collections
import random
import threading
import time


def run_once(wait_before_write):
    bucket = [1, 2]              # 这一批梯度
    sent = []                    # 通信流实际读到、发出去的值
    reduction_done = threading.Event()   # 类比 CUDA event：通信流用完 bucket 后打的标记

    def comm_stream():
        for i in range(2):
            time.sleep(random.uniform(0, 0.001))   # 读得有快有慢
            sent.append(bucket[i])
        reduction_done.set()

    def default_stream():
        if wait_before_write:
            reduction_done.wait()                  # 修复：写之前先等通信流做完
        time.sleep(random.uniform(0, 0.001))
        bucket[0], bucket[1] = 5, 6                # 往同一个 bucket 写下一批梯度

    threads = [threading.Thread(target=comm_stream), threading.Thread(target=default_stream)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return tuple(sent)


for wait in (False, True):
    title = "加了等待（像 #5606 / debug.py 的修复）" if wait else "没有等待（#5545 的情况）"
    counts = collections.Counter(run_once(wait) for _ in range(200))
    print(f"== {title}：跑 200 次，通信流发出去的值 ==")
    for value, n in counts.most_common():
        mark = "正确" if value == (1, 2) else "错了：读到了被覆盖的值"
        print(f"  {value}：{n} 次（{mark}）")
    print()
print("每次运行这个脚本，没有等待那一组的次数分布都可能不一样，这正是\"不确定\"。")
