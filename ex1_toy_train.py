"""练习 1：玩具模型 y = w * x 的训练。
对应《OpGuard导读》第 0 阶段 0.1~0.7。只用 Python 自带的功能，不需要 GPU。
"""


def forward(w, x):
    """前向：用输入 x 算出输出 y。"""
    return w * x


def loss_fn(y, t):
    """loss：输出 y 离目标 t 有多远（平方误差）。"""
    return (y - t) ** 2


def grad_fn(w, x, t):
    """梯度：loss 对 w 的导数 = 2 * (y - t) * x。"""
    return 2 * (forward(w, x) - t) * x


print("== 1. 一条数据，训练一步（0.1~0.4）==")
w, x, t, lr = 0.5, 2.0, 3.0, 0.01
y = forward(w, x)
print(f"前向：y = {y}    loss = {loss_fn(y, t)}    梯度 = {grad_fn(w, x, t)}")
w_new = w - lr * grad_fn(w, x, t)
print(f"更新：w = {w} - {lr} * ({grad_fn(w, x, t)}) = {w_new:.4f}    新的 loss = {loss_fn(forward(w_new, x), t):.4f}")

print("\n== 2. 一个 batch 有两条数据，取平均（0.5）==")
data = [(2.0, 3.0), (1.0, 1.0)]   # (输入 x, 目标 t)
losses = [loss_fn(forward(0.5, x), t) for x, t in data]
grads = [grad_fn(0.5, x, t) for x, t in data]
print(f"每条数据的 loss = {losses}    梯度 = {grads}")
print(f"batch 平均 loss = {sum(losses) / 2}    平均梯度 = {sum(grads) / 2}")

print("\n== 3. 梯度累积：把 batch 拆成 2 个 micro batch（0.6）==")
acc = 0.0
for i, (x, t) in enumerate(data, 1):
    acc += grad_fn(0.5, x, t)
    print(f"第 {i} 个 micro batch 之后，累积的梯度 = {acc}")
print(f"攒够 2 个，除以 2 = {acc / 2}（和上面一次算完的平均梯度一样）")

print("\n== 4. 两张卡做数据并行，用 all-reduce 求平均（0.7）==")
gpu_grads = {"卡 0": grad_fn(0.5, 2.0, 3.0), "卡 1": grad_fn(0.5, 1.0, 1.0)}
total = sum(gpu_grads.values())
print(f"各卡自己算出的梯度：{gpu_grads}")
print(f"all-reduce 之后，每张卡都拿到总和 {total}；再除以卡数 2 得 {total / 2}，两张卡用同一个值更新")

print("\n== 5. 多训练几步，看 loss 怎么下降（学习率 0.1）==")
w = 0.5
for step in range(1, 11):
    g = sum(grad_fn(w, x, t) for x, t in data) / len(data)
    w -= 0.1 * g
    avg_loss = sum(loss_fn(forward(w, x), t) for x, t in data) / len(data)
    print(f"第 {step:2d} 步：w = {w:.4f}    平均 loss = {avg_loss:.4f}")
