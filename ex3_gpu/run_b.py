"""练习 3 的 B 边：和 run_a.py 完全一样，只把 GELU 换成 tanh 近似。"""
import torch
import torch.nn.functional as F

DEV = "cuda"

APPROX = "tanh"          # 和 run_a.py 唯一的不同

torch.manual_seed(0)
w1 = torch.randn(16, 32, device=DEV, requires_grad=True)
w2 = torch.randn(32, 1, device=DEV, requires_grad=True)
opt = torch.optim.SGD([w1, w2], lr=0.01)
x = torch.randn(8, 16, device=DEV)
y = torch.randn(8, 1, device=DEV)

for step in range(3):
    h = x @ w1
    h = F.gelu(h, approximate=APPROX)
    out = h @ w2
    loss = F.mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {step} loss {loss.item():.8f}")
