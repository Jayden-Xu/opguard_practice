import os

import torch
import torch.nn.functional as F

approx = os.environ.get("EX3_GELU_APPROX", "none")
if approx not in ("none", "tanh"):
    raise ValueError(approx)

torch.set_num_threads(1)
torch.manual_seed(0)

device = "cuda"
w1 = torch.randn(16, 32, device=device, requires_grad=True)
w2 = torch.randn(32, 1, device=device, requires_grad=True)
opt = torch.optim.SGD([w1, w2], lr=0.01)
x = torch.randn(8, 16, device=device)
y = torch.randn(8, 1, device=device)

for step in range(3):
    h = x @ w1
    h = F.gelu(h, approximate=approx)
    out = h @ w2
    loss = F.mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {step} loss {loss.item():.8f}")
