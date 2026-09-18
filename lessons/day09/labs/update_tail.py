# ---
# title: 10 个 batch、accumulation_steps=3：参数更新了几次
# timeout: 60
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把 iters 从 10 改成 9（正好整除 3）：尾巴那一次还会触发吗？总更新次数变成几次？"
#   - "把 accumulation_steps 从 3 改成 4：循环内更新几次、尾巴补几次？总共还是 4 次吗？"
# ---
import torch
from learnkit import *

accumulation_steps, iters, start_step = 3, 10, 0      # 👉 改这三个数
model, log = torch.nn.Linear(4, 2), []
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
last_step = start_step

for step in range(start_step + 1, iters + 1):         # train_epoch: enumerate(loader, start=start_step+1)
    last_step = step
    model(torch.randn(3, 4)).sum().backward()         # 代替一次前向 + backward，梯度攒进 .grad
    if step % accumulation_steps == 0:                # train_pretrain.py:42
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        log.append([step, "✅ 更新", "循环内 step % accumulation_steps == 0", model.weight.grad])
    else:
        log.append([step, "—— 只累积", f"{step} % {accumulation_steps} = {step % accumulation_steps}", "梯度留在 .grad 里"])

if last_step > start_step and last_step % accumulation_steps != 0:   # train_pretrain.py:75，epoch 末尾的尾巴
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    log.append([f"{last_step} 之后", "✅ 补一次更新", f"last_step % {accumulation_steps} = "
                                                  f"{last_step % accumulation_steps} ≠ 0", model.weight.grad])

table([[s, what, why, "grad=None" if g is None else str(g)[:24]] for s, what, why, g in log],
      headers=["step", "发生了什么", "触发条件", "更新后的 .grad"],
      title=f"iters={iters}, accumulation_steps={accumulation_steps} → "
            f"循环内更新 {iters // accumulation_steps} 次 + 尾巴 {1 if iters % accumulation_steps else 0} 次")
