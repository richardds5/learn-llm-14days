# ---
# title: clip_grad_norm_ 前后的全局梯度范数
# timeout: 90
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把 grad_clip 从 1.0 改成 100：两条曲线是不是完全重合？标题里说的裁剪次数变成几次？"
#   - "把 lr 从 1e-2 改成 1e-4（模型几乎学不动）：裁剪前的范数还会掉到阈值以下吗？20 步里被裁几步？"
# ---
import torch
from learnkit import *

grad_clip = 1.0                          # 👉 真实脚本的默认值就是 1.0
model = build_model(num_hidden_layers=2, hidden_size=128).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
input_ids = torch.randint(0, 6400, (3, 11))                # B=3, T=11，20 步都用同一个 batch

pre, post = [], []
for step in range(1, 21):
    res = model(input_ids, labels=input_ids.clone())
    (res.loss + res.aux_loss).backward()
    # clip_grad_norm_ 的返回值 = 裁剪「前」把所有 .grad 拼成一个长向量算出来的 L2 范数
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 裁剪是原地改 .grad，所以这里再算一遍就是裁剪「后」的范数
    after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    pre.append(total_norm.item()); post.append(after.item())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)

clipped = sum(p > grad_clip for p in pre)
plot({"裁剪前（clip_grad_norm_ 的返回值）": pre, "裁剪后（重新算一遍）": post},
     x=list(range(1, 21)), title=f"grad_clip={grad_clip}：20 步里有 {clipped} 步超过阈值被裁剪"
                                 f"（这些步裁剪后恰好等于 {grad_clip}），其余 {20 - clipped} 步两条曲线重合",
     xlabel="step", ylabel="全局梯度 L2 范数")
print("裁剪前:", [f"{x:.3f}" for x in pre[:6]])
print("裁剪后:", [f"{x:.3f}" for x in post[:6]])
