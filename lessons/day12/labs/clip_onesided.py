# ---
# title: clip 目标在哪一侧截断了梯度
# timeout: 60
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 clip_epsilon 从 0.2 改成 0.5：整张表里还有梯度被削成 0 的行吗？「计入 clipfrac」那一列呢？"
#   - "把 A 的取值从 ±1 改成 ±3：梯度量级怎么变？两个平台的位置变了吗？"
# ---
import torch
from learnkit import *

clip_epsilon = 0.2                      # 👉 源码默认 args.clip_epsilon=0.2


def per_token_policy_loss(ratio, A):    # train_ppo.py:210-212 里的逐 token 项（是 loss，越小越好）
    return torch.max(-A * ratio, -A * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon))


rows = []
for r in [0.5, 0.79, 0.81, 1.0, 1.19, 1.21, 1.5]:
    cell = []
    for A in (1.0, -1.0):
        rt = torch.tensor(r, requires_grad=True)
        loss = per_token_policy_loss(rt, A)
        loss.backward()
        cell += [round(loss.item(), 4), round(rt.grad.item(), 4)]
    rows.append([r, cell[0], cell[1], cell[2], cell[3], bool(abs(r - 1.0) > clip_epsilon)])
table(rows, headers=["ratio", "loss (A=+1)", "d loss/d ratio (A=+1)", "loss (A=-1)", "d loss/d ratio (A=-1)",
                     "计入 clipfrac"],
      title=f"clip 是单边的：A>0 和 A<0 被削平的区间正好相反（clip_epsilon={clip_epsilon}）")
print("A=+1 时梯度为 0 的区间是 ratio > 1.2；A=-1 时是 ratio < 0.8；"
      "而 clipfrac 用的是对称判据 |ratio-1| > 0.2，两者在表里错开了两行。")
