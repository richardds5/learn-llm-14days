# ---
# title: 四种 (load, scores) 组合下的 aux_loss
# timeout: 30
# tasks:
#   - "把 E 从 4 改成 8（其余不变）：『两边都均匀』那一行还等于 coef 吗？『两边都坍缩』那一行的倍数变成多少？"
#   - "把 collapse 里的 0.97 改成 0.4（其余 3 个 expert 平分 0.6）：『两边都坍缩』那一行的倍数掉到多少？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

E, coef, BT = 4, 5e-4, 21


def aux_loss(topk_idx, scores):                       # 照抄 model_minimind.py:172-173
    load = F.one_hot(topk_idx, E).float().mean(0)     # [k, E]
    return ((load * scores.mean(0)).sum() * E * coef).item()


uniform_idx = (torch.arange(BT) % E).unsqueeze(-1)    # 21 个 token 轮流分给 4 个 expert
collapse_idx = torch.zeros(BT, 1, dtype=torch.long)   # 全部挤到 expert 0
uniform_P = torch.full((BT, E), 1.0 / E)              # gate 对每个 expert 都给 0.25
collapse_P = torch.full((BT, E), 0.01); collapse_P[:, 0] = 0.97   # 👉 改这个 0.97 看惩罚强度

rows = []
for fname, idx in [("均匀 (轮流)", uniform_idx), ("坍缩 (全去 expert0)", collapse_idx)]:
    for pname, P in [("均匀 (每个 0.25)", uniform_P), ("坍缩 (expert0 拿 0.97)", collapse_P)]:
        a = aux_loss(idx, P)
        rows.append([fname, pname, f"{a:.6f}", f"{a / coef:.2f} × coef"])

table(rows, headers=["load f（谁真的拿到 token）", "scores 均值 P（gate 的信心）", "aux_loss", "相对 coef"],
      title=f"aux_loss = E·Σ f_i·P_i·coef  (E={E}, coef={coef}, B*T={BT})")
