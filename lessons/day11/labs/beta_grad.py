# ---
# title: beta 怎么改变 loss 曲线，以及 margin=0 处的梯度
# timeout: 120
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "在 BETAS 里加一个 3.0：曲线在 margin > 0 那一侧塌得多快？beta 越大是不是越早「学饱」？"
#   - "把 mask[:, 5:] 改成 mask[:, 3:]（每条 10 个有效 token）：per-token 梯度变了吗？整条序列拿到的梯度总量呢？"
# ---
import math

import torch
import torch.nn.functional as F

from learnkit import *
from trainer.train_dpo import dpo_loss

BETAS = [0.1, 0.15, 1.0]                        # 👉 仓库默认是 0.15
margin = torch.linspace(-10, 10, 201)           # 源码里那个也叫 logits 的量：pi_logratios - ref_logratios
plot({f"beta={b}": (-F.logsigmoid(b * margin)).tolist() for b in BETAS},
     title=f"-logsigmoid(beta · margin)：三条曲线在 margin=0 处都等于 ln2 = {math.log(2):.4f}",
     xlabel="margin（从 -10 到 +10）", ylabel="loss")

N, T, BETA = 6, 13, 0.1                         # 👉 3 对样本
torch.manual_seed(0)
ref_lp = torch.randn(N, T)
mask = torch.zeros(N, T, dtype=torch.long); mask[:, 5:] = 1      # 每条 8 个有效 token
policy_lp = ref_lp.clone().requires_grad_(True)                  # policy 初始 == ref
dpo_loss(ref_lp, policy_lp, mask, beta=BETA).backward()
g, n_pairs = policy_lp.grad, N // 2

print(f"beta={BETA}, {n_pairs} 对：理论上 per-token 梯度 = ∓beta/(2·n) = ∓{BETA / 2 / n_pairs:.6f}")
print(f"  chosen  行 mask=1 处：{g[0, 5].item():+.6f}   同一行 mask=0 处：{abs(g[0, 0].item()):.6f}")
print(f"  rejected 行 mask=1 处：{g[n_pairs, 5].item():+.6f}   整条 chosen 行梯度之和：{g[0].sum().item():+.6f}")
print(f"  每一行的梯度都一样大吗：{bool((g[0, 5:] == g[0, 5]).all())}")
