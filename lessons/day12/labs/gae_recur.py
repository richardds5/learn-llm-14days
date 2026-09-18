# ---
# title: GAE 倒序递推：一条 T=7 的序列手算对照
# timeout: 60
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 lam 改成 1.0：advantage_t 是不是正好等于表格最后一列的 2.0 - V_t？"
#   - "把 lam 改成 0.0：advantage_t 退化成哪一列？为什么 t=0..5 全是 0.1？"
# ---
import torch
from learnkit import *

T = 7
gamma, lam = 1.0, 0.95                                                  # 👉 源码默认 gamma=1.0, lam=0.95
old_resp_values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])   # critic 的逐位置估值 [1, T]
token_rewards = torch.zeros(1, T); token_rewards[0, T - 1] = 2.0        # reward 只落在末位

# ===== train_ppo.py:140-147，逐行照抄 =====
lastgaelam = torch.zeros(1); advs_rev = []; rows = []
for t in reversed(range(T)):                                            # 👈 必须倒序
    nv = old_resp_values[:, t + 1] if t < T - 1 else 0.0                # 末位不 bootstrap
    delta = token_rewards[:, t] + gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    advs_rev.append(lastgaelam)
    rows.append([t, round(float(token_rewards[0, t]), 4), round(float(old_resp_values[0, t]), 4),
                 round(float(nv) if t < T - 1 else 0.0, 4), round(float(delta), 4),
                 round(float(lastgaelam), 4), round(2.0 - float(old_resp_values[0, t]), 4)])
advantages = torch.stack(advs_rev[::-1], dim=1)                         # [1, T]
returns = advantages + old_resp_values

table(list(reversed(rows)),
      headers=["t", "token_rewards[t]", "V_t", "V_{t+1}", "delta_t", "advantage_t", "对照：2.0 - V_t"],
      title=f"GAE 递推表（按 t 正序显示，实际计算顺序是从最后一行往上）gamma={gamma}, lam={lam}")
print("advantages =", [round(v, 4) for v in advantages[0].tolist()])
print("returns    =", [round(v, 4) for v in returns[0].tolist()], "← returns = advantages + old_resp_values")
