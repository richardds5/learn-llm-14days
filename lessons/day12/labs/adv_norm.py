# ---
# title: advantage 的全局归一化与 returns 的算账顺序
# timeout: 60
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把归一化改成 per-sequence（对每一行各自算 mean/var）：表里「本行均值(归一化后)」还都不是 0 吗？"
#   - "把 returns = advantages + old_resp_values 这一行挪到归一化三行的后面：最后那个 allclose 变成什么？"
# ---
import torch
from learnkit import *

B, R = 3, 11
resp_lengths = torch.tensor([11, 7, 4])
resp_idx = torch.arange(R).unsqueeze(0)
resp_policy_mask = (resp_idx < resp_lengths.unsqueeze(1)).float()
torch.manual_seed(0)
old_resp_values = torch.linspace(0.1, 0.7, R).repeat(B, 1) * resp_policy_mask
advantages = (torch.randn(B, R) * 0.8 + 1.2) * resp_policy_mask       # 假装 GAE 已经跑完
returns = advantages + old_resp_values                                # 👈 train_ppo.py:147，在归一化之前

# ===== train_ppo.py:149-151：跨整个 batch 的有效 token 算一个标量 mean/var =====
adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
adv_norm = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

row_mean = lambda a, i: float((a[i] * resp_policy_mask[i]).sum() / resp_policy_mask[i].sum())
table([[i, int(resp_lengths[i]), round(row_mean(advantages, i), 4), round(row_mean(adv_norm, i), 4),
        round(float(returns[i, 0]), 4), round(float((adv_norm + old_resp_values)[i, 0]), 4)] for i in range(B)],
      headers=["样本", "有效长度", "本行均值（归一化前）", "本行均值（归一化后）", "returns[i,0]（源码）",
               "若用归一化后的 adv 重算 returns"],
      title="归一化是跨 batch 全局的；returns 用的是归一化之前的 advantage")
print(f"全局 adv_mean = {adv_mean.item():.4f}，adv_std = {adv_var.sqrt().item():.4f}；"
      f"归一化后的全局均值 = {((adv_norm * resp_policy_mask).sum() / resp_policy_mask.sum()).item():.2e}")
print("returns - old_resp_values 等于归一化后的 advantages 吗：",
      bool(torch.allclose(returns - old_resp_values, adv_norm)))
