# ---
# title: 序列级 reward 落到哪一个 token 上
# timeout: 60
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 resp_lengths 改成 [11, 11, 11]：三个非零位置分别挪到哪里？"
#   - "改成平均分摊 token_rewards += (rewards / resp_lengths).unsqueeze(1) * resp_policy_mask：非零个数变成多少？"
# ---
import torch
from learnkit import *

B, R = 3, 11
resp_lengths = torch.tensor([11, 7, 4])          # 👉 第 2/3 条提前遇到 EOS
rewards = torch.tensor([2.0, -1.0, 0.5])         # [B]，calculate_rewards 给的序列级标量
resp_idx = torch.arange(R).unsqueeze(0)
resp_policy_mask = (resp_idx < resp_lengths.unsqueeze(1)).float()

# ===== train_ppo.py:136-138：只在最后一个有效 token 上 += rewards =====
token_rewards = torch.zeros(B, R)
last_idx = resp_lengths - 1                      # [B]
valid_resp = resp_lengths > 0
token_rewards[torch.arange(B)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]

heatmap(token_rewards, title="token_rewards [B, R]：每条序列只有最后一个有效 token 非零",
        xlabels=[f"t={i}" for i in range(R)], ylabels=[f"样本{i} (len={int(l)})" for i, l in enumerate(resp_lengths)])
print("非零位置 (行, 列) =", token_rewards.nonzero().tolist(), " = [样本, resp_lengths-1]")
print(f"非零个数 = {int((token_rewards != 0).sum())} / {B * R}；"
      f"而 resp_policy_mask 里的有效位有 {int(resp_policy_mask.sum())} 个")
print("每一行的和 =", [round(v, 4) for v in token_rewards.sum(1).tolist()], "= rewards 本身（没有被摊薄）")
