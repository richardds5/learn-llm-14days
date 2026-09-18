# ---
# title: 组内归一化：view(-1, G) 之后在 dim=1 上求 mean/std
# timeout: 60
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 1e-4 改成 0 再跑：g1 那一组的 advantage 变成什么？这个值会怎么顺着 loss 传染出去？"
#   - "把 unbiased=False 改成 True：g0 组 advantage 的绝对值变大还是变小？（G=4 时差一个 sqrt(4/3)≈1.1547 的因子）"
# ---
import torch
from learnkit import *

B, G = 3, 4  # 3 个 prompt，每个采 4 条 → 12 条轨迹；下标 k = i*G + j

rewards = torch.tensor([
    2.0, 0.5, -1.0, 1.5,   # g0：有高有低，学习信号充足
    1.0, 1.0, 1.0, 1.0,    # g1：四条一样好
    3.0, -0.2, -0.1, 0.0,  # g2：一枝独秀
])  # [B*G]

# ↓↓↓ 下面 4 行与 trainer/train_grpo.py:122-125 完全一致 ↓↓↓
grouped_rewards = rewards.view(-1, G)                                    # [B, G] —— 一行 = 一组
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(G)                # [B*G]
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(G)  # [B*G]
advantages = (rewards - mean_r) / (std_r + 1e-4)                         # 👉 1e-4 在这里

show(grouped_rewards=grouped_rewards, std_r=std_r, advantages=advantages)
table([[k, k // G, round(rewards[k].item(), 2), round(mean_r[k].item(), 4), round(std_r[k].item(), 4),
        round(advantages[k].item(), 4)] for k in range(B * G)],
      headers=["下标 k", "组 i", "reward", "mean_r（组内）", "std_r（组内）", "advantage"],
      title="每条轨迹的组内相对优势：baseline 只和同组的 G-1 个兄弟有关")
note(f"每组 advantage 之和 = {[round(v, 4) for v in advantages.view(B, G).sum(1).tolist()]}，"
     f"恒等于 0 —— 这就是「相对」两个字的含义：组内有人被抬，必然有人被压。")
