# ---
# title: 组内归一化：四种 reward 分布分别算出什么 advantage
# timeout: 60
# sources:
#   - trainer/train_agent.py
# tasks:
#   - "把 ② 改成 [3.0, 3.0, 3.0, 2.999]：GrpStd 变成多少？advantages 从全 0 变成什么？（注意分母里那个 +1e-4 在这里挡住了多大的爆炸）"
#   - "把 G 改成 2、GROUPS 里每条只留前两个数：有梯度的那几组 advantages 是不是都变成了 ±1"
# ---
import torch
from learnkit import *

G = 4                                              # 👉 --num_generations
GROUPS = [
    ("① 有对有错", [3.0, 0.5, -1.0, 0.5]),
    ("② 全对（模型太确定）", [3.0, 3.0, 3.0, 3.0]),
    ("③ 全错", [-1.5, -1.5, -1.5, -1.5]),
    ("④ 只有一条对", [3.0, -1.0, -1.0, -1.0]),
]
rows = []
for tag, rw in GROUPS:
    rewards = torch.tensor(rw)                      # [B*G]，这里 B=1
    grouped_rewards = rewards.view(-1, G)           # [B, G]
    mean_r = grouped_rewards.mean(dim=1).repeat_interleave(G)
    std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(G)
    advantages = (rewards - mean_r) / (std_r + 1e-4)
    rows.append([tag, rw, round(mean_r[0].item(), 3), round(std_r[0].item(), 4),
                 [round(v, 3) for v in advantages.tolist()],
                 "没有梯度" if advantages.abs().max() < 1e-3 else "有梯度"])
table(rows, headers=["组内 reward 分布", "rewards [B·G]", "GrpMean", "GrpStd", "advantages [B·G]", "这一步"],
      title="advantage = (自己的 reward - 组内均值) / (组内标准差 + 1e-4)")

# ---- 排列顺序：rollout_batch 是「样本0 的 G 条、样本1 的 G 条…」，所以只能 repeat_interleave ----
B = 2
rewards = torch.tensor([3.0, 0.5, -1.0, 1.0])       # 样本0 的两条、样本1 的两条
grouped_rewards = rewards.view(B, 2)
ri = grouped_rewards.mean(dim=1).repeat_interleave(2)
rp = grouped_rewards.mean(dim=1).repeat(2)
print(f"grouped_rewards = {grouped_rewards.tolist()}  → 每组均值 {grouped_rewards.mean(dim=1).tolist()}")
print(f"repeat_interleave(2) = {ri.tolist()}  ← 和 rewards 的排列对齐")
print(f"         .repeat(2)  = {rp.tolist()}  ← shape 一样是 [4]，不报错，但每条减错了别人的均值")
print(f"两者算出的 advantages: {[round(v,3) for v in ((rewards-ri)/(grouped_rewards.std(1,unbiased=False).repeat_interleave(2)+1e-4)).tolist()]}"
      f" vs {[round(v,3) for v in ((rewards-rp)/(grouped_rewards.std(1,unbiased=False).repeat_interleave(2)+1e-4)).tolist()]}")
