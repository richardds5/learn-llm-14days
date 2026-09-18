# ---
# title: 48 个频率分量各自转一圈要走多少个位置
# timeout: 60
# tasks:
#   - "把 rope_theta=1e4 那条曲线的底数改成 1e2：最低频那一端掉到多少？还够不够覆盖 32768？"
#   - "把 dim 从 96 改成 48（相当于 num_attention_heads=16）：曲线的终点（i 最大那一端）变高还是变低？"
# ---
import math
import torch
from learnkit import *

dim = 96                                           # 👉 head_dim，真实模型是 768 // 8 = 96
i = torch.arange(0, dim, 2)[: dim // 2].float()    # 0, 2, 4, ..., 94 —— 和 L63 里的一模一样
wave = lambda base: 2 * math.pi * (base ** (i / dim))   # 波长 = 2π / freq_i = 2π · base^(2i/D)

plot({"rope_theta=1e6（MiniMind）": wave(1e6),
      "rope_theta=1e4（Llama2 惯例）": wave(1e4),
      "max_position_embeddings=32768": torch.full((dim // 2,), 32768.0)},
     title="每个频率分量转满一圈需要多少个位置（波长，纵轴对数）",
     xlabel="频率分量下标 i（0 = 最高频）", ylabel="波长（位置数）", logy=True)

for base in (1e6, 1e4):
    w = wave(base)
    print(f"rope_theta={base:.0e}:  i=0 波长 {w[0]:.2f}   i=47 波长 {w[-1]:,.0f}   "
          f"波长 > 32768 的分量有 {(w > 32768).sum().item()} / {dim // 2} 个")
