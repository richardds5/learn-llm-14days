# ---
# title: 点积热力图：同一条对角线上的格子颜色一样
# timeout: 60
# tasks:
#   - "把 q = torch.randn(...).expand(...) 换成 q = torch.randn(1, N, 1, D)（每个位置换一份新内容）：对角线条纹还在吗？说明这条性质要求的是什么？"
#   - "给 precompute_freqs_cis 加上 rope_scaling=dict(beta_fast=32, beta_slow=1, factor=16, original_max_position_embeddings=16, attention_factor=1.0)（orig_max 要小于 end=48 才会触发 YaRN）：条纹会消失吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import apply_rotary_pos_emb, precompute_freqs_cis

torch.manual_seed(0)
N, D = 48, 96
cos, sin = precompute_freqs_cis(dim=D, end=N, rope_base=1e6)
q = torch.randn(1, 1, 1, D).expand(1, N, 1, D)   # 👉 同一个内容向量，摆在 0..N-1 每一个位置上
k = torch.randn(1, 1, 1, D).expand(1, N, 1, D)
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
dots = q_rot[0, :, 0] @ k_rot[0, :, 0].T          # dots[m, n] = <R_m q, R_n k>

heatmap(dots, title="dots[m, n] = ⟨R_m q, R_n k⟩：行 = q 的位置 m，列 = k 的位置 n",
        xlabels=[f"n={n}" if n % 8 == 0 else "" for n in range(N)],
        ylabels=[f"m={m}" if m % 8 == 0 else "" for m in range(N)])

print(f"同一条对角线（m-n = -15）：dots[5,20] = {dots[5, 20]:.5f}   dots[25,40] = {dots[25, 40]:.5f}"
      f"   差 {abs(dots[5, 20] - dots[25, 40]):.2e}")
print(f"隔壁一格（m-n = -16）：   dots[5,21] = {dots[5, 21]:.5f}   "
      f"和上面差 {abs(dots[5, 20] - dots[5, 21]):.4f}  ← 相对距离一变，值就变")
