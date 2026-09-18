# ---
# title: 两种配对约定之间差的那个 permutation
# timeout: 60
# tasks:
#   - "把 D 从 8 改成 6（3 对频率）：perm 变成什么？两行结论变吗？"
#   - "把 perm 改成 torch.arange(D)（什么都不重排）再跑：第二行的最大差会回到第一行的量级吗？"
# ---
import torch
from learnkit import *

torch.manual_seed(0)
D, pos = 8, 3
x = torch.randn(D)
angles = pos * (1.0 / (10000 ** (torch.arange(0, D, 2).float() / D)))   # [D/2] 个角度

# 约定 A：MiniMind / HF —— 配对 (x[i], x[i+D/2])，用 rotate_half
rotate_half = lambda v: torch.cat((-v[D // 2:], v[: D // 2]))
cos, sin = torch.cat([angles.cos()] * 2), torch.cat([angles.sin()] * 2)
rope_A = lambda v: v * cos + rotate_half(v) * sin

# 约定 B：Meta 原版 llama —— 配对 (x[2i], x[2i+1])，用复数乘法
rope_B = lambda v: torch.view_as_real(torch.view_as_complex(v.view(-1, 2)) *
                                      torch.polar(torch.ones_like(angles), angles)).flatten()

perm = torch.cat([torch.arange(0, D, 2), torch.arange(1, D, 2)])   # 👉 偶数下标搬前半，奇数下标搬后半
bridged = rope_A(x[perm])[torch.argsort(perm)]                     # 先重排通道，再用约定 A，最后换回原顺序

table([["直接比：rope_A(x) vs rope_B(x)", f"{(rope_A(x) - rope_B(x)).abs().max():.3f}", "两种约定的输出不一样"],
       [f"先按 perm={perm.tolist()} 重排再比", f"{(bridged - rope_B(x)).abs().max():.2e}", "完全一致"]],
      headers=["比较方式", "最大绝对差", "结论"], title="rotate_half 约定 ↔ 复数相邻配对约定：只差一次通道重排")
