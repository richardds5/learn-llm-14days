# ---
# title: rotate_half 等于 D/2 个 2D 旋转
# timeout: 60
# tasks:
#   - "把手写公式里的 x1 * theta.sin() 改成 -x1 * theta.sin()（把旋转方向弄反）：最大差还会是 0 吗？大概是什么量级？"
#   - "把配对方式改成相邻两个下标：x1 = q[..., 0::2]、x2 = q[..., 1::2]，theta 不变。最大差变成多少？（这就是 GPT-J / Meta 原版的约定）"
# ---
import torch
from learnkit import *
from model.model_minimind import apply_rotary_pos_emb, precompute_freqs_cis

torch.manual_seed(0)
D, pos = 8, 3                                  # 👉 D 取 8 只是为了肉眼看清配对，真实模型是 96
x = torch.arange(1.0, D + 1)                   # [1,2,...,8]
rotate_half = lambda v: torch.cat((-v[..., D // 2:], v[..., : D // 2]), dim=-1)   # 和 L81 同一行

cos, sin = precompute_freqs_cis(dim=D, end=8, rope_base=1e6)
q = torch.randn(1, 1, 1, D)
q_rot, _ = apply_rotary_pos_emb(q, q, cos[pos:pos + 1], sin[pos:pos + 1])         # 仓库里的真实实现

theta = pos * (1.0 / (1e6 ** (torch.arange(0, D, 2)[: D // 2].float() / D)))      # [D/2] 个角度
x1, x2 = q[..., : D // 2], q[..., D // 2:]                                        # 👉 配对方式就写在这两行
manual = torch.cat([x1 * theta.cos() - x2 * theta.sin(),
                    x1 * theta.sin() + x2 * theta.cos()], dim=-1)

table([["x（当成 1~8 的编号）", str([int(v) for v in x.tolist()])],
       ["rotate_half(x)", str([int(v) for v in rotate_half(x).tolist()])],
       ["于是第 i 对通道是", str([(i, i + D // 2) for i in range(D // 2)])],
       ["手写 2D 旋转 vs apply_rotary_pos_emb 的最大差", f"{(manual - q_rot).abs().max().item():.2e}"]],
      headers=["检查项", "结果"], title=f"位置 pos={pos}：rotate_half 一次做掉 D/2={D // 2} 个 2D 旋转")
