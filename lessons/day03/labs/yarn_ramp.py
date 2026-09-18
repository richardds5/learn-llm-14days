# ---
# title: YaRN 的 ramp：哪些频率不动、哪些除以 factor
# timeout: 60
# tasks:
#   - "把 beta_fast 从 32 改成 8：low 变大还是变小？完全不缩放的那一段（γ=0）变宽还是变窄？"
#   - "把 factor 从 16 改成 4：low/high 变吗？曲线 f'/f 的右端落到多少？"
# ---
import math
import torch
from learnkit import *

dim, base = 96, 1e6                                              # head_dim 和 rope_theta
orig_max, factor, beta_fast, beta_slow = 2048, 16, 32.0, 1.0     # 👉 MiniMindConfig L32-L39 写死的那一组

inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(base))   # L70
low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
ramp = torch.clamp((torch.arange(dim // 2).float() - low) / max(high - low, 0.001), 0, 1)   # L72

plot({"γ = ramp（0 = 完全不动，1 = 完全压缩）": ramp,
      "f'(i) / f(i) = 1 - γ + γ/factor": 1 - ramp + ramp / factor},
     title=f"YaRN 的分段缩放：low={low}, high={high}, factor={factor}",
     xlabel="频率分量下标 i（0 = 最高频 / 最短波长）")

print(f"inv_dim(beta_fast={beta_fast:.0f}) = {inv_dim(beta_fast):.3f}  → low  = floor(...) = {low}")
print(f"inv_dim(beta_slow={beta_slow:.0f})  = {inv_dim(beta_slow):.3f} → high = ceil(...)  = {high}")
print(f"γ=0（频率原封不动）的分量: {int((ramp == 0).sum())} 个  |  "
      f"0<γ<1（线性过渡）: {int(((ramp > 0) & (ramp < 1)).sum())} 个  |  "
      f"γ=1（频率除以 {factor}，波长 ×{factor}）: {int((ramp == 1).sum())} 个")
