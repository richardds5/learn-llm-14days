# ---
# title: freqs_cos 热力图：左右两半是同一张图
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "在源码标签页把 L76 改成 `freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1) * attn_factor`（GPT-J 那种交错约定）：热力图的条纹变成什么样？「左半 == 右半」那一行变成什么？"
#   - "把 end 从 64 改成 512：最右边那几列（最低频）在图上还看得出明暗变化吗？对照上一节的波长表想想为什么。"
# ---
import torch
from learnkit import *
from model.model_minimind import precompute_freqs_cis

dim, end = 32, 64                       # 👉 画图用的小表；真实模型是 dim=96, end=32768
cos_small, _ = precompute_freqs_cis(dim=dim, end=end, rope_base=1e6)

heatmap(cos_small, title=f"freqs_cos：行 = 位置 t（0~{end - 1}），列 = 通道 j（0~{dim - 1}）",
        xlabels=[f"j={j}" if j % 4 == 0 else "" for j in range(dim)],
        ylabels=[f"t={t}" if t % 8 == 0 else "" for t in range(end)])

# 真实尺寸的那张表：cat([cos, cos]) 的直接后果是「第 j 列 == 第 j + D/2 列」
cos, sin = precompute_freqs_cis(dim=96, end=32768, rope_base=1e6)
print("freqs_cos.shape =", list(cos.shape), " = [max_position_embeddings, head_dim]",
      f"  fp32 占 {cos.numel() * 4 / 1e6:.1f} MB")
print("左半 == 右半 ?", torch.equal(cos[:, :48], cos[:, 48:]),
      "  ← torch.cat([cos, cos], dim=-1) 的直接后果")
print("第 0 行全是 cos(0)=1 吗 ?", torch.equal(cos[0], torch.ones(96)),
      "  ← 位置 0 不旋转；model_minimind.py:216 就是靠 freqs_cos[0,0] 判断 buffer 有没有丢")
