# ---
# title: 开关 YaRN 的真正条件：end / orig_max
# timeout: 60
# tasks:
#   - "再加一行 (4096, True)：end/orig_max = 2 > 1，表还和不开时一样吗？"
#   - "把 (1024, True) 那一行改成 (2049, True)：刚刚超过 orig_max=2048 一点点，YaRN 就生效了吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import MiniMindConfig, precompute_freqs_cis

def rope_table(max_pos, yarn):
    cfg = MiniMindConfig(max_position_embeddings=max_pos, inference_rope_scaling=yarn)
    cos, _ = precompute_freqs_cis(dim=cfg.head_dim, end=cfg.max_position_embeddings,     # 和 L205 一模一样
                                  rope_base=cfg.rope_theta, rope_scaling=cfg.rope_scaling)
    return cfg, cos

_, plain = rope_table(32768, False)          # 基准：完全不开 YaRN 的那张表
rows = []
for max_pos, yarn in ((32768, False), (32768, True), (1024, True)):   # 👉 加一行 (4096, True) 试试
    cfg, cos = rope_table(max_pos, yarn)
    rows.append([f"max_position_embeddings={max_pos}", str(yarn),
                 "None" if cfg.rope_scaling is None else cfg.rope_scaling["type"],
                 f"{cfg.max_position_embeddings / 2048:g}",
                 str(torch.equal(cos, plain[:cos.shape[0]])),
                 round(cos[1000, 24].item(), 4)])

table(rows, headers=["config", "inference_rope_scaling", "rope_scaling", "end / orig_max",
                     "整张表和不开 YaRN 时逐元素相同", "cos[1000, 24]（位置 1000 · 第 24 个分量）"],
      title="L69 的 if 只看 end（= config.max_position_embeddings），和输入多长无关")
