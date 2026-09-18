# ---
# title: intermediate_size：π 倍 vs LLaMA 的 8/3 倍
# timeout: 60
# tasks:
#   - "把 RULE 里的 math.pi 改成 8/3，看 MiniMind 这一列会不会和 LLaMA 那一列完全重合（验证公式只差一个系数）"
#   - "加一行 C=4096（LLaMA-7B 的宽度）：两种规则算出来的 I 差多少个参数？（第 6 列是单层 FFN 的 3·C·I）"
# ---
import math
from learnkit import *
from model.model_minimind import MiniMindConfig

RULE = math.pi  # 👉 MiniMind 用的系数；LLaMA 系惯例是 8/3 ≈ 2.667

rows = []
for C in (512, 768, 1024, 2048):
    mine = math.ceil(C * RULE / 64) * 64          # 和源码 L26 同一个式子
    assert mine == MiniMindConfig(hidden_size=C).intermediate_size
    llama = math.ceil(C * 8 / 3 / 64) * 64        # LLaMA 惯例：8/3·C 再对齐到 64
    rows.append([C, f"{C * RULE / 64:.2f}", mine, llama, f"+{(mine / llama - 1) * 100:.1f}%", f"{3 * C * mine:,}"])

table(rows, headers=["C = hidden_size", "C·π/64", "MiniMind I = ceil(·)·64", "LLaMA 惯例 I", "MiniMind 宽了", "单层 FFN 参数 3·C·I"],
      title="ceil(C·π/64)·64：向上取整到 64 的倍数，比 8/3 惯例宽约 18%")
