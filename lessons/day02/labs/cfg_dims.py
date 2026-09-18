# ---
# title: 七个符号：config 里哪几个数字决定了所有 tensor 的 shape
# timeout: 60
# tasks:
#   - "把第二列的 MiniMindConfig(hidden_size=512) 改成 MiniMindConfig(hidden_size=512, head_dim=96)：D 那一行变回 96，其余行变吗？（head_dim 是可以和 C/H 解耦的）"
#   - "再加一列 MiniMindConfig(num_attention_heads=12)：D 变成多少？C 变吗？"
# ---
from learnkit import *
from model.model_minimind import MiniMindConfig

# 👉 想看别的形状，就改这一行
a, b = MiniMindConfig(), MiniMindConfig(hidden_size=512)

SYMS = [
    ("C", "hidden_size", "残差流宽度：hidden_states 永远是 [B, T, C]"),
    ("L", "num_hidden_layers", "MiniMindBlock 堆几层"),
    ("H", "num_attention_heads", "query 头数"),
    ("KV", "num_key_value_heads", "key/value 头数 (GQA)"),
    ("D", "head_dim", "每个头的宽度，默认 = C // H"),
    ("I", "intermediate_size", "FFN 中间层宽度，默认 = ceil(C·π/64)·64"),
    ("V", "vocab_size", "词表大小：embed_tokens/lm_head 的那一维"),
]
table([[sym, attr, getattr(a, attr), getattr(b, attr), desc] for sym, attr, desc in SYMS],
      headers=["符号", "config 字段", "MiniMindConfig()", "MiniMindConfig(hidden_size=512)", "含义"],
      title="七个数字 → 全模型的 shape（只有 C 被改了，D 和 I 跟着变）")
