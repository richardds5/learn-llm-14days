# ---
# title: 第 124 行：一行里同时完成 repeat_kv 和 transpose
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 num_key_value_heads 改成 1：repeat_kv 那两个子表达式的输入/输出头数分别是多少？xq 那一支变吗？"
#   - "把 num_key_value_heads 改成 8：n_rep 变成 1，repeat_kv 直接 return x——子表达式的 shape 还会变吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

model = build_model(num_hidden_layers=2, num_key_value_heads=4, flash_attn=False)  # 👉
input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # B=3, T=7

# 这一行没有任何副作用，可以放心 expand：先 repeat_kv 再 transpose，顺序看得一清二楚
with trace(model, fns=[Attention.forward], dims=dict(B=3, T=7),
           title="L124：三个变量在同一行里被对齐到 [B, H, T, D]",
           focus=(124, 124), expand=True, tree=False, max_calls=1):
    model(input_ids)
