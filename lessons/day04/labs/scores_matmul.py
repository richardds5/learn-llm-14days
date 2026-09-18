# ---
# title: 第 128 行：q @ k^T 把 D 消掉，换来一张 T×T_kv 的表
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把输入改成 (3, 11)、dims 里的 T 也改成 11：scores 的后两维变成什么？它和 D=96 有关系吗？"
#   - "把 head_dim 改成 48（build_model(..., head_dim=48)）：scores 的 shape 变吗？除的那个 sqrt 变吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

model = build_model(num_hidden_layers=2, flash_attn=False)  # 👉 手写分支才有 scores 这个局部变量
input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # B=3, T=7

# 这一行只是 transpose + matmul + 除法，没有副作用，可以 expand
with trace(model, fns=[Attention.forward], dims=dict(B=3, T=7),
           title="L128：[B,H,T,D] @ [B,H,D,T_kv] → [B,H,T,T_kv]",
           focus=(128, 128), expand=True, tree=False, max_calls=1):
    model(input_ids)
