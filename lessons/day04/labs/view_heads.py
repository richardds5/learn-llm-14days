# ---
# title: 第 114~116 行：一次 view 把最后一维拆成 (头数, head_dim)
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 num_key_value_heads 改成 2：第 115/116 行的第三维变成几？xq 那一行变吗？"
#   - "把 (3, 7) 改成 (3, 11)，dims 里的 T 也一起改成 11：三行里哪几个数字跟着变了？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

# 👉 KV 头数是这一节的主角：改它，只有 xk / xv 那两行会变
model = build_model(num_hidden_layers=2, num_key_value_heads=4, flash_attn=False)
input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # B=3, T=7

with trace(model, fns=[Attention.forward], dims=dict(B=3, T=7),
           title="L114~L116：[B,T,H*D] → [B,T,H,D]，[B,T,KV*D] → [B,T,KV,D]",
           focus=(114, 116), tree=False, max_calls=1):
    model(input_ids)
