# ---
# title: 第 132~133 行：把 H 个头拼回一条 [B,T,C] 的总线
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 L132 的 transpose(1, 2) 去掉（直接 output.reshape(bsz, seq_len, -1)）：报错了吗？shape 变了吗？（reshape 是按内存顺序读数的）"
#   - "加一个 head_dim=128：L132 之后的最后一维变成多少？L133 之后呢？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

model = build_model(num_hidden_layers=2, flash_attn=False)  # 👉 加 head_dim=128 看第二个 task
input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # B=3, T=7

with trace(model, fns=[Attention.forward], dims=dict(B=3, T=7),
           title="L132~L133：[B,H,T,D] → [B,T,H,D] → [B,T,H*D] → [B,T,C]",
           focus=(132, 133), expand=True, tree=False, max_calls=1):
    model(input_ids)
