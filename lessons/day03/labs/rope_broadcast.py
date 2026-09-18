# ---
# title: 拆开 L82：[T,D] 的 cos 怎么乘到 [B,T,H,D] 的 q 上
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "在源码标签页把 L82 的 unsqueeze(unsqueeze_dim) 改成 unsqueeze(2)（HF Llama 那种 [B,H,T,D] 布局下的写法）：cos 变成什么 shape？和 q 广播时在哪一维对不齐、报什么错？"
#   - "换成 HF Llama 的布局再跑：apply_rotary_pos_emb(q.transpose(1,2), k.transpose(1,2), cos[None], sin[None])。cos.unsqueeze(1) 这次变成几维？q_embed 的 shape 变成什么？"
# ---
import torch
from learnkit import *
from model.model_minimind import apply_rotary_pos_emb, precompute_freqs_cis

B, T, H, KV, D = 3, 7, 8, 4, 96
q = torch.randn(B, T, H, D)     # Attention.forward L114 之后的布局：还没 transpose(1,2)
k = torch.randn(B, T, KV, D)    # GQA：k 只有 KV=4 个头，照样广播
cos, sin = precompute_freqs_cis(dim=D, end=64, rope_base=1e6)
cos, sin = cos[:T], sin[:T]     # L219 从大表里切出来的就是这个 [T, D]，没有 batch 维

with trace(fns=[apply_rotary_pos_emb], dims=dict(B=B, T=T, H=H, KV=KV, D=D), tree=False,
           focus=(82, 82), expand=True,
           title="L82：q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)"):
    apply_rotary_pos_emb(q, k, cos, sin)
