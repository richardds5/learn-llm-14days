# ---
# title: MOEFeedForward.forward 的前两行：[B,T,C] → [B*T,C]
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 B, T 改成 5, 11：x_flat 的第一维变成多少？最后一维 C 变吗？"
#   - "把 focus=(157, 158) 改成 focus=(157, 161)：多看到哪几行？（focus 只决定网页上默认展开哪几行，不影响执行）"
# ---
import torch
from learnkit import *
from model.model_minimind import MOEFeedForward

# 默认 MoE 配置：num_experts=4 (E), num_experts_per_tok=1 (k)，和官方 full_sft_768_moe.pth 一致
model = build_model(num_hidden_layers=2, use_moe=True)

B, T = 3, 7                                  # 👉 改这里，看 x_flat 的第一维怎么跟着动
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

with trace(model, fns=[MOEFeedForward.forward], dims=dict(B=B, T=T),
           focus=(157, 158), tree=False, max_calls=1,
           title=f"batch_size/seq_len/hidden_dim 被拆出来，x 随即被拉平成 [B*T={B * T}, C]") as tr:
    model(input_ids)
