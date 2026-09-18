# ---
# title: 第 113 行：一份 x 喂给三个投影，出来两种宽度
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 num_key_value_heads 改成 1（MQA 的极端情况）：xk / xv 的最后一维变成多少？xq 跟着变吗？"
#   - "在 build_model 里加一个 head_dim=128：xq 的最后一维变成多少？它还等于 hidden_size 吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

# 👉 想看别的 GQA 配置，就改这一行（KV=4 是默认值）
model = build_model(num_hidden_layers=2, num_key_value_heads=4, flash_attn=False)
input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # B=3, T=7

# expand=True 把这一行的三个子表达式拆开，逐个标出输出宽度
with trace(model, fns=[Attention.forward], dims=dict(B=3, T=7),
           title="L113：q_proj / k_proj / v_proj 各自输出多宽",
           focus=(113, 113), expand=True, tree=False, max_calls=1):
    model(input_ids)
