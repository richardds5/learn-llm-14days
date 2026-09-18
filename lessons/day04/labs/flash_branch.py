# ---
# title: 同一个 flash_attn=True 的模型，连着两次调用
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 flash_attn 改成 False 再跑：#1 这次调用里，125~131 之间亮着的是哪几行？"
#   - "给第一次调用加上 attention_mask=torch.ones(3, 11)，再把它的 [0, 0] 改成 0：#1 亮的行换了吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

model = build_model(num_hidden_layers=1, flash_attn=True)  # 👉 config 默认就是 True
V = model.config.vocab_size
prefill, step = torch.randint(0, V, (3, 11)), torch.randint(0, V, (3, 1))

# 网页上用 #1 / #2 两个按钮切换这两次调用；没被执行的行是暗的
with trace(model, fns=[Attention.forward], dims=dict(B=3, T=11, Tkv=12),
           title="#1 一次性喂 11 个 token；#2 带着 cache 只喂 1 个新 token",
           focus=(125, 131), tree=False, max_calls=2):
    out = model(prefill, use_cache=True)
    model(step, past_key_values=out.past_key_values, use_cache=True)
