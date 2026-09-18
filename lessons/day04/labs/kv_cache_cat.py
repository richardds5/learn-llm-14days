# ---
# title: 第 120~123 行：新算的 KV 接在旧 cache 后面
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把续写的 5 个 token 改成 1 个（generate 的单步 decode 就是这样）：第 121 行之后 xk 的第二维变成多少？"
#   - "把第二次 forward 的 use_cache 改成 False：第 123 行的 past_kv 变成什么？xk / xv 还拼接吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

model = build_model(num_hidden_layers=2, flash_attn=False)
V = model.config.vocab_size
out = model(torch.randint(0, V, (3, 7)), use_cache=True)   # 先 prefill 7 个 token，建立 cache
new = torch.randint(0, V, (3, 5))                          # 👉 再一次性续写 5 个新 token

with trace(model, fns=[Attention.forward], dims=dict(B=3, T=5, Tkv=12),
           title="带 cache 的第二次调用：T=5 个新 token，T_kv=7+5=12",
           focus=(120, 123), tree=False, max_calls=1):
    model(new, past_key_values=out.past_key_values, use_cache=True)
