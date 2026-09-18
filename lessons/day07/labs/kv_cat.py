# ---
# title: 逐行追踪 Attention.forward：xk 的 T 维在 prefill 和 decode 里各是多少
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "用 trace 面板上的调用切换器翻到第 2、3 次调用（decode）：`torch.cat` 那一行之后 xk 的第 1 维是多少？和第 1 次调用（prefill）差几？"
#   - "把 generate 的 use_cache 改成 False：第 2 次调用里 `torch.cat` 那一行还会执行吗？xk 的 T 维变成什么？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

model = build_model(num_hidden_layers=1)       # 只留 1 层：Attention.forward 的调用次数 = generate 的步数
B, T = 3, 11                                    # T=11，这样 11/12/13 都不会和 H=8、KV=4 撞符号
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

# focus 只看 RoPE → cat → past_kv → repeat_kv 这几行；max_calls=3 = 记录 prefill + 前两次 decode
with trace(model, fns=[Attention.forward], dims=dict(B=B, T=T), tree=False,
           focus=(119, 124), max_calls=3,
           title="第 1 次调用 = prefill，第 2、3 次 = decode（用上方的调用切换器翻页）"):
    model.generate(input_ids, max_new_tokens=3, do_sample=False, eos_token_id=None, use_cache=True)
