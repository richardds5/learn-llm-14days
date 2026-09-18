# ---
# title: 逐行追踪 MiniMindModel.forward：start_pos 与 RoPE 切片
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "用调用切换器翻到第 2、3 次调用：start_pos 各是多少？position_embeddings 里两个 tensor 的第 0 维呢？"
#   - "把 generate 的 use_cache 改成 False：三次调用的 start_pos 分别变成什么？RoPE 切出来的段还对得上真实位置吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import MiniMindModel

model = build_model(num_hidden_layers=1)
B, T = 3, 11
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

# focus 按变量名收窄：只看 start_pos 和它切出来的 position_embeddings
with trace(model, fns=[MiniMindModel.forward], dims=dict(B=B, T=T), tree=False,
           focus=["start_pos", "position_embeddings"], max_calls=3,
           title="start_pos = cache 里已有多少个位置；RoPE 表就从这里开始切"):
    model.generate(input_ids, max_new_tokens=3, do_sample=False, eos_token_id=None, use_cache=True)
