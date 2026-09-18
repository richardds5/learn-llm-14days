# ---
# title: 逐行追踪 for expert 分发循环
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "给 build_model 加上 num_experts=8：`mask = ...` 那一行的 ×N 变成 8，但 `token_idx = ...` 那几行呢？（有的 expert 一个 token 都没分到）"
#   - "给 build_model 加上 num_experts_per_tok=2：mask 的 shape 变成什么？token_idx 的长度之和变成多少？"
# ---
import torch
from learnkit import *
from model.model_minimind import MOEFeedForward

model = build_model(num_hidden_layers=2, use_moe=True)   # E=4, k=1
B, T = 3, 7
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

# focus 到循环体：mask → token_idx → weight → index_add_ 写回
# 这几行每次 forward 都会被执行 E 次，网页上每行标着 ×N，用分页器可以逐次翻看每个 expert
with trace(model, fns=[MOEFeedForward.forward], dims=dict(B=B, T=T),
           focus=(163, 168), tree=False, max_calls=1,
           title=f"一次 forward 里，循环体被执行 E={model.config.num_experts} 次，"
                 f"每次的 token_idx 长度都不一样（B*T={B * T} 个 token 被瓜分）") as tr:
    model(input_ids)
