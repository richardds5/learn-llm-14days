# ---
# title: 逐行看一个 MiniMindBlock：存一份、归一化一份、加回去
# timeout: 60
# tasks:
#   - "把 num_hidden_layers 改成 3：trace 抬头的『共调用 N 次』变成几？（这就是 MiniMindModel.forward 那个 for 循环）"
#   - "把 T 从 7 改成 13（input_ids 和 dims 一起改）：哪些格子跟着变了？带 C 的那些呢？"
# ---
import torch
from learnkit import *
from model.model_minimind import MiniMindBlock

B, T = 3, 7
model = build_model(num_hidden_layers=2)
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

# focus 到 forward 的函数体：residual / self_attn / += / mlp 这四拍
# （注意 L192 是 in-place，不能用 expand 去重放它）
with trace(model, fns=[MiniMindBlock.forward], dims=dict(B=B, T=T),
           tree=False, focus=(186, 194), max_calls=1,
           title="MiniMindBlock.forward：进出都是 [B,T,C]，中间读写了两次总线"):
    model(input_ids)
