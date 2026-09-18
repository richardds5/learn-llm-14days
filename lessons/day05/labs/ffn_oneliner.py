# ---
# title: 把 FeedForward 的一行流拆开：C → I → C
# timeout: 60
# tasks:
#   - "把 build_model 改成 build_model(num_hidden_layers=2, hidden_size=512)：I 变成多少？中间那几个子表达式的 shape 跟着变了吗？"
#   - "把 x 换成 torch.randn(3, 7, 768) 之外的形状，比如 torch.randn(11, 768)（去掉 batch 维）：三个投影还能跑吗？输出是什么 shape？"
# ---
import torch
from learnkit import *
from model.model_minimind import FeedForward

model = build_model(num_hidden_layers=2)
ffn = model.model.layers[0].mlp          # 稠密路径下 mlp 就是 FeedForward
C = model.config.hidden_size
x = torch.randn(3, 7, C)                 # 假装是从 post_attention_layernorm 出来的那份副本

# expand=True 把 L146 那一行按求值顺序拆开，逐个子表达式标 shape
with trace(model, fns=[FeedForward.forward], dims=dict(B=3, T=7),
           tree=False, focus=(145, 146), expand=True,
           title="down_proj(act_fn(gate_proj(x)) * up_proj(x))：一行里的四步"):
    ffn(x)
