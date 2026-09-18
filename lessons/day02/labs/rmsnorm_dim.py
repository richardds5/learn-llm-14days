# ---
# title: 把 RMSNorm 那一行拆开：mean 沿最后一维，keepdim 撑出那个 1
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "在源码标签页把 L57 的 keepdim=True 改成 False 再跑：报错发生在 norm 还是 forward？错误信息里的 768 和 7 分别是谁？"
#   - "把 rn 和 x 换成 Attention 里 q_norm 看到的样子：rn = RMSNorm(96)、x = torch.randn(3, 7, 8, 96)。拆开后 x.pow(2).mean(-1, keepdim=True) 那一步变成什么 shape？"
# ---
import torch
from learnkit import *
from model.model_minimind import RMSNorm

rn = RMSNorm(768, eps=1e-6)           # 和 model.norm / input_layernorm 同一个类，dim = C = 768
x = torch.randn(3, 7, 768)             # B=3, T=7, C=768

# expand=True 会把 L56-57 这一行流按求值顺序拆开，逐个子表达式标出 shape
with trace(fns=[RMSNorm.forward, RMSNorm.norm], dims=dict(B=3, T=7, C=768),
           tree=False, focus=(56, 57), expand=True,
           title="RMSNorm.norm：[B,T,C] 怎么先被压成 [B,T,1] 再广播回去"):
    rn(x)
