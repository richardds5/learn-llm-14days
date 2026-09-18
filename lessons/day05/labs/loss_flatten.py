# ---
# title: 切片之后为什么必须 .contiguous() 才能 .view
# timeout: 60
# tasks:
#   - "把切片换成在最后一维上切 `logits[..., :-1]`（切 V 而不是 T）：is_contiguous() 还是 False 吗？"
#   - "把 B 改成 1：`logits[..., :-1, :]` 的 is_contiguous() 变成什么？（只有一条序列时，切掉尾巴不会留下空洞）"
# ---
import torch
from learnkit import *

B, T = 3, 7
model = build_model(num_hidden_layers=2)
V = model.config.vocab_size
input_ids = torch.randint(0, V, (B, T))
labels = input_ids.clone()

with torch.no_grad():
    logits = model(input_ids).logits          # [B, T, V]
x, y = logits[..., :-1, :], labels[..., 1:]   # 源码 L251 里两个切片（先不加 .contiguous()）


def try_(fn):                                 # 跑得通就报 shape，跑不通就报错误信息
    try: return str(list(fn().shape))
    except RuntimeError as e: return f"RuntimeError: {str(e)[:60]}…"


table([["logits", str(list(logits.shape)), str(logits.is_contiguous()), try_(lambda: logits.view(-1, V))],
       ["logits[..., :-1, :]", str(list(x.shape)), str(x.is_contiguous()), try_(lambda: x.view(-1, V))],
       ["↑ 先 .contiguous()", str(list(x.shape)), "True", try_(lambda: x.contiguous().view(-1, V))],
       ["↑ 改用 .reshape", str(list(x.shape)), str(x.is_contiguous()), try_(lambda: x.reshape(-1, V))],
       ["labels[..., 1:]", str(list(y.shape)), str(y.is_contiguous()), try_(lambda: y.view(-1))]],
      headers=["张量", "shape", "is_contiguous()", ".view(-1, V) / .view(-1) 的结果"],
      title=f"B={B}, T={T}, V={V}：cross_entropy 要的是 [B*(T-1), V] 和 [B*(T-1)]")
