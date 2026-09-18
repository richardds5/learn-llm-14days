# ---
# title: 带 cache 续写时，-inf 落在 scores 的哪些格子上
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把续写长度 5 改成 1（单步 decode）：图变成一行，这一行里还有黄格子吗？"
#   - "把 prefill 的 7 改成 3：图变成几列？黄格子的个数跟着变了吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import Attention

model = build_model(num_hidden_layers=2, flash_attn=False)
V = model.config.vocab_size
out = model(torch.randint(0, V, (3, 7)), use_cache=True)   # 👉 prefill 7 个
new = torch.randint(0, V, (3, 5))                          # 👉 再续写 5 个

with trace(model, fns=[Attention.forward], dims=dict(B=3, T=5, Tkv=12),
           focus=(129, 129), capture=["scores"], tree=False, max_calls=1) as tr:
    model(new, past_key_values=out.past_key_values, use_cache=True)

sc = tr.captured["Attention.forward"][0]["scores"]         # 掩码加完、softmax 之前
mask = torch.isinf(sc)[0, 0].float()                       # 第 0 个样本、第 0 个 head
heatmap(mask, ylabels=[f"q{7 + i}" for i in range(5)], xlabels=[f"k{j}" for j in range(12)],
        title=f"scores{list(sc.shape)} 里 -inf 的位置（黄 = 被屏蔽）：行 = 5 个新 query，列 = 12 个 key")
