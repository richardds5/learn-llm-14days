# ---
# title: Embedding 就是查表：embed_tokens(ids) == embed_tokens.weight[ids]
# timeout: 60
# tasks:
#   - "把 input_ids[0, 0] 改成 model.config.vocab_size（= 6400，刚好越界一个）再跑：报什么错？在哪一行？"
#   - "把 input_ids 换成一维的 torch.randint(0, 6400, (7,))：输出 shape 变成什么？（查表对任意形状的下标都成立）"
# ---
import torch
from learnkit import *

model = build_model(num_hidden_layers=2)
embed_tokens = model.model.embed_tokens                       # nn.Embedding(V=6400, C=768)

input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # B=3, T=7
input_ids[0, 0] = 1                                            # 👉 改成 6400 试试越界
out = embed_tokens(input_ids)                                  # 正规调用
gathered = embed_tokens.weight[input_ids]                      # 纯粹的高级索引（查表）

show(input_ids=input_ids, weight=embed_tokens.weight, out=out,
     dims=dict(B=3, T=7, C=768, V=6400))
print("embed_tokens(input_ids) 和 embed_tokens.weight[input_ids] 完全相等吗：",
      torch.equal(out, gathered))
print("第 0 个 token 的向量 == weight 的第 1 行吗：",
      torch.equal(out[0, 0], embed_tokens.weight[1]))
