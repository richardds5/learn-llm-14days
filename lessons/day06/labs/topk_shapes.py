# ---
# title: torch.topk 把 [B*T,E] 切成两个 [B*T,k]
# timeout: 60
# tasks:
#   - "把 sorted=False 改成 sorted=True：表格最后两列（第 0 个 token 选中的 expert 和权重）有变化吗？"
#   - "把 k 的取值加上 5（大于 num_experts=4）：在哪一行报什么错？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

model = build_model(num_hidden_layers=2, use_moe=True)
gate = model.model.layers[0].mlp.gate
B, T = 3, 7
box = {}
h = gate.register_forward_hook(lambda m, i, o: box.__setitem__("s", F.softmax(o.detach(), -1)))
with torch.no_grad():
    model(torch.randint(0, model.config.vocab_size, (B, T)))
h.remove()
scores = box["s"]                                              # [B*T, E]

rows = []
for k in (1, 2, 4):                                            # 👉 加一个 5 试试（E 只有 4）
    topk_weight, topk_idx = torch.topk(scores, k=k, dim=-1, sorted=False)   # model_minimind.py:160
    rows.append([k, str(list(topk_weight.shape)), str(list(topk_idx.shape)),
                 str(topk_idx.dtype).replace("torch.", ""),
                 str(topk_idx[0].tolist()),
                 str([round(v, 3) for v in topk_weight[0].tolist()])])

table(rows, headers=["k", "topk_weight.shape", "topk_idx.shape", "topk_idx.dtype",
                     "第 0 个 token 选中的 expert", "对应权重"],
      title=f"scores {list(scores.shape)} --topk--> 两个 [B*T={B * T}, k]（权重表和身份证表分开）")
