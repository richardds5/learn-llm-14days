# ---
# title: topk_weight / (sum + 1e-20) 在 k=1 和 k=2 下的区别
# timeout: 60
# tasks:
#   - "把 1e-20 改成 1e-2 再跑：k=1 那两行的『归一化后』还等于 1 吗？差多少？"
#   - "把 scores 那一行改成 F.softmax(gate_logits * 10, -1)（让 gate 更自信）：k=2 归一化后的两个权重会更接近还是更分开？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

model = build_model(num_hidden_layers=2, use_moe=True)
gate = model.model.layers[0].mlp.gate
box = {}
h = gate.register_forward_hook(lambda m, i, o: box.__setitem__("g", o.detach()))
with torch.no_grad():
    model(torch.randint(0, model.config.vocab_size, (3, 7)))
h.remove()
scores = F.softmax(box["g"], dim=-1)                            # [B*T, E]

rows = []
for k in (1, 2):
    topk_weight, _ = torch.topk(scores, k=k, dim=-1, sorted=False)
    normed = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)   # 👉 model_minimind.py:161
    for t in (0, 1):                                            # 只看前两个 token
        rows.append([k, t, str([round(v, 4) for v in topk_weight[t].tolist()]),
                     str([round(v, 6) for v in normed[t].tolist()]),
                     f"{(normed[t] - 1).abs().max().item():.3e}" if k == 1 else "—",
                     bool((normed[t] == topk_weight[t]).all()) if k == 1 else "—"])

table(rows, headers=["k", "token", "归一化前 topk_weight", "归一化后", "与 1 的偏差", "和归一化前相同吗"],
      title="k=1 时这一行把一个 0.2~0.4 的概率变成常数 1.0；k=2 时才是两个数之间的真除法")
