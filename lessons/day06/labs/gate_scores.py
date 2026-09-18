# ---
# title: scores = softmax(gate(x_flat))：每个 token 一行 E 维的概率
# timeout: 60
# tasks:
#   - "给 build_model 加上 num_experts=6：热力图变成几列？gate.weight 的 shape 变成什么？"
#   - "把 F.softmax(..., dim=-1) 改成 dim=0（沿 token 维归一化）：行和还等于 1 吗？标题里的最大偏差会变成多少？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

model = build_model(num_hidden_layers=2, use_moe=True)
gate = model.model.layers[0].mlp.gate          # nn.Linear(C, E, bias=False)
B, T, E = 3, 7, model.config.num_experts
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

box = {}                                       # 用 hook 把第 0 层 gate 的输出 (logits) 抓出来
h = gate.register_forward_hook(lambda m, i, o: box.__setitem__("logits", o.detach()))
with torch.no_grad():
    model(input_ids)
h.remove()

scores = F.softmax(box["logits"], dim=-1)      # 👉 复刻 model_minimind.py:159 这一行
row_err = (scores.sum(-1) - 1).abs().max().item()

heatmap(scores, cmap="viridis",
        xlabels=[f"e{i}" for i in range(E)],
        ylabels=[f"b{i // T}t{i % T}" for i in range(B * T)],
        title=f"scores {list(scores.shape)} = [B*T={B * T}, E={E}]："
              f"gate.weight {list(gate.weight.shape)}, bias={gate.bias}；每行和与 1 的最大偏差 {row_err:.1e}")
