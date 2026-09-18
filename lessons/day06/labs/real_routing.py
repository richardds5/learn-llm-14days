# ---
# title: full_sft_768_moe 在 8 层上的 expert 负载
# timeout: 90
# tasks:
#   - "把 TEXT 换成一段纯代码或纯英文：哪些层的偏斜格子跟着变？总量还接近均衡吗？"
#   - "把 argmax 换成 topk(2, dim=-1).indices（假装 k=2 统计）：每行合计从 44 变成 88，最偏斜那一层的 0 还在吗？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

model, tok = load_model("full_sft", use_moe=True), get_tokenizer()   # 真实 SFT 过的 MoE 权重
TEXT = "MiniMind is a small language model. 我们用 MiniMind 来学习大模型的实现细节，MoE 让每个 token 只激活一部分参数。"
ids = tok(TEXT, return_tensors="pt")["input_ids"]                    # [1, T]
L, E, T = model.config.num_hidden_layers, model.config.num_experts, ids.shape[1]

logits = {}                                                          # 每层 mlp.gate 的输出 [T, E]
hs = [l.mlp.gate.register_forward_hook(lambda m, i, o, k=n: logits.__setitem__(k, o.detach()))
      for n, l in enumerate(model.model.layers)]
with torch.no_grad():
    model(ids)
for h in hs: h.remove()

load = torch.zeros(L, E)
for li, lg in logits.items():
    top1 = F.softmax(lg, dim=-1).argmax(dim=-1).flatten()            # k=1 时 topk 就是 argmax
    load[li] = torch.bincount(top1, minlength=E).float()

heatmap(load, cmap="viridis", xlabels=[f"e{i}" for i in range(E)], ylabels=[f"L{i}" for i in range(L)],
        title=f"每层 (行) × 每个 expert (列) 接到的 token 数，每行合计 T={T}；"
              f"{L} 层加总 = {[int(v) for v in load.sum(0)]}（共 {int(load.sum())} 次路由）")
