# ---
# title: index_add_ 与「改成赋值」的差别（k=2）
# timeout: 60
# tasks:
#   - "把 num_experts_per_tok 改成 1：第 2 行的最大差变成多少？为什么 k=1 时『加』和『赋值』等价？"
#   - "把 num_experts_per_tok 改成 4（等于 num_experts，每个 token 都过一遍全部 expert）：第 3 行变成几次？第 2 行的差变大还是变小？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *
from model.model_minimind import MiniMindConfig, MOEFeedForward

torch.manual_seed(0)
cfg = MiniMindConfig(use_moe=True, num_experts_per_tok=2, moe_intermediate_size=256)  # 👉 改 k
moe = MOEFeedForward(cfg).eval()
x = torch.randn(3, 7, cfg.hidden_size)

with torch.no_grad():
    y_real = moe(x).view(-1, cfg.hidden_size)                       # 仓库实现（index_add_）
    x_flat = x.view(-1, cfg.hidden_size)
    scores = F.softmax(moe.gate(x_flat), dim=-1)
    tw, ti = torch.topk(scores, k=cfg.num_experts_per_tok, dim=-1, sorted=False)
    if cfg.norm_topk_prob: tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)
    y_sum, y_assign, hits = torch.zeros_like(x_flat), torch.zeros_like(x_flat), torch.zeros(x_flat.shape[0])
    for i, expert in enumerate(moe.experts):
        mask = (ti == i)
        idx = mask.any(-1).nonzero().flatten()
        out = expert(x_flat[idx]) * tw[mask].view(-1, 1)
        y_sum.index_add_(0, idx, out)                               # 累加：同一行可以被写多次
        y_assign[idx] = out                                         # 👉 赋值：后写的覆盖先写的
        hits[idx] += 1

table([["真实 y  vs  手工「k 个 expert 加权和」", f"{(y_real - y_sum).abs().max():.2e}"],
       ["真实 y  vs  把 index_add_ 换成赋值", f"{(y_real - y_assign).abs().max():.2e}"],
       [f"每个 token 被写回的次数 (min/max)", f"{int(hits.min())} / {int(hits.max())}"]],
      headers=["对比", "最大逐元素差"], title=f"k={cfg.num_experts_per_tok}：y 的每一行是 k 个 expert 输出的加权和")
