# ---
# title: 同一批 token，dense FFN 与 MoE FFN 的耗时
# timeout: 120
# tasks:
#   - "把 B, T 从 9, 53 改成 9, 400（token 多 7 倍）：MoE 相对 dense 的倍数变大还是变小？"
#   - "给 MOEFeedForward 的 config 加上 num_experts_per_tok=2：E=4 那一行的耗时涨多少？（这次连 FLOPs 也真的翻倍了）"
# ---
import time
import torch
from learnkit import *
from model.model_minimind import MiniMindConfig, FeedForward, MOEFeedForward

torch.set_num_threads(1)                      # 关掉多线程，减少计时噪声
torch.manual_seed(0)
B, T = 9, 53                                  # 👉 改 T：token 越多，固定调度开销占比越小
cfg = MiniMindConfig(use_moe=True)
x = torch.randn(B, T, cfg.hidden_size)


def bench(module, reps=15):
    module.eval()
    with torch.no_grad():
        for _ in range(3): module(x)          # warmup
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); module(x); ts.append(time.perf_counter() - t0)
    return min(ts) * 1000, sum(p.numel() for p in module.parameters()) / 1e6


rows, (dense_ms, dense_n) = [], bench(FeedForward(cfg))
rows.append(["dense FeedForward", "—", f"{dense_n:.1f}M", f"{dense_n:.1f}M", f"{dense_ms:.1f}", "1.00×"])
for E in (4, 16):                             # 👉 加一个 64 试试
    c = MiniMindConfig(use_moe=True, num_experts=E)
    module = MOEFeedForward(c)
    ms, n = bench(module)
    act = sum(p.numel() for p in module.experts[0].parameters()) * c.num_experts_per_tok / 1e6
    rows.append(["MOEFeedForward", f"{E} / top-{c.num_experts_per_tok}", f"{n:.1f}M", f"{act:.1f}M",
                 f"{ms:.1f}", f"{ms / dense_ms:.2f}×"])

table(rows, headers=["前馈层", "experts", "这一层的总参数", "一个 token 实际过的参数", "最快一次 (ms)", "相对 dense"],
      title=f"只跑前馈层本身，B={B}, T={T}, C={cfg.hidden_size}：第 4 列相同 = 浮点运算量相同")
