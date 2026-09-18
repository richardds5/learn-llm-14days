# ---
# title: 三个矩阵的账：SwiGLU 比两矩阵 MLP 贵在哪
# timeout: 60
# tasks:
#   - "把 build_model() 改成 build_model(intermediate_size=2048)（LLaMA 的 8/3·C 对齐到 64）：FFN 参数量变成多少？和 4C 两矩阵 MLP 谁大？"
#   - "把 build_model() 改成 build_model(hidden_size=1536)：三个矩阵各自涨了几倍？（每个都是 C·I，而 I 也跟着 C 走）"
# ---
from learnkit import *

model = build_model(num_hidden_layers=2)     # 👉 改 config 看这张账单怎么变
cfg = model.config
C, I = cfg.hidden_size, cfg.intermediate_size
mlp, attn = model.model.layers[0].mlp, model.model.layers[0].self_attn

rows = [[n, str(list(getattr(mlp, n).weight.shape)), f"{getattr(mlp, n).weight.numel():,}"]
        for n in ("gate_proj", "up_proj", "down_proj")]
ffn = sum(p.numel() for p in mlp.parameters())
att = sum(p.numel() for p in attn.parameters() if p.dim() == 2)
rows += [
    ["FFN 小计 = 3·C·I", f"3×{C}×{I}", f"{ffn:,}"],
    ["同层 attention 的 q/k/v/o_proj", "2·C·H·D + 2·C·KV·D", f"{att:,}"],
    ["假想的两矩阵 MLP（中间维 4C）", f"2×{C}×{4 * C}", f"{2 * C * 4 * C:,}"],
    ["SwiGLU / 两矩阵 MLP", "", f"{ffn / (2 * C * 4 * C):.4f}  →  多花 {ffn / (2 * C * 4 * C) - 1:.2%}"],
]
table(rows, headers=["矩阵 / 小计", "形状或公式", "参数量"],
      title=f"C={C}, I={I}：一层 FFN 的参数账（FFN 是 attention 的 {ffn / att:.2f} 倍）")
