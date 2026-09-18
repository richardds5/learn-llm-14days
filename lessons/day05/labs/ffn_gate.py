# ---
# title: 真实权重下，gate 支路和 up 支路的数值分布对照
# timeout: 60
# tasks:
#   - "把 LAYER 从 3 改成 0 或 7：最小值那一行会不会跌破 -0.2785？（SiLU 的全局下界）"
#   - "把 act_fn 临时换成 torch.relu：`gate = torch.relu(mlp.gate_proj(h))`，再看『恰好 = 0』那一行——硬门控和软门控的差别就在这一格"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
LAYER = 3                                    # 👉 看第几层的 FFN
mlp = model.model.layers[LAYER].mlp

box = {}
hk = mlp.register_forward_pre_hook(lambda m, inp: box.setdefault("h", inp[0].detach()))
with torch.no_grad():
    model(tok("人工智能的发展离不开大量的数据和算力。", return_tensors="pt").input_ids)
hk.remove()

h = box["h"]                                 # post_attention_layernorm 的输出，[B,T,C]
with torch.no_grad():
    gate, up = mlp.act_fn(mlp.gate_proj(h)), mlp.up_proj(h)   # 两条支路，同一个输入

stat = lambda t: [f"{t.min().item():.4f}", f"{t.max().item():.2f}",
                  f"{(t == 0).float().mean().item():.2%}",
                  f"{(t.abs() < 0.1).float().mean().item():.1%}",
                  f"{(t < 0).float().mean().item():.1%}"]
table([["act_fn(gate_proj(x))  门"] + stat(gate), ["up_proj(x)  内容"] + stat(up)],
      headers=["支路", "最小值", "最大值", "恰好 = 0", "|·| < 0.1", "< 0"],
      title=f"layer {LAYER} 的 FFN：{gate.numel()} 个门控值的分布（SiLU 下界 = -0.2785）")
