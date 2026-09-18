# ---
# title: 64M 参数的四个去处：FFN / attention / embedding / norm
# timeout: 60
# tasks:
#   - "把 build_model() 改成 build_model(vocab_size=32000)（Qwen 量级词表）：embedding 占比涨到多少？总参数量变成多少？"
#   - "把 build_model() 改成 build_model(hidden_size=1536)：总量涨了几倍？FFN 和 attention 的占比谁动得更小（两者都是 O(C²)）？"
# ---
import re
from learnkit import *

model = build_model()          # 👉 默认 C=768, L=8, V=6400；改这里看占比怎么变
cfg = model.config

def bucket(name):
    if "embed_tokens" in name: return "embedding（与 lm_head 共享一份）"
    if re.search(r"self_attn\.(q|k|v|o)_proj", name): return "attention 的 q/k/v/o_proj"
    if re.search(r"mlp\.(gate|up|down)_proj", name): return "FFN 的 gate/up/down_proj"
    return "所有 RMSNorm 的 weight"

sizes = {}
for name, p in model.named_parameters():
    sizes[bucket(name)] = sizes.get(bucket(name), 0) + p.numel()
total = sum(sizes.values())

table([[k, f"{v:,}", f"{v / total * 100:.2f}%"] for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])],
      headers=["去处", "参数量", "占比"],
      title=f"C={cfg.hidden_size}, L={cfg.num_hidden_layers}, V={cfg.vocab_size} → 总参数 {total:,} ({total / 1e6:.2f}M)")
