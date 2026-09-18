# ---
# title: total / base / active 三个数怎么算出来
# timeout: 60
# sources:
#   - trainer/trainer_utils.py
# tasks:
#   - "给 build_model 加上 num_experts=8（k 仍是 1）：total 涨了多少？active 几乎不动，只多了哪一项？"
#   - "给 build_model 加上 num_experts_per_tok=2：active 涨了多少？正好等于表里哪一行？"
# ---
from learnkit import *
from trainer.trainer_utils import get_model_params

model = build_model(use_moe=True)          # 👉 加 num_experts=8 或 num_experts_per_tok=2
cfg = model.config
get_model_params(model, cfg)               # 仓库原函数，打印 "Model Params: xxxM-Axxx M"

total = sum(p.numel() for p in model.parameters())
expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n)   # 已跨全部层加总
base = total - expert * cfg.num_experts
active = base + expert * cfg.num_experts_per_tok
one = sum(p.numel() for n, p in model.named_parameters() if n.startswith("model.layers.0.mlp.experts.0."))

table([["total（全部参数）", f"{total:,}", f"{total / 1e6:.2f}M"],
       [f"expert：0 号 expert 槽位 × {cfg.num_hidden_layers} 层", f"{expert:,}", f"{expert / 1e6:.2f}M"],
       ["  其中单层单个 expert（3·C·moe_I）", f"{one:,}", f"{one / 1e6:.2f}M"],
       [f"base = total − expert × num_experts({cfg.num_experts})", f"{base:,}", f"{base / 1e6:.2f}M"],
       [f"active = base + expert × k({cfg.num_experts_per_tok})", f"{active:,}", f"{active / 1e6:.2f}M"],
       ["（对照）同配置的 dense 模型 total", f"{sum(p.numel() for p in build_model().parameters()):,}", ""]],
      headers=["分项", "参数量", "百万"],
      title=f"C={cfg.hidden_size}, L={cfg.num_hidden_layers}, E={cfg.num_experts}, k={cfg.num_experts_per_tok}")
