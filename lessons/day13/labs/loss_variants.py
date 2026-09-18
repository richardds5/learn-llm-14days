# ---
# title: 两个变体的梯度权重 ∂loss/∂logπ 随 ratio 的变化
# timeout: 60
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 epsilon 从 0.2 改成 0.6：grpo 的两条线各自从哪个 ratio 开始归零？"
#   - "把 epsilon_high 从 5.0 改成 1.2：cispo 的两条线在 ratio > 1.2 之后是继续张开还是走平？"
# ---
import torch
from learnkit import *

epsilon, epsilon_high = 0.2, 5.0  # train_grpo.py 的默认值
ratios = torch.linspace(0.05, 6.0, 120)
curves = {}

for loss_type in ("grpo", "cispo"):
    for A in (1.0, -1.0):  # advantage 的正负决定这条 token 是被鼓励还是被压制
        old_logp = torch.zeros_like(ratios)
        logp = (old_logp + ratios.log()).clone().requires_grad_(True)  # 反推出能给出这些 ratio 的 logπ_θ
        ratio = torch.exp(logp - old_logp)
        if loss_type == "cispo":  # 👉 .detach() 让 ratio 只当权重，梯度只从末尾的 logp 走
            per_token_loss = -(torch.clamp(ratio, max=epsilon_high).detach() * A * logp)
        else:                     # 👉 min + clamp：一旦被 clamp 的那一支被选中，梯度就断了
            per_token_loss = -torch.min(ratio * A, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A)
        per_token_loss.sum().backward()
        curves[f"{loss_type}  A={A:+.0f}"] = logp.grad.clone()

plot(curves, title=f"∂(per_token_loss)/∂logπ_θ（β=0，只看策略项；ε={epsilon}, ε_high={epsilon_high}）",
     xlabel="ratio = exp(logπ_θ − logπ_old)", ylabel="梯度权重", x=ratios)
for name, g in curves.items():
    zero = (g.abs() < 1e-9)
    rng = f"ratio ∈ [{ratios[zero].min():.2f}, {ratios[zero].max():.2f}]" if zero.any() else "从不归零"
    print(f"{name}:  梯度为 0 的区间 → {rng}")
