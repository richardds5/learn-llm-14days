# ---
# title: get_lr 的两端：起点 ≈ lr，终点 = 0.1·lr
# timeout: 60
# sources:
#   - trainer/trainer_utils.py
# tasks:
#   - "把 total_steps 从 1000 改成 100：曲线的形状和终点数值变了吗？（get_lr 只看 current_step/total_steps 这个比值）"
#   - "把 0.1+0.45*(...) 改成 0.0+0.5*(...)：终点变成多少？训练最后几步还会更新参数吗？"
# ---
from learnkit import *
from trainer.trainer_utils import get_lr

lr, total_steps = 5e-4, 1000          # 👉 改这两个数，看曲线怎么动
lrs = [get_lr(s, total_steps, lr) for s in range(total_steps + 1)]

# 系数 = get_lr 的返回值 / lr，也就是公式里 0.1 + 0.45*(1 + cos(π·s/total)) 那一坨
coef = lambda s: lrs[s] / lr
half = total_steps // 2
plot({"get_lr（纯 cosine，没有 warmup）": lrs}, x=list(range(total_steps + 1)),
     title=f"lr={lr}, total_steps={total_steps}："
           f"起点 {lrs[0]:.2e}（系数 {coef(0):.2f}）→ 半程 {lrs[half]:.2e}（系数 {coef(half):.2f}）"
           f"→ 终点 {lrs[-1]:.2e}（系数 {coef(total_steps):.2f}）",
     xlabel="current_step", ylabel="lr")
print(f"起点/lr = {lrs[0] / lr:.4f}   半程/lr = {lrs[half] / lr:.4f}   终点/lr = {lrs[-1] / lr:.4f}")
print(f"最大值出现在 step={max(range(total_steps + 1), key=lambda s: lrs[s])}（若有 warmup，最大值不会在 step 0）")
