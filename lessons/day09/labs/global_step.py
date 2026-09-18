# ---
# title: epoch * iters + step：两个 epoch 拼成一条连续的余弦
# timeout: 60
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把 get_lr 的第一个参数从 epoch*iters+step 改成 step（只用 epoch 内的步号）：曲线变成什么样？第二个 epoch 的 lr 是接着降还是回到最大值？"
#   - "把 args.epochs 从 2 改成 1（iters 不变）：第 1 个 epoch 结束时 lr 掉到多少？"
# ---
from types import SimpleNamespace
from learnkit import *
from trainer.trainer_utils import get_lr

args = SimpleNamespace(epochs=2, learning_rate=5e-4)
iters = 300                                     # 一个 epoch 有多少个 batch（真实脚本里 = len(loader)）

xs, ys, marks = [], [], []
for epoch in range(args.epochs):
    for step in range(1, iters + 1):            # ← 注意 train_epoch 的 step 从 start_step+1 开始，不是 0
        current_step = epoch * iters + step      # 👉 train_pretrain.py:31 的第一个实参
        xs.append(current_step)
        ys.append(get_lr(current_step, args.epochs * iters, args.learning_rate))
    marks.append((epoch, xs[-1], ys[-1]))

plot({"lr": ys}, x=xs, title=f"epochs={args.epochs}, iters={iters} → 总步数 {args.epochs * iters}："
                            f"所有 epoch 共用同一条余弦", xlabel="epoch*iters+step", ylabel="lr")
for epoch, x, y in marks:
    print(f"epoch {epoch} 最后一步: current_step={x}, lr={y:.3e} ({y / args.learning_rate:.2f}·lr)")
if len(ys) > iters:
    print(f"epoch 1 的第一步: current_step={xs[iters]}, lr={ys[iters]:.3e}"
          f"；对比 epoch 0 的第一步 lr={ys[0]:.3e}")
