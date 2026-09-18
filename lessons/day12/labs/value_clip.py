# ---
# title: value loss 的双重裁剪
# timeout: 60
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 cliprange_value 从 0.2 改成 2.0：还有哪几行的梯度是 0？"
#   - "把 returns_ 从 1.5 改成 0.55（离 old_value 只差 0.05）：'哪一支胜出' 那一列变成什么？"
# ---
import torch
from learnkit import *

cliprange_value = 0.2               # 👉 源码默认 args.cliprange_value=0.2
old_value = torch.tensor(0.5)       # rollout 时 critic 给的旧估值，是裁剪的中心
returns_ = torch.tensor(1.5)        # GAE 算出来的拟合目标


def value_loss(v):                  # train_ppo.py:214-217 的逐 token 项
    return 0.5 * torch.max((v - returns_) ** 2,
                           (torch.clamp(v, old_value - cliprange_value, old_value + cliprange_value) - returns_) ** 2)


rows = []
for x in [0.1, 0.5, 0.69, 0.71, 1.5, 2.4]:
    v = torch.tensor(float(x), requires_grad=True)
    loss = value_loss(v)
    loss.backward()
    unclipped = 0.5 * float((v.detach() - returns_) ** 2)
    rows.append([x, round(unclipped, 4), round(float(loss), 4), round(v.grad.item(), 4),
                 "未裁剪支" if abs(loss.item() - unclipped) < 1e-6 else "裁剪支（常数）"])
table(rows, headers=["mb_resp_values", "0.5*(v-returns)^2（无 clip）", "0.5*max(两支)（源码）",
                     "d loss/d v", "哪一支胜出"],
      title=f"value clip：old_value={float(old_value)}, cliprange_value={cliprange_value}, returns={float(returns_)}")
print("裁剪支在 |v - old_value| > cliprange_value 时是常数，梯度为 0；"
      "取 max 之后，只要它比未裁剪支更大，整项的梯度就被削成 0 —— critic 每一步最多只能挪 cliprange_value。")
