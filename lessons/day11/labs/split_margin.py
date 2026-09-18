# ---
# title: 逐行拆开 dpo_loss：[N,T] → [N] → 两个 [N/2] → 标量
# timeout: 120
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "把 N 从 6 改成 10（5 对）：trace 里 chosen_ref_log_probs 的 shape 变成什么？loss 呢？"
#   - "把 policy[:N // 2] += 0.5 改成 policy[N // 2:] += 0.5（改成抬高 rejected）：trace 里 logits 那一行的值是正是负？loss 比刚才大还是小？"
# ---
import torch

from learnkit import *
from trainer.train_dpo import dpo_loss

torch.manual_seed(0)
N, T, BETA = 6, 13, 0.15               # 👉 N = 2 × 3 对，前 3 行 chosen、后 3 行 rejected
base = torch.randn(N, T)               # 假装是 logits_to_log_probs 的输出
mask = torch.zeros(N, T, dtype=torch.long)
mask[:, 5:] = 1                        # 👉 前 5 个位置当作 prompt/padding，不计 loss
policy = base.clone()
policy[:N // 2] += 0.5                 # 让 chosen 比 ref 高一点，margin 才不是 0

with trace(fns=["trainer.train_dpo:dpo_loss"], dims=dict(N=N, T=T),
           expand=True, tree=False, title="mask 求和 → 切两半 → margin → logsigmoid → 标量") as tr:
    loss = dpo_loss(base, policy, mask, beta=BETA)

print(f"loss = {loss.item():.6f}（标量，shape={tuple(loss.shape)}）")
print("注意 ref_log_probs / policy_log_probs 这两个形参在函数里被就地重新赋值："
      f"进来是 [{N},{T}]，第 36~37 行之后变成 [{N}]。")
