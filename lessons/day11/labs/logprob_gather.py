# ---
# title: 逐行拆开 logits_to_log_probs：[N,T,V] 怎么塌成 [N,T]
# timeout: 120
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "把 F.log_softmax 那一行的 dim=2 改成 dim=1（用 overrides 或直接读源码想）：gather 出来的还是 [N,T] 吗？数值还是 log p(y_t) 吗？"
#   - "把 labels.unsqueeze(2) 去掉，直接传 labels：报的是哪一个 RuntimeError？和 -100 那个错区分开。"
# ---
import torch

from learnkit import *
from trainer.train_dpo import logits_to_log_probs

torch.manual_seed(42)
N, T, V = 6, 13, 6400                 # 👉 N = 2 × 3 对；T 用 13 是为了和 config 里的数字区分开
logits = torch.randn(N, T, V)         # 模型吐出来的 logits
labels = torch.randint(0, V, (N, T))  # DPODataset 里已经 shift 过的 y

with trace(fns=["trainer.train_dpo:logits_to_log_probs"],
           dims=dict(N=N, T=T, V=V), expand=True, tree=False,
           title="log_softmax → gather → squeeze：每个子表达式的 shape") as tr:
    log_probs_per_token = logits_to_log_probs(logits, labels)

print("返回值 shape =", tuple(log_probs_per_token.shape),
      "  它的每个元素就是 log p(y_t | x_<=t)，还没有乘 mask")
print("手工核对第 (0, 0) 个元素：",
      torch.allclose(log_probs_per_token[0, 0],
                     torch.log_softmax(logits[0, 0], dim=-1)[labels[0, 0]]))
