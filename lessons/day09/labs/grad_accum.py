# ---
# title: 累积 4 步 vs 一次大 batch：梯度差多少
# timeout: 90
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把 lengths 全部改成 7（每条样本有效 token 数相同）：第二行的 max|diff| 会掉到和第一行一个量级吗？"
#   - "把 accumulation_steps 从 4 改成 8（micro batch=1）：第二行的差异是变大还是变小？"
# ---
import torch
from learnkit import *

B, T, accumulation_steps = 8, 11, 4        # 8 条样本，累积 4 步 = 每个 micro-batch 2 条


def accum_grads(input_ids, labels, n):
    """完全照搬 train_epoch：loss 除以 n 再 backward，重复 n 次，梯度自动累加进 .grad"""
    model = build_model(num_hidden_layers=2, hidden_size=128).train()
    micro = B // n
    for i in range(n):
        res = model(input_ids[i * micro:(i + 1) * micro], labels=labels[i * micro:(i + 1) * micro])
        ((res.loss + res.aux_loss) / n).backward()
    return {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}


def max_diff(labels):
    g1, g2 = accum_grads(input_ids, labels, 1), accum_grads(input_ids, labels, accumulation_steps)
    return max((g1[k] - g2[k]).abs().max().item() for k in g1)


torch.manual_seed(0)
input_ids = torch.randint(0, 6400, (B, T))
same = input_ids.clone()                                      # 每条样本 11 个 token 全参与 loss
uneven = input_ids.clone()
lengths = [3, 5, 9, 11, 4, 10, 6, 7]                          # 👉 各不相同的有效长度
for i, n in enumerate(lengths): uneven[i, n:] = -100          # 模仿 PretrainDataset 把 pad 位置设成 -100

table([["每条都是 11 个有效 token", f"{max_diff(same):.2e}"],
       [f"有效 token 数 {lengths}", f"{max_diff(uneven):.2e}"]],
      headers=["labels 里的有效 token 分布", "max|grad(累积4步) - grad(一次8条)|"],
      title="梯度累积 ≡ 大 batch 的前提：每个 micro-batch 的有效 token 数相等")
