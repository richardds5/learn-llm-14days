# ---
# title: 把每个 token 的 log prob 统一压低 δ 之后，两种聚合方式各算出什么
# timeout: 180
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "把 DELTA 从 0.3 调到 1.0：sum 版最低那一对的 loss 掉到多少？mean 版呢？"
#   - "把 dpo_loss_mean 里的 denom 改成 mask.shape[1]（按 padding 后的满长度归一化）：还能消掉长度差带来的 margin 吗？"
# ---
import itertools
import math
import os
import random
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learnkit import *
from dataset.lm_dataset import DPODataset
from trainer.train_dpo import dpo_loss

N_PAIRS, MAX_LEN, BETA, DELTA = 4, 256, 0.1, 0.3
random.seed(0); torch.manual_seed(0)

def dpo_loss_mean(ref_lp, pol_lp, mask, beta):
    """和 trainer/train_dpo.py:36 只差一处：序列 log prob 再除以有效 token 数。"""
    d = mask.sum(dim=1).clamp(min=1)
    r, p = (ref_lp * mask).sum(1) / d, (pol_lp * mask).sum(1) / d
    n = r.shape[0] // 2
    return -F.logsigmoid(beta * ((p[:n] - p[n:]) - (r[:n] - r[n:])))

rows = sorted(itertools.islice(open("dataset/dpo.jsonl", encoding="utf-8"), 60), key=len)[:N_PAIRS]
path = os.path.join(tempfile.mkdtemp(), "dpo_mini.jsonl")
open(path, "w", encoding="utf-8").writelines(rows)
b = next(iter(DataLoader(DPODataset(path, get_tokenizer(), max_length=MAX_LEN), batch_size=N_PAIRS)))
mask = torch.cat([b['mask_chosen'], b['mask_rejected']])
ref_lp = torch.randn(mask.shape)                 # ref 的具体数值不影响结论：下面全是 policy - ref 的差
L = mask.sum(1)

pol_lp = ref_lp - DELTA                          # 👉 每个 token 的 log prob 统一压低 DELTA，零偏好信息
per_pair_sum = -F.logsigmoid(BETA * ((pol_lp * mask).sum(1)[:N_PAIRS] - (pol_lp * mask).sum(1)[N_PAIRS:]
                                     - (ref_lp * mask).sum(1)[:N_PAIRS] + (ref_lp * mask).sum(1)[N_PAIRS:]))
per_pair_mean = dpo_loss_mean(ref_lp, pol_lp, mask, BETA)
table([[i, int(L[i]), int(L[i + N_PAIRS]), round(DELTA * (int(L[i + N_PAIRS]) - int(L[i])), 2),
        round(per_pair_sum[i].item(), 4), round(per_pair_mean[i].item(), 4)] for i in range(N_PAIRS)],
      headers=["pair", "len_c", "len_r", f"sum 版白赚的 margin = δ·(len_r−len_c)", "loss_sum", "loss_mean"],
      title=f"δ={DELTA}，β={BETA}，ln2={math.log(2):.4f}；整批平均：sum={per_pair_sum.mean():.4f}  mean={per_pair_mean.mean():.4f}")
print(f"用仓库的 dpo_loss 直接算整批：{dpo_loss(ref_lp, pol_lp, mask, BETA).item():.4f}")
