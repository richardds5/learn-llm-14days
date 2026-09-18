# ---
# title: torch.cat 把一对样本摊成 [2B, T] 的 6 行
# timeout: 120
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "把 torch.cat([x_chosen, x_rejected]) 的顺序对调（y、mask 同改）：表格里「前一半 / 后一半」这两列换到哪去了？"
#   - "把 N_PAIRS 从 3 改成 5：拼出来是几行？dpo_loss 里的 batch_size // 2 切在第几行之后？"
# ---
import itertools
import os
import random
import tempfile

import torch
from torch.utils.data import DataLoader

from learnkit import *
from dataset.lm_dataset import DPODataset

N_PAIRS, MAX_LEN = 3, 256
random.seed(0)
tok = get_tokenizer()

rows = sorted(itertools.islice(open("dataset/dpo.jsonl", encoding="utf-8"), 60), key=len)[:N_PAIRS]
path = os.path.join(tempfile.mkdtemp(), "dpo_mini.jsonl")
open(path, "w", encoding="utf-8").writelines(rows)
b = next(iter(DataLoader(DPODataset(path, tok, max_length=MAX_LEN), batch_size=N_PAIRS)))

x = torch.cat([b['x_chosen'], b['x_rejected']], dim=0)          # 👉 顺序就是 [:N//2] / [N//2:] 的唯一依据
mask = torch.cat([b['mask_chosen'], b['mask_rejected']], dim=0)
n = x.shape[0] // 2

table([[i, "chosen" if i < n else "rejected", f"x[{i}] == x_{'chosen' if i < n else 'rejected'}[{i % n}] ?",
        bool(torch.equal(x[i], (b['x_chosen'] if i < n else b['x_rejected'])[i % n])), int(mask[i].sum())]
       for i in range(x.shape[0])],
      headers=["cat 之后的行号", "来自哪一半", "对照", "逐元素相同", "mask=1 的 token 数"],
      title=f"x_chosen{tuple(b['x_chosen'].shape)} + x_rejected{tuple(b['x_rejected'].shape)} → x{tuple(x.shape)}")
print(f"batch_size // 2 = {n}：dpo_loss 把第 0~{n - 1} 行当 chosen，第 {n}~{2 * n - 1} 行当 rejected")
