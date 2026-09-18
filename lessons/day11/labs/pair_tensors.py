# ---
# title: DPODataset 一次吐出六个张量，而且已经 shift 过
# timeout: 120
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 MAX_LEN 从 256 改成 128：六个张量的 T 变成多少？每行 mask=1 的 token 数跟着变了吗？"
#   - "把最后一行的 x_chosen[:, 1:] / y_chosen[:, :-1] 换成 x_chosen[:, :-1] / y_chosen[:, 1:]：还是 True 吗？想清楚 y 是往哪边挪的。"
# ---
import itertools
import os
import random
import tempfile

import torch
from torch.utils.data import DataLoader

from learnkit import *
from dataset.lm_dataset import DPODataset

N_PAIRS, MAX_LEN = 3, 256          # 👉 3 对偏好样本，每条单独 padding 到 256
random.seed(0)                      # DPODataset 内部会随机删 <think>…</think>，不固定就不可复现
tok = get_tokenizer()

rows = sorted(itertools.islice(open("dataset/dpo.jsonl", encoding="utf-8"), 60), key=len)[:N_PAIRS]
path = os.path.join(tempfile.mkdtemp(), "dpo_mini.jsonl")
open(path, "w", encoding="utf-8").writelines(rows)
batch = next(iter(DataLoader(DPODataset(path, tok, max_length=MAX_LEN), batch_size=N_PAIRS)))

table([[k, str(tuple(v.shape)), str(v.dtype).replace("torch.", ""),
        str(v.sum(1).tolist()) if k.startswith("mask") else "—"] for k, v in batch.items()],
      headers=["返回的键", "shape", "dtype", "每行 mask=1 的 token 数"],
      title=f"__getitem__ 的六个张量：max_length={MAX_LEN} 经过 [:-1] / [1:] 之后 T={MAX_LEN - 1}")

print("x_chosen[:, 1:] 与 y_chosen[:, :-1] 逐元素相同？",
      torch.equal(batch['x_chosen'][:, 1:], batch['y_chosen'][:, :-1]), "  ← y 就是 x 往左挪一格")
