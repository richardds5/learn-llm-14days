# ---
# title: mask 框住 assistant 段，而 y 里一个 -100 都不能有
# timeout: 120
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把切片 sl 改成 slice(0, 20)：开头这 20 个 token 是 chat 模板的哪一段？mask 全是 0 吗？"
#   - "把 y_bad[0, 0] = -100 改成 = 6400（刚好越过词表上界）：报错信息里的数字变成什么？"
# ---
import itertools
import os
import random
import tempfile

import torch
from torch.utils.data import DataLoader

from learnkit import *
from dataset.lm_dataset import DPODataset
from trainer.train_dpo import logits_to_log_probs

N_PAIRS, MAX_LEN = 3, 256
random.seed(0)
tok = get_tokenizer()

rows = sorted(itertools.islice(open("dataset/dpo.jsonl", encoding="utf-8"), 60), key=len)[:N_PAIRS]
path = os.path.join(tempfile.mkdtemp(), "dpo_mini.jsonl")
open(path, "w", encoding="utf-8").writelines(rows)
b = next(iter(DataLoader(DPODataset(path, tok, max_length=MAX_LEN), batch_size=N_PAIRS)))
y, mask = b['y_chosen'][0], b['mask_chosen'][0]

i0 = int(mask.argmax())                       # 👉 mask 第一次从 0 翻到 1 的位置
sl = slice(max(i0 - 7, 0), i0 + 9)
token_strip([tok.decode([t]) for t in y[sl]], mask[sl].tolist(),
            title=f"y[{sl.start}:{sl.stop}] 与 mask：谁算 loss（mask 从下标 {i0} 开始变 1）", legend="loss mask")

y_bad = b['y_chosen'].clone(); y_bad[0, 0] = -100      # 👉 如果像 SFT 那样用 -100 当 ignore_index
try:
    logits_to_log_probs(torch.randn(N_PAIRS, y.shape[0], 6400), y_bad)
except RuntimeError as e:
    print("把 -100 塞进 y 之后：RuntimeError:", e)
