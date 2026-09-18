# ---
# title: (log_probs * mask).sum(dim=1)：真实权重下 [6,255] → [6]
# timeout: 180
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "把 (lp * mask).sum(1) 改成 lp.sum(1)（不乘 mask）：序列 log prob 掉到多少？padding 位置贡献了多少？"
#   - "把 mask 的 dtype 从 int64 改成 float32（mask.float()）：结果变了吗？说明 float * int64 的类型提升是安全的。"
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
tok, dev = get_tokenizer(), best_device()

rows = sorted(itertools.islice(open("dataset/dpo.jsonl", encoding="utf-8"), 60), key=len)[:N_PAIRS]
path = os.path.join(tempfile.mkdtemp(), "dpo_mini.jsonl")
open(path, "w", encoding="utf-8").writelines(rows)
b = next(iter(DataLoader(DPODataset(path, tok, max_length=MAX_LEN), batch_size=N_PAIRS)))
x = torch.cat([b['x_chosen'], b['x_rejected']]).to(dev)
y = torch.cat([b['y_chosen'], b['y_rejected']]).to(dev)
mask = torch.cat([b['mask_chosen'], b['mask_rejected']]).to(dev)

model = load_model("full_sft", device=dev)
with torch.no_grad():
    lp = logits_to_log_probs(model(x).logits, y)        # [2B, T]
seq = (lp * mask).sum(dim=1)                            # 👉 dpo_loss 的第一行：[2B, T] → [2B]

table([[i, "chosen" if i < N_PAIRS else "rejected", int(mask[i].sum()),
        round(seq[i].item(), 2), round((seq[i] / mask[i].sum()).item(), 3)] for i in range(2 * N_PAIRS)],
      headers=["行", "来自哪一半", "mask=1 的 token 数", "序列 log prob = Σ mask·logp", "平均每 token"],
      title=f"lp{tuple(lp.shape)} * mask{tuple(mask.shape)} → sum(dim=1) → seq{tuple(seq.shape)}  "
            f"（lp.dtype={str(lp.dtype).replace('torch.', '')}, mask.dtype={str(mask.dtype).replace('torch.', '')}）")
