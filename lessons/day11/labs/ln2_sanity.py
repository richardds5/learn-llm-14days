# ---
# title: 真实权重下第一步的 loss
# timeout: 300
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "把 BETA 从 0.15 改成 10.0：前四行变了吗？最后一行（policy=pretrain）变成多少？"
#   - "把最后一行改成 dpo_loss(pol_lp, ref_lp, mask, BETA)（policy 和 ref 对调）：margin 整体变号，loss 变成多少？"
# ---
import itertools
import math
import os
import random
import tempfile

import torch
from torch.utils.data import DataLoader

from learnkit import *
from dataset.lm_dataset import DPODataset
from trainer.train_dpo import logits_to_log_probs, dpo_loss

N_PAIRS, MAX_LEN, BETA = 3, 256, 0.15          # 👉 BETA 的仓库默认值就是 0.15
random.seed(0)
tok, dev = get_tokenizer(), best_device()

rows = sorted(itertools.islice(open("dataset/dpo.jsonl", encoding="utf-8"), 60), key=len)[:N_PAIRS]
path = os.path.join(tempfile.mkdtemp(), "dpo_mini.jsonl")
open(path, "w", encoding="utf-8").writelines(rows)
b = next(iter(DataLoader(DPODataset(path, tok, max_length=MAX_LEN), batch_size=N_PAIRS)))
x, y = torch.cat([b['x_chosen'], b['x_rejected']]).to(dev), torch.cat([b['y_chosen'], b['y_rejected']]).to(dev)
mask = torch.cat([b['mask_chosen'], b['mask_rejected']]).to(dev)

with torch.no_grad():
    ref_lp = logits_to_log_probs(load_model("full_sft", device=dev)(x).logits, y)      # ref
    pol_lp = logits_to_log_probs(load_model("pretrain", device=dev)(x).logits, y)      # 一个偏离了 ref 的 policy
    L = lambda p, m, beta=BETA: round(dpo_loss(ref_lp, p, m, beta).item(), 10)
    table([["policy = ref = full_sft，真实 mask", BETA, L(ref_lp, mask)],
           ["同上，beta 换成 1.0", 1.0, L(ref_lp, mask, 1.0)],
           ["同上，mask 全部置 1（连 padding 都算）", BETA, L(ref_lp, torch.ones_like(mask))],
           ["同上，mask 全部置 0（一个 token 都不算）", BETA, L(ref_lp, torch.zeros_like(mask))],
           ["policy = pretrain，ref = full_sft", BETA, L(pol_lp, mask)]],
          headers=["设置", "beta", "dpo_loss"], title=f"ln 2 = {math.log(2):.10f}")
