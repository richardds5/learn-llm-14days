# ---
# title: 单卡的 randperm 和多卡的 DistributedSampler，谁在决定数据顺序
# timeout: 60
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把两个 rank 的 set_epoch(0) 改成不同的值（rank0 用 0、rank1 用 1）：两张卡还会各训各的一半吗？会不会有样本被训两次？"
#   - "把 args.seed + epoch 改回固定的 args.seed：两个 epoch 的 indices 还不一样吗？"
# ---
import torch
from types import SimpleNamespace
from torch.utils.data import DistributedSampler
from learnkit import *
from trainer.trainer_utils import setup_seed

args = SimpleNamespace(seed=42)
train_ds = list(range(12))                     # 假装数据集里有 12 条样本

rows = []
for epoch in range(2):                         # ---- 非 DDP：train_sampler 是 None ----
    setup_seed(args.seed + epoch)              # train_pretrain.py:160，每个 epoch 换一个种子
    indices = torch.randperm(len(train_ds)).tolist()
    rows.append([f"单卡 epoch={epoch}", "train_sampler=None", str(indices)])
setup_seed(args.seed + 0)                      # 再种一次同样的种子 → 复现 epoch 0
rows.append(["单卡 epoch=0（重跑）", "同样的 seed+epoch", str(torch.randperm(len(train_ds)).tolist())])

train_sampler = DistributedSampler(train_ds, num_replicas=2, rank=0)   # ---- DDP：2 张卡 ----
other = DistributedSampler(train_ds, num_replicas=2, rank=1)
train_sampler.set_epoch(0); other.set_epoch(0)                          # train_pretrain.py:159
rows.append(["双卡 rank=0", "DistributedSampler", str(list(train_sampler))])
rows.append(["双卡 rank=1", "DistributedSampler", str(list(other))])

# train_pretrain.py:162 的 `train_sampler or indices`：非空的 sampler 会让上面刚算出来的 indices 完全失效
chosen = train_sampler or indices
table(rows, headers=["场景", "真正生效的 sampler", "这个 epoch 的样本顺序"],
      title=f"`train_sampler or indices` 在 DDP 下取到的是 {type(chosen).__name__}，indices 被短路掉了")
