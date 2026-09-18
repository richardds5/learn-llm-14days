# ---
# title: SkipBatchSampler 丢掉前 skip 个 batch，剩下的顺序原样不动
# timeout: 60
# sources:
#   - trainer/trainer_utils.py
# tasks:
#   - "把 skip 从 3 改成 10（总共只有 10 个 batch）：len() 是多少？for 循环还会吐出东西吗？"
#   - "把 indices 的长度从 40 改成 38（batch_size=4 除不尽）：不跳过时最后一个 batch 有几个样本？len() 变成多少？"
# ---
from learnkit import *
from trainer.trainer_utils import SkipBatchSampler

indices = list(range(40))         # 这个 epoch 的样本顺序（真实脚本里是 randperm 出来的，或 DistributedSampler）
batch_size, skip = 4, 3           # 👉 skip = 续训时从 checkpoint 里读回来的 start_step

full = list(SkipBatchSampler(indices, batch_size, 0))
resumed = list(SkipBatchSampler(indices, batch_size, skip))

rows = []
for i in range(len(full)):
    j = i - skip
    rows.append([f"batch {i}", str(full[i]),
                 "🗑 被丢弃（续训前已经训过）" if j < 0 else f"第 {j} 个 yield 出来 → {resumed[j]}"])
aligned = resumed[0] == full[skip] if resumed else "（一个 batch 都没剩下）"
table(rows, headers=["原始第几个 batch", "内容", f"skip_batches={skip} 时"],
      title=f"len(不跳过)={len(full)}，len(跳过{skip}个)={len(resumed)}；"
            f"续训后的第 0 个 batch == 原来的第 {skip} 个：{aligned}")

# train_epoch 收到的 iters 是「补齐过」的：len(loader)+skip，这样 lr 调度和日志里的分母才和第一次训练一致
print(f"续训时 train_epoch(epoch, loader, iters={len(resumed)}+{skip}={len(resumed) + skip}, start_step={skip})")
print(f"→ step 从 {skip + 1} 数到 {len(resumed) + skip}，共 {len(resumed)} 步；编号和不中断训练时完全一致")
