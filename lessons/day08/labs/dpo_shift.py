# ---
# title: DPODataset 返回的 x / y / mask 是怎么错开的
# timeout: 90
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 MAX_LENGTH 从 128 改成 129：x/y/mask 的 shape 跟着变成多少？"
#   - "把倒数第二行的 mask_chosen 换成 item['mask_rejected']、x/y 也换成 rejected 那一组：mask 开始变 1 的下标一样吗？"
# ---
import random, tempfile
from itertools import islice
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import DPODataset

tok = get_tokenizer()
dst = Path(tempfile.mkdtemp()) / "toy_dpo.jsonl"  # dpo.jsonl 有 53MB，只取前 4 行
with open("dataset/dpo.jsonl", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in islice(fin, 4): fout.write(line)

random.seed(0)  # post_processing_chat 会随机删空 think 块，固定种子好复现
MAX_LENGTH = 128  # 👉 改这里；train_dpo.py 的默认值是 1024
item = DPODataset(str(dst), tok, max_length=MAX_LENGTH)[0]
x, y, m = item["x_chosen"], item["y_chosen"], item["mask_chosen"]

print(f"max_length={MAX_LENGTH}，但 x/y/mask 的 shape 是 {tuple(x.shape)}")
print("x[1:] == y[:-1] 恒成立:", bool((x[1:] == y[:-1]).all()))
s = int((m == 1).nonzero()[0])
table([[i, repr(tok.decode([x[i].item()])), repr(tok.decode([y[i].item()])), int(m[i])] for i in range(s - 5, s + 4)],
      headers=["i", "x[i]（喂进模型）", "y[i]（gather 的目标）", "mask[i]"],
      title=f"mask 从 i={s} 开始变成 1：此处 x[i] 还是 assistant 头部的最后一个 token，y[i] 已经是回复正文了")
