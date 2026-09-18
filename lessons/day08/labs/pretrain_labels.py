# ---
# title: PretrainDataset 的 labels：哪些位置是 -100
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 MAX_LEN 从 20 改成 13：彩带右边会怎么变？「参与 loss」的计数变成多少？"
#   - "把 print 里的切片从 [:4] 改成 [:8]，逐位对照 input_ids 和 labels：它们是错开一位的，还是逐位相等的？"
# ---
import json, tempfile
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import PretrainDataset

tok = get_tokenizer()
# 👉 注意这条正文里带了一个字面的 <|endoftext|>（真实语料里用它分隔文档，很常见）
text = "论文里用 <|endoftext|> 分隔文档。"
path = Path(tempfile.mkdtemp()) / "toy_pretrain.jsonl"
path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")

MAX_LEN = 20  # 👉 改这里
ds = PretrainDataset(str(path), tok, max_length=MAX_LEN)
input_ids, labels = ds[0]

tokens = [tok.decode([i]) for i in input_ids.tolist()]
mask = [0 if l == -100 else 1 for l in labels.tolist()]
token_strip(tokens, mask, title=f"max_length={MAX_LEN}：labels 是不是 -100",
            legend="1 = labels[j] 是真 token（参与 loss）, 0 = labels[j] == -100")
print("参与 loss 的位置数:", sum(mask), "/", MAX_LEN)
print("input_ids[:4] =", input_ids[:4].tolist())
print("labels   [:4] =", labels[:4].tolist())
