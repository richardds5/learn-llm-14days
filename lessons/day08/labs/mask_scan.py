# ---
# title: 三轮对话的 loss mask 彩带
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把第 2 轮 assistant 的 content 改成空字符串 ''：这一段还会被 generate_labels 找到吗？它还剩几个亮格？"
#   - "把最后一条 assistant 消息删掉（对话以 user 结尾）：彩带右边少了什么？"
# ---
import json, tempfile
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import SFTDataset

tok = get_tokenizer()
conv = [  # 👉 改这里
    {"role": "user", "content": "1+1="}, {"role": "assistant", "content": "2"},
    {"role": "user", "content": "再加 3 呢"}, {"role": "assistant", "content": "5"},
    {"role": "user", "content": "谢谢"}, {"role": "assistant", "content": "不客气"},
]
path = Path(tempfile.mkdtemp()) / "toy_multi.jsonl"
path.write_text(json.dumps({"conversations": conv}, ensure_ascii=False), encoding="utf-8")

MAX_LENGTH = 96
ds = SFTDataset(str(path), tok, max_length=MAX_LENGTH)
# 走 __getitem__ 的同一条路，只是跳过两处随机扰动，好让输出可复现
input_ids = tok(ds.create_chat_prompt(conv)).input_ids[:MAX_LENGTH]
input_ids += [tok.pad_token_id] * (MAX_LENGTH - len(input_ids))
labels = ds.generate_labels(input_ids)

token_strip([tok.decode([i]) for i in input_ids], [0 if l == -100 else 1 for l in labels],
            title=f"max_length={MAX_LENGTH}：3 轮对话的 loss mask",
            legend="1 = labels[j] = input_ids[j]（参与 loss）, 0 = labels[j] = -100")
print("参与 loss 的 token 数:", sum(1 for l in labels if l != -100), "/", MAX_LENGTH)
