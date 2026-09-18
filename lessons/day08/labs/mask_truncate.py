# ---
# title: 两个 max_length 下 generate_labels 找到的 loss span
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 MLS 改成 [96, 48]：这次找到几段？最后一段的解码内容结尾是不是 '<|im_end|>\\n'？"
#   - "把 MLS 里的 64 依次改成 63 和 62：最后一段分别剩几个 token？62 的时候为什么一段都找不到了？"
# ---
import json, tempfile
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import SFTDataset

tok = get_tokenizer()
conv = [{"role": "user", "content": "1+1="}, {"role": "assistant", "content": "2"},
        {"role": "user", "content": "再加 3 呢"}, {"role": "assistant", "content": "5"},
        {"role": "user", "content": "谢谢"}, {"role": "assistant", "content": "不客气"}]
path = Path(tempfile.mkdtemp()) / "toy_multi.jsonl"
path.write_text(json.dumps({"conversations": conv}, ensure_ascii=False), encoding="utf-8")
full = tok(SFTDataset(str(path), tok, max_length=4096).create_chat_prompt(conv)).input_ids

MLS = [96, 64]  # 👉 改这里：完整 prompt 需要 len(full) 个 token
rows = []
for ML in MLS:
    ids = full[:ML] + [tok.pad_token_id] * max(0, ML - len(full))
    labels, seg = SFTDataset(str(path), tok, max_length=ML).generate_labels(ids), None
    for j, l in enumerate(labels + [-100]):
        if l != -100: seg = [j, j] if seg is None else [seg[0], j]
        elif seg: rows.append([ML, f"[{seg[0]},{seg[1]}]", seg[1] - seg[0] + 1, repr(tok.decode(ids[seg[0]:seg[1] + 1]))]); seg = None
table(rows, headers=["max_length", "区间", "token 数", "解码内容"],
      title=f"完整 prompt 需要 {len(full)} 个 token；两个 max_length 各找到几段 assistant 回复")
