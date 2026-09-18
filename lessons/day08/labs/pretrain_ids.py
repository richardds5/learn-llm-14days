# ---
# title: PretrainDataset：一条 text 的长度账本
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 MAX_LEN 从 24 改成 16：哪几行的「正文 token 数」和「实际装进去」不再相等？「末尾那个特殊 token」还是 <|im_end|> 吗？"
#   - "把第 2 条 text 改成空字符串 ''：这一行的「实际装进去」是几？input_ids 的总长度还是 MAX_LEN 吗？"
# ---
import json, tempfile
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import PretrainDataset

tok = get_tokenizer()
# dataset/ 下没有 pretrain 主数据（太大没下载），这三条长短不一的文本是现造的，格式就是 {"text": ...}
texts = [
    "MiniMind 是一个从 0 开始训练的极小语言模型。",
    "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构，几乎所有主流模型都沿用了它。",
    "你好。",
]
path = Path(tempfile.mkdtemp()) / "toy_pretrain.jsonl"
path.write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts), encoding="utf-8")

MAX_LEN = 24  # 👉 改这里
ds = PretrainDataset(str(path), tok, max_length=MAX_LEN)

rows = []
for i in range(len(ds)):
    ids = ds[i][0].tolist()
    n_pad = sum(1 for t in ids if t == tok.pad_token_id)  # 右侧补齐的 <|endoftext|>
    rows.append([i, len(tok(texts[i], add_special_tokens=False).input_ids), MAX_LEN - 2,
                 len(ids) - n_pad - 2, n_pad, tok.decode([ids[0]]), tok.decode([ids[len(ids) - n_pad - 1]])])
table(rows, headers=["idx", "正文 token 数", "预算 max_length-2", "实际装进去", "末尾 padding 数", "第 0 个 token", "正文后那个特殊 token"],
      title=f"max_length={MAX_LEN} 时三条样本的长度账本（每行加起来恒等于 {MAX_LEN}）")
