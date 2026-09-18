# ---
# title: 一句话被 ByteLevel BPE 切成了什么
# timeout: 60
# tasks:
#   - "把 TEXT 换成 'Large language models'：一个英文单词占几个 token？对照最后一行的压缩率"
#   - "把 TEXT 换成一个生僻字 '龘'：它在词表里有对应子词吗？decode 还原得回来吗？"
# ---
from learnkit import *

tok = get_tokenizer()
TEXT = "我是谁"                                   # 👉 换成任意文本，中文 / 英文 / emoji 都行

ids = tok.encode(TEXT)
rows = []
for k, i in enumerate(ids):
    piece = tok.convert_ids_to_tokens(i)          # byte-level 的「伪字符」形式，不是给人读的
    rows.append([piece, i,
                 repr(tok.decode([i])),           # 单独 decode：字节不完整就是 '�'
                 repr(tok.decode(ids[:k + 1]))])  # 前 k+1 个一起 decode：凑齐字节才显出汉字

table(rows, headers=["token（byte-level 显示）", "id", "单独 decode", "前 k 个一起 decode"],
      title=f"{TEXT!r} → {len(ids)} 个 token")
print("拼起来 decode 和原文完全一致：", tok.decode(ids) == TEXT)
print(f"压缩率（字符 / token）：{len(TEXT)} / {len(ids)} = {len(TEXT) / len(ids):.2f}")
