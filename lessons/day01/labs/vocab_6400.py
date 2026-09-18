# ---
# title: 把 model/tokenizer.json 拆开数一遍：6400 是怎么凑出来的
# timeout: 60
# tasks:
#   - "把 tokenizer.json 换成统计 merges 的前 10 条：print(merges[:10])。最早学会的合并是哪些字节对？"
#   - "数一数 vocab 里长度 ≥ 6 的 token 有多少个（len(t) >= 6）：占 6400 的百分之几？"
# ---
import json
from learnkit import *

d = json.load(open("model/tokenizer.json", encoding="utf-8"))
vocab, merges, added = d["model"]["vocab"], d["model"]["merges"], d["added_tokens"]

n_byte = len([t for t in vocab if len(t) == 1])      # ByteLevel 的 256 个单字符（= 256 个字节的可打印替身）
n_merge = len(merges)                                # BpeTrainer 贪心合并出来的子词
n_special = len(added)                               # SPECIAL_TOKENS_NUM = 36

rows = [
    ["ByteLevel 初始字母表", n_byte, "pre_tokenizers.ByteLevel.alphabet()：每个字节先占一个 id，保证任何文本都编得出来"],
    ["BPE 学到的 merges", n_merge, "在 sft_t2t_mini.jsonl 前 10000 行上贪心合并出的子词"],
    ["特殊符号", n_special, f"tokenizer.json 的 added_tokens，占 id 0~{max(t['id'] for t in added)}"],
    ["合计", n_byte + n_merge + n_special, f"= len(vocab) = {len(vocab)}，也就是 MiniMindConfig.vocab_size"],
]
table(rows, headers=["组成部分", "数量", "来源"], title="6400 个格子的分账（直接数 model/tokenizer.json）")
print("三部分相加 == len(vocab)：", n_byte + n_merge + n_special == len(vocab))
