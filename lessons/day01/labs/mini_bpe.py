# ---
# title: 用同一套 API 现场训练一个迷你 BPE
# timeout: 90
# tasks:
#   - "把 VOCAB_SIZE 从 400 改成 300：merges 少了多少条？最长的那几个子词还在吗？"
#   - "把 initial_alphabet=... 那一行删掉再训一次：最后一行的单字符个数从 256 掉到多少？（只剩语料里真出现过的字节）"
# ---
import itertools
import json

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from learnkit import *

VOCAB_SIZE = 400  # 👉 官方是 6400；这里只用 40 行语料，几秒训完


def corpus():                                            # 和 train_tokenizer.py:get_texts 同样的读法
    with open("dataset/lora_identity.jsonl", encoding="utf-8") as f:
        for line in itertools.islice(f, 40):
            for msg in json.loads(line).get("conversations", []):
                if msg.get("content"): yield msg["content"]


tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.train_from_iterator(corpus(), trainer=trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE, show_progress=False,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))   # 256 个字节先占坑
tok.decoder = decoders.ByteLevel()

vocab = tok.get_vocab()
longest = sorted(vocab.items(), key=lambda kv: (len(kv[0]), -kv[1]))[-10:]
table([[t, i, repr(tok.decode([i]))] for t, i in longest],
      headers=["token（byte-level 显示）", "id", "decode 回原文"],
      title=f"{len(vocab)} 词表里合并次数最多的 10 个子词（= 语料里最高频的片段）")
n_char = len([t for t in vocab if len(t) == 1])
print(f"{n_char} 个单字符（字节替身）+ {len(vocab) - n_char} 条 merge = {len(vocab)}")
