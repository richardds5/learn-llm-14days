# ---
# title: 36 个特殊符号里，谁会被 skip_special_tokens 吃掉
# timeout: 60
# tasks:
#   - "把 SAMPLE 里的 <think> 换成 <|im_start|>：最后一行 print（skip_special=True）里这个位置还剩什么？"
#   - "把 specials[:3] 改成 specials[:21]：21 个 special=True 的全列出来了，里面有多少是为多模态预留的？"
# ---
from learnkit import *

tok = get_tokenizer()
SAMPLE = "<|im_start|>assistant\n<think>\n想一下\n</think>\n\n<tool_call>\n{}\n</tool_call><|im_end|>"

specials = [(i, t.content, t.special) for i, t in sorted(tok.added_tokens_decoder.items())]
n_true = sum(1 for _, _, s in specials if s)
rows = [[i, repr(c), "True" if s else "False", "decode 时被吃掉" if s else "decode 时保留"]
        for i, c, s in specials[:3] + [x for x in specials if not x[2]][:6]]
table(rows, headers=["id", "token", "special", "skip_special_tokens=True 时"],
      title=f"added_tokens_decoder 共 {len(specials)} 个：{n_true} 个 special=True，{len(specials) - n_true} 个被改回 False")

ids = tok.encode(SAMPLE)
print("原文            :", repr(SAMPLE))
print("skip_special=False:", repr(tok.decode(ids, skip_special_tokens=False)))
print("skip_special=True :", repr(tok.decode(ids, skip_special_tokens=True)))
