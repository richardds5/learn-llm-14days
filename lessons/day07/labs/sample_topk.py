# ---
# title: top-k 之后，词表里还剩几个候选
# timeout: 60
# tasks:
#   - "把 TOP_K 改成 1：第二张图只剩一根柱子吗？这时候 do_sample=True 和 argmax 有区别吗？"
#   - "把 TOP_K 改成 50（eval_llm.py 走的默认值）：第二张图和第一张图看起来还有差别吗？被置 -inf 的个数变成多少？"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
text = tok.apply_chat_template([{"role": "user", "content": "写一句关于秋天的诗"}],
                               tokenize=False, add_generation_prompt=True) + "秋"
ids = tok(text, return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = model(ids, use_cache=False).logits[0, -1, :]

TOP_K = 3                                                     # 👉 改这里
cut = logits.clone()
cut[cut < torch.topk(cut, TOP_K)[0][..., -1, None]] = -float('inf')   # generate L272 原样搬过来

idx = torch.topk(logits, 8).indices
labels = [repr(tok.decode([i]))[1:-1] for i in idx.tolist()]
n_inf = int(torch.isinf(cut).sum())
bars(labels, torch.softmax(logits, -1)[idx], highlight=[0], title="原始分布（前 8 个候选）")
bars(labels, torch.softmax(cut, -1)[idx], highlight=[0],
     title=f"top_k={TOP_K} 之后：{n_inf} / {len(logits)} 个位置被置成 -inf，剩下的重新归一")
