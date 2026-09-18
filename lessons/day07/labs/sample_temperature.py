# ---
# title: 同一条分布，被 temperature 拉尖和摊平
# timeout: 60
# tasks:
#   - "把 2.0 改成 10.0：8 根柱子会不会几乎一样高？再改成 0.05 呢（接近 argmax）？"
#   - "把 8 改成 3，只看前 3 个候选：temperature 变化时，这 3 根柱子的高矮顺序有没有换过？"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
text = tok.apply_chat_template([{"role": "user", "content": "写一句关于秋天的诗"}],
                               tokenize=False, add_generation_prompt=True) + "秋"
ids = tok(text, return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = model(ids, use_cache=False).logits[0, -1, :]    # [V]：已经生成「秋」，下一个 token 的原始 logits

idx = torch.topk(logits, 8).indices                          # 👉 固定盯住概率最高的这 8 个候选
labels = [repr(tok.decode([i]))[1:-1] for i in idx.tolist()]

for temperature in (1.0, 0.5, 2.0):                          # 👉 generate L267 就是 logits / temperature
    p = torch.softmax(logits / temperature, dim=-1)
    bars(labels, p[idx], highlight=[0],
         title=f"temperature={temperature}：top1 概率 {p[idx[0]]:.3f}，这 {len(idx)} 个候选合计 {p[idx].sum():.3f}")
