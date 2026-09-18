# ---
# title: 同一个 top_p，用在两条尖锐程度不同的分布上
# timeout: 60
# tasks:
#   - "把 TOP_P 改成 0.95：两条分布留下的候选数各变成多少？哪一条涨得更猛？"
#   - "把 TOP_P 改成 1.0：候选数会回到 6400 吗？（源码里 `if top_p < 1.0` 会整段跳过这几行，实验没有这层守卫——想想 float32 的 cumsum 在长尾上会提前变成多少）"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
TOP_P = 0.85                                                   # 👉 改这里

def top_p_filter(logits):                                      # generate L274-L277 原样搬过来
    lg = logits.clone().unsqueeze(0)
    sorted_logits, sorted_indices = torch.sort(lg, descending=True)
    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > TOP_P
    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
    lg[mask.scatter(1, sorted_indices, mask)] = -float('inf')
    return lg[0]

for user, written in [("写一句关于秋天的诗", "秋"), ("为什么天空是蓝色的", "天空是蓝色的，因为")]:
    text = tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True) + written
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0, -1, :]
    kept = top_p_filter(logits)
    n_kept = int((kept > -float('inf')).sum())
    p, top = torch.softmax(kept, -1), torch.topk(logits, 10).indices
    bars([repr(tok.decode([i]))[1:-1] for i in top.tolist()], p[top], highlight=[0],
         title=f"已写「{written}」，原始 top1={torch.softmax(logits, -1).max():.3f} → top_p={TOP_P} 留下 {n_kept} 个候选")
