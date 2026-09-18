# ---
# title: 真实权重的 attention：第 0 层贴对角线，最后一层盯着第 0 列
# timeout: 90
# tasks:
#   - "把 caps[-1] 换成 caps[3]（第 3 层）：右图标题里第 0 列的占比变成多少？落在两端之间吗？"
#   - "去掉 apply_chat_template，直接用 tok(text, return_tensors='pt').input_ids（开头没有 <|im_start|>）：第 0 列还是一样亮吗？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *
from model.model_minimind import Attention

model, tok = load_model("full_sft", flash_attn=False), get_tokenizer()  # 手写分支才抓得到 scores
ids = tok.apply_chat_template([{"role": "user", "content": "今天天气怎么样"}],
                              tokenize=True, add_generation_prompt=True, return_tensors="pt")
labels = [tok.decode([i]) for i in ids[0].tolist()]

with trace(model, fns=[Attention.forward], capture=["scores"], tree=False,
           focus=(131, 131), max_calls=model.config.num_hidden_layers) as tr:
    model(ids)
caps = tr.captured["Attention.forward"]                    # 每层一份 scores（掩码后、softmax 前）

# 复现源码 L131 的那次 softmax，再对 8 个 head 取平均
attn = lambda c: F.softmax(c["scores"].float(), dim=-1)[0].mean(0)   # [T, T]
a0, aL = attn(caps[0]), attn(caps[-1])                     # 👉 换成 caps[3] 看中间层
sink = lambda a: a[1:, 0].mean().item()                    # 其余 token 分给第 0 个 token 的平均注意力
heatmap(torch.stack([a0, aL]), labels=labels,
        facet_titles=[f"第 0 层（第 0 列占 {sink(a0):.0%}）",
                      f"第 {len(caps) - 1} 层（第 0 列占 {sink(aL):.0%}）"],
        title="softmax 之后的 attention 权重，8 个 head 取平均；行 = query，列 = key")
