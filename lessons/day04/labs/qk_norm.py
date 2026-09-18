# ---
# title: QK-Norm 前后：每个 head 的 RMS 被拉到同一个数
# timeout: 60
# tasks:
#   - "把 attn.q_norm.weight 先置成全 1（加一行 attn.q_norm.weight.data.fill_(1.0)）：after 那一列变成多少？"
#   - "把 q_proj/q_norm 换成 k_proj/k_norm，头数换成 attn.n_local_kv_heads：4 个 KV 头的 RMS 也被拉齐吗？"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft", flash_attn=False), get_tokenizer()
attn = model.model.layers[0].self_attn                      # 👉 换层看看：layers[7]
H, D = attn.n_local_heads, attn.head_dim

ids = tok("今天天气怎么样，要不要带伞", return_tensors="pt").input_ids
x = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids))
xq = attn.q_proj(x).view(1, -1, H, D)                       # [B,T,H,D]，和源码 L114 一模一样
xq_n = attn.q_norm(xq)                                      # L117：shape 不变，变的是每个 head 的模长

rms = lambda t: t.pow(2).mean(-1).sqrt()[0]                 # 沿最后一维 D 求 RMS → [T, H]
before, after = rms(xq), rms(xq_n)
w_rms = attn.q_norm.weight.pow(2).mean().sqrt().item()
table([[f"head{h}", f"{before[:, h].min():.3f} ~ {before[:, h].max():.3f}",
        f"{after[:, h].min():.3f} ~ {after[:, h].max():.3f}"] for h in range(H)],
      headers=["head", "q_norm 之前：RMS 在各 token 上的范围", "q_norm 之后"],
      title=f"shape 不变 {tuple(xq.shape)} → {tuple(xq_n.shape)}；"
            f"归一化后全部贴着 RMS(q_norm.weight) = {w_rms:.3f}")
