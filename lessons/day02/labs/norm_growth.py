# ---
# title: 真实权重下，残差流的范数随深度怎么变
# timeout: 60
# tasks:
#   - "把 load_model('pretrain') 换成 load_model('full_sft')：曲线形状变了吗？终点数值呢？"
#   - "把 h, _ = layer(h, pe) 换成 MiniMindBlock.forward 的两半，中间各测一次范数：`h = h + layer.self_attn(layer.input_layernorm(h), pe)[0]` 和 `h = h + layer.mlp(layer.post_attention_layernorm(h))`。attn 和 mlp 谁贡献的增量更大？有没有哪一层的增量是负的？"
# ---
import torch
from learnkit import *

model, tok = load_model("pretrain"), get_tokenizer()          # 真实预训练权重，C=768, L=8
mm = model.model
input_ids = tok("床前明月光，疑是地上霜。", return_tensors="pt").input_ids

# 手动走一遍 MiniMindModel.forward 的骨架循环，只是每层结束后多测一次范数
h = mm.dropout(mm.embed_tokens(input_ids))
pe = (mm.freqs_cos[:input_ids.shape[1]], mm.freqs_sin[:input_ids.shape[1]])
norms, labels = [h.norm(dim=-1).mean().item()], ["embed 之后"]
for i, layer in enumerate(mm.layers):
    h, _ = layer(h, pe)                                        # 👉 每层内部都是 hidden_states += 子层输出
    norms.append(h.norm(dim=-1).mean().item())
    labels.append(f"layer{i} 之后")
norms.append(mm.norm(h).norm(dim=-1).mean().item())
labels.append("model.norm 之后")

table([[l, f"{n:.2f}", f"×{n / norms[0]:.1f}"] for l, n in zip(labels, norms)],
      headers=["位置", "‖hidden_states‖ 对 token 取平均", "相对 embed"],
      title=f"pretrain_768：残差流范数 {norms[0]:.1f} → {norms[-2]:.1f}（√C = {model.config.hidden_size ** 0.5:.1f}）")
