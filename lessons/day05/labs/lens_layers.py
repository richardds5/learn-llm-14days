# ---
# title: 用最终的 norm + lm_head 提前读每一层的 hidden_states
# timeout: 60
# tasks:
#   - "把 model.model.norm(h) 换成 h（跳过最后那次 RMSNorm）：各层的 top-1 还读得出来吗？概率那一列变成什么样"
#   - "换一句你自己的话：模型从第几层开始锁定最终答案（后面几层 top-1 不再变化）？"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
mm = model.model
text = "小猫喜欢吃鱼，小狗喜欢吃"                      # 👉 换句话试试
ids = [tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids
input_ids = torch.tensor([ids])

# 照抄 MiniMindModel.forward 的循环，只是每层跑完都顺手过一次最终的 norm + lm_head
h = mm.embed_tokens(input_ids)
pe = (mm.freqs_cos[:input_ids.shape[1]], mm.freqs_sin[:input_ids.shape[1]])
rows = []
with torch.no_grad():
    for i, layer in enumerate(mm.layers):
        h, _ = layer(h, pe)
        prob = torch.softmax(model.lm_head(mm.norm(h))[0, -1], dim=-1)   # [V]，只看最后一个位置
        top3 = torch.topk(prob, 3)
        rows.append([f"layer{i} 之后", " ".join(repr(tok.decode([j])) for j in top3.indices.tolist()),
                     f"{top3.values[0].item():.1%}", f"{h.norm(dim=-1).mean().item():.1f}"])

table(rows, headers=["hidden_states 来自", "提前解码的 top-3", "top-1 概率", "‖hidden_states‖"],
      title=f"logit lens：{text!r} 的下一个 token，逐层看一遍")
