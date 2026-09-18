# ---
# title: T=6 的小例子：每一格 logits 被拿去对哪一个 label
# timeout: 60
# tasks:
#   - "把 y 换成不做 shift 的 labels[..., :-1]（让 logits[t] 去对 labels[t] 自己）：每一格的 loss 变成多少？和『top-1』那一列对照着看"
#   - "把开头的 bos_token_id 去掉（直接用 tok(text).input_ids）：t=0 那一格的 loss 变成多少？（PretrainDataset 里每条样本都是以 BOS 开头的）"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

model, tok = load_model("pretrain"), get_tokenizer()
text = "小猫喜欢吃鱼"                                    # 👉 换一句话试试
ids = [tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids   # PretrainDataset 也是这么拼的
input_ids = torch.tensor([ids])                              # [1, T]
labels = input_ids.clone()                                   # 和 input_ids 逐位对齐，没有任何错位
toks = [tok.decode([i]) for i in ids]
T = len(toks)

with torch.no_grad():
    logits = model(input_ids).logits                         # [1, T, V]
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()      # 源码 L251 的 shift
per = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100, reduction="none")

rows = [[t, repr(toks[t]), repr(tok.decode([logits[0, t].argmax().item()])),
         repr(toks[t + 1]), f"{per[t].item():.2f}"] for t in range(T - 1)]
rows.append([T - 1, repr(toks[T - 1]), "—", "没有下一个 token 了", "被 [..., :-1, :] 切掉"])
table(rows, headers=["t", "input_ids[t]", "logits[t] 的 top-1", "监督目标 = labels[t+1]", "这一格的 loss"],
      title=f"{text!r}：T={T} 个位置，只有 T-1={T - 1} 项 loss（labels[0]={toks[0]!r} 没有任何 logits 去预测它）")
