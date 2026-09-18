# ---
# title: 逐 token loss 彩带：同一句话里哪些字最难猜
# timeout: 60
# tasks:
#   - "把 text 换成一句词序打乱的病句：彩带整体是不是明显变深？平均 loss 离 ln(6400)≈8.76 还有多远"
#   - "把 load_model('pretrain') 换成 load_model('full_sft')：同一句裸文本上彩带变深还是变浅？（SFT 权重是在 chat 模板下训练的）"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

model, tok = load_model("pretrain"), get_tokenizer()
text = "今天天气很好，我和朋友一起去公园散步，看到很多人在跑步和聊天。"      # 👉 换句话试试
ids = [tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids
input_ids = torch.tensor([ids])
toks = [tok.decode([i]) for i in ids]

with torch.no_grad():
    logits = model(input_ids).logits                     # [1, T, V]
x, y = logits[..., :-1, :].contiguous(), input_ids[..., 1:].contiguous()
per = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1),
                      ignore_index=-100, reduction="none")    # 👉 reduction='none' 就是不做那次平均

# per[t] = 「看过 toks[0..t] 之后预测 toks[t+1] 有多意外」，所以标在 toks[1:] 上
token_strip(toks[1:], per,
            title=f"逐 token loss（平均 {per.mean().item():.2f}，完全猜不到的上限是 ln(6400)≈8.76）",
            legend="颜色越深 = loss 越高 = 模型越没想到这个 token")
