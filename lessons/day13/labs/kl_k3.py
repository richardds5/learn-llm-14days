# ---
# title: 三个 KL 估计器 vs 对全词表求和算出来的真 KL
# timeout: 120
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把噪声 5e-4 改成 2e-3（policy 和 ref 拉得更开）：真 KL 涨到多少？k1 的 std 跟着涨到多少？"
#   - "把每个位置的采样数 16 改成 1：三个估计器的 mean 离「真 KL」那一行差多远？（样本少时方差大的估计器最先跑偏）"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

tok = get_tokenizer()
policy = load_model("full_sft")
ref_model = load_model("full_sft")
torch.manual_seed(0)
with torch.no_grad():  # 👉 给 ref 加一点点噪声，模拟 policy 训了几十步之后和 ref 拉开的小差距
    for p in ref_model.parameters(): p.add_(torch.randn_like(p) * 5e-4)

texts = [tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False)
         for q, a in [("你是谁？", "我是 MiniMind，一个很小的语言模型。"),
                      ("水为什么会结冰？", "温度降到零度以下时，水分子排成晶格，就结成了冰。"),
                      ("写一句鼓励的话。", "慢一点没关系，只要还在往前走。")]]
enc = tok(texts, return_tensors="pt", padding=True, return_token_type_ids=False, padding_side="left", add_special_tokens=False)
ids, mask = enc["input_ids"], enc["attention_mask"]

with torch.no_grad():  # 每个有效位置上两个模型的完整分布，[N, V]
    lp_theta = F.log_softmax(policy(ids, attention_mask=mask).logits, -1)[mask.bool()]
    lp_ref = F.log_softmax(ref_model(ids, attention_mask=mask).logits, -1)[mask.bool()]
exact = (lp_theta.exp() * (lp_theta - lp_ref)).sum(-1)      # 真 KL(π_θ‖π_ref)：对 6400 个词求和
torch.manual_seed(1)
a = torch.multinomial(lp_theta.exp(), 16, replacement=True)  # 估计器的前提：token 是从 π_θ 采样出来的
lr = (lp_ref.gather(1, a) - lp_theta.gather(1, a)).flatten()  # 源码里的 kl_div = log π_ref − log π_θ
k1, k2, k3 = -lr, lr.pow(2) / 2, torch.exp(lr) - lr - 1      # 👉 k3 就是 train_grpo.py:134

table([[n, round(t.mean().item(), 5), round(t.std().item(), 4), round(t.min().item(), 4), int((t < 0).sum())]
       for n, t in [("k1 = −lr", k1), ("k2 = lr²/2", k2), ("k3 = e^lr − lr − 1", k3)]]
      + [["真 KL（全词表求和）", round(exact.mean().item(), 5), "—", "—", 0]],
      headers=["估计器", "mean", "std", "min", "出现负值的样本数"],
      title=f"{lr.numel()} 个从 π_θ 采样出来的 token 上，三个估计器对同一个真 KL 的三种估法")
