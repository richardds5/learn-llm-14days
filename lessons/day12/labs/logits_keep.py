# ---
# title: compute_per_token_logps 的 logits_to_keep 与切片
# timeout: 180
# sources:
#   - trainer/rollout_engine.py
# tasks:
#   - "把 n_keep 从 R 改成 R-3：trace 里 logits 那一行之后的 shape 变成什么？"
#   - "把源码第 29 行的 logits_to_keep=n_keep+1 改成 n_keep（去掉 +1）：在哪一行报什么错？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *
from trainer.rollout_engine import compute_per_token_logps

B, P, R = 3, 9, 13
model = load_model("full_sft")
torch.manual_seed(0)
gen_out = torch.randint(3, 6400, (B, P + R))             # 模拟 rollout 拼好的 prompt+response
attention_mask = torch.ones(B, P + R, dtype=torch.long)
with trace(fns=[compute_per_token_logps], dims=dict(B=B, P=P, R=R, PR=P + R, R1=R + 1, V=6400),
           title="compute_per_token_logps 逐行（PR=P+R，R1=n_keep+1）", focus=(29, 36), tree=False):
    old_logp = compute_per_token_logps(model, gen_out, n_keep=R, attention_mask=attention_mask)

# ppo_train_epoch 用的是等价的另一种写法：全量 logits → 错一位 → gather 到 logp_pos (train_ppo.py:135/179)
with torch.no_grad():
    logits = model(gen_out, attention_mask=attention_mask).logits                  # [B, P+R, V]
labels = gen_out[:, 1:]                                                            # [B, P+R-1]
logp_pos = torch.full((B, 1), P) - 1 + torch.arange(R).unsqueeze(0)                # [B, R] = P-1 … P+R-2
full_logp = F.log_softmax(logits[:, :-1], -1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
print(f"两种写法的最大差值 = {(full_logp.gather(1, logp_pos) - old_logp).abs().max().item():.3e}")

# 把 [:, :-1, :] 换成 [:, 1:, :]：shape 一样是 [B, R, V]，但整体错了一位
with torch.no_grad():
    shifted = model(gen_out, attention_mask=attention_mask, logits_to_keep=R + 1).logits[:, 1:, :]
wrong = shifted.log_softmax(-1).gather(2, gen_out[:, -R:].unsqueeze(-1)).squeeze(-1)
print(f"改成 [:, 1:, :] 之后的最大差值 = {(wrong - old_logp).abs().max().item():.3f}  ← 不报错，只是静默算错")
