# ---
# title: 第一个 minibatch 的 log_ratio
# timeout: 180
# sources:
#   - trainer/rollout_engine.py
# tasks:
#   - "把 actor_model 换成 load_model('full_sft', dropout=0.1) 并在后面加 .train()：max|log_ratio| 变成多少？"
#   - "在两次计算之间插一次真实更新（optim.AdamW(actor_model.parameters(), lr=1e-3) 上跑 mb_resp_logp.sum().backward() + step()）：ratio 偏离 1 多少？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *
from trainer.rollout_engine import compute_per_token_logps

B, P, R = 3, 9, 13
actor_model = load_model("full_sft")
torch.manual_seed(0)
gen_out = torch.randint(3, 6400, (B, P + R))
full_mask = torch.ones(B, P + R, dtype=torch.long)
labels = gen_out[:, 1:]
logp_pos = torch.full((B, 1), P) - 1 + torch.arange(R).unsqueeze(0)

# rollout 阶段的 old_resp_logp：走 logits_to_keep 那条路 (rollout_engine.py:88)
with torch.no_grad():
    old_resp_logp = compute_per_token_logps(actor_model, gen_out, R, attention_mask=full_mask)
# 更新阶段的 mb_resp_logp：走全量 logits + 两次 gather 那条路 (train_ppo.py:179)，此时 actor 还没被 step 过
res = actor_model(input_ids=gen_out, attention_mask=full_mask)
mb_resp_logp = F.log_softmax(res.logits[:, :-1], -1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

log_ratio = (mb_resp_logp - old_resp_logp).detach()          # train_ppo.py:181
ratio = torch.exp(log_ratio)
print(f"max|log_ratio| = {log_ratio.abs().max().item():.3e}"
      f"   {log_ratio.numel()} 个元素里严格等于 0.0 的有 {int((log_ratio == 0).sum())} 个")
print(f"ratio: min={ratio.min().item():.8f}  max={ratio.max().item():.8f}")
print(f"approx_kl = mean(0.5*log_ratio^2) = {(0.5 * log_ratio ** 2).mean().item():.3e}"
      f"   clipfrac = mean(|ratio-1|>0.2) = {((ratio - 1).abs() > 0.2).float().mean().item():.3f}")
