# ---
# title: 更新循环：ppo_update_iters × minibatch，一次 backward 喂两张图
# timeout: 240
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 ppo_update_iters 从 2 改成 1：少了哪两行日志？剩下的两行里 ratio 还偏离 1 吗？"
#   - "把 loss 换成只有 policy_loss（去掉 + 0.5 * value_loss）：critic.value_head.weight.grad 变成什么？（提示：不是 0）"
# ---
import torch
import torch.nn.functional as F
from torch import optim
from learnkit import *
from model.model_minimind import MiniMindConfig
import trainer.train_ppo as tp

cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)      # 👉 小模型，只看更新循环的机制
torch.manual_seed(0)
actor = build_model(hidden_size=128, num_hidden_layers=2).train()
critic = tp.CriticModel(cfg).train()
actor_opt, critic_opt = optim.AdamW(actor.parameters(), lr=1e-3), optim.AdamW(critic.parameters(), lr=1e-3)

B, P, R, ppo_update_iters, mb_size = 3, 9, 11, 2, 2             # 👉 ppo_update_iters=2 是源码默认
gen_out = torch.randint(3, 6400, (B, P + R)); full_mask = torch.ones(B, P + R, dtype=torch.long)
labels = gen_out[:, 1:]; logp_pos = torch.full((B, 1), P) - 1 + torch.arange(R).unsqueeze(0)
mask = torch.ones(B, R); adv = torch.randn(B, R); ret = torch.randn(B, R); old_v = torch.zeros(B, R)
logp = lambda m, ids: F.log_softmax(m(ids, attention_mask=full_mask[:ids.size(0)]).logits[:, :-1], -1)
with torch.no_grad():
    old_logp = logp(actor, gen_out).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

for ppo_epoch in range(ppo_update_iters):
    for i, inds in enumerate(torch.randperm(B).split(mb_size)):
        mb_v = critic(input_ids=gen_out[inds], attention_mask=full_mask[inds]).gather(1, logp_pos[inds])
        mb_logp = logp(actor, gen_out[inds]).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])
        ratio = torch.exp(mb_logp - old_logp[inds])
        policy_loss = (torch.max(-adv[inds] * ratio, -adv[inds] * ratio.clamp(0.8, 1.2)) * mask[inds]).sum() / mask[inds].sum()
        value_loss = 0.5 * (torch.max((mb_v - ret[inds]) ** 2,
                                      (mb_v.clamp(old_v[inds] - 0.2, old_v[inds] + 0.2) - ret[inds]) ** 2) * mask[inds]).sum() / mask[inds].sum()
        (policy_loss + 0.5 * value_loss).backward()             # 👈 一次 backward，梯度同时进两张图
        ga = actor.model.layers[0].self_attn.q_proj.weight.grad.norm().item()
        gc = critic.value_head.weight.grad.norm().item()
        live("ratio 随更新偏离 1", "ratio.mean", ppo_epoch * 2 + i, ratio.mean().item())
        print(f"ppo_epoch={ppo_epoch} mb={inds.tolist()} ratio.mean={ratio.mean().item():.6f} "
              f"policy={policy_loss.item():+.4f} value={value_loss.item():.4f} "
              f"|grad actor.q_proj|={ga:.3e} |grad critic.value_head|={gc:.3e}")
        actor_opt.step(); critic_opt.step(); actor_opt.zero_grad(); critic_opt.zero_grad()
