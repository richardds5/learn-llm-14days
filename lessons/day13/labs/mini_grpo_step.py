# ---
# title: 迷你 GRPO：用规则 reward 完整跑通 2 个 step
# timeout: 300
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 temperature 从 0.9 改成 0.2：同组 4 条回答变得更像之后，Adv Std 从 1.04 降到了多少？"
#   - "把 beta 从 0.1 改成 5.0：哪一步的 Actor Loss 变了、哪一步没变？为什么第 1 步不受影响？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *
from trainer.rollout_engine import create_rollout_engine
from trainer.train_grpo import rep_penalty  # 直接复用仓库里的 3-gram 重复惩罚

B, G, R = 3, 4, 32
beta, epsilon, epsilon_high, loss_type = 0.1, 0.2, 5.0, "cispo"  # 👉 train_grpo 的 4 个关键超参
tok = get_tokenizer()
model = load_model("full_sft")                                    # policy
ref_model = load_model("full_sft").eval().requires_grad_(False)   # reference，冻结
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
engine = create_rollout_engine("torch", policy_model=model, tokenizer=tok, device="cpu")
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in ["你是谁？", "介绍一下你的能力。", "你能帮我做什么？"]]
enc = tok(prompts, return_tensors="pt", padding=True, return_token_type_ids=False, padding_side="left", add_special_tokens=False)
# reward model（internlm2-1_8b-reward）的替身：长度区间分 + 「越啰嗦扣越多」+ 仓库原版的重复惩罚
rule_reward = lambda cs: torch.tensor([(0.5 if 20 <= len(c.strip()) <= 800 else -0.5) - len(c.strip()) / 100 - rep_penalty(c) for c in cs])

for step in (1, 2):
    torch.manual_seed(step)
    rr = engine.rollout(prompt_ids=enc["input_ids"], attention_mask=enc["attention_mask"], num_generations=G, max_new_tokens=R, temperature=0.9)
    outputs, comp = rr.output_ids, rr.completion_ids
    full_mask = (outputs != tok.pad_token_id).long()
    logp_pos = rr.prompt_lens.unsqueeze(1) - 1 + torch.arange(comp.size(1)).unsqueeze(0)
    full_mask.scatter_(1, logp_pos + 1, rr.completion_mask.to(full_mask.dtype))
    lp = lambda m: F.log_softmax(m(outputs, attention_mask=full_mask).logits[:, :-1, :], -1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
    per_token_logps = lp(model)
    with torch.no_grad(): ref_per_token_logps = lp(ref_model)
    rewards = rule_reward(rr.completions)
    gr = rewards.view(-1, G)
    advantages = (rewards - gr.mean(1).repeat_interleave(G)) / (gr.std(1, unbiased=False).repeat_interleave(G) + 1e-4)
    is_eos = (comp == tok.eos_token_id) & rr.completion_mask.bool()
    eos_idx = torch.full((B * G,), comp.size(1) - 1, dtype=torch.long)
    eos_idx[is_eos.any(1)] = is_eos.int().argmax(1)[is_eos.any(1)]
    completion_mask = ((torch.arange(comp.size(1)).expand_as(comp) <= eos_idx.unsqueeze(1)) & rr.completion_mask.bool()).int()
    kl_div = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(kl_div) - kl_div - 1
    ratio = torch.exp(per_token_logps - rr.per_token_logps.detach())
    per_token_loss = -(torch.clamp(ratio, max=epsilon_high).detach() * advantages.unsqueeze(1) * per_token_logps - beta * per_token_kl)
    policy_loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)).mean()
    optimizer.zero_grad(); policy_loss.backward(); optimizer.step(); engine.update_policy(model)
    live("迷你 GRPO", "Actor Loss", step, policy_loss.item())
    print(f"({step}/2) Reward: {rewards.mean():.4f}, KL_ref: {(kl_div * completion_mask).sum().item() / max(completion_mask.sum().item(), 1):.6f}, "
          f"Adv Std: {advantages.std():.4f}, Actor Loss: {policy_loss.item():.4f}, Avg Response Len: {completion_mask.sum(1).float().mean():.2f}, ratio_max: {ratio.max():.4f}")
