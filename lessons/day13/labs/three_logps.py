# ---
# title: 同一批 outputs 上的三份 per-token logps
# timeout: 120
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 ref_model 换成 load_model('pretrain')：kl_div 的绝对值最大变成多少？per_token_kl 还全是 0 吗？"
#   - "把 old_per_token_logps 整体减掉 0.3 再跑一次：ratio 的 min/max 变成多少？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *
from trainer.rollout_engine import create_rollout_engine

tok = get_tokenizer()
model = load_model("full_sft")                                   # policy：唯一要梯度的模型
ref_model = load_model("full_sft").eval().requires_grad_(False)  # reference：和 policy 同一份权重（训练第 1 步的真实状态）
B, G = 3, 4

prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in ["你是谁？", "水为什么会结冰？", "写一句鼓励的话。"]]
enc = tok(prompts, return_tensors="pt", padding=True, return_token_type_ids=False, padding_side="left", add_special_tokens=False)
engine = create_rollout_engine("torch", policy_model=model, tokenizer=tok, device="cpu")
torch.manual_seed(0)
rr = engine.rollout(prompt_ids=enc["input_ids"], attention_mask=enc["attention_mask"], num_generations=G, max_new_tokens=16, temperature=0.9)

outputs, R = rr.output_ids, rr.completion_ids.size(1)
old_per_token_logps = rr.per_token_logps.to(torch.float32).detach()      # train_grpo.py:90
full_mask = (outputs != tok.pad_token_id).long()
logp_pos = rr.prompt_lens.unsqueeze(1) - 1 + torch.arange(R).unsqueeze(0)
full_mask.scatter_(1, logp_pos + 1, rr.completion_mask.to(full_mask.dtype))

lp = lambda m: F.log_softmax(m(outputs, attention_mask=full_mask).logits[:, :-1, :], -1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
per_token_logps = lp(model)                       # 👉 train_grpo.py:102，在 autocast 里做的那次 forward
with torch.no_grad():
    ref_per_token_logps = lp(ref_model)           # train_grpo.py:105
ratio = torch.exp(per_token_logps - old_per_token_logps)
kl_div = ref_per_token_logps - per_token_logps

table([[n, str(tuple(t.shape)), t.requires_grad, s] for n, t, s in [
    ("old_per_token_logps", old_per_token_logps, "rollout 里算的，立刻 .detach()"),
    ("per_token_logps", per_token_logps, "训练 forward，带梯度"),
    ("ref_per_token_logps", ref_per_token_logps, "ref_model，在 no_grad 里")]],
      headers=["名字", "shape", "requires_grad", "来源"], title=f"三份 logps（B*G={B * G}, R={R}）")
print(f"ratio = exp(per_token_logps - old_per_token_logps):  min={ratio.min():.6f}  max={ratio.max():.6f}")
print(f"kl_div = ref - policy:  绝对值最大 {kl_div.abs().max():.6f}   per_token_kl = exp(kl)-kl-1 最大 {(torch.exp(kl_div) - kl_div - 1).max():.6f}")
