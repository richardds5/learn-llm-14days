# ---
# title: 打包：两条长短不一的轨迹 → 一批右 padding 的 tensor
# timeout: 60
# sources:
#   - trainer/train_agent.py
# tasks:
#   - "把 MAX_TOTAL_LEN 改成 300：哪几条被左截断？seq_lens 和 old_per_token_logps 的第二维分别变成多少"
#   - "把 old_logps 那行的 `max(len(p) - 1, 0)` 改成 `len(p)`：在哪一行、报什么错？（提示：老老实实数一下这一行该有多长）"
# ---
import torch
from learnkit import *
from trainer.train_agent import rollout_single
from trainer.rollout_engine import RolloutResult

MAX_TOTAL_LEN = 2500                                   # 👉 train_agent.py 的 --max_total_len 默认值
CALL = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'
SCRIPTS = [[CALL, "7109 乘以 2920 等于 20758280。"], ["我算一下：7109 乘以 2920 等于 20758280，这个结果可以直接用。"]]

class FakeEngine:                                      # 脚本化引擎：两条轨迹分别演「调工具」和「直接回答」
    def __init__(self, tok, script): self.tok, self.script, self.i = tok, script, 0
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
        text = self.script[self.i]; self.i += 1
        t = torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]])
        return RolloutResult(t, t, torch.full(t.shape, -0.7), [text], torch.tensor([0]), torch.ones_like(t))

tok = get_tokenizer()
tools = [{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
base = [{"role": "system", "content": ""}, {"role": "user", "content": "算算 7109*2920"}]
traj = [rollout_single(FakeEngine(tok, s), tok, [dict(m) for m in base], tools, 3, 64, 0.0, "cpu") for s in SCRIPTS]

packed = []                                            # 照抄 trainer/train_agent.py:260-270
for _, _, p, r, m, old_lp, _, _ in traj:
    ids, mask = p + r, [0] * len(p) + m
    old_logps = [0.0] * max(len(p) - 1, 0) + old_lp    # 👉 len(p)-1：per-token 量的下标比 token 下标小 1
    if len(ids) > MAX_TOTAL_LEN:                       # 超长就砍掉最老的一段，保留最新的上下文
        ids, mask = ids[-MAX_TOTAL_LEN:], mask[-MAX_TOTAL_LEN:]
        old_logps = old_logps[-(len(ids) - 1):]
    packed.append((ids, mask, old_logps))

seq_lens = torch.tensor([len(i) for i, _, _ in packed])
L = seq_lens.max().item()
input_ids = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i, _, _ in packed])
full_response_masks = torch.tensor([m + [0] * (L - len(m)) for _, m, _ in packed], dtype=torch.float32)
old_per_token_logps = torch.tensor([o + [0.0] * ((L - 1) - len(o)) for _, _, o in packed], dtype=torch.float32)
full_mask = (torch.arange(L).unsqueeze(0) < seq_lens.unsqueeze(1)).long()
show(input_ids=input_ids, full_mask=full_mask, full_response_masks=full_response_masks,
     old_per_token_logps=old_per_token_logps, seq_lens=seq_lens, dims=dict(BG=len(packed)))

p0 = next(i for i, v in enumerate(packed[0][1]) if v == 1)     # train_agent.py:269 的 prompt_len
print(f"两条轨迹真实长度 {seq_lens.tolist()}，右 padding 到 L = {L}")
print(f"对齐自检：mask 里第一个 1 在下标 {p0}（= response 第 0 个 token），它的 logprob 该落在 {p0-1}")
print(f"  full_response_masks[0][{p0-1}:{p0+1}] = {full_response_masks[0][p0-1:p0+1].tolist()}，"
      f"old_per_token_logps[0][{p0-2}:{p0+1}] = {[round(v,2) for v in old_per_token_logps[0][p0-2:p0+1].tolist()]}")
print("  👆 old_logps 前面填的 0.0 正好在下标 p0-1 处结束，下一格开始才是 rollout 攒下来的真实 logprob")
