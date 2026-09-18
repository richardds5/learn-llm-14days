# ---
# title: 从 full_response_masks 到 completion_mask：那段 eos 截断做了什么
# timeout: 60
# sources:
#   - trainer/train_agent.py
# tasks:
#   - "把 `[[0] * len(p) + m]` 改成 `[[0] * len(p) + [1] * len(m)]`（假装整段 response 都算 loss）：is_eos.any() 变成什么？token_counts 被截掉多少"
#   - "把 SCRIPT 改成只有一轮 `[\"直接回答：20758280。\"]`：token_counts 变成多少？valid_rows 还是 True 吗"
# ---
import torch
from learnkit import *
from trainer.train_agent import rollout_single
from trainer.rollout_engine import RolloutResult

CALL = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'
SCRIPT = [CALL, "7109 乘以 2920 等于 20758280。"]

class FakeEngine:
    def __init__(self, tok): self.tok, self.i = tok, 0
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
        text = SCRIPT[self.i]; self.i += 1
        t = torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]])
        return RolloutResult(t, t, torch.full(t.shape, -0.7), [text], torch.tensor([0]), torch.ones_like(t))

tok = get_tokenizer()
tools = [{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
messages = [{"role": "system", "content": ""}, {"role": "user", "content": "算算 7109*2920"}]
_, _, p, r, m, _, _, _ = rollout_single(FakeEngine(tok), tok, messages, tools, 3, 64, 0.0, "cpu")

input_ids = torch.tensor([p + r])
full_response_masks = torch.tensor([[0] * len(p) + m], dtype=torch.float32)
completion_mask = full_response_masks[:, 1:]                       # 👉 左移一位就是 per-token loss 的 mask
before = completion_mask.sum(dim=1).clone()

is_eos = (input_ids[:, 1:] == tok.eos_token_id) & completion_mask.bool()   # train_agent.py:292-297
eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, dtype=torch.long)
has_eos = is_eos.any(dim=1)
eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
pos = torch.arange(completion_mask.size(1)).unsqueeze(0)
completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
token_counts = completion_mask.sum(dim=1)

n_eos = int((input_ids[:, 1:] == tok.eos_token_id).sum())
table([["序列里 eos 的个数", n_eos],
       ["这些位置上 completion_mask 的值", [completion_mask[0, i].item() for i in (input_ids[0, 1:] == tok.eos_token_id).nonzero().flatten().tolist()]],
       ["is_eos.any()", is_eos.any().item()],
       ["eos_idx（没找到就是最后一列）", f"{eos_idx.tolist()} / 最后一列下标 {completion_mask.size(1) - 1}"],
       ["截断前后的 token_counts", f"{before.tolist()} → {token_counts.tolist()}"],
       ["valid_rows = token_counts > 0", (token_counts > 0).tolist()]],
      headers=["检查项", "值"], title=f"input_ids {tuple(input_ids.shape)} → completion_mask {tuple(completion_mask.shape)}")
