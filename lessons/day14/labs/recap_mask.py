# ---
# title: 同一段轨迹，SFT 和 Agent RL 分别监督哪些 token
# timeout: 90
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "盯住 `<|im_end|>` + `\\n` 那一段：SFT 是 1、Agent RL 是 0。把 generate_labels 的 `min(end + len(self.eos_id), ...)` 改成 `min(end, ...)`，SFT 那一列会怎么变"
#   - "把 SCRIPT 改成一轮 `[\"直接回答：20758280。\"]`（不调工具）：表里还剩几段「👉 不一致」？"
# ---
import json, os, tempfile, torch
from learnkit import *
from dataset.lm_dataset import SFTDataset
from trainer.train_agent import rollout_single
from trainer.rollout_engine import RolloutResult

CALL = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'
SCRIPT = [CALL, "7109 乘以 2920 等于 20758280。"]

class FakeEngine:                                          # 同 response_mask 那个实验，脚本化省掉 generate
    def __init__(self, tok): self.tok, self.i = tok, 0
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
        text = SCRIPT[self.i]; self.i += 1
        t = torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]])
        return RolloutResult(t, t, torch.full(t.shape, -0.7), [text], torch.tensor([0]), torch.ones_like(t))

tok = get_tokenizer()
tools = [{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
messages = [{"role": "system", "content": ""}, {"role": "user", "content": "算算 7109*2920"}]
_, _, p, r, agent_mask, _, _, _ = rollout_single(FakeEngine(tok), tok, messages, tools, 3, 64, 0.0, "cpu")
ids, agent = p + r, [0] * len(p) + agent_mask               # rl_train_epoch 打包出来的那一条

d = tempfile.mkdtemp()                                     # SFTDataset 要一个 jsonl 才能构造
with open(os.path.join(d, "s.jsonl"), "w", encoding="utf-8") as f:
    f.write(json.dumps({"conversations": messages[:2]}, ensure_ascii=False) + "\n")
ds = SFTDataset(os.path.join(d, "s.jsonl"), tok, max_length=len(ids))
sft = [int(v != -100) for v in ds.generate_labels(ids)]    # 👉 拿同一条 input_ids 走一遍 SFT 的打标签逻辑

rows, i = [], 0                                            # 按 (sft, agent) 相同的连续段落折叠
while i < len(ids):
    j = i
    while j < len(ids) and (sft[j], agent[j]) == (sft[i], agent[i]): j += 1
    rows.append([repr(tok.decode(ids[i:j]))[:52], j - i, sft[i], agent[i],
                 "一致" if sft[i] == agent[i] else "👉 不一致"])
    i = j
table(rows, headers=["这一段文本", "token 数", "SFT labels≠-100", "Agent response_mask", ""],
      title=f"同一条 {len(ids)} token 的序列：SFT 监督 {sum(sft)} 个，Agent RL 只算 {sum(agent)} 个")
