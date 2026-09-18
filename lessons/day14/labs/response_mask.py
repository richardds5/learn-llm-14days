# ---
# title: response_mask 逐 token 上色：1 = 模型自己写的，0 = 环境喂进来的
# timeout: 60
# sources:
#   - trainer/train_agent.py
# tasks:
#   - "把 SCRIPT[0] 末尾的 `</tool_call>` 删掉：parse_tool_calls 解析不出调用，这条轨迹几轮就结束？彩带里还剩几段 0"
#   - "把 FakeEngine 里的 `+ [self.tok.eos_token_id]` 去掉（假装模型这一轮没吐 eos）：response_ids 从 90 变成多少？彩带**末尾**少了哪一格？（train_agent.py:153 的去重只有在模型真吐了 eos 时才触发）"
# ---
import torch
from learnkit import *
from trainer.train_agent import rollout_single
from trainer.rollout_engine import RolloutResult

CALL = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'
SCRIPT = [CALL, "7109 乘以 2920 等于 20758280。"]      # 👉 假装模型两轮分别吐了这两段（省掉 generate，结果可复现）

class FakeEngine:                                      # 只实现 rollout_single 真正用到的那几个字段
    def __init__(self, tok): self.tok, self.i = tok, 0
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
        text = SCRIPT[self.i]; self.i += 1
        t = torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]])
        return RolloutResult(t, t, torch.full(t.shape, -0.7), [text], torch.tensor([0]), torch.ones_like(t))

tok = get_tokenizer()
tools = [{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
messages = [{"role": "system", "content": ""}, {"role": "user", "content": "算算 7109*2920"}]

_, _, p_ids, r_ids, r_mask, r_lp, turns, unfinished = rollout_single(
    FakeEngine(tok), tok, messages, tools, max_turns=3, max_new_tokens=64, thinking_ratio=0.0, device="cpu")

token_strip(tok.convert_ids_to_tokens(r_ids), r_mask,
            title=f"response_ids 共 {len(r_ids)} 个 token，其中 mask=1 的 {sum(r_mask)} 个",
            legend={"1": "模型生成 → 算 loss", "0": "eos / 工具结果 / 模板补的 → mask 掉"})
eos_pos = [i for i, t in enumerate(r_ids) if t == tok.eos_token_id]
print(f"prompt={len(p_ids)} tok（不在 response_ids 里）, response={len(r_ids)} tok, mask=1 的 {sum(r_mask)} 个")
print(f"eos(id={tok.eos_token_id}) 落在 response 的 {eos_pos} 位，对应 mask = {[r_mask[i] for i in eos_pos]}")
print(f"old_logps 里被填成 0.0 的有 {sum(1 for v in r_lp if v == 0.0)} 个 —— 正好是观测 token 数")
