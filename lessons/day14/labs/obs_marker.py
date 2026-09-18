# ---
# title: 「模板新增了哪些文本」：直接减前缀 vs 插 marker 再 partition
# timeout: 60
# tasks:
#   - "把 case B 的 '\\n\\n</think>' 改回 '\\n</think>'、结尾 '\\n\\n\\n' 改回 '\\n\\n'：naive 那一列会不会变回 True？（模板 rstrip/lstrip 的正是这些换行）"
#   - "在 CASES 里加一条 open_thinking=True、内容里完全没有 </think> 的：naive 为什么必然失败？marker 还找得到吗"
# ---
from learnkit import *

tok = get_tokenizer()
tools = [{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
base = [{"role": "system", "content": ""}, {"role": "user", "content": "算算 7109*2920"}]
CALL = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'

CASES = [                                   # (说明, 模型这一轮的 new_text, open_thinking)
    ("A 规范的 think + 调用", "好的，我算一下。\n</think>\n\n" + CALL, True),
    ("B think 后多一个空行", "好的，我算一下。\n\n</think>\n\n\n" + CALL, True),
    ("C 无 think，直接调用", CALL, False),
    ("D 正文以换行开头", "\n\n" + CALL, False),
    ("E 输出里有两个 </think>", "想一下。\n</think>\n\n再想想</think>\n\n" + CALL, True),
]
rows, obs_demo = [], ""
for tag, new_text, ot in CASES:
    am = {"role": "assistant", "content": new_text}
    msgs = base + [am, {"role": "tool", "content": '{"result": "20758280"}'}]
    render = lambda: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=ot)
    ctx = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=ot)
    naive_ok = render().startswith(ctx + new_text)              # 「渲染结果 = 旧 context + 原样的 new_text」？
    marker = f"<|agent_observation_{id(msgs)}_0|>"              # 👉 train_agent.py:145 的哨兵
    am["content"] = new_text + marker
    _, found, observation = render().partition(marker)
    am["content"] = new_text                                    # 立刻还原，别把 marker 留在 messages 里
    obs_demo = obs_demo or observation
    rows.append([tag, ot, naive_ok, bool(found), len(tok(observation, add_special_tokens=False)["input_ids"])])

table(rows, headers=["模型这一轮输出长什么样", "open_thinking", "naive 减前缀成立", "marker 找得到", "观测增量 token 数"],
      title="模板会重排 assistant 内容，所以「减前缀」时灵时不灵；marker + partition 永远成立")
note("case A 拿到的观测增量：\n```\n" + obs_demo + "\n```")
