# ---
# title: tools 分支与连续 tool 消息的合并
# timeout: 60
# tasks:
#   - "在 messages 最前面加一条 system：# Tools 那段说明前面会多出什么？（tools 分支接管了整个 system 段）"
#   - "把两条 tool 消息中间插一条 assistant：它们还会被裹进同一个 user 轮次吗？<|im_end|> 出现了几次？"
# ---
from learnkit import *

tok = get_tokenizer()
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询城市天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]

messages = [                                       # 👉 一轮完整的工具调用：问 → 调用 → 两条结果 → 回答
    {"role": "user", "content": "杭州天气"},
    {"role": "assistant", "content": "稍等",
     "tool_calls": [{"name": "get_weather", "arguments": {"city": "杭州"}}]},
    {"role": "tool", "content": '{"t": 26}'},
    {"role": "tool", "content": '{"w": "晴"}'},
]

with_tools = tok.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=True)
without = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

note("**传了 tools 的渲染结果：**\n```text\n" + with_tools + "\n```")
print("传 tools 比不传多出的字符数：", len(with_tools) - len(without), "（全都加在开头那个 system 段里）")
print("<|im_start|>user 出现次数：", with_tools.count("<|im_start|>user"),
      "= 1 个真正的提问 +", with_tools.count("<|im_start|>user") - 1, "个装工具结果的轮次")
print("<tool_response> 段数：", with_tools.count("<tool_response>"), "（每条 tool 消息一段）")
