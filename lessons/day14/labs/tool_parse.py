# ---
# title: parse_tool_calls → CHECK_ARGS → execute_tool：一段模型输出能走到哪一步
# timeout: 60
# sources:
#   - trainer/train_agent.py
# tasks:
#   - "把 case ⑤ 的 `{name: calculate_math}` 改成合法 JSON `{\"name\": \"calculate_math\", \"arguments\": {\"expression\": \"1+1\"}}`：解析出几条？parse_tool_calls 里那个裸 except 吞掉的正是这种错误"
#   - "把 case ⑨ 的 expression 改成 `__import__('os').getcwd()`：execute_tool 返回什么？回 trainer/train_agent.py:58 看 eval 的第二个参数 {\"__builtins__\": {}} 起了什么作用"
# ---
import json
from learnkit import *
from trainer.train_agent import parse_tool_calls, execute_tool, CHECK_ARGS, TOOLS

VALID_NAMES = {t["function"]["name"] for t in TOOLS}   # reward 里的 valid_names 就是这么来的
C = lambda name, args: '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (name, json.dumps(args, ensure_ascii=False))

OUTPUTS = [
    ("① 标准单次调用", C("calculate_math", {"expression": "7109*2920"})),
    ("② 一次两个调用", C("calculate_math", {"expression": "771-242"}) + "\n" + C("get_current_weather", {"location": "Tokyo"})),
    ("③ 夹在自然语言里", "好的，我先算一下。\n" + C("unit_converter", {"value": 100, "from_unit": "km", "to_unit": "miles"})),
    ("④ 全角符号表达式", C("calculate_math", {"expression": "（771−242）+84×27"})),  # 👉 train_agent.py:58 做了 6 次 replace
    ("⑤ JSON 非法", '<tool_call>\n{name: calculate_math}\n</tool_call>'),
    ("⑥ 标签不闭合", '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "1+1"}}'),
    ("⑦ 工具名不存在", C("search_web", {"query": "minimind"})),
    ("⑧ 必填参数缺失", C("get_exchange_rate", {"from_currency": "USD"})),
    ("⑨ eval 越权", C("calculate_math", {"expression": '__import__("os").system("ls")'})),
]

rows = []
for tag, text in OUTPUTS:
    calls = parse_tool_calls(text)                      # 👉 只认 <tool_call>…</tool_call> 里能 json.loads 的内容
    if not calls:
        rows.append([tag, 0, "—", "—", "—", "— (轨迹到此结束)"]); continue
    for call in calls:
        name, args = call.get("name", ""), call.get("arguments", {})
        check = CHECK_ARGS.get(name)
        ok_args = bool(name in VALID_NAMES and check and check(args))   # valid_call_count 的判据
        result = execute_tool(name, args)
        rows.append([tag, len(calls), name, json.dumps(args, ensure_ascii=False)[:34], ok_args,
                     json.dumps(result, ensure_ascii=False)[:38] if result else str(result)])

table(rows, headers=["模型输出", "解析出几条", "name", "arguments", "CHECK_ARGS 通过", "execute_tool 返回"],
      title="解析(parse_tool_calls) / 校验(CHECK_ARGS) / 执行(MOCK_RESULTS) 是三张互不相干的表")
