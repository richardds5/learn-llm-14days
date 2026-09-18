# ---
# title: parse_response：一段自回归文本 → OpenAI 的 content / reasoning_content / tool_calls
# timeout: 60
# sources:
#   - scripts/serve_openai_api.py
# tasks:
#   - "把 case ⑤ 的 JSON 补成合法的：tool_calls 从 0 变成 1 条，content 从「带标签的原文」变成什么？（关键在 `if tool_calls:` 之后才 re.sub）"
#   - "把 case ② 的 `</think>` 删掉（只留 `<think>`）：走 if 分支还是 elif 分支？reasoning_content 会是 None 吗"
# ---
import ast, json, pathlib, re, time
from learnkit import *

# venv 里没装 fastapi，`import scripts.serve_openai_api` 会在第 12 行 import uvicorn 就炸，
# 所以用 ast 把 parse_response 这一个函数单独抠出来 exec（它只依赖 re / json / time）。
src = pathlib.Path("scripts/serve_openai_api.py").read_text(encoding="utf-8")
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "parse_response")
ns = {"re": re, "json": json, "time": time}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "scripts/serve_openai_api.py", "exec"), ns)
parse_response = ns["parse_response"]
print(f"从 scripts/serve_openai_api.py:{fn.lineno}-{fn.end_lineno} 抠出 parse_response，依赖只有 re/json/time")

CASES = [
    ("① 闭合 think + 普通回答", "<think>\n先算乘法，再回答。\n</think>\n\n7109 乘以 2920 等于 20758280。"),
    ("② think + tool_call", '<think>\n需要调用计算器。\n</think>\n\n<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'),
    ("③ 没有 think", "我直接回答：今天天气不错。"),
    ("④ 只有闭合标签（模板开了 <think>）", "好的，我想想。\n</think>\n\n答案是 42。"),
    ("⑤ tool_call 里 JSON 非法", '<tool_call>{"name": "x", "arguments": }</tool_call>后半段'),
    ("⑥ 两个 tool_call", '<tool_call>\n{"name": "a", "arguments": {"i": 1}}\n</tool_call>\n<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'),
]
rows = []
for tag, text in CASES:
    content, reasoning, tool_calls = parse_response(text)
    rows.append([tag, repr(text)[:40], repr(content)[:34], repr(reasoning)[:26],
                 0 if tool_calls is None else len(tool_calls), "tool_calls" if tool_calls else "stop"])
table(rows, headers=["case", "原始 text", "content", "reasoning_content", "tool_calls 条数", "finish_reason"],
      title="parse_response 把一段文本拆成三个字段；解析失败时标签原样留在 content 里")

content, reasoning, tool_calls = parse_response(CASES[1][1])       # 完整看一条
message = {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": tool_calls}
note("非流式 `/v1/chat/completions` 返回的 `choices[0]` 长这样：\n```json\n" + json.dumps(
    {"index": 0, "message": message, "finish_reason": "tool_calls"}, ensure_ascii=False, indent=2) + "\n```")
