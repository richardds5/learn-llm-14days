# ---
# title: tools / tool_calls 少一次 json.loads 会怎样
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 conversations[0] 的 role 从 'system' 改成 'user'：create_chat_prompt 还找得到 tools 吗？渲染结果里 <tools> 那一段还在吗？"
#   - "把 conversations[2] 的 content 从 '' 改成 '我来算一下'：渲染结果里 <tool_call> 那一段前面多了什么？"
# ---
import json, tempfile
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import SFTDataset

tok = get_tokenizer()
tools_json = json.dumps([{"function": {"name": "calculate_math", "description": "计算数学表达式",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}], ensure_ascii=False)
calls_json = json.dumps([{"name": "calculate_math", "arguments": {"expression": "12*34"}}], ensure_ascii=False)
conversations = [  # 真实 jsonl 里 tools / tool_calls 都是 JSON **字符串**
    {"role": "system", "content": "", "tools": tools_json},
    {"role": "user", "content": "算算 12*34"},
    {"role": "assistant", "content": "", "tool_calls": calls_json},
]
path = Path(tempfile.mkdtemp()) / "toy_tool.jsonl"
path.write_text(json.dumps({"conversations": conversations}, ensure_ascii=False), encoding="utf-8")
ds = SFTDataset(str(path), tok, max_length=1024)

print(ds.create_chat_prompt(ds.samples[0]["conversations"]))  # 正确路径：两个字段都被 json.loads
rows = []
for name, kw in [("tools 传字符串（跳过 json.loads）", dict(tools=tools_json)),
                 ("tool_calls 传字符串（跳过 json.loads）", dict(tools=json.loads(tools_json)))]:
    msgs = [dict(m) for m in conversations]
    if "tool_calls" in name: msgs[2]["tool_calls"] = calls_json
    else: msgs[2]["tool_calls"] = json.loads(calls_json)
    try: out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw); rows.append([name, "渲染成功", len(tok(out).input_ids)])
    except Exception as e: rows.append([name, f"{type(e).__name__}: {str(e)[:70]}", "-"])
table(rows, headers=["少做的那一次 json.loads", "结果", "token 数"], title="create_chat_prompt 里那两行反序列化少一行会怎样")
