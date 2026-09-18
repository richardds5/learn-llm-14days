# ---
# title: 毕业演示：full_sft 权重跑一遍完整的工具调用闭环
# timeout: 180
# sources:
#   - scripts/eval_toolcall.py
# tasks:
#   - "把 PROMPT 换成 '现在几点了？'、tools 换成 get_tools(['get_current_time'])：模型调对工具了吗？观察它有没有把工具返回的时间抄进最终回答"
#   - "把 tools 传成 []（等于没给工具）：prompt 里的 # Tools 整块消失，模型这一轮还会吐 <tool_call> 吗"
# ---
import json, random, torch
from learnkit import *
from scripts.eval_toolcall import get_tools, parse_tool_calls, execute_tool

PROMPT = "帮我算一下 256 乘以 37 等于多少"                  # 👉 eval_toolcall.py 的 TEST_CASES[0]
tools = get_tools(["calculate_math", "get_current_time"])
tok, dev = get_tokenizer(), best_device()
model = load_model("full_sft", device=dev)

random.seed(0); torch.manual_seed(0)
messages, trace_rows = [{"role": "user", "content": PROMPT}], []
for turn in range(3):                                      # eval_toolcall.run_case 的 while True 循环
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tok(text, return_tensors="pt").to(dev)
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=96,
                         do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    content = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    calls = parse_tool_calls(content)
    trace_rows.append([f"turn {turn}", "assistant", repr(content)[:90], len(calls)])
    if not calls:
        break
    messages.append({"role": "assistant", "content": content})
    for call in calls:                                     # 真去执行，再以 role=tool 塞回对话
        result = execute_tool(call)
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
        trace_rows.append(["  ↳ 工具", call.get("name", "?"), json.dumps(result, ensure_ascii=False)[:90], "—"])

table(trace_rows, headers=["步骤", "角色 / 工具名", "内容", "解析出的 tool_call 数"],
      title=f"用户：{PROMPT} | 可用工具 {[t['function']['name'] for t in tools]}")
print(f"最终对话里有 {len(messages)} 条消息；模型眼里 role=tool 的那几条都是 <tool_response> 包起来的 user 消息")
note("这就是 14 天的终点：**jsonl → tokenizer → 64M 的 Transformer → pretrain / SFT → RL → 能在多轮里调工具的模型**。"
     "生成靠 Day 7 的 `generate`，模板靠 Day 1，`parse_tool_calls` 和 `execute_tool` 靠今天。")
