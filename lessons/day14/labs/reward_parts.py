# ---
# title: calculate_rewards 拆解：七条轨迹各拿到多少分
# timeout: 60
# sources:
#   - trainer/train_agent.py
# tasks:
#   - "把 case A 的答案改成 '结果是 20,758,280。'（带千分位逗号）：GT 还命中吗？看 validate_gt_in_text 里 `text.replace(',', '')` 那一步"
#   - "把 case A 的 gt 改成 ['20758280', '529']（假装要两次计算）：tool_gap 变成多少？GT 分变成多少？总分呢"
# ---
from learnkit import *
from trainer.train_agent import calculate_rewards, parse_tool_calls, validate_gt_in_text, rep_penalty, CHECK_ARGS, TOOLS

TL = [t for t in TOOLS if t["function"]["name"] in ("calculate_math", "get_current_weather")]
VALID = {t["function"]["name"] for t in TL}
CALL = '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'
BAD = '<tool_call>\n{"name": "search_web", "arguments": {"q": "x"}}\n</tool_call>'

CASES = [                                        # (说明, 每一轮 assistant 的原始输出, gt, unfinished)
    ("A 调对工具 + 答对", [CALL, "7109 乘以 2920 等于 20758280。"], ["20758280"], False),
    ("B 调对工具 + 答错", [CALL, "7109 乘以 2920 等于 20758000。"], ["20758280"], False),
    ("C <tool_call> 不闭合", ['<tool_call>\n{"name": "calculate_math"}', "算完了 20758280"], ["20758280"], False),
    ("D 打满 max_turns", [CALL, CALL, CALL], ["20758280"], True),
    ("E 工具名不存在 + 答对", [BAD, "结果 20758280"], ["20758280"], False),
    ("F 不调工具 + 规范思考", ["<think>\n用户在闲聊，直接回答即可，不需要调用任何工具。\n</think>\n\n今天天气不错，适合散步。"], [], False),
    ("G 不调工具 + 复读", ["<think>\n用户在闲聊，直接回答即可，不需要调用任何工具。\n</think>\n\n" + "好的好的好的 " * 40], [], False),
]

rows = []
for name, turns, gt, unf in CASES:
    total = calculate_rewards([""], [turns[-1]], [gt], [TL], 1, None, device="cpu",
                              turn_outputs_batch=[turns], unfinished_batch=[unf]).item()
    ta = [t.split("</think>", 1)[-1].strip() if "</think>" in t else t.strip() for t in turns]
    answer, calls = ta[-1], [c for t in ta for c in parse_tool_calls(t)]
    tag = -0.5 * sum(abs(t.count("<tool_call>") - t.count("</tool_call>")) for t in ta)
    if calls:                                    # ---- 分支 B：有工具调用 ----
        valid = sum(int(c.get("name") in VALID and CHECK_ARGS.get(c.get("name"), lambda a: 0)(c.get("arguments", {}))) for c in calls)
        gap = abs(valid - len(gt)) + max(0, len(calls) - valid)
        ft = "" if unf else (answer.split("</tool_call>")[-1] if "</tool_call>" in answer else answer)
        hit = validate_gt_in_text(ft, gt) if gt else set()
        rows.append([name, "有工具", tag or 0, f"valid {valid}/{len(calls)}, gap={gap}", round(0.5 if gap == 0 else -0.5 * gap, 2),
                     f"{len(hit)}/{len(gt)} → +{round(2.5 * len(hit) / len(gt), 2)}", -0.5 if unf else 0,
                     round(-rep_penalty(ft or answer), 3) or 0, round(total, 3)])
    else:                                        # ---- 分支 A：一条都没解析出来 ----
        resp, think = turns[-1], 0.0
        if "</think>" in resp:
            think = (1.0 if 20 <= len(resp.split("</think>", 1)[0].strip()) <= 300 else -0.5) + (0.25 if resp.count("</think>") == 1 else -0.25)
        rows.append([name, "无工具", tag or 0, "—", f"长度分 {0.5 if 5 <= len(resp.strip()) <= 800 else -0.5}",
                     f"think 分 {round(think, 2)}", 0, round(-rep_penalty(answer), 3) or 0, round(total, 3)])

table(rows, headers=["轨迹", "走哪个分支", "标签扣分", "tool 校验", "tool 对齐分 / 长度分", "GT 分 / think 分", "未完成", "复读扣分", "总分"],
      title="总分来自源码 calculate_rewards；其余列用源码里的子函数复算，方便对账（总分硬夹在 [-3, 3]）")
