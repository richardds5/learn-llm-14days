# ---
# title: 四种输入下，assistant 段和 prompt 结尾长什么样
# timeout: 60
# tasks:
#   - "把 A 里 assistant 的 content 改成 '<think>\\n先算一下\\n</think>\\n\\n等于2'：第 1 行的 <think> 里出现了什么？第 2 行为什么多出一对 <think>（提示：reasoning_content 优先，content 里的 think 就没人去切了）"
#   - "给第 3 条 CASE 的参数里再加一个 open_thinking=False：和完全不传这个参数的结果一样吗？"
# ---
from learnkit import *

tok = get_tokenizer()
A = [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "等于2"}]
G = [{"role": "user", "content": "1+1"}]

CASES = [                                          # 👉 (说明, messages, apply_chat_template 的参数)
    ("训练态：一条普通 assistant 消息", A, {}),
    ("训练态：assistant 自带 reasoning_content",
     [A[0], dict(A[1], reasoning_content="1 加 1")], {}),
    ("推理态：add_generation_prompt", G, dict(add_generation_prompt=True)),
    ("推理态：再加 open_thinking=True", G, dict(add_generation_prompt=True, open_thinking=True)),
]

rows = []
for name, msgs, kw in CASES:
    text = tok.apply_chat_template(msgs, tokenize=False, **kw)
    tail = text[text.rindex("<|im_start|>assistant"):]      # 只看最后那个 assistant 段
    rows.append([name, repr(tail), "闭合" if "</think>" in tail else "没闭合，等模型自己写"])

table(rows, headers=["调用方式", "最后一个 assistant 段渲染成什么", "<think> 是否闭合"],
      title="assistant 段的四种渲染结果")
