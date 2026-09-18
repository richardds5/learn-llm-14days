# ---
# title: 每条 message 各自渲染成了哪一段字符串
# timeout: 60
# tasks:
#   - "把第一条 system 挪到最后（messages 末尾）：它还会被渲染进开头的 system 段吗？变成什么了？"
#   - "把 add_generation_prompt 那一行改成 False：表格最后一行的尾巴消失了——训练数据该用哪一种？"
# ---
from learnkit import *

tok = get_tokenizer()
messages = [                                        # 👉 随便增删改，观察每一行的渲染片段
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好呀"},
    {"role": "user", "content": "再见"},
]

prev, rows = "", []
for k in range(1, len(messages) + 1):               # 逐条累加地渲染，差量就是这条 message 的贡献
    cur = tok.apply_chat_template(messages[:k], tokenize=False)
    rows.append([k - 1, messages[k - 1]["role"], repr(cur[len(prev):])])
    prev = cur
full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
rows.append(["—", "add_generation_prompt=True", repr(full[len(prev):])])

table(rows, headers=["messages 下标", "role", "渲染出来的片段"],
      title=f"4 条 message → {len(full)} 个字符 / {len(tok.encode(full))} 个 token")
