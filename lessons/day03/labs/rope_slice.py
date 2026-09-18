# ---
# title: start_pos 切片：这一次 forward 要表里的哪一段
# timeout: 60
# tasks:
#   - "把 next_id 从 (B, 1) 改成 (B, 3)（一次喂 3 个新 token）：第二行的 start_pos 变吗？cos 的第一维变成几？"
#   - "把第二次调用的 past_key_values=out1.past_key_values 删掉（当成没有 cache）：start_pos 变成多少？这时候新 token 会被当成第几个位置？"
# ---
import torch
from learnkit import *

B, T = 3, 7
model = build_model(num_hidden_layers=2)
mm = model.model
prompt = torch.randint(0, model.config.vocab_size, (B, T))
next_id = torch.randint(0, model.config.vocab_size, (B, 1))   # 👉 改成 (B, 3) 试试

rows = []
def step(tag, input_ids, past_key_values):
    past = past_key_values or [None] * len(mm.layers)
    start_pos = past[0][0].shape[1] if past[0] is not None else 0      # 和 L213 同一个表达式
    cos = mm.freqs_cos[start_pos:start_pos + input_ids.shape[1]]       # 和 L219 同一个表达式
    out = model(input_ids, past_key_values=past_key_values, use_cache=True)
    rows.append([tag, str(list(input_ids.shape)), start_pos, str(list(cos.shape)),
                 str(list(out.past_key_values[0][0].shape))])
    return out

out1 = step("① 完整 prompt，无 cache", prompt, None)
step("② 带 cache 再喂新 token", next_id, out1.past_key_values)

table(rows, headers=["这一次 forward", "input_ids", "start_pos", "cos 的 shape", "之后 cache 里 xk 的 shape"],
      title="freqs_cos[start_pos : start_pos + seq_length] 只覆盖「本次新增的位置」")
