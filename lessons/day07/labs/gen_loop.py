# ---
# title: generate 每一步到底喂进去多长的 input_ids
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 generate 的 use_cache 改成 False：past_len 那一列变成什么？「这一步喂进 forward 的 input_ids」那一列呢？"
#   - "把 max_new_tokens 从 5 改成 8：多出来的几行是 prefill 还是 decode？第 0 行会变吗？"
# ---
import torch
from learnkit import *

model = build_model(num_hidden_layers=2)      # 只看喂进去的形状，2 层就够
B, T = 3, 11                                   # 避开 config 里的 8/4/96/768…
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

rows = []
def watch(module, args):                       # MiniMindModel.forward(input_ids, attention_mask, past_key_values, use_cache)
    ids, past = args[0], args[2]
    past_len = past[0][0].shape[1] if past else 0   # 👉 和 generate L264 算 past_len 的方式一模一样
    rows.append([len(rows), str(tuple(ids.shape)), past_len, past_len + ids.shape[1]])

h = model.model.register_forward_pre_hook(watch)
model.generate(input_ids, max_new_tokens=5, do_sample=False, eos_token_id=None, use_cache=True)
h.remove()

table(rows, headers=["generate 第几步", "这一步喂进 forward 的 input_ids", "past_len", "模型这一步看到的总长度"],
      title="同一个 for 循环：第 0 步喂 [B,T]，之后每步只喂 [B,1]")
