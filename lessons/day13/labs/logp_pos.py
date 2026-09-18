# ---
# title: logp_pos 的值域，以及 full_mask 被 scatter_ 改回来的那一格
# timeout: 60
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 completion 里那个 pad_token_id（0）换成普通 id（比如 1968）：scatter_ 之前的热力图还有被误判成 padding 的格子吗？"
#   - "把 prompt_lens 改成 [P, P-2, P-1]（模拟 sglang 的变长 prompt）：logp_pos 的三行还一样吗？scatter_ 写进去的位置跟着动了没有？"
# ---
import torch
from learnkit import *

tok = get_tokenizer()
PAD, P, R = tok.pad_token_id, 6, 5  # PAD = 0 = <|endoftext|>；prompt 左 padding 到 P，最多生成 R 个 token

# [B*G, P+R]：前 P 列是左 padding 过的 prompt，后 R 列是 completion
outputs = torch.tensor([
    [PAD, PAD, 1, 832, 311, 2] + [456, 789, 321, 654, 987],
    [PAD, 1, 832, 311, 457, 2] + [111, PAD, 222, 333, 444],  # 👉 模型真的吐出了一个 id==PAD 的 token
    [1, 832, 311, 457, 458, 2] + [555, 666, 777, 888, 999],
])
prompt_lens = torch.full((3,), P)                  # torch 引擎里每行都是同一个 P
completion_mask_from_engine = torch.ones(3, R).long()  # torch 引擎给的是全 1

full_mask = (outputs != PAD).long()                # ← 只按「等于 pad id 就算 padding」推，会误伤
before = full_mask.clone()
logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(R).unsqueeze(0)  # [B*G, R]，train_grpo.py:93
full_mask.scatter_(1, logp_pos + 1, completion_mask_from_engine)        # 👉 train_grpo.py:94

for i in range(3):
    print(f"logp_pos[{i}] = {logp_pos[i].tolist()}   →  scatter_ 写进去的列 = {(logp_pos[i] + 1).tolist()}")
print(f"公式：logp_pos[:, j] = prompt_lens - 1 + j（预测第 j 个 completion token 的那个位置）；scatter_ 写的是 logp_pos + 1（token 自己占的列）")
heatmap(torch.stack([before, full_mask]).float(), title="full_mask：黄=1 参与 attention，紫=0 当 padding",
        facet_titles=["scatter_ 之前（第 1 行第 7 列被误判）", "scatter_ 之后（引擎给的真值覆盖回来）"])
