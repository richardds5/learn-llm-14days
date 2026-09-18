# ---
# title: 变长结果补成矩形之后，prompt_lens 和 completion_mask 才真的有用
# timeout: 60
# sources:
#   - trainer/rollout_engine.py
# tasks:
#   - "把三条 completion 改成一样长：completion_mask 还有 0 吗？logp_pos 的三行还互不相同吗？"
#   - "把第 1 条 prompt 砍成 [1, 811, 322, 2]（4 个 token）：max_out_len 变成多少？第 1 行的 completion 挪到了哪几列？"
# ---
import torch
from learnkit import *

PAD = get_tokenizer().pad_token_id
# 模拟 sglang server 返回的东西：prompt 已经去掉左 padding，completion 是变长 list
all_input_ids = [[1, 811, 322, 764, 2], [1, 811, 322, 764, 900, 431, 2], [1, 811, 512, 2]]
all_completion_ids = [[41, 42, 43, 44, 45, 46], [71, 72], [91, 92, 93, 94]]
all_output_ids = [p + c for p, c in zip(all_input_ids, all_completion_ids)]

# ↓↓↓ 和 rollout_engine.py:159-172 一致：自己把变长结果 pad 回矩形 ↓↓↓
max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
pad_to = lambda seqs, n, v=PAD: torch.tensor([s + [v] * (n - len(s)) for s in seqs])
output_ids = pad_to(all_output_ids, max_out_len)                        # [B*G, max_out_len]
completion_ids = pad_to(all_completion_ids, max_comp_len)               # [B*G, max_comp_len]
prompt_lens = torch.tensor([len(ids) for ids in all_input_ids])         # 👉 每行都不一样
completion_mask = torch.tensor([[1] * len(c) + [0] * (max_comp_len - len(c)) for c in all_completion_ids])

logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(max_comp_len).unsqueeze(0)  # train_grpo.py:93
table([[i, prompt_lens[i].item(), len(all_completion_ids[i]),
        f"{prompt_lens[i].item()} … {prompt_lens[i].item() + len(all_completion_ids[i]) - 1}",
        str(logp_pos[i].tolist()), completion_mask[i].sum().item()] for i in range(3)],
      headers=["行", "prompt_len", "completion 真实长度", "completion 占 output_ids 的哪几列", "logp_pos", "mask 里 1 的个数"],
      title=f"output_ids {tuple(output_ids.shape)}：三行的 completion 起点各不相同（max_out_len={max_out_len}）")
note("torch 引擎里 `prompt_lens` 是常量、`completion_mask` 全 1，所以 `gather(1, logp_pos)` 看着像脱裤子放屁；"
     "在这里它是**必需**的：每行的 completion 从不同的列开始，还带着右 padding。")
