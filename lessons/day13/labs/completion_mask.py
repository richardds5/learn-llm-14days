# ---
# title: 用第一个 EOS 的下标切出 completion mask
# timeout: 60
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 eos_idx 那一行的花式索引去掉，直接写 eos_idx = is_eos.int().argmax(dim=1)：第 1 行保留几个 token？丢了几个 token 的梯度？"
#   - "把 <= eos_idx 改成 < eos_idx：第 0 行保留几个 token？模型还学得到「该在这里停」吗？"
# ---
import torch
from learnkit import *

tok = get_tokenizer()
EOS, R = tok.eos_token_id, 9  # EOS = 2 = <|im_end|>；每条轨迹最多 R 个 token
ids = lambda s, n: tok(s, add_special_tokens=False).input_ids[:n]

# generate 里某条序列一旦 finished，后面会被无脑塞满 EOS（model_minimind.py:279），
# 所以一行里可能有一长串 EOS，只有第一个是真的结束符。
completion_ids = torch.tensor([
    ids("你好呀朋友", 3) + [EOS] * 6,  # 第 3 步就结束，后面全是补出来的 EOS
    ids("今天天气真的很不错啊哈", 9),    # 一直没 EOS，被 max_new_tokens 截断
    [EOS] * 9,                        # 第 0 步就 EOS（空回答）
])  # [B*num_gen, R]
completion_pad_mask = torch.ones_like(completion_ids).bool()  # torch 引擎恒为全 1

# ↓↓↓ 下面 4 行与 trainer/train_grpo.py:128-131 完全一致 ↓↓↓
is_eos = (completion_ids == tok.eos_token_id) & completion_pad_mask
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long)  # 默认 R-1 = 全保留
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]     # 👉 .any() 兜底
completion_mask = ((torch.arange(R).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()

table([[i, ["提前 EOS + 填充", "没有 EOS（被截断）", "首 token 就 EOS"][i], bool(is_eos[i].any()),
        is_eos[i].int().argmax().item(), eos_idx[i].item(), completion_mask[i].sum().item()]
       for i in range(is_eos.size(0))],
      headers=["行", "这一行是什么情况", "is_eos.any()", "裸 argmax 会给出", "最终 eos_idx", "计入 loss 的 token 数"],
      title="第 4 列和第 5 列只有第 1 行不一样 —— 那一行就是 .any() 兜底救回来的")
for i in range(is_eos.size(0)):
    token_strip([tok.decode([t]) for t in completion_ids[i].tolist()], completion_mask[i],
                title=f"行 {i}", legend="1 = 计入 loss，0 = 被屏蔽")
