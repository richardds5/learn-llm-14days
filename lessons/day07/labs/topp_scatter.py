# ---
# title: top-p 的四行代码：sort → cumsum → 右移一位 → scatter
# timeout: 30
# tasks:
#   - "把 top_p 改成 0.3（比 'cat' 自己的 0.5 还小），再把第 3 步整行换成 `mask[..., 1:] = mask[..., :-1].clone()`（只右移、去掉第 0 位保底）：最终还留得下几个候选？"
#   - "把 top_p 改成 1.0：mask 那两列会不会全是 False？（源码里 `if top_p < 1.0` 就是为了跳过这一整段）"
# ---
import torch
from learnkit import *

vocab = ["the", "cat", "sat", "on", "mat", "dog"]     # 6 个词的玩具词表，方便一眼看完
logits = torch.tensor([[2.0, 3.5, 1.0, 0.5, 0.2, 3.0]])   # [1, V]，模拟 generate 里某一行的 logits
top_p = 0.9                                                # 👉 改这里

sorted_logits, sorted_indices = torch.sort(logits, descending=True)      # 第 1 步
probs = torch.softmax(sorted_logits, dim=-1)
cumsum = torch.cumsum(probs, dim=-1)
mask = cumsum > top_p                                                     # 第 2 步：累计概率越过阈值的位置
before = mask.clone()
mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0                   # 第 3 步：整体右移一位 + 第 0 名保底
final = mask.scatter(1, sorted_indices, mask)                             # 第 4 步：散回原始词表顺序

order = [vocab[i] for i in sorted_indices[0].tolist()]
table([[i, order[i], f"{probs[0, i]:.3f}", f"{cumsum[0, i]:.3f}", bool(before[0, i]), bool(mask[0, i]),
        vocab[i], bool(final[0, i]), "-inf" if final[0, i] else f"{logits[0, i]:.2f}"] for i in range(len(vocab))],
      headers=["排序后位次", "token（按概率降序）", "概率", "累计概率", "cumsum>top_p", "右移后的 mask",
               "token（原始词表顺序）", "scatter 回来的 mask", "最终 logit"],
      title=f"左半边按「排序后」的顺序，右半边按「原始词表」的顺序（top_p={top_p}，最终留下 "
            f"{int((~final[0]).sum())} 个候选）")
