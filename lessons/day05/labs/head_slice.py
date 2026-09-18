# ---
# title: 四种 logits_to_keep 传值下的 slice_indices 与 logits shape
# timeout: 60
# tasks:
#   - "在 KEEPS 里再加一个 7（正好等于 T）：它的 slice_indices 长什么样？"
#   - "把 torch.tensor([0, 2, 4]) 换成 torch.tensor([-1])：logits 的第 1 维是几？和传 int 的 1 比，走的是哪一支？"
# ---
import torch
from learnkit import *

B, T = 3, 7
model = build_model(num_hidden_layers=2)
input_ids = torch.randint(0, model.config.vocab_size, (B, T))

KEEPS = [0, 1, 3, torch.tensor([0, 2, 4])]      # 👉 想试别的传值就改这里
rows = []
for k in KEEPS:
    # 完全照抄 model_minimind.py:L247 那一行三元表达式
    slice_indices = slice(-k, None) if isinstance(k, int) else k
    with torch.no_grad():
        logits = model(input_ids, logits_to_keep=k).logits
    rows.append([repr(k).replace("\n", ""), str(isinstance(k, int)),
                 repr(slice_indices).replace("\n", ""), str(list(logits.shape))])

table(rows, headers=["传入的 logits_to_keep", "isinstance(_, int)", "slice_indices", "logits.shape"],
      title=f"B={B}, T={T}, V={model.config.vocab_size}：同一次 forward，只有最后一步 lm_head 的输入位置数不同")
