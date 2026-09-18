# ---
# title: 两层 mask：full_mask 与 resp_policy_mask
# timeout: 60
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 eos_at 改成 [0, 5, 10]：三行的 resp_lengths 和 resp_policy_mask 的 1 的个数各变成多少？"
#   - "把 resp_pad_mask 第 2 行的后 4 位改成 False（模拟 SGLang 引擎的右 padding）：那一行的 full_mask 在 response 段还是全 1 吗？"
# ---
import torch
from learnkit import *

tokenizer = get_tokenizer()
pad_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id    # 0 和 2
B, P, R = 3, 5, 11
eos_at = [3, -1, 7]                       # 👉 每条 response 第一次出现 EOS 的下标，-1 = 一直没出现
torch.manual_seed(0)
prompt = torch.randint(3, 6400, (B, P)); prompt[1, :2] = pad_token_id          # 第 1 条左边补了 2 格 pad
resp = torch.randint(3, 6400, (B, R))
for i, e in enumerate(eos_at):
    if e >= 0: resp[i, e:] = eos_token_id       # generate 在 finished 之后一直填 eos_token_id
gen_out = torch.cat([prompt, resp], dim=1)                                     # [B, P+R]

# ===== 逐行复刻 train_ppo.py:116-128 =====
full_mask = (gen_out != pad_token_id).long()                                   # [B, P+R]
resp_idx = torch.arange(R).unsqueeze(0)                                        # [1, R]
logp_pos = torch.full((B, 1), P) - 1 + resp_idx                                # [B, R]
resp_pad_mask = torch.ones(B, R, dtype=torch.bool)                             # torch 引擎恒为全 1
full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.long())                      # 👈 response 段被整段覆盖
resp_lengths = resp_pad_mask.sum(1); eos_mask = resp.eq(eos_token_id) & resp_pad_mask
has_eos = eos_mask.any(1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()

table([[i, bool(has_eos[i]), int(eos_pos[i]), int(resp_lengths[i]), int(full_mask[i, :P].sum()),
        int(full_mask[i, P:].sum()), int(resp_policy_mask[i].sum())] for i in range(B)],
      headers=["样本", "has_eos", "eos_pos", "resp_lengths", "full_mask 在 prompt 段的 1",
               "full_mask 在 response 段的 1", "resp_policy_mask 的 1"],
      title=f"两层 mask 各自数出来的有效长度（P={P}, R={R}）")
print("logp_pos[0] =", logp_pos[0].tolist(), "← 第 j 个 response token 在「错一位坐标系」里的下标")
