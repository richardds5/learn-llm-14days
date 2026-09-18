# ---
# title: 两级平均下，每条轨迹在 loss 里的权重
# timeout: 60
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 lens 里第 0 条的有效长度从 4 改成 40：它在两列里的权重分别怎么变？"
#   - "把 lens 第 5 条改成 0（这条轨迹一个 token 都没留下），再把 clamp(min=1) 删掉：policy_loss 变成什么？"
# ---
import torch
from learnkit import *

BG, R = 6, 40
lens = torch.tensor([4, 20, 7, 40, 11, 3])  # 6 条轨迹的有效长度差得很远
completion_mask = (torch.arange(R).expand(BG, -1) < lens.unsqueeze(1)).int()  # [B*G, R]
per_token_loss = torch.ones(BG, R, requires_grad=True)  # 每个 token 的 loss 都设成 1，梯度就直接读出「权重」

# 源码 train_grpo.py:144：先序列内平均 → [B*G]，再序列间平均 → 标量
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
g_two = torch.autograd.grad(policy_loss, per_token_loss, retain_graph=True)[0]
# 对照：全局 token 平均（DAPO 等变体改的就是这里）
flat_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
g_flat = torch.autograd.grad(flat_loss, per_token_loss)[0]

table([[k, lens[k].item(), round(g_two[k].sum().item(), 4), round(g_flat[k].sum().item(), 4)] for k in range(BG)],
      headers=["轨迹 k", "有效 token 数", "两级平均下的权重（源码）", "全局平均下的权重（对照）"],
      title=f"两个 loss 的数值都是 {policy_loss.item():.1f}（每个 token 的 loss 都设成了 1），但后两列完全不同")
note(f"源码那一列恒等于 1/{BG} = {1 / BG:.4f}：**每条轨迹等权，和长度无关**。\n\n"
     f"对照那一列是 `len_k / {int(lens.sum())}`：最长的那条拿到 {g_flat.sum(1).max().item():.4f}，"
     f"最短的那条只有 {g_flat.sum(1).min().item():.4f} —— 全局平均下，长回答会主导梯度。")
