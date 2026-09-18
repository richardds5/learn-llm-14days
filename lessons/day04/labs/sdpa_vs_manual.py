# ---
# title: 两条分支算出来的 logits 差多少
# timeout: 60
# tasks:
#   - "把 attention_mask 那一行的 am[0, :2] = 0 取消注释：两个模型此刻都被赶去手写分支，最大误差变成多少？"
#   - "把 num_hidden_layers 改成 4：最大误差还在 1e-6 这个量级吗？和 logits 本身的量级差几个数量级？"
# ---
import torch
from learnkit import *

# 同一个 seed 构造两个模型，权重逐位相同；唯一的差别是 forward 里走哪条分支
m_flash = build_model(num_hidden_layers=2, flash_attn=True)
m_manual = build_model(num_hidden_layers=2, flash_attn=False)
same_w = torch.equal(m_flash.model.layers[0].self_attn.q_proj.weight,
                     m_manual.model.layers[0].self_attn.q_proj.weight)

input_ids = torch.randint(0, m_flash.config.vocab_size, (3, 7))  # B=3, T=7，无 cache 无 padding
am = torch.ones(3, 7)
# am[0, :2] = 0        # 👉 取消注释：有 padding 之后 m_flash 也会被赶去手写分支

a = m_flash(input_ids, attention_mask=am).logits
b = m_manual(input_ids, attention_mask=am).logits
table([["两个模型的 q_proj 权重逐位相同", str(same_w)],
       ["logits 形状", str(list(a.shape))],
       ["allclose(atol=1e-5)", str(torch.allclose(a, b, atol=1e-5))],
       ["最大绝对误差", f"{(a - b).abs().max().item():.2e}"],
       ["logits 本身的量级（最大绝对值）", f"{a.abs().max().item():.2e}"]],
      headers=["检查项", "结果"], title="SDPA 分支 vs 手写分支：只差在浮点累加顺序上")
