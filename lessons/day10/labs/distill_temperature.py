# ---
# title: 同一份 logits 在 T = 1 / 2 / 4 下的三张图和三个数
# timeout: 60
# sources:
#   - trainer/train_distillation.py
# tasks:
#   - "把 distillation_loss 的返回值从 `(temperature ** 2) * kl` 改成 `kl`：表格最后一列还稳得住吗？"
#   - "把 teacher_logits 的缩放从 `* 3` 改成 `* 0.5`（一个本来就很平的 teacher）：T 从 1 升到 4 时 entropy 的变化幅度还有这么大吗？"
# ---
import torch
import torch.nn.functional as F
from learnkit import bars, table
from trainer.train_distillation import distillation_loss

M, V = 21, 6400
torch.manual_seed(0)
student_logits = torch.randn(M, V) * 3
teacher_logits = torch.randn(M, V) * 3

rows = []
for T in (1.0, 2.0, 4.0):
    probs = F.softmax(teacher_logits[0] / T, dim=-1)  # 只看第 0 个 token 的分布
    top = probs.topk(12)
    bars([str(i) for i in top.indices.tolist()], top.values.tolist(),
         title=f"T={T:.0f}：teacher 在第 0 个 token 上的 top-12 概率")
    kl = F.kl_div(F.log_softmax(student_logits / T, -1), F.softmax(teacher_logits / T, -1), reduction="batchmean")
    rows.append([T, round(-(probs * probs.clamp_min(1e-9).log()).sum().item(), 3),
                 round(probs.max().item(), 4), round(kl.item(), 4),
                 round(distillation_loss(student_logits, teacher_logits, temperature=T).item(), 4)])

table(rows, headers=["T", "teacher 熵", "max prob", "裸 KL", "T²·KL（真正返回的 loss）"],
      title="温度把分布拉平 → 裸 KL 塌下去 → 乘回 T² 才稳住量级")
