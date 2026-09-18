# ---
# title: 逐行追踪 distillation_loss
# timeout: 60
# sources:
#   - trainer/train_distillation.py
# tasks:
#   - "把 F.kl_div 的两个参数对调（`F.kl_div(teacher_probs, student_log_probs, ...)`）：loss 变成什么？"
#   - "把 reduction 从 'batchmean' 改成 'sum' 和 'mean'：数值分别是 batchmean 的几倍？和 M=21、V=6400 对得上吗？"
# ---
import torch
from learnkit import trace
from trainer.train_distillation import distillation_loss

M, V = 21, 6400  # M = 一个 batch 里参与蒸馏的有效 token 数；V = 词表大小
torch.manual_seed(0)
student_logits = torch.randn(M, V) * 3
teacher_logits = torch.randn(M, V) * 3

with trace(fns=["trainer.train_distillation:distillation_loss"], dims=dict(M=M, V=V),
           title="distillation_loss：一路都是 [M, V]", tree=False) as tr:
    loss = distillation_loss(student_logits, teacher_logits, temperature=2.0)

print("loss =", round(loss.item(), 4))
print("teacher 换成 student 自己（完美拟合）:",
      distillation_loss(student_logits, student_logits, temperature=2.0).item())
