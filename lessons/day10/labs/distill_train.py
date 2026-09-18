# ---
# title: 8 层 teacher 带 2 层 student 跑 20 步
# timeout: 200
# sources:
#   - trainer/train_distillation.py
# tasks:
#   - "把 alpha 改成 0.0（纯蒸馏）和 1.0（纯 CE）各跑一次：20 步下来 ce 和 distill 各降了多少？纯 CE 时 distill 这一项明显吃亏在哪。"
#   - "把 teacher 也换成 build_model(num_hidden_layers=2)（随机初始化的 teacher）：distill 仍然在降，但 student 这次在学什么？"
# ---
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from learnkit import build_model, load_model, get_tokenizer, best_device, live
from dataset.lm_dataset import SFTDataset
from trainer.train_distillation import distillation_loss

dev, tok = best_device(), get_tokenizer()
teacher = load_model("full_sft").to(dev).eval().requires_grad_(False)  # 8 层，训练好的
student = build_model(num_hidden_layers=2).to(dev)                      # 2 层，随机初始化
print(f"teacher {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M（0 个参数在更新）  "
      f"student {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M（100% 都在更新）")

loader = DataLoader(SFTDataset("dataset/lora_identity.jsonl", tok, max_length=128), batch_size=8, shuffle=True)
opt = torch.optim.AdamW(student.parameters(), lr=5e-5)  # 👉 收的是整个 student，不是一小撮
alpha, T = 0.5, 2.0

student.train()
for step, (input_ids, labels) in enumerate(itertools.cycle(loader), start=1):
    input_ids, labels = input_ids.to(dev), labels.to(dev)
    mask = (labels[..., 1:] != -100).float().view(-1)
    s_logits = student(input_ids).logits[..., :-1, :].contiguous()
    with torch.no_grad():
        t_logits = teacher(input_ids).logits[..., :-1, :].contiguous()
    ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), labels[..., 1:].reshape(-1),
                         ignore_index=-100, reduction="none")
    ce = (ce * mask).sum() / (mask.sum() + 1e-8)
    kd = distillation_loss(s_logits.view(-1, s_logits.size(-1))[mask == 1],
                           t_logits.reshape(-1, t_logits.size(-1))[mask == 1], temperature=T)
    loss = alpha * ce + (1 - alpha) * kd  # 👉 唯一的混合旋钮
    opt.zero_grad(); loss.backward(); opt.step()
    for k, v in (("total", loss), ("ce", ce), ("distill", kd)):
        live("蒸馏 loss", k, step, v.item())
    if step >= 20: break
print("done")
