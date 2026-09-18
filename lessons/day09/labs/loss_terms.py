# ---
# title: res.loss、res.aux_loss、除以 N、再乘回来
# timeout: 60
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把 accumulation_steps 从 8 改成 1：表格里哪几行会变、哪几行不变？"
#   - "把 model.train() 改成 model.eval() 再看 MoE 那一行的 aux_loss：为什么变成 0？（提示：model_minimind.py:171 的 if self.training）"
# ---
import torch
from learnkit import *

accumulation_steps = 8                      # 👉 train_pretrain.py 的默认值
input_ids = torch.randint(0, 6400, (3, 7))  # B=3, T=7

rows, aux = [], {}
for tag, use_moe in [("稠密 (use_moe=False)", False), ("MoE (use_moe=True)", True)]:
    model = build_model(num_hidden_layers=2, hidden_size=128, use_moe=use_moe).train()
    res = model(input_ids, labels=input_ids.clone())          # ← labels 一给，forward 内部就算好 loss
    loss = res.loss + res.aux_loss                            # train_pretrain.py:37
    scaled = loss / accumulation_steps                        # train_pretrain.py:38，真正拿去 backward 的就是它
    aux[use_moe] = res.aux_loss
    rows.append([tag,
                 f"{res.loss.item():.4f} {tuple(res.loss.shape)}",
                 f"{res.aux_loss.item():.6f} {tuple(res.aux_loss.shape)}",
                 f"{scaled.item():.4f}",
                 f"{scaled.item() * accumulation_steps:.4f}"])

table(rows, headers=["模型", "res.loss（交叉熵）", "res.aux_loss（路由）", f"loss/{accumulation_steps} ← backward 用这个",
                     f"日志 loss.item()*{accumulation_steps}"],
      title="两项相加 → 除以 accumulation_steps → 日志里再乘回去（稠密模型的 aux_loss 是一个值为 0 的标量）")
print(f"稠密 aux_loss = {aux[False].item()}, requires_grad = {aux[False].requires_grad}（加上它不引入任何梯度）")
print(f"MoE   aux_loss = {aux[True].item():.6f}, requires_grad = {aux[True].requires_grad}"
      f"（只有 self.training 为真时才非 0，见 model_minimind.py:171）")
