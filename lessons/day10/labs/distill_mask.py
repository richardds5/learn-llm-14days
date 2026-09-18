# ---
# title: 从 [B,T] 的 labels 到 [M,V] 的蒸馏输入
# timeout: 90
# sources:
#   - trainer/train_distillation.py
# tasks:
#   - "把 `loss_mask_flat == 1` 这个布尔索引去掉，直接把整个 [B*T, V] 喂给 distillation_loss：M 变成多少？loss 数值往哪个方向偏？"
#   - "把切片 `[..., :-1, :]` 去掉（保留最后一个位置）：哪一行会因为 shape 对不上而报错？"
# ---
import random
import torch
from torch.utils.data import DataLoader
from learnkit import build_model, get_tokenizer, token_strip
from dataset.lm_dataset import SFTDataset

random.seed(0)  # SFTDataset 会随机决定加不加 system prompt / 留不留空 think，固定住才可复现
tok = get_tokenizer()
loader = DataLoader(SFTDataset("dataset/lora_identity.jsonl", tok, max_length=128), batch_size=3, shuffle=False)
input_ids, labels = next(iter(loader))

loss_mask = (labels[..., 1:] != -100).float()          # 👉 和 Day 8 的 SFT mask 同一个写法
student_logits = build_model(num_hidden_layers=2)(input_ids).logits[..., :-1, :].contiguous()
loss_mask_flat = loss_mask.view(-1)
selected = student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1]

print(f"input_ids {tuple(input_ids.shape)} → logits 切掉最后一位 → student_logits {tuple(student_logits.shape)}")
print(f"labels {tuple(labels.shape)} → loss_mask {tuple(loss_mask.shape)} → flat {tuple(loss_mask_flat.shape)}")
print(f"布尔索引后喂给 distillation_loss 的是 {tuple(selected.shape)}，"
      f"M = {int(loss_mask_flat.sum())} / {loss_mask_flat.numel()} = {loss_mask_flat.mean() * 100:.1f}%")

toks = [tok.decode([i]) for i in input_ids[0, 1:].tolist()]
token_strip(toks, loss_mask[0].tolist(), title="第 0 条样本：1 = 参与蒸馏和 CE，0 = prompt / padding",
            legend="loss_mask")
