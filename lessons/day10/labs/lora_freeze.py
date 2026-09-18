# ---
# title: 25 步之后，哪些张量真的变了
# timeout: 200
# sources:
#   - trainer/train_lora.py
# tasks:
#   - "把 `p.requires_grad = False` 那一支改成 True，并把 optimizer 换成 optim.AdamW(model.parameters(), lr=1e-4)：跑完之后 q_proj.weight 的变化量还是 0 吗？"
#   - "把 lr 从 1e-4 调到 1e-3：loss 曲线是下得更快，还是开始震荡？"
# ---
import itertools
import torch
from torch.utils.data import DataLoader
from learnkit import load_model, get_tokenizer, best_device, live
from model.model_lora import apply_lora
from dataset.lm_dataset import SFTDataset

dev = best_device()
tok = get_tokenizer()
model = load_model("full_sft").to(dev)
apply_lora(model, rank=16)

lora_params = []  # 对照 train_lora.py：名字里带 'lora' 的才解冻
for name, p in model.named_parameters():
    if "lora" in name:
        p.requires_grad = True
        lora_params.append(p)
    else:
        p.requires_grad = False
total, trainable = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in lora_params)
print(f"可训练 {trainable} / 总 {total} = {trainable / total * 100:.3f}%")

q = model.model.layers[0].self_attn.q_proj
w0, b0 = q.weight.detach().clone(), q.lora.B.weight.detach().clone()

loader = DataLoader(SFTDataset("dataset/lora_identity.jsonl", tok, max_length=128), batch_size=8, shuffle=True)
opt = torch.optim.AdamW(lora_params, lr=1e-4)  # 👉 optimizer 只收 lora_params
model.train()
for step, (input_ids, labels) in enumerate(itertools.cycle(loader), start=1):
    loss = model(input_ids.to(dev), labels=labels.to(dev)).loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)  # 👉 clip 也只收 lora_params
    opt.step()
    live("LoRA 训练 loss", "train", step, loss.item())
    if step >= 25: break

print("基座 q_proj.weight 的变化量:", (q.weight - w0).abs().max().item())
print("旁路 q_proj.lora.B 的变化量:", (q.lora.B.weight - b0).abs().max().item())
