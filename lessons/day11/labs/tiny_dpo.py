# ---
# title: 16 步迷你 DPO：loss、梯度和两条 log prob 曲线
# timeout: 300
# sources:
#   - trainer/train_dpo.py
# tasks:
#   - "把 LR 从 2e-6 改成 2e-8（接近仓库默认的 4e-8）：16 步里两条曲线还分得开吗？"
#   - "把 LR 改成 1e-5：loss 几步就到 0，两条曲线各自往哪走？这就是 DPO 常说的模型退化。"
# ---
import itertools, os, random, tempfile, time

import torch
from torch.utils.data import DataLoader

from learnkit import *
from dataset.lm_dataset import DPODataset
from trainer.train_dpo import logits_to_log_probs, dpo_loss

N_PAIRS, MAX_LEN, BETA, LR, EPOCHS, BS = 8, 256, 0.1, 2e-6, 4, 2      # 👉 LR 是仓库默认 4e-8 的 50 倍
random.seed(0); torch.manual_seed(0)
tok, dev = get_tokenizer(), best_device()

rows = sorted(itertools.islice(open("dataset/dpo.jsonl", encoding="utf-8"), 60), key=len)[:N_PAIRS]
path = os.path.join(tempfile.mkdtemp(), "dpo_mini.jsonl")
open(path, "w", encoding="utf-8").writelines(rows)
ds = DPODataset(path, tok, max_length=MAX_LEN)
batches = [(torch.cat([b['x_chosen'], b['x_rejected']]).to(dev), torch.cat([b['y_chosen'], b['y_rejected']]).to(dev),
            torch.cat([b['mask_chosen'], b['mask_rejected']]).to(dev)) for b in DataLoader(ds, batch_size=BS)]

model = load_model("full_sft", device=dev).train()                   # policy
ref_model = load_model("full_sft", device=dev)                       # ref：同一个 full_sft 起点
ref_model.eval(); ref_model.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
px, py, pm = batches[0][0], batches[0][1], batches[0][2]              # 固定探针：每步都量同一批样本

def probe(step):
    model.eval()
    with torch.no_grad():
        s = (logits_to_log_probs(model(px).logits, py) * pm).sum(1)
    model.train()
    live("探针：序列级 log prob", "chosen", step, s[:BS].mean().item())
    live("探针：序列级 log prob", "rejected", step, s[BS:].mean().item())

probe(0); t0 = time.time(); step = 0
for _ in range(EPOCHS):
    for x, y, mask in batches:
        with torch.no_grad():
            ref_lp = logits_to_log_probs(ref_model(x).logits, y)
        loss = dpo_loss(ref_lp, logits_to_log_probs(model(x).logits, y), mask, beta=BETA)
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 返回的是裁剪前的 grad norm
        opt.step(); step += 1
        live("dpo loss", "train", step, loss.item())
        if step == 1: print(f"第 1 步：loss = {loss.item():.4f}，裁剪前的 grad norm = {gnorm.item():.2f}")
        probe(step)
print(f"{step} 步用时 {time.time() - t0:.1f}s，device={dev}")
