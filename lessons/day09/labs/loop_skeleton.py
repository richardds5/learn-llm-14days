# ---
# title: 照着 train_epoch 的七拍，跑一遍迷你预训练
# timeout: 180
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把 args.accumulation_steps 从 2 改成 8：forward 次数不变，optimizer.step() 少了 4 倍，loss 曲线变平滑还是变抖？"
#   - "把 args.learning_rate 从 3e-3 改成 3e-1：loss 曲线在第几步开始不降反升？"
# ---
import json, os, tempfile
from contextlib import nullcontext
from itertools import islice
from types import SimpleNamespace
import torch
from torch import optim
from torch.utils.data import DataLoader
from learnkit import *
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_lr

# 仓库里没有 pretrain 主数据，临时造一份玩具语料（80 条医疗问答的回答文本）
path = os.path.join(tempfile.mkdtemp(), "toy.jsonl")
with open("dataset/lora_medical.jsonl", encoding="utf-8") as f, open(path, "w", encoding="utf-8") as w:
    for line in islice(f, 80):
        w.write(json.dumps({"text": json.loads(line)["conversations"][1]["content"][:80]}, ensure_ascii=False) + "\n")

# 下面这些名字在真实脚本里全是 __main__ 里的模块级变量，train_epoch 直接靠全局查找引用它们
args = SimpleNamespace(device=best_device(), accumulation_steps=2, grad_clip=1.0, learning_rate=3e-3)
autocast_ctx = nullcontext()                                    # ← train_pretrain.py:123 在 CPU/MPS 上的取值
torch.manual_seed(42)
lm_config = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
model = MiniMindForCausalLM(lm_config).to(args.device).train()
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=False)
loader = DataLoader(PretrainDataset(path, get_tokenizer(), max_length=64), batch_size=8, shuffle=True)
epochs = 12
iters, step = len(loader) * epochs, 0

for epoch in range(epochs):                                     # ↓↓↓ 和 train_epoch 逐拍对应 ↓↓↓
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(args.device), labels.to(args.device)   # ① 取 batch 上设备
        step += 1
        lr = get_lr(step, iters, args.learning_rate)                            # ② 算 lr 并写回
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        with autocast_ctx:                                                      # ③ 前向
            res = model(input_ids, labels=labels)
            loss = (res.loss + res.aux_loss) / args.accumulation_steps          # ④ 两项相加再除以 N
        scaler.scale(loss).backward()                                           # ⑤ 反传，梯度攒进 .grad
        if step % args.accumulation_steps == 0:                                 # ⑥ 每 N 步才更新一次
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        live("迷你预训练", "loss", step, loss.item() * args.accumulation_steps)  # ⑦ 日志里乘回去
