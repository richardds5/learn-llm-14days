# ---
# title: 同样 12 步 SFT，5e-4 和 1e-5 两条 loss 曲线
# timeout: 180
# sources:
#   - trainer/train_full_sft.py
# tasks:
#   - "把 5e-4 那一组换成 5e-5：还会把 loss 顶上去吗？在这个数据上大概从哪个量级开始崩？"
#   - "把 load_model('full_sft') 换成 load_model('pretrain')：两条曲线的起点 loss 差多少？（pretrain 权重还没学过对话格式）"
# ---
import torch
from torch.utils.data import DataLoader
from learnkit import *
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr

dev, tok = best_device(), get_tokenizer()
train_ds = SFTDataset("dataset/lora_identity.jsonl", tok, max_length=128)   # Day 8 见过：只有回答部分算 loss
g = torch.Generator().manual_seed(0)
batches = [b for b, _ in zip(DataLoader(train_ds, batch_size=4, shuffle=True, generator=g), range(12))]

for tag, learning_rate in [("full_sft 默认 lr=1e-5", 1e-5), ("pretrain 默认 lr=5e-4", 5e-4)]:
    model = load_model("full_sft", device=dev).train()      # 每组都从同一份 out/full_sft_768.pth 重新开始
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for step, (input_ids, labels) in enumerate(batches, start=1):
        input_ids, labels = input_ids.to(dev), labels.to(dev)
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step, len(batches), learning_rate)
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss                      # accumulation_steps 在 SFT 里默认就是 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        live("在已经会说话的模型上继续训练", tag, step, loss.item())
    del model, optimizer
