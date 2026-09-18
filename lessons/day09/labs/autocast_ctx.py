# ---
# title: 三行代码决定 autocast_ctx 是什么
# timeout: 60
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把判断改成 device_type = args.device.split(':')[0]：mps 那一行的 device_type 变成什么？再想想 torch.cuda.amp.autocast 在 mps 上能不能用"
#   - "把最后实测那一行的 nullcontext() 换成 torch.autocast(device_type=dev, dtype=torch.bfloat16)：logits.dtype 变成什么？"
# ---
from contextlib import nullcontext
import torch
from learnkit import *


def script_logic(device, dtype_str="bfloat16"):
    """原样照搬 train_pretrain.py:121-123 的三行"""
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    return device_type, type(autocast_ctx).__name__

rows = [[d, *script_logic(d), ""] for d in ["cuda:0", "cuda:3", "cpu", "mps", "xpu"]]

# 再按脚本的逻辑真跑一次 forward，看这台机器上 hidden_states 到底是什么精度
dev = best_device()
model = build_model(num_hidden_layers=2, hidden_size=128).to(dev)
with nullcontext() if script_logic(dev)[0] == "cpu" else torch.cuda.amp.autocast(dtype=torch.bfloat16):
    logits = model(torch.randint(0, 6400, (3, 7), device=dev)).logits
rows.append([f"{dev}（本机实测）", *script_logic(dev), f"logits.dtype = {logits.dtype}"])

table(rows, headers=["args.device", "device_type", "autocast_ctx 的类型", "实测"],
      title='device_type = "cuda" if "cuda" in args.device else "cpu" —— 只认 cuda，其余全归到 cpu 分支')
