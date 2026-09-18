# ---
# title: save_lora 存下来的到底是什么
# timeout: 60
# sources:
#   - model/model_lora.py
# tasks:
#   - "把 save_lora 里的 `.cpu().half()` 改成 `.cpu()`（保持 float32）：文件体积变成多少？"
#   - "把 dst 上的 `apply_lora(dst, rank=16)` 这一行注释掉，直接 load_lora：会报错吗？max diff 变成多少？（提示：load_lora 靠 hasattr(module, 'lora') 找目标）"
# ---
import os, tempfile
import torch
from learnkit import build_model, table
from model.model_lora import apply_lora, save_lora, load_lora

torch.manual_seed(0)
src = build_model(num_hidden_layers=2)
apply_lora(src, rank=16)
for _, m in src.named_modules():  # 模拟「训练过」：让 B 不再是 0
    if hasattr(m, "lora"): m.lora.B.weight.data.normal_(std=0.02)

path = os.path.join(tempfile.mkdtemp(), "lora.pth")
save_lora(src, path)  # 👉 只遍历 hasattr(module, 'lora') 的模块
sd = torch.load(path, map_location="cpu")
table([[k, str(tuple(v.shape)), str(v.dtype)] for k, v in sd.items()],
      headers=["key", "shape", "dtype"], title=f"{len(sd)} 个 key，文件 {os.path.getsize(path) / 1024:.1f} KB")

x = torch.randint(0, 6400, (3, 7))
dst = build_model(num_hidden_layers=2)          # 另一份同样的基座
apply_lora(dst, rank=16)                         # 先把空壳结构搭好
load_lora(dst, path)                             # 👉 再把 A/B 填进去
with torch.no_grad():
    print("load_lora 之后两个模型的 logits max diff:",
          (src(x).logits - dst(x).logits).abs().max().item(), " (fp16 存盘的舍入)")
