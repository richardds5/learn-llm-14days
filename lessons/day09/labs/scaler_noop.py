# ---
# title: GradScaler 到底有没有在缩放
# timeout: 60
# sources:
#   - trainer/train_pretrain.py
# tasks:
#   - "把第三行的 enabled=True 改成 enabled=False：scale(loss) 还等于 loss 吗？unscale_ 之后的梯度呢？"
#   - "把对照组换成 torch.amp.GradScaler('cpu', enabled=True, init_scale=2**20)：backward 后的 |grad|max 变成多少？unscale_ 之后那一列变吗？"
# ---
import torch
from learnkit import *

model = build_model(num_hidden_layers=2, hidden_size=128).train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
grad_of = lambda: model.model.embed_tokens.weight.grad.abs().max().item()

rows = []
for tag, scaler in [("--dtype bfloat16（脚本默认）", torch.cuda.amp.GradScaler(enabled=False)),
                    ("--dtype float16（本机无 CUDA）", torch.cuda.amp.GradScaler(enabled=True)),
                    ("真·启用的 scaler（对照）", torch.amp.GradScaler("cpu", enabled=True))]:
    model.zero_grad(set_to_none=True)
    res = model(torch.randint(0, 6400, (3, 7)), labels=torch.randint(0, 6400, (3, 7)))
    loss = res.loss + res.aux_loss
    scaled = scaler.scale(loss)                 # train_pretrain.py:40
    scaled.backward()
    before = grad_of()
    scaler.unscale_(optimizer)                  # train_pretrain.py:43，把放大的梯度换算回真实尺度
    rows.append([tag, scaler.is_enabled(), f"{scaler.get_scale():.0f}" if scaler.is_enabled() else "-",
                 scaled is loss, f"{before:.3e}", f"{grad_of():.3e}"])

table(rows, headers=["scaler = GradScaler(...)", "is_enabled()", "get_scale()", "scale(loss) is loss",
                     "backward 后的 |grad|max", "unscale_ 之后"],
      title="只有真正启用时 scale/unscale_ 才动数值；bfloat16 和「没有 CUDA」两种情况下它都是透明包装")
