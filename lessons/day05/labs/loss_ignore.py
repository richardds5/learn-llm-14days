# ---
# title: 四种 -100 分布下，loss 的分母各是多少
# timeout: 60
# tasks:
#   - "把 v[:, :3] = -100 改成 v[:, :T-1] = -100（每条序列只剩最后一个位置有效）：有效位置数变成几？loss 还是有限数吗？"
#   - "把最后一行的 labels 改成只有 1 个位置有效：`v = torch.full_like(base, -100); v[0, 1] = base[0, 1]`，loss 变成什么？"
# ---
import torch
from learnkit import *

B, T = 3, 7
model = build_model(num_hidden_layers=2)
input_ids = torch.randint(0, model.config.vocab_size, (B, T))
base = input_ids.clone()                       # PretrainDataset 的做法：labels = input_ids.clone()

variants = [("原样，不 mask 任何位置", base.clone())]
v = base.clone(); v[:, :3] = -100; variants.append(("每条序列的前 3 个位置 = -100", v))
v = base.clone(); v[0] = -100; variants.append(("batch 里第 0 条整条 = -100", v))
variants.append(("整个 batch 全是 -100", torch.full_like(base, -100)))   # 👉 分母会变成 0

total, rows = B * (T - 1), []
for name, labels in variants:
    with torch.no_grad():
        loss = model(input_ids, labels=labels).loss
    n = (labels[..., 1:] != -100).sum().item()          # shift 之后还剩几个有效位置
    rows.append([name, f"{n} / {total}", f"{loss.item():.4f}",
                 f"{loss.item() * n / total:.4f}" if n else "0.0000"])

table(rows, headers=["labels 的情形", "参与 loss 的位置数", "模型返回的 loss", "若 -100 只是「贡献一个 0」"],
      title=f"B={B}, T={T}：ignore_index 同时改掉了分子和分母")
