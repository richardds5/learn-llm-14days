# ---
# title: aux_loss 在 train/eval、逐层、dense 三种情况下的取值
# timeout: 60
# tasks:
#   - "给 build_model 加上 router_aux_loss_coef=0.0：train 那两行走的是哪个分支？值是多少？"
#   - "把 num_hidden_layers 从 4 改成 8：train 模式那一行的 aux_loss 大约翻几倍？"
# ---
import torch
from learnkit import *

L = 4                                              # 👉 改层数，看总 aux_loss 怎么跟着变
moe = build_model(num_hidden_layers=L, use_moe=True)
dense = build_model(num_hidden_layers=L)           # use_moe=False：一层 MOEFeedForward 都没有
ids = torch.randint(0, moe.config.vocab_size, (3, 7))

rows = []
for name, model in [("MoE", moe), ("dense", dense)]:
    for mode in ("eval", "train"):
        getattr(model, mode)()
        with torch.no_grad():
            aux = model(ids).aux_loss
        per_layer = [round(l.mlp.aux_loss.item(), 6) for l in model.model.layers if hasattr(l.mlp, "aux_loss")]
        rows.append([name, f"model.{mode}()", f"{aux.item():.6f}", str(tuple(aux.shape)),
                     str(per_layer) if per_layer else "（没有 MoE 层）"])

table(rows, headers=["模型", "模式", "res.aux_loss", "shape", "每层 mlp.aux_loss"],
      title=f"L={L} 层：train 下每层各算一份、在 MiniMindModel.forward 里求和；"
            f"eval 和 dense 都返回标量 0（coef={moe.config.router_aux_loss_coef}）")
