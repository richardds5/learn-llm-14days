# ---
# title: 把 forward 拆成「transformer 主体」和「lm_head」两段分别计时
# timeout: 90
# tasks:
#   - "把 T 从 300 改成 900：主体那一行和 lm_head 全量那一行，各自涨了几倍？"
#   - "把 build_model(num_hidden_layers=8) 改成 num_hidden_layers=1：lm_head 全量在总耗时里的占比变成多少？"
# ---
import time
import torch
from learnkit import *

B, T = 3, 300                                  # 👉 模拟“送进一段长上文，只要下一个词”
model = build_model(num_hidden_layers=8)
C, V = model.config.hidden_size, model.config.vocab_size
input_ids = torch.randint(0, V, (B, T))


def ms(fn, n=5):
    fn()                                       # 预热一次，别让首次调用的初始化开销混进来
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return f"{sorted(ts)[n // 2] * 1000:.2f} ms"


with torch.no_grad():
    h = model.model(input_ids)[0]              # [B, T, C]：主体吐出来的 hidden_states
    rows = [[f"{model.config.num_hidden_layers} 层 transformer 主体（两种写法算的完全一样）", ms(lambda: model.model(input_ids))],
            ["lm_head：hidden_states[:, slice(0, None), :]  (keep=0)", ms(lambda: model.lm_head(h[:, slice(0, None), :]))],
            ["lm_head：hidden_states[:, slice(-1, None), :] (keep=1)", ms(lambda: model.lm_head(h[:, slice(-1, None), :]))]]
rows.append(["logits 张量显存 (fp32)", f"{B * T * V * 4 / 1024 ** 2:.2f} MB  →  {B * 1 * V * 4 / 1024 ** 2:.2f} MB"])
table(rows, headers=["这一步", "耗时 / 占用"],
      title=f"B={B}, T={T}, C={C}, V={V}：logits_to_keep 只动得了下面第 2、3 行")
