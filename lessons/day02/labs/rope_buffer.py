# ---
# title: persistent=False：是 buffer，但不进 state_dict
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "在源码标签页把 L206-L207 的 persistent=False 改成 persistent=True 再跑：state_dict 的 key 数变成多少？多出来的是哪两个？"
#   - "把 max_position_embeddings 从默认的 32768 改成 2048（build_model 的参数）：freqs_cos 的 shape 变成什么？如果它进了 checkpoint，会多占多少个 float？"
# ---
from learnkit import *

model = build_model(num_hidden_layers=2)   # 👉 加 max_position_embeddings=2048 看 shape 怎么变
mm = model.model

buf_names = [n for n, _ in mm.named_buffers()]
sd_keys = list(model.state_dict().keys())

inside = [n for n in buf_names if f"model.{n}" in sd_keys]
print("named_buffers() 里有：", buf_names)
print("其中进了 state_dict() 的：", inside, "← 空的" if not inside else "← persistent=True 之后它们就进去了")
print("state_dict() 一共", len(sd_keys), "个 key，带 'freqs' 的有", sum("freqs" in k for k in sd_keys), "个")
print()
print("freqs_cos.shape =", list(mm.freqs_cos.shape),
      "= [max_position_embeddings, head_dim] =", [mm.config.max_position_embeddings, mm.config.head_dim],
      "（D/2 个频率被 torch.cat 复制成 D，见 L78）")
print("freqs_cos[0, 0] =", mm.freqs_cos[0, 0].item(), "← cos(0)=1；L216 就是靠它判断 buffer 有没有在 meta device 上丢掉")
print("它占的内存：", f"{mm.freqs_cos.numel() * 4 / 1e6:.1f} MB", "（不进 checkpoint，每次 __init__ 现算）")
