# ---
# title: 打完补丁之后的模块树与 state_dict
# timeout: 60
# sources:
#   - model/model_lora.py
# tasks:
#   - "把 `setattr(module, \"lora\", lora)` 这一行删掉（forward 里仍然用闭包里的 lora）：state_dict 还多出那 8 个 key 吗？前向结果变了吗？"
#   - "在模块树里数一数 q_proj 节点底下有几个子节点：原始 Linear 的那次矩阵乘法为什么没有作为独立节点出现？"
# ---
import torch
from learnkit import build_model, trace
from model.model_lora import apply_lora

model = build_model(num_hidden_layers=2)
before = set(model.state_dict())
apply_lora(model, rank=16)  # 👉 setattr 让 lora 变成 q_proj 的真子模块
after = set(model.state_dict())

print(f"state_dict key 数：{len(before)} → {len(after)}，新增 {len(after - before)} 个")
for k in sorted(after - before)[:3]:
    print("  ", k)
print("   …")
print("q_proj.weight 这个 key 还在吗:", "model.layers.0.self_attn.q_proj.weight" in after)
print("q_proj.forward 是实例属性吗:",
      "forward" in model.model.layers[0].self_attn.q_proj.__dict__)

# 模块树靠 forward hook 画：看 q_proj 底下挂着谁
x = torch.randint(0, 6400, (3, 7))
with trace(model, dims=dict(B=3, T=7), title="apply_lora 之后的模块调用树", max_roots=1):
    model(x)
