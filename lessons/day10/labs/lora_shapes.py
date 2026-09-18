# ---
# title: 旁路里的三个 shape：768 → r → 768
# timeout: 60
# sources:
#   - model/model_lora.py
# tasks:
#   - "把下面 apply_lora 的 rank 从 16 改成 4：trace 里中间那一步的 shape 变成什么？单个 LoRA 的参数量是不是正好降到 1/4？"
#   - "把 rank 改成 768：这时 B @ A 还是『低秩』的吗？参数量和直接训练一个 q_proj.weight（768×768）比，谁更大？"
# ---
import torch
from learnkit import build_model, trace
from model.model_lora import apply_lora, LoRA

model = build_model(num_hidden_layers=2)
apply_lora(model, rank=16)  # 👉 rank 是这一节唯一的旋钮

lora = model.model.layers[0].self_attn.q_proj.lora
n = sum(p.numel() for p in lora.parameters())
print(f"A.weight {tuple(lora.A.weight.shape)}   B.weight {tuple(lora.B.weight.shape)}")
print(f"单个 LoRA 参数量 = r*(in+out) = 16*(768+768) = {n}；"
      f"原 q_proj.weight = 768*768 = {model.model.layers[0].self_attn.q_proj.weight.numel()}")

# 逐行追踪 LoRA.forward：expand=True 把 self.B(self.A(x)) 这一行流拆开，
# 中间那个 [B, T, r] 就是「低秩瓶颈」本身
x = torch.randint(0, 6400, (3, 7))
with trace(model, fns=[LoRA.forward], dims=dict(B=3, T=7, r=16),
           title="LoRA.forward：x 穿过旁路时的 shape", max_calls=1, tree=False, expand=True):
    model(x)
