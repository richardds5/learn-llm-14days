# ---
# title: apply_lora 前后的 logits 差
# timeout: 60
# sources:
#   - model/model_lora.py
# tasks:
#   - "把 LoRA.__init__ 里 `self.B.weight.data.zero_()` 改成和 A 一样的 `normal_(mean=0.0, std=0.02)`：torch.equal 还是 True 吗？max diff 大概多大？"
#   - "只把 A 改成 zero_()（B 仍然全零）：输出一样不变。再想一想，为什么论文选『A 随机 + B 全零』而不是『A 全零 + B 随机』（提示：A、B 同时为 0 时两边的梯度都是 0）。"
# ---
import torch
from learnkit import build_model, show
from model.model_lora import apply_lora

torch.manual_seed(0)
model = build_model(num_hidden_layers=2)
x = torch.randint(0, 6400, (3, 7))

with torch.no_grad():
    out_before = model(x).logits.clone()

apply_lora(model, rank=16)  # 👉 A 高斯、B 全零 → ΔW = B @ A ≡ 0
with torch.no_grad():
    out_after = model(x).logits.clone()

print("apply_lora 前后 torch.equal:", torch.equal(out_before, out_after))
show(diff=(out_before - out_after).abs(), title="|apply_lora 前 − 后|：注意 max 是精确的 0，不是『很小』")

# 只把其中一个 B 改成非零，模拟「训练了一步」
model.model.layers[0].self_attn.q_proj.lora.B.weight.data.normal_(std=0.02)
with torch.no_grad():
    out_changed = model(x).logits.clone()
print("只把 layers.0.q_proj 的 B 改非零后 max diff:",
      (out_before - out_changed).abs().max().item())
