# ---
# title: 合并之后的干净模型 vs 基座 + 旁路
# timeout: 60
# sources:
#   - model/model_lora.py
# tasks:
#   - "把 save_lora 和 merge_lora 里所有的 `.half()` 都删掉（全程 float32）再跑：atol=1e-4 这一行变成 True 了吗？"
#   - "把 merge_lora 里的 `state_dict[f'{name}.weight'] += (...)` 改成 `-=`：max diff 从 2e-3 跳到什么量级？"
# ---
import os, tempfile
import torch
from learnkit import build_model
from model.model_lora import apply_lora, save_lora, merge_lora

torch.manual_seed(0)
model = build_model(num_hidden_layers=2)
apply_lora(model, rank=16)
for _, m in model.named_modules():  # 模拟「训练过」
    if hasattr(m, "lora"):
        m.lora.A.weight.data.normal_(std=0.02); m.lora.B.weight.data.normal_(std=0.02)

x = torch.randint(0, 6400, (3, 7))
with torch.no_grad():
    out_lora = model(x).logits.clone()  # 基座 forward + 旁路 forward

d = tempfile.mkdtemp()
lora_path, merged_path = os.path.join(d, "lora.pth"), os.path.join(d, "merged.pth")
save_lora(model, lora_path)
merge_lora(model, lora_path, merged_path)  # 👉 W += B @ A，存成没有 .lora. 的普通权重
merged = torch.load(merged_path, map_location="cpu")

clean = build_model(num_hidden_layers=2)  # 完全没有 lora 子模块的干净结构
clean.load_state_dict({k: v.float() for k, v in merged.items()}, strict=True)
clean.eval()
with torch.no_grad():
    out_merged = clean(x).logits.clone()

print("merged 里还有 .lora. 的 key 吗:", any(".lora." in k for k in merged))
print("logits 的 std:", round(out_lora.std().item(), 4), " max abs diff:", (out_lora - out_merged).abs().max().item())
for atol in (1e-2, 1e-3, 1e-4):
    print(f"allclose(atol={atol}): {torch.allclose(out_lora, out_merged, atol=atol)}")
