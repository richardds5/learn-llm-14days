# ---
# title: 模型里 15 个 Linear，谁通过了这道筛子
# timeout: 60
# sources:
#   - model/model_lora.py
# tasks:
#   - "把 build_model 换成 build_model(num_hidden_layers=2, num_attention_heads=4)（于是 head_dim=192）：表格里哪几行的『方阵?』从 ❌ 翻成了 ✅？"
#   - "把最后那段复现里的 LoRA(...) 换成 nn.Identity()：循环能跑完吗？由此判断问题出在『条件』上还是出在『挂上去的东西』上。"
# ---
from torch import nn
from learnkit import build_model, table
from model.model_lora import apply_lora, LoRA

model = build_model(num_hidden_layers=2)
linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
apply_lora(model, rank=16)  # 👉 条件：isinstance(m, nn.Linear) and m.in_features == m.out_features

table([[n.replace("model.layers.", "L"), m.in_features, m.out_features,
        "✅" if m.in_features == m.out_features else "❌", "lora" if hasattr(m, "lora") else ""]
       for n, m in linears], headers=["module", "in", "out", "方阵?", "注入"],
      title=f"2 层模型里的 {len(linears)} 个 nn.Linear")
lora_n = sum(p.numel() for n, p in model.named_parameters() if "lora" in n)
print(f"被注入 {sum(hasattr(m, 'lora') for _, m in linears)} 个；LoRA 参数占比 "
      f"{lora_n / sum(p.numel() for p in model.parameters()) * 100:.3f}%")

# 如果把 in_features == out_features 这个条件去掉，只留 isinstance 判断：
try:
    for name, module in build_model(num_hidden_layers=2).named_modules():
        if isinstance(module, nn.Linear):  # 👉 少了 and module.in_features == module.out_features
            setattr(module, "lora", LoRA(module.in_features, module.out_features, rank=16))
    print("循环跑完了")
except Exception as e:
    print(f"去掉条件之后：{type(e).__name__}: {str(e)[:70]}")
