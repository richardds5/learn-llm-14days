# ---
# title: 把 layer1= / layer2= 两个默认参数去掉之后
# timeout: 60
# tasks:
#   - "把 `def forward_with_lora(x):` 改回仓库里的写法 `def forward_with_lora(x, layer1=original_forward, layer2=lora): return layer1(x) + layer2(x)`：max diff 变成多少？"
#   - "把 num_hidden_layers 改成 8：max diff 还是这个量级吗？最后两行打印里『闭包指向谁』的答案变成哪个模块？"
# ---
import torch
from torch import nn
from learnkit import build_model
from model.model_lora import LoRA


def apply_lora_no_default(model, rank=16):
    """和仓库里的 apply_lora 逐字相同，只有 def 那一行去掉了两个默认参数"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x):  # 👉 少了 layer1=original_forward, layer2=lora
                return original_forward(x) + lora(x)

            module.forward = forward_with_lora


torch.manual_seed(0)
model = build_model(num_hidden_layers=2)
x = torch.randint(0, 6400, (3, 7))
with torch.no_grad():
    before = model(x).logits.clone()
apply_lora_no_default(model, rank=16)
with torch.no_grad():
    after = model(x).logits.clone()  # 这一行会抛异常吗？

print("前向跑完了，没有任何异常。allclose:", torch.allclose(before, after),
      "  max diff:", (before - after).abs().max().item())

cells = dict(zip((fw := model.model.layers[0].self_attn.q_proj.forward).__code__.co_freevars, fw.__closure__))
last = model.model.layers[-1].self_attn.o_proj
print("layers.0.q_proj 闭包里的 lora 其实是 layers.1.o_proj.lora:", cells["lora"].cell_contents is last.lora)
print("闭包里的 original_forward 绑在 layers.1.o_proj 上:", cells["original_forward"].cell_contents.__self__ is last)
