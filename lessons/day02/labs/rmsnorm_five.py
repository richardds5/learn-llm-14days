# ---
# title: 数一数模型里的 RMSNorm：有几个、各自归一化多宽
# timeout: 60
# tasks:
#   - "把 num_hidden_layers 从 2 改成 4：实例数从 9 变成几个？凑出关于 L 的公式"
#   - "加一个参数 build_model(num_hidden_layers=2, head_dim=64)：q_norm/k_norm 那几行变成多少？input_layernorm 变吗？"
# ---
from learnkit import *
from model.model_minimind import RMSNorm

model = build_model(num_hidden_layers=2)   # 👉 改层数 / head_dim 看下表怎么变
cfg = model.config

rows = []
for name, m in model.named_modules():
    if isinstance(m, RMSNorm):
        dim = m.weight.shape[0]
        sym = "C = hidden_size" if dim == cfg.hidden_size else ("D = head_dim" if dim == cfg.head_dim else "?")
        rows.append([name, dim, sym, m.eps])

table(rows, headers=["模块名", "len(weight)：归一化的宽度", "对应符号", "实际 eps"],
      title=f"{cfg.num_hidden_layers} 层模型里共 {len(rows)} 个 RMSNorm 实例，分 5 类")
print(f"RMSNorm 类定义里的默认 eps 是 {RMSNorm(8).eps}，但上表里没有一个实例用到它"
      f"——每处都显式传了 config.rms_norm_eps = {cfg.rms_norm_eps}")
