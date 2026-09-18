# ---
# title: 一次 forward 的模块调用树：从 input_ids 到 logits
# timeout: 90
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 (3, 7) 和 dims 里的 T 一起改成 11：树里哪些格子跟着变了？`position_embeddings tuple(cos …, sin …)` 里那两个 [T, D] 变了吗？"
#   - "把 build_model() 改成 build_model(num_hidden_layers=2)：末尾那句『后面 N 个 MiniMindBlock 结构相同，省略』里的 N 变成几？"
# ---
import torch
from learnkit import *

# 不带 num_hidden_layers 参数 = 和 out/*.pth 真实权重一样的 8 层架构（随机初始化，只看 shape）
model = build_model()
input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # 👉 B=3, T=7，避开 config 里的数字

with trace(model, dims=dict(B=3, T=7),
           title="MiniMindForCausalLM: input_ids [B,T] → logits [B,T,V]"):
    model(input_ids)
