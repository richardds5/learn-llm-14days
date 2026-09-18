# ---
# title: 模块调用树：hidden_states 从头到尾都是 [B, T, C]
# timeout: 60
# tasks:
#   - "把 build_model 改成 build_model(num_hidden_layers=2, hidden_size=512)：树里哪些 shape 跟着变了？除了 C，还有哪个符号被带着变（看 self_attn 里的 D）？"
#   - "把 input_ids 的 (3, 7) 和 dims 里的 T 一起改成 11：树里带 T 的那些格子全变了，带 C 的呢？"
# ---
import torch
from learnkit import *

# 只看骨架，不看 Attention/FFN 内部，所以 2 层就够
model = build_model(num_hidden_layers=2)
input_ids = torch.randint(0, model.config.vocab_size, (3, 7))  # B=3, T=7

with trace(model.model, dims=dict(B=3, T=7),
           title="MiniMindModel 骨架：embed_tokens → dropout → L 层 MiniMindBlock → norm"):
    model.model(input_ids)
