# ---
# title: embed_tokens.weight 和 lm_head.weight 是不是同一块内存
# timeout: 60
# tasks:
#   - "在 print 之前加一句 `with torch.no_grad(): model.lm_head.weight[3, 5] = 999.`，再打印 model.model.embed_tokens.weight[3, 5]：会是多少？"
#   - "把 untied 那次的 data_ptr() 也打印出来，确认 tie=False 之后两者不再指向同一块内存"
# ---
import torch
from learnkit import *

model = build_model(num_hidden_layers=2)            # 默认 tie_word_embeddings=True
e, h = model.model.embed_tokens.weight, model.lm_head.weight

print("embed_tokens.weight.shape =", list(e.shape), "  (nn.Embedding(V, C) → [V, C])")
print("lm_head.weight.shape      =", list(h.shape), "  (nn.Linear(C, V) → [out, in] = [V, C])")
print("是同一个 nn.Parameter 对象 (is)：", e is h, "   data_ptr() 相同：", e.data_ptr() == h.data_ptr())
print("'lm_head.weight' 出现在 named_parameters() 里吗：", "lm_head.weight" in dict(model.named_parameters()))

tied = sum(p.numel() for p in model.parameters())
untied = sum(p.numel() for p in build_model(num_hidden_layers=2, tie_word_embeddings=False).parameters())
V, C = model.config.vocab_size, model.config.hidden_size
print(f"\ntie=True  参数量 {tied:,}")
print(f"tie=False 参数量 {untied:,}")
print(f"多出来 {untied - tied:,} == V*C == {V}*{C} == {V * C:,}")
