# ---
# title: 两处残差相加分别改成 in-place，backward 各是什么结果
# timeout: 60
# tasks:
#   - "在 make_forward 里把 `normed = self.post_attention_layernorm(hidden_states)` 换成 `normed = hidden_states.clone()`（跳过 RMSNorm），再跑第三行：结果变了吗？这说明报错是谁引起的"
#   - "把最后一行 run(True, True) 的 labels 去掉、改成只做 forward 不 backward：还会报错吗？"
# ---
import types
import torch
from learnkit import *


def make_forward(ip192, ip193):              # 👉 两个开关分别控制 L192 / L193 写不写 in-place
    def fwd(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, pkv = self.self_attn(self.input_layernorm(hidden_states),
                                            position_embeddings, past_key_value, use_cache, attention_mask)
        if ip192: hidden_states += residual                 # 源码 L192 就是这一句
        else: hidden_states = hidden_states + residual
        normed = self.post_attention_layernorm(hidden_states)
        if ip193: hidden_states += self.mlp(normed)         # 源码 L193 写的是 `=`，这里换成 `+=`
        else: hidden_states = hidden_states + self.mlp(normed)
        return hidden_states, pkv
    return fwd


def run(ip192, ip193):
    m = build_model(num_hidden_layers=2)
    for layer in m.model.layers: layer.forward = types.MethodType(make_forward(ip192, ip193), layer)
    ids = torch.randint(0, m.config.vocab_size, (3, 7))
    try:
        m(ids, labels=ids).loss.backward(); return "backward 正常跑完"
    except RuntimeError as e:
        return f"{type(e).__name__}: {str(e)[:78]}…"


table([["源码原样：L192 `+=`，L193 `=`", run(True, False)],
       ["L192 也改成 `=`（两处都不 in-place）", run(False, False)],
       ["L193 也改成 `+=`（两处都 in-place）", run(True, True)]],
      headers=["MiniMindBlock.forward 的写法", "loss.backward() 的结果"],
      title="同样是 `+=`，两处残差相加的后果并不一样")
