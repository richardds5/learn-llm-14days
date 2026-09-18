# ---
# title: 把 L63 那一行流拆开：从 arange 到 D/2 个频率
# timeout: 60
# sources:
#   - model/model_minimind.py
# tasks:
#   - "把 dim 从 96 改成 97：arange(0, dim, 2) 这一步出来几个元素？[: (dim // 2)] 这个切片这时候起作用了吗？"
#   - "把 rope_base 从 1e6 改成 1e4：拆开后每个子表达式的 shape 有变化吗？最后那个 freqs 的数值呢？"
# ---
from learnkit import *
from model.model_minimind import precompute_freqs_cis

# end 只给 64：这一节只看 L63 这一行，表算多大无所谓（focus 把视野收窄到 L63）
with trace(fns=[precompute_freqs_cis], dims=dict(D=96), tree=False,
           focus=(63, 63), expand=True,
           title="L63：1.0 / (rope_base ** (arange(0, dim, 2)[:dim//2].float() / dim))"):
    precompute_freqs_cis(dim=96, end=64, rope_base=1e6)   # 👉 dim 改成 97 试试
