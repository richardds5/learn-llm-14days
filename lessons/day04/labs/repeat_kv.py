# ---
# title: repeat_kv：expand 不拷贝内存，reshape 才拷贝
# timeout: 60
# tasks:
#   - "把 KV 改成 2、n_rep 改成 4：expand 那一行的 stride 元组里，值为 0 的还是同一个位置吗？"
#   - "把 n_rep 改成 1：标题里 `== x.repeat(1,1,n_rep,1)` 那一项为什么也变成 True 了？reshape 那一行还拷贝内存吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import repeat_kv

B, T, KV, D, n_rep = 3, 7, 4, 6, 2      # 👉 D 用 6 只是为了 stride 好读，真实是 96
torch.manual_seed(42)
x = torch.randn(B, T, KV, D)

expanded = x[:, :, :, None, :].expand(B, T, KV, n_rep, D)   # 源码 L89 的前半句
reshaped = expanded.reshape(B, T, KV * n_rep, D)            # 源码 L89 的后半句
same = lambda t: "✔ 是同一块" if t.data_ptr() == x.data_ptr() else "✘ 已经拷贝走了"

eq_interleave = torch.equal(repeat_kv(x, n_rep), torch.repeat_interleave(x, n_rep, dim=2))
eq_repeat = torch.equal(repeat_kv(x, n_rep), x.repeat(1, 1, n_rep, 1))
table([["x（KV 个头）", str(list(x.shape)), str(x.stride()), same(x)],
       ["expand 之后", str(list(expanded.shape)), str(expanded.stride()), same(expanded)],
       ["reshape 之后 = repeat_kv 输出", str(list(reshaped.shape)), str(reshaped.stride()), same(reshaped)]],
      headers=["tensor", "shape", "stride", "底层存储"],
      title=f"repeat_kv == repeat_interleave(dim=2)? {eq_interleave} ；"
            f"== x.repeat(1,1,n_rep,1)? {eq_repeat} ；repeat_kv(x,1) is x? {repeat_kv(x, 1) is x}")
