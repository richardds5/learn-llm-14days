# ---
# title: 省掉 .float()：fp16 下 x.pow(2) 溢出之后会怎样
# timeout: 60
# tasks:
#   - "把放大倍数 300 改成 15：还会出现 inf 吗？（fp16 上限 65504，|x| ≥ 256 时 x² 就溢出）"
#   - "把 .half() 改成 .bfloat16() 再跑：bf16 的指数位和 fp32 一样宽，清零行数会变成几行？"
# ---
import torch
from learnkit import *
from model.model_minimind import RMSNorm

torch.manual_seed(0)
rn = RMSNorm(768, eps=1e-6)
x16 = (torch.randn(3, 7, 768) * 300).half()   # 👉 300² = 90000 > fp16 上限 65504

def forward_no_upcast(x):                      # 「忘记 .float()」的写法：全程留在 fp16 里
    return (rn.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rn.eps))).type_as(x)

naive, real = forward_no_upcast(x16), rn(x16)  # real 是仓库里的真实实现（先 .float()）
rows = lambda t: (t.abs().sum(-1) == 0).sum().item()

table([
    ["x.pow(2) 里 inf 的元素数", f"{torch.isinf(x16.pow(2)).sum().item()} / {x16.numel()}"],
    ["rsqrt(inf) 等于", f"{torch.rsqrt(torch.tensor(float('inf'))).item()}   ← 不是 nan，是 0"],
    ["不 .float()：整行输出全 0 的 token 数", f"{rows(naive)} / {naive.shape[0] * naive.shape[1]}"],
    ["不 .float()：输出里有 nan 吗", str(torch.isnan(naive).any().item())],
    ["真实实现（先 .float()）：全 0 的 token 数", f"{rows(real)} / {real.shape[0] * real.shape[1]}"],
], headers=["检查项", "结果"], title="fp16 溢出不报错、不 nan —— 只是悄悄把 token 清零")
