# ---
# title: num_return_sequences=3 之后，哪几行属于哪个 prompt
# timeout: 30
# tasks:
#   - "把 n 从 3 改成 4：prompt1 的 4 个候选落在哪几行？规律是「从第 1 行起，步长 = 原始 batch size」吗？"
#   - "把 prompt 改成只有一行（去掉 901 那一行）：两种写法的结果还有区别吗？"
# ---
import torch
from learnkit import *

prompt = torch.tensor([[101, 102, 103],       # prompt0
                       [901, 902, 903]])      # prompt1
n = 3                                          # 👉 num_return_sequences
rep = prompt.repeat(n, 1)                      # generate L258 用的就是这一行
inter = prompt.repeat_interleave(n, dim=0)     # 很多人以为 .repeat 是这个效果

whose = lambda row: "prompt0" if row[0].item() == 101 else "prompt1"
table([[i, str(rep[i].tolist()), whose(rep[i]), str(inter[i].tolist()), whose(inter[i])]
       for i in range(n * prompt.shape[0])],
      headers=["行号", ".repeat(3, 1) 的内容", "来自", ".repeat_interleave(3, 0) 的内容", "来自"],
      title="源码用的是 .repeat：整块首尾接 3 遍，同一个 prompt 的候选不连续")
