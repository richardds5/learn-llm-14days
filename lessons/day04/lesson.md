---
day: 4
format: points
title: "Attention：GQA、QK-Norm 与因果掩码"
subtitle: "一份 [B,T,C] 拆成 8 个头、和 4 组 KV 对齐、被掩码切成三角形，再拼回 [B,T,C]"
minutes: 90
mainline: "一份 x [B,T,C] 走完 Attention.forward 的 24 行：拆成 [B,T,H,D] 和 [B,T,KV,D]，对齐成 [B,H,T,D]，算出一张 [B,H,T,T_kv] 的分数表，掩码 + softmax 之后再合并回 [B,T,C]"
files:
  - model/model_minimind.py#L86-L134
goals:
  - 能报出 xq/xk/xv 在 Attention.forward 每一行之后的 shape，并说清 view / QK-Norm / repeat_kv / transpose 各自动的是哪一维
  - 能解释 GQA 为什么需要 repeat_kv、n_rep 怎么算、KV cache 里存的是 repeat 之前还是之后的版本
  - 能说出第 125 行那四个条件各自在挡什么，以及为什么自回归单步 decode 永远走不到 SDPA
  - 能读懂因果掩码那一行的 `-seq_len:` 切片，和 padding mask 的 `[B,1,1,T_kv]` 广播
  - 能在真实权重上认出 attention sink，并说清它是训练涌现的现象而不是结构规定
---

Day 2 把 `self.self_attn(...)` 当黑盒放过去了，Day 3 钻进 RoPE 时 xq/xk 还停在 `[B,T,H,D]` 布局上。今天把这个黑盒彻底拆开：`Attention.forward` 从 `def` 到 `return` 一共 24 行（L111~L134），我们跟着同一份 `x [B,T,C]` 逐行走完。

这一天的重点全在 shape 上。一个张量先被拆成头（而且是**两种**头数），再被对齐成统一的 `[B,H,T,D]`，然后变成一张 `[B,H,T,T_kv]` 的分数表，被掩码挖掉一半，softmax 之后加权求和，最后拼回 `[B,T,C]` 交还给残差流。每一小节只盯住其中一两行。

五章的分工：第一章是进门的三个投影和一次 view；第二章让 q 和 k/v 在头数和布局上重新对齐，这是 GQA 的核心；第三章是那条长得离谱的分支判断，它决定用 SDPA 还是手写；第四章拆开手写分支的三行——分数、因果掩码、padding mask；第五章出门，并用真实权重看看训练出来的 attention 长什么样。

除了讲分支判断的那一节，本课的实验一律传 `flash_attn=False`：只有手写分支里才有 `scores` 这个局部变量可以抓。

{{flow: x [B,T,C] | *q/k/v_proj* | [B,T,H,D] + [B,T,KV,D] | *repeat_kv + transpose* | [B,H,T,D] | *scores + mask + softmax* | [B,H,T,D] | *o_proj* | [B,T,C]}}

# 进门：从一条总线拆成多头

残差流上流下来的是一条宽 `C` 的向量。Attention 要做的第一件事是把它变成 H 个各管各的小向量——中间只有三个矩阵乘法、一次 view 和一次归一化。但 q 和 k/v 从这里开始就走上了不同宽度的路，这一章走完 L113~L119。

## 三个投影：Q 宽 768，K/V 只有 384 {#qkv-proj}

`forward` 的第一行实质工作是 L113：同一份 `x` 同时喂给三个 `nn.Linear`。三个投影的**输入**宽度都是 `hidden_size=768`，但**输出**宽度不一样——这是 GQA 在代码里留下的第一个痕迹。

{{source:model/model_minimind.py#L100-L103}}

`q_proj` 输出 `num_attention_heads * head_dim = 8 × 96 = 768`；`k_proj` / `v_proj` 输出 `num_key_value_heads * head_dim = 4 × 96 = 384`，正好是一半。`o_proj` 反过来，把 `H·D` 收回 `hidden_size`。四个投影一律 `bias=False`（Qwen3、LLaMA 也都这么写）。

值得抠一下 L100 的写法：`q_proj` 的输出宽度写的是 `H·D`，**不是** `hidden_size`。默认配置下两者都等于 768 纯属巧合（`head_dim` 默认就是 `C // H`），连实验里的 trace 都会被这个巧合骗到，把这一维标成 `C=768`。显式传 `head_dim=128` 就露馅了：`xq` 变成 `[B,T,1024]`，而 `x` 还是 `[B,T,768]`。

GQA 省下的不只是这半个矩阵的参数。`k_proj` / `v_proj` 变窄意味着算出来的 K/V 也只有一半，推理时 KV cache 直接减半——**省显存带宽才是 GQA 的真正目的**，省参数只是顺带。把 `Attention` 这一个模块的参数量数出来：`num_key_value_heads=8`（退化成标准 MHA）是 2,359,488，默认的 4 是 1,769,664，`=1`（MQA）是 1,327,296。

{{lab:qkv_proj}}

{{quiz:q1}}

> [!KEY]
> `q_proj` 输出 `H·D`，`k_proj`/`v_proj` 只输出 `KV·D`（默认 768 vs 384）；`H·D == hidden_size` 只是默认配置下的巧合。

> [!MORE] 为什么 K/V 能少、Q 不能少
> Q 决定「有多少种提问方式」，K/V 决定「有多少份可检索的底库」。实验上把 Q 头数砍掉掉点很明显，而让多个 Q 头共享同一份 K/V 掉点很小——因为共享的只是检索的底库，每个 Q 头仍然用自己的投影去提问，仍然能学出不同的关注模式。GQA 就是踩在这个不对称性上：Q 保持 8 个，K/V 砍到 4 组。

## view 拆头：两个头数字段，两条分岔 {#view-heads}

L114~L116 连着三行 `view`，做的事完全一样：把最后一维 `H·D` 或 `KV·D` 拆成两维 `(头数, head_dim)`。

{{source:model/model_minimind.py#L114-L116}}

`xq` 变成 `[B,T,H=8,D=96]`，`xk` / `xv` 变成 `[B,T,KV=4,D=96]`。注意三行分别读了两个不同的字段：`self.n_local_heads`（= `num_attention_heads`）和 `self.n_local_kv_heads`（= `num_key_value_heads`）。**q 和 k/v 的 shape 从这里正式分岔，一直要到 L124 才重新合上。**

这一步是纯 reshape，**不搬动任何数据**：最后一维本来就是连续的 `H·D` 个浮点，把它重新解释成 `H` 行 `D` 列即可，`view` 返回的是同一块存储的另一种 stride 视图。之所以敢直接用 `view` 而不是 `reshape`，是因为 `nn.Linear` 的输出天然 contiguous。

为什么要先拆再做别的？因为紧接着的 QK-Norm 和 RoPE 都是**按「每个头内部的 D 个分量」**算的，必须先把 `D` 单独拎成一维，它们才能靠「作用在最后一维」这条统一规则生效。

一个容易踩的坑：`num_key_value_heads` 必须整除 `num_attention_heads`。传 3 的话构造阶段一声不吭（`n_rep = 8 // 3 = 2`），这三行的 shape 看着也正常，`repeat_kv` 之后却只有 6 个头，一路要到 L128 的 `xq @ xk.transpose(-2,-1)` 才炸出 `RuntimeError: The size of tensor a (8) must match the size of tensor b (6)`。

{{lab:view_heads}}

{{quiz:q2}}

> [!KEY]
> 三行 `view` 把最后一维拆成 `(头数, head_dim)`，是零拷贝的重新解释；`xq` 走 `H=8`、`xk/xv` 走 `KV=4`，两条路要到 L124 才汇合。

## QK-Norm：归一化落在 head_dim 上 {#qk-norm}

L117 对 `xq` / `xk` 各做一次 RMSNorm。这两个 norm 在 `__init__` 里是 `RMSNorm(self.head_dim)`——传的是 **96，不是 768**。

{{source:model/model_minimind.py#L117-L119}}

Day 2 讲过，`RMSNorm.norm()` 里的 `mean(-1, keepdim=True)` 永远沿**最后一维**。此刻最后一维是 `D=96`，所以归一化的统计量是「某个 batch、某个位置、某个 head 内部那 96 个分量」的均方——和别的 head、别的 token 完全无关。shape 一个字都不变。

为什么要归？attention logits 是 `q·k/√D`，而点积的量级由 q 和 k 的**模长**直接决定。某个 head 的 q 模长在训练中跑飞，这个 head 的 softmax 就会提前饱和成近似 one-hot，梯度基本消失，这个 head 就废了。QK-Norm 把每个 head 的模长钉死在 `RMS(weight)` 上，只把**方向**信息留给点积——这是 Qwen3 引入、现在越来越常见的做法。

实测 `full_sft` 第 0 层：归一化前 8 个 head 的 RMS 在各 token 上从 0.55 晃到 0.91；归一化后全部贴在 1.204~1.265 之间，也就是 `RMS(q_norm.weight) = 1.229` 附近。注意残留的那点浮动来自可学习的 `weight` 逐通道缩放，`norm()` 那一步本身是精确的。

还有个顺序问题：QK-Norm 在 RoPE（L119）**之前**。RoPE 是旋转，不改模长，两者理论上可交换；放在前面意味着它作用在还没加位置信息的 q/k 上。

{{lab:qk_norm}}

{{quiz:q3}}

> [!KEY]
> `q_norm`/`k_norm` 是 `RMSNorm(head_dim)`，作用在 `[B,T,H,D]` 的最后一维：每个 head 内部单独归一，shape 不变，模长被钉住、只留方向。

> [!MORE] 这就是 Day 3 里 `unsqueeze_dim=1` 的前提
> L119 调 `apply_rotary_pos_emb(xq, xk, cos, sin)` 没传 `unsqueeze_dim`，用的是默认值 1。`cos`/`sin` 是 `[T, D]`，`unsqueeze(1)` 之后变成 `[T, 1, D]`，正好对上此刻的 `[B, T, H, D]`：`T` 对 `T`、`1` 广播到 `H`、`D` 对 `D`、`B` 作为多出来的前导维自动广播。**如果 RoPE 挪到 L124 之后（布局变成 `[B,H,T,D]`），`unsqueeze_dim` 就得改成 2。**

# 对齐到 [B,H,T,D]：cache、repeat_kv、transpose

上一章结束时 `xq` 是 `[B,T,8,D]`、`xk/xv` 是 `[B,T,4,D]`，头数对不上，布局也不适合做矩阵乘法。这一章的三行代码（L120~L124）把它们对齐，顺带把 KV cache 接进来——而 cache 存的是对齐**之前**还是**之后**的版本，是一个价值 2 倍显存的设计决定。

## cache 拼接：在 dim=1 上接上旧的 KV {#kv-cache-cat}

推理时每生成一个 token 都重算全部历史的 K/V 是纯浪费，所以算过的要存下来。L120~L122 干的就是这件事。

{{source:model/model_minimind.py#L120-L123}}

如果调用方传了 `past_key_value`（一个 `(xk, xv)` 元组），就在 **`dim=1`** 上把旧的接在新的前面。此刻 `xk` 还是 `[B, T, KV, D]` 布局（`transpose` 要到下一行才发生），所以 `dim=1` 正是时间维。prefill 7 个 token、再一次性续写 5 个的话，拼完 `xk` 的第二维从 5 变成 12。

只有 k 和 v 参与拼接，**q 不拼**：query 永远只是这一轮新喂进来的 token。这就决定了后面那张分数表是个「矮而宽」的矩形——5 行（新 query）× 12 列（全部 key），而不是正方形。

L123 把拼好的结果原样存回 `past_kv`，由上层塞进 `past_key_values` 列表带到下一步。`use_cache=False` 时这里直接是 `None`，但注意 L121/L122 的拼接**照样会执行**——拼不拼取决于传没传 `past_key_value`，存不存才取决于 `use_cache`。

L123 在整行代码里的**位置**决定了 cache 的大小。它排在 L124 之前——所以存进去的 `xk` 是几个头？

{{predict:p1}}

{{lab:kv_cache_cat}}

实验里 `past_kv` 是 `[B=3, T_kv=12, KV=4, D=96]`：存的是 **repeat 之前的 4 个头**。这是刻意的：如果存 repeat 之后的 8 个头，cache 直接翻倍，GQA 省下来的显存又被吃回去了。复制这一步被推迟到「每次真正算 attention 时现算」。generate 循环怎么维护这个列表，Day 7 细讲。

{{quiz:q4}}

> [!KEY]
> `torch.cat(..., dim=1)` 在时间维上接旧 KV（此刻布局还是 `[B,T,KV,D]`）；`past_kv` 存的是 repeat **之前**的 `KV` 个头，省下 `n_rep` 倍 cache。

## repeat_kv：expand 不拷贝，reshape 才拷贝 {#repeat-kv}

4 组 KV 要给 8 个 Q 头用，缺的那一半得补出来。`n_rep = n_local_heads // n_local_kv_heads = 8 // 4 = 2`，每组 KV 要被复制 2 份。

{{source:model/model_minimind.py#L86-L89}}

复制的方式是「组内连续复制」，等价于 `torch.repeat_interleave(x, n_rep, dim=2)`，而**不是** `x.repeat(1,1,n_rep,1)`（后者是把整个头维度整体重复一遍，顺序完全不同）。所以映射关系是 `KV 组号 = Q head 号 // n_rep`：

| Q head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 用的 KV 组 | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |

实现上是两步。`x[:, :, :, None, :].expand(...)` 先插一个长度 1 的新维再撑成 `n_rep`——`expand` **不拷贝内存**，只是把这一维的 stride 设成 0，让 `n_rep` 个「逻辑位置」都指向同一块数据（实验里能看到 stride 从 `(168,24,6,1)` 变成 `(168,24,6,0,1)`，`data_ptr` 不变）。紧接着的 `.reshape(...)` 把 `[KV, n_rep]` 压成一维 `H`，**这一步才真的复制**：stride 为 0 的维度没法重新解释成合法的连续布局，`reshape` 只好先 `contiguous()` 一份，`data_ptr` 随之改变。

`n_rep == 1` 时（`num_key_value_heads == num_attention_heads`，即标准 MHA）函数第一行就 `return x`，连一次拷贝都不做。

{{lab:repeat_kv}}

{{quiz:q5,q6}}

> [!KEY]
> `repeat_kv` == `repeat_interleave(dim=2)`：`expand` 只把 stride 设成 0（零拷贝），后面的 `reshape` 才真正复制一份 `[B,T,H,D]` 出来。

## 第 124 行：一行里做完三件事 {#transpose-layout}

到这里 q 和 k/v 终于要合流了。源码用一行元组赋值同时处理三个变量，把它拆开看是理解整个 forward 的关键。

{{source:model/model_minimind.py#L124}}

`xq` 这一支只有 `transpose(1, 2)`：`[B,T,H,D]` → `[B,H,T,D]`。`xk`/`xv` 这两支先 `repeat_kv(·, n_rep)` 把头数从 `KV=4` 补到 `H=8`（此刻布局仍是 `[B,T,H,D]`），**再** `transpose(1,2)` 换成 `[B,H,T,D]`。实验用 `expand=True` 把这五个子表达式逐个标出 shape，顺序一目了然。

为什么必须转成 `[B,H,T,D]`？因为下一步是 `torch.matmul`，它把**最后两维**当矩阵、前面所有维当 batch。只有 `(T, D)` 落在最后两维，`q @ k^T` 才是「每个头内部，T 个 query 对 T_kv 个 key」的那个矩阵乘法。留在 `[B,T,H,D]` 的话最后两维是 `(H, D)`，算出来的东西毫无意义。

`transpose` 和 `view`/`reshape` 不一样：它**不搬数据**，只是交换两个维度的 stride，所以做完之后 `xq` 已经不 contiguous 了。这不影响 `matmul`（内部会按需处理），但会影响到 L132——那里的 `reshape` 因此不得不复制一份，见[「合并多头」一节](#merge-heads)。

顺带注意 `repeat_kv` 排在 `transpose` **前面**：它的实现是按 `[bs, slen, num_key_value_heads, head_dim]` 解包的，换了顺序就得改写。

{{lab:transpose_layout}}

{{quiz:q7}}

> [!KEY]
> L124 = `xq` 转置 + `xk/xv` 先 repeat 后转置，三者统一成 `[B,H,T,D]`；转成这个布局是为了让 `matmul` 的「最后两维」正好是 `(T, D)`。

# 两条分支：SDPA 与手写实现

L125 是全文件最长的一行判断。它决定这一次 attention 是交给 PyTorch 的 `scaled_dot_product_attention`（融合 kernel，省显存），还是老老实实自己算 `q@k^T`。这一章先看条件，再验证两条路算出来的结果到底差多少。

## 第 125 行的四个开关 {#flash-branch}

四个条件用 `and` 串起来，任何一个不满足就落到 `else`。

{{source:model/model_minimind.py#L125-L126}}

1. `self.flash`：构造时就定死了，`hasattr(F, 'scaled_dot_product_attention') and config.flash_attn`。
2. `seq_len > 1`：`seq_len = x.shape[1]`，是**这一次调用新喂进来**的 token 数，不含 cache 里的历史。
3. `not self.is_causal or past_key_value is None`：`self.is_causal` 在 `__init__` 里硬编码成 `True`、没有配置项能关，所以整项等价于 `past_key_value is None`——**没有 KV cache**（为什么必须这样，见下面的折叠块）。
4. `attention_mask is None or torch.all(attention_mask == 1)`：batch 里不存在 padding。注意这一项每次 forward 都要真的扫一遍 mask。

下面的实验用同一个 `flash_attn=True` 的模型连着调用两次：#1 一次性喂 11 个 token，#2 带着 cache 只喂 1 个新 token。**第 2 次调用里，126~131 这几行哪些会被执行？**

{{predict:p2}}

{{lab:flash_branch}}

第 2 次调用 `seq_len == 1`，条件 2 直接不成立（条件 3 也不成立），于是 126 行暗着、128~131 亮着。也就是说**自回归 generate 的每一个 decode 步都走手写分支，不管 `flash_attn` 开没开**——flash 只在 prefill 这种一次性吃多个 token 的场合才可能生效。

{{quiz:q8}}

> [!KEY]
> 四个条件同时满足才走 SDPA：PyTorch 支持、`seq_len>1`、没有 KV cache、没有 padding。单步 decode 恒定走手写分支。

> [!MORE] 有 cache 为什么就不能用 SDPA
> `F.scaled_dot_product_attention(..., is_causal=True)` 的语义是「下三角**对齐到矩阵左上角**」，隐含假设 `T_q == T_kv`。一旦有 cache，`T_kv = T_old + T_new > T_q`，下三角就贴错了位置——等价于假装这一轮的新 token 是从序列第 0 位开始的，语义完全不对。SDPA 本身可以接受显式的 `attn_mask` 张量来表达任意掩码（传一个 `[T_q, T_kv]` 的布尔矩阵即可），但这份实现选了更省事的做法：有 cache 就整个放弃 flash，退回手写分支自己构造正确的矩形掩码。
> 实测四种场景：`T=11` 无 cache 无 mask → SDPA；`T=11` + 全 1 的 mask → 仍是 SDPA；`T=11` + 含 0 的 mask → 手写；`T=1` + cache → 手写。

## 两条分支的数值差异 {#sdpa-vs-manual}

既然有两条路径，一个很自然的担心是：它们算出来的结果一样吗？数学上等价，但浮点加法不满足结合律，SDPA 内部的分块累加顺序和 `xq @ xk.transpose(-2,-1)` 不同，逐位相同是不可能的。

要做这个对比得先排除权重差异。实验里用同一个 seed 构造两个模型（`build_model` 内部固定了 `torch.manual_seed`），两份权重逐位相同，唯一的差别就是 `flash_attn` 这个运行时开关——表格第一行验证了这一点。

实测 `B=3, T=7, L=2` 时最大绝对误差 `2.98e-06`，而 logits 本身的量级是 `2.44`，相差约六个数量级；`allclose(atol=1e-5)` 通过。层数加到 4 时误差是 `3.58e-06`，同一量级，没有雪崩式放大。这是典型的浮点舍入误差，不代表哪条路径「更准」。

更有意思的是把 padding 加进来：给 `attention_mask` 塞两个 0 之后，两个模型**都**被条件 4 赶去了手写分支，执行的是完全相同的指令序列，误差精确地变成 `0.00e+00`。这个 0 反过来证明了前面那个 `2.98e-06` 确实来自分支差异，而不是别的什么随机性。

工程上的含义：不要用「换个分支结果没变」来判断代码正确，也不要因为 1e-6 量级的差异去怀疑实现——但如果你在做数值回归测试（比如对比 PyTorch 和 ONNX 导出的结果），阈值就得按这个量级来定，`atol=1e-8` 一定会误报。

{{lab:sdpa_vs_manual}}

{{quiz:q9}}

> [!KEY]
> 两条分支数学等价、浮点不等价：实测最大误差 ~3e-6（logits 量级 2.4）；一旦两边都被赶去手写分支，误差精确为 0。

# 手写分支的三行

`else` 里只有四行代码（L128~L131），却是整个 attention 的数学本体：算分、挡住不该看的、归一化成概率、加权求和。这一章逐行拆开前三件事，最后一步的加权求和留到下一章的出口一起看。

## scores：D 被消掉，换来一张 T×T_kv 的表 {#scores-matmul}

`xq [B,H,T,D] @ xk.transpose(-2,-1) [B,H,D,T_kv]` → `[B,H,T,T_kv]`。`matmul` 把前两维 `(B,H)` 当 batch，对每个 `(样本, head)` 独立做一次 `(T,D) @ (D,T_kv)` 的矩阵乘法。

{{source:model/model_minimind.py#L128}}

`D=96` 是被求和消掉的那一维——`scores[b,h,i,j]` 就是第 `i` 个 query 向量和第 `j` 个 key 向量在 96 维上的点积。换来的 `T × T_kv` 是 attention 里**唯一随序列长度平方增长**的张量：默认配置 `B=3, T=T_kv=7` 时它有 1176 个数；换成 `T=1024` 就是 2500 万个数、fp32 下 100 MB。FlashAttention 存在的全部理由就是**不把这张表完整写进显存**，而是分块算完立刻消费掉——这也是走 SDPA 分支时 trace 里看不到 `scores` 的根本原因：那条路上它根本没有以完整形态存在过。

除以 `math.sqrt(self.head_dim)` 是标准的 scaled dot-product：两个各分量方差为 1 的 `D` 维向量，点积的方差正比于 `D`，不除的话 logits 的量级会随 `D` 一起涨，softmax 提前饱和。注意除的是 `sqrt(96)` 而不是 `sqrt(768)`——缩放跟着 `head_dim` 走，不跟 `hidden_size` 走。这也意味着改 `head_dim` 只会改这个系数，不会改 `scores` 的 shape。

{{lab:scores_matmul}}

{{quiz:q10}}

> [!KEY]
> `scores = q @ k^T / √D` 的 shape 是 `[B, H, T, T_kv]`：`D` 被点积消掉，代价是一张随序列长度平方增长的表。

## 因果掩码只加在最后 seq_len 列 {#causal-mask}

`torch.full((seq_len, seq_len), -inf).triu(1)` 造出一个严格上三角（对角线保留为 0）的 `-inf` 矩阵，然后**只加到 `scores` 的最后 `seq_len` 列上**。

{{source:model/model_minimind.py#L129}}

`scores[:, :, :, -seq_len:]` 这个切片是理解整行的钥匙。没有 cache 时 `T_kv == seq_len`，「最后 seq_len 列」就是整张表，退化成人人熟悉的下三角因果掩码。有 cache 时 `T_kv > seq_len`，前面那 `T_kv - seq_len` 列对应的是**已经在 cache 里的旧 token**——它们在时间上全部早于这一轮的任何新 query，因果关系天然成立，一个都不用挡。真正需要挡的只有「新 token 之间」那个 `seq_len × seq_len` 的右下角子块。

另外两个细节。一是这里用的是 `+=`：**原地修改** `scores`，所以 trace 时这一行不能加 `expand=True`（子表达式会被重复求值，掩码就加了两遍）。二是这里用纯 `-inf` 而不是大负数——`exp(-inf)` 精确等于 0，softmax 之后这些位置是干净的 0，不会留下 `1e-40` 之类的残渣。

下面的实验 prefill 7 个 token、再一次性续写 5 个，把加完掩码的 `scores` 抓出来画成「哪些格子是 `-inf`」。**图里黄色（被屏蔽）的格子会怎么分布？**

{{predict:p3}}

{{lab:causal_mask}}

图上前 7 列（旧 cache）一片深色，一个 `-inf` 都没有；黄色只出现在最后 5 列的严格上三角里，每个 head 上共 `5×4/2 = 10` 个。

{{quiz:q11}}

> [!KEY]
> 因果掩码只写进最后 `seq_len` 列：旧 cache 列天然全部可见，只有新 token 之间需要 `triu(1)` 的 `-inf`。没有 cache 时退化成整张下三角。

## padding mask：按列屏蔽，广播到所有 head {#padding-mask}

`attention_mask` 的 shape 是 `[B, T_kv]`（1 = 真 token，0 = padding）。`unsqueeze(1).unsqueeze(2)` 把它变成 `[B, 1, 1, T_kv]`，广播到 `[B, H, T, T_kv]`。

{{source:model/model_minimind.py#L130-L131}}

那两个长度为 1 的维度就是广播的位置：`1 → H`（所有 head 用同一份 mask）、`1 → T`（所有 query 行用同一份 mask）。换句话说 padding mask 是**按 key 列**屏蔽的——它挡的是「谁不能被看」，而不是「谁不许看」。pad 位置上的 query 照样会算出一行输出，只是没人要它。

为什么用 `-1e9` 而不是 `-inf`？因为这里是**乘法式**写法：`(1.0 - attention_mask) * X`。真 token 处 `1 - 1 = 0`，而 `0 * float("-inf")` 在 IEEE 754 里等于 `nan`——整张 scores 会被一次性毁掉（实测一个 5×5 的例子里 25 个格子全变 `nan`）。换成有限的 `-1e9`，真 token 处得到 `-0.0`，安全无副作用。

漏传 `attention_mask` 的后果是**按行独立**的：某一行如果 mask 全是 1，`(1-1)*-1e9` 恒为 0，传不传完全一样；只有真的含 padding 的行才会被污染。实测一个左 padding 的 batch（真实权重）：没有 padding 的那句 logits 最大绝对误差 `0.000`、next token 不变；padded 了 9 个位置的 `"你好"` 那句误差飙到 `11.709`，next token 从 `'你好'` 直接变成 `'<|im_end|>'`——模型被 9 个 `<|endoftext|>` 带偏，以为该收尾了。

{{lab:padding_mask}}

{{quiz:q12}}

> [!KEY]
> `[B,T_kv] → [B,1,1,T_kv]` 广播到所有 head 和所有 query 行，按 key 列屏蔽；用 `-1e9` 而非 `-inf` 是因为 `0 * -inf = nan`。

> [!MORE] L131 为什么要 `.float()` 再 `.type_as(xq)`
> `F.softmax(scores.float(), dim=-1).type_as(xq)`：即使模型跑在 fp16/bf16 下，softmax 也强制升到 fp32 算完再降回去。理由和 Day 2 里 RMSNorm 的 `.float()` 一样——低精度下 `exp` 和求和容易丢精度；再加上这里满屏的 `-inf` 和 `-1e9`，fp16 的窄动态范围更容易出问题。注意 `-1e9` 本身超出了 fp16 的表示范围（上限 65504），在 fp16 里会直接变成 `-inf`，所以这一步升精度不是可有可无的装饰。本课的实验全跑在 fp32 上，这一行是空操作。

# 出门：合并多头，以及真实权重下的样子

softmax 之后 `@ xv` 得到 `[B,H,T,D]`——每个 head 各自算出了一份 `D` 维的输出。最后一章先把它们拼回残差流的宽度交差，再换上真正训练过的权重，看看这套机制在真实模型里学成了什么样。

## 合并多头，投影回残差流 {#merge-heads}

`output` 此刻是 `[B, H, T, D]`。L132 先 `transpose(1,2)` 换回 `[B, T, H, D]`，再 `reshape(bsz, seq_len, -1)` 把最后两维压成一维。

{{source:model/model_minimind.py#L132-L134}}

这两步正好是 L114 和 L124 的逆运算，把「按 head 分开」的结果重新串成每个 token 一条长向量。默认配置下 `H·D = 8×96 = 768` 恰好等于 `hidden_size`，但 L133 的 `o_proj` 是 `Linear(H·D, C)`，写的是 `H·D` 而不是 `C`——`head_dim=128` 时 L132 之后是 `[B,T,1024]`，`o_proj` 照样能把它收回 `[B,T,768]`。

顺序不能颠倒，而且颠倒了**不会报错**。`output` 是 `[B,H,T,D]`，元素总数 `B·H·T·D`；直接 `reshape(bsz, seq_len, -1)` 时 `B·H·T·D / (B·T) = H·D`，shape 算出来一模一样，程序照跑不误，只是 `reshape` 按内存顺序读数，把「head 维」和「时间维」搅在了一起——**结果是安静的、纯数值上的错误**。这类 bug 在 shape 检查里完全看不出来，只能靠对照实现或者 loss 不降来发现。

还有个性能细节：`transpose` 之后张量不 contiguous，所以这次 `reshape` 必然触发一次真实拷贝（等价于 `contiguous().view(...)`）。这是多头 attention 里除 `repeat_kv` 之外的第二次不可避免的数据搬运。最后 `resid_dropout` 在 `.eval()` 下是恒等映射，`return output, past_kv` 交还给 `MiniMindBlock` 去做残差相加——今天的黑盒到此完全打开。

{{lab:merge_heads}}

{{quiz:q13}}

> [!KEY]
> `transpose(1,2)` + `reshape` 把 `[B,H,T,D]` 拼回 `[B,T,H·D]`，再由 `o_proj: Linear(H·D, C)` 回到残差流宽度；少了 transpose 不会报错，只会安静算错。

## 真实权重下的 attention：从对角线到 attention sink {#attention-sink}

前面所有实验都是随机初始化的模型：shape 全对，数值毫无意义。换上训练过的 `full_sft` 权重，把每一层的 `scores` 抓出来手动做一次 softmax（复现 L131），就能看到语言模型真正学到了什么样的注意力模式。

输入是一句走过 chat 模板的 `"今天天气怎么样"`，共 21 个 token，第 0 个是 `<|im_start|>`。把 8 个 head 平均之后画热力图，第 0 层和第 7 层是两个极端：**第 0 层紧贴对角线**（每个 token 主要看自己和前一两个），其余 token 分给第 0 列的注意力平均只有 5%；**第 7 层几乎整列发亮**，第 0 列独占 61%。逐层看这个比例是 0.05 → 0.19 → 0.20 → 0.25 → 0.47 → 0.54 → 0.51 → 0.61，基本单调上升。

这就是文献里的 **attention sink**。成因可以从 softmax 的约束倒推：每一行的输出必须和为 1，当某个 query 位置其实「没有特别想关注谁」时，它也必须把这一份概率质量倒给某个 key。而序列的第一个 token 对所有位置都可见、又几乎不携带语义（`<|im_start|>` 这类固定开头），于是成了最方便的「垃圾桶」。这不是结构设计出来的，是优化过程自己涌现的——实验里把 chat 模板去掉、直接编码裸文本之后，最后一层的这个比例从 61% 掉到 17%，因为开头那个 token 变成了有实义的字。

它有很实际的后果：StreamingLLM 那类滑动窗口方案必须**永久保留最前面几个 token** 的 KV，窗口一旦滑过第 0 个 token，模型的输出会立刻崩坏；KV cache 量化时，sink 位置也通常要单独留高精度。

{{lab:attention_sink}}

{{quiz:q14}}

> [!KEY]
> 真实权重下浅层贴对角线、深层把过半注意力倒回第 0 个 token（实测 5% → 61%）；attention sink 是训练涌现的，不是结构规定的。
