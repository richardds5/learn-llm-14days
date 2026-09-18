---
day: 3
format: points
title: "位置编码：RoPE 与 YaRN"
subtitle: "一张 [32768, 96] 的常量表：怎么算出来、怎么切、怎么乘进 q/k、长文本时怎么改"
minutes: 85
mainline: "跟着 freqs_cos / freqs_sin 这张表走：precompute_freqs_cis 算出 [32768, D] → 按 start_pos 切成 [T, D] → 广播乘到 [B, T, H, D] 的 q/k 上 → 长文本时 YaRN 改写它的频率"
files:
  - model/model_minimind.py#L62-L84
  - model/model_minimind.py#L209-L219
  - model/model_minimind.py#L27-L39
goals:
  - 能默写 precompute_freqs_cis 里 freqs → outer → cat 三步之后的 shape，并说清为什么是 cat([cos, cos]) 而不是交错重复
  - 能解释 cos.unsqueeze(1) 在 MiniMind 的 [B,T,H,D] 布局下一次广播掉了哪两个维度，和 HF Llama 的同名参数差在哪
  - 能说出 start_pos 从哪来、为什么读的是 shape[1]，以及 KV cache 解码时切出来的 cos 有多长
  - 能用一张热力图证明 ⟨R_m q, R_n k⟩ 只依赖 m − n，并说清 rotate_half 约定和 GPT-J 交错约定之间差的那个 permutation
  - 能算出 YaRN 的 low / high，说清哪些频率分量被压、压多少，以及 `if end / orig_max > 1.0` 判断的到底是什么
---

Day 2 搭好的残差流上没有任何「第几个 token」的信息——`Attention` 里的 `xq @ xk.transpose(-2, -1)` 是纯粹的向量点积，把两个 token 调个个儿，分数一模一样。MiniMind 的位置信息不加在 `hidden_states` 上，而是在 `q_proj`/`k_proj` 之后、算分之前，把 q 和 k **各自转一个角度**。

今天只跟一张表走：`freqs_cos` / `freqs_sin`。它在 `MiniMindModel.__init__` 里算一次（`[32768, 96]`），每次 forward 按 `start_pos` 切出 `[T, 96]`，再广播乘到 `[B, T, H, 96]` 的 q/k 上。四章就是这张表的一生：怎么被算出来（频率 → 波长 → outer 与 cat）；怎么被切、被广播、被乘进去（`rotate_half` 的「前后两半」配对约定）；它带来的核心性质——点积只依赖相对距离；以及输入变长时，YaRN 怎么把表里的频率改窄。Attention 本身（GQA、QK-Norm、因果掩码）是 Day 4 的事。

{{flow: head_dim / rope_theta | *precompute_freqs_cis* | freqs_cos [32768, D] | *切片 [start_pos : start_pos+T]* | cos [T, D] | *apply_rotary_pos_emb* | q, k [B, T, H, D]}}

# 频率表：precompute_freqs_cis

RoPE 的全部「参数」就是一组固定的转速。这一章把 `precompute_freqs_cis` 的三步拆开：先看 `[D/2]` 个频率是怎么来的，再把频率翻译成更好理解的波长（顺带解释 `rope_theta=1e6`），最后看 `outer` 和 `cat` 怎么把它撑成一张 `[32768, 96]` 的大表。

## D/2 个频率：每两个通道共用一个转速 {#rope-freqs}

RoPE 的做法是把一个 head 内部的 `D=96` 个通道**两两配成一对**，每一对当成平面上的一个点，然后按 token 所在的位置把这个点**绕原点转一个角度**。96 个通道配成 48 对，于是需要 48 个不同的转速——转得快的那几对负责区分「隔壁」和「隔两个」，转得慢的负责区分「几千个位置之外」。

{{source:model/model_minimind.py#L62-L63}}

L63 就是在算这 48 个转速。`torch.arange(0, dim, 2)` 取的是 `0, 2, 4, …, 94`，正好 48 个数；除以 `dim` 得到指数 `2i/D ∈ [0, 1)`；`rope_base ** (...)` 再取倒数，得到 `freqs[i] = rope_base^(-2i/D)`，**shape 是 `[D/2] = [48]`**。第 0 个是 `1e6^0 = 1.0`（每往后走一个位置就转 1 弧度，非常快），最后一个是 `1e6^(-94/96) ≈ 1.33e-6`（几乎不动）。

有两个容易看漏的细节。一是 `[: (dim // 2)]` 这个切片：`dim` 是偶数时 `arange(0, dim, 2)` 已经恰好是 `dim//2` 个元素，切片什么也没切掉；只有 `dim` 是奇数时（比如 97，`arange` 给 49 个）它才真的起作用，是一句防御性写法。二是这一行同时把 `attn_factor` 初始化成 `1.0`——它是 YaRN 分支才会改写的量，第四章会用到。

注意这 48 个频率是**纯常量**：只由 `head_dim` 和 `rope_theta` 决定，跟权重、跟输入都无关，所以才敢在 `__init__` 里算一次就当 buffer 挂着（Day 2 的 `persistent=False` 讲的就是这张表）。

{{lab:rope_freqs}}

{{quiz:q1,q2}}

> [!KEY]
> `freqs[i] = rope_theta^(-2i/D)`，shape 是 `[D/2] = [48]`：96 个通道两两一对，每对共用一个转速，从 1.0 一直降到 1.33e-6。

> [!MORE] 为什么是「两两配对做 2D 旋转」而不是别的
> 位置编码要解决的是「让点积感知相对距离」。正交变换（旋转）是唯一既不改变向量长度、又能让 `R_m^T R_n` 只依赖 `m-n` 的线性变换，而任意维度的旋转都可以分解成若干个互不干扰的 2D 平面旋转——所以 `D` 维向量最自然的做法就是拆成 `D/2` 个平面，每个平面给一个角速度。用不同的角速度是为了让不同的通道对负责不同的距离尺度，下一节的波长讲的就是这件事。

## 波长：rope_theta 定下能走多远 {#rope-wavelength}

频率不好直觉，把它倒过来看成**波长**就清楚了：第 `i` 个分量转满一圈（2π）需要走过 `2π / freqs[i] = 2π · rope_base^(2i/D)` 个位置。波长短的分量像秒针，隔壁 token 之间就有明显区别，但走几十个位置就转回原处；波长长的像时针，相邻位置几乎看不出差别，却能在几万个位置的尺度上单调地变化。

{{source:model/model_minimind.py#L27-L29}}

为什么要关心「转回原处」？因为一旦某个分量在模型见过的长度范围内转了好几圈，位置 `p` 和位置 `p + 波长` 在这个分量上就长得一模一样，模型没法靠它区分远近——这就是位置编码的「周期混淆」。MiniMind 的窗口是 `max_position_embeddings = 32768`，所以需要有相当一部分分量的波长**大于 32768**，让它们在整个窗口里连一圈都转不完，充当「绝对的、单调的」远距离信号。

`rope_theta`（源码里叫 `rope_base`）就是调这件事的旋钮，MiniMind 给的是 `1e6`，比 Llama2 惯用的 `1e4` 大两个数量级。下面的实验把两组波长画在同一张对数坐标图上，并数了数各有几个分量的波长超过 32768。

在看图之前先押一个注：`rope_theta` 从 `1e6` 换成 `1e4`，**最高频那个分量（`i=0`）的波长**会怎么变？

{{predict:p1}}

{{lab:rope_wavelength}}

实验给出的答案是：`i=0` 那一端两条曲线**完全重合**，都是 `2π ≈ 6.28`——因为 `freqs[0] = rope_base^0 = 1`，指数是 0，底数再怎么改都没用。调 `rope_theta` 只会撬动低频那一端：`1e6` 下 48 个分量里有 **18 个**波长超过 32768，`1e4` 下只有 **3 个**。这正是长上下文模型普遍把 theta 调大的原因——既延长了远距离的量程，又不动相邻 token 的分辨率。

{{quiz:q3}}

> [!KEY]
> 波长 = `2π · rope_theta^(2i/D)`：`i=0` 恒为 2π 与 theta 无关，调大 theta 只拉长低频端。`1e6` 让 18/48 个分量在 32768 的窗口内转不满一圈。

## outer 与 cat：从 [48] 到 [32768, 96] {#rope-table}

有了 48 个转速，剩下的就是「每个位置各转了多少」。这三行把 `[D/2]` 的频率撑成一张覆盖所有位置的大表。

{{source:model/model_minimind.py#L74-L78}}

`t = torch.arange(end)` 是位置序列 `[0, 1, …, 32767]`，shape `[32768]`。`torch.outer(t, freqs)` 做外积，得到 `[32768, 48]`：第 `p` 行第 `i` 列就是「位置 `p` 在第 `i` 个分量上转过的角度」`p · freqs[i]`。取 `cos` / `sin` 之后 shape 不变，最后 `torch.cat([cos, cos], dim=-1)` 把 `[32768, 48]` **整体复制一份接在后面**，变成 `[32768, 96]`，正好和 `head_dim` 对齐。

关键在最后这一步为什么是「整体复制」而不是 `repeat_interleave`（每个值紧挨着重复两次）。答案在下一章的 `rotate_half`：它把 `D` 维向量切成前后两半，第 `i` 对是 `(x[i], x[i+48])`。要让这一对用同一个角度旋转，`cos` 向量的第 `i` 位和第 `i+48` 位就必须相等——这正是 `cat([cos, cos])` 的效果，而 `repeat_interleave` 给出的是 `θ₀, θ₀, θ₁, θ₁, …`，对应的是 GPT-J 那种「相邻两个下标配对」的约定，两者不能混用。

顺带记住这张表的两个数：`[32768, 96]` 的 fp32 张量是 **12.6 MB**，`cos`/`sin` 两张就是 25 MB（所以 Day 2 里用 `persistent=False` 把它挡在 checkpoint 外面）；第 0 行全是 `cos(0) = 1`，`model/model_minimind.py:216` 那个「buffer 有没有在 meta device 上丢掉」的自检就是靠它。

{{lab:rope_table}}

{{quiz:q4}}

> [!KEY]
> `outer(t, freqs)` 给出 `[end, D/2]` 的角度表，`cat([cos, cos], -1)` 复制成 `[end, D]`——复制而不是交错，是为了配合 `rotate_half` 的前后两半配对。

> [!MORE] attn_factor 乘在哪
> L76-L77 末尾还有一个 `* attn_factor`。不开 YaRN 时它恒为 `1.0`（L63 初始化），整行等于没乘；开了 YaRN 也只有 `rope_scaling["attention_factor"]` 不为 1 时才起作用，而 MiniMind 的默认配置给的正是 `1.0`。YaRN 论文里它用来补偿频率被压缩后 attention logits 的幅度漂移，实现上因为 `cos`/`sin` 被同比例缩放，效果等价于给所有 attention 分数乘一个常数。

# 把表用到 q、k 上：切片与广播

表建好之后，每次 forward 只用得上其中很小一段。这一章按数据流往下走：`MiniMindModel.forward` 怎么按 `start_pos` 切、`apply_rotary_pos_emb` 里那个 `unsqueeze` 怎么把 `[T, D]` 广播到 `[B, T, H, D]`、以及 `rotate_half` 到底在旋转什么。

## start_pos：这次 forward 要表里的哪一段 {#rope-slice}

`position_embeddings` 不是整张表，而是从表里切出来的一小段。切哪一段，全看 `start_pos`。

{{source:model/model_minimind.py#L212-L219}}

L213 是唯一的计算：`past_key_values[0][0]` 是第 0 层缓存的 `xk`，它的 shape 是 `[B, T_cached, KV, D]`，所以序列长度在 **`shape[1]`**——这一点值得留意，HF 的 Cache 类普遍把 KV 存成 `[B, H, T, D]`（序列维在 `shape[2]`），MiniMind 因为在 `Attention.forward` L121 里是沿 `dim=1` 做的 `torch.cat`，序列维就留在第 1 维。没有 cache 时（L212 把 `None` 铺成一个列表），`start_pos = 0`。

L219 用它切出 `(cos, sin)` 元组：`self.freqs_cos[start_pos : start_pos + seq_length]`。切出来的这一对会**原样传给每一层**，L 层共用同一段 `cos/sin`——RoPE 表不是逐层参数，Day 2 的残差流一节已经提过。

这套设计的前提是：进 cache 的 `xk` **已经是旋转过的**。看 `Attention.forward` 的顺序，L119 先 `apply_rotary_pos_emb`，L121 才 `torch.cat([past_key_value[0], xk], dim=1)`——缓存里存的是「已经带好位置」的 k，历史位置永远不需要重算 RoPE。代价是这份 cache 和它当初所在的绝对位置绑死了，不能挪到别的起点复用。

{{lab:rope_slice}}

实验第二行给出了答案：`start_pos = 7`、`input_ids` 是 `[3, 1]`，切出来的 `cos` 是 **`[1, 96]`**——只覆盖这一个新位置。

{{quiz:q5,q6}}

> [!KEY]
> `start_pos = past_key_values[0][0].shape[1]`（cache 的 `xk` 是 `[B,T,KV,D]`，序列维在 dim 1）；切片长度等于本次新增的 token 数，逐 token 解码时 `cos` 退化成 `[1, D]`。

## cos.unsqueeze(1)：一次广播掉 B 和 H {#rope-broadcast}

`apply_rotary_pos_emb` 拿到的 `cos` 是 `[T, D]`，要乘的 `q` 是 `[B, T, H, D]`——两者差了两个维度，而源码里只写了**一个** `unsqueeze`。

{{source:model/model_minimind.py#L82}}

答案藏在调用时机上。回到 `Attention.forward`：L114 把 `xq` reshape 成 `[B, T, H, D]`，L119 就在这里调用 `apply_rotary_pos_emb`，而 `xq.transpose(1, 2)` 要到 L124 才发生。所以 RoPE 看到的 `q` 是 **`[B, T, H, D]`**，heads 在第 2 维而不是第 1 维。

于是 `cos.unsqueeze(1)` 把 `[T, D]` 变成 `[T, 1, D]`，PyTorch 的广播规则再从右往左对齐、在最前面补一维，等价于 `[1, T, 1, D]`：`1↔B`、`T↔T`、`1↔H`、`D↔D` 全部对上。**一个 `unsqueeze` 同时广播掉了 `B` 和 `H` 两个维度**，因为 `cos` 既没有 batch 维（一批样本共用同一段位置），也和 head 无关（所有 head 用同一组转速）。同样的 `cos` 直接乘到 `k` 的 `[B, T, KV=4, D]` 上也成立——GQA 下 q 和 k 的头数不同，但 RoPE 与头数无关。

这里有个跨仓库的坑：HF Llama 里 `q` 已经是 `[B, H, T, D]`、`cos` 是 `[B, T, D]`，同样写 `unsqueeze(1)`，广播掉的只有 `H` 一个维度。**同名参数、同默认值，语义却随张量布局而变**。把 MiniMind 的 `unsqueeze_dim` 照着 HF 的教程改成 2，得到的 `cos` 是 `[T, D, 1]`，会直接在第 2 维上撞车：`The size of tensor a (8) must match the size of tensor b (96)`。

{{lab:rope_broadcast}}

{{quiz:q7,q8}}

> [!KEY]
> RoPE 在 `transpose(1,2)` 之前做，`q` 是 `[B,T,H,D]`、`cos` 是 `[T,D]`，所以 `cos.unsqueeze(1) → [T,1,D]` 一次同时广播掉 `B` 和 `H`。

> [!MORE] 末尾那个 `.to(q.dtype)` 在补什么
> `cos`/`sin` 永远是 fp32（L75 的 `.float()` 定死了）。混合精度下 `q` 是 bf16，`q * cos` 会触发类型提升，整条表达式在 fp32 里算完——这是好事，旋转的精度不会被吃掉。但如果不转回去，`q_embed` 就是 fp32，后面的 `scaled_dot_product_attention` 要么报 dtype 不一致，要么背着一个 fp32 的 q 去算矩阵乘法。`.to(q.dtype)` 就是把这份临时的高精度收回来。实测：`q` 是 bf16 时，`(q * cos.unsqueeze(1))` 的 dtype 是 `torch.float32`，而 `apply_rotary_pos_emb` 的返回值是 `torch.bfloat16`。

## rotate_half：一行做掉 48 个 2D 旋转 {#rope-rotate-half}

把广播搞清楚之后，剩下的就是那个公式本身。`rotate_half` 只有一行，却是整个 RoPE 里最容易记反的一行。

{{source:model/model_minimind.py#L81}}

它把 `D` 维向量切成前后两半，**后半取负搬到前面，前半原样搬到后面**：`[x₀…x₄₇, x₄₈…x₉₅]` 变成 `[-x₄₈…-x₉₅, x₀…x₄₇]`。配上 L82 的 `q * cos + rotate_half(q) * sin`，逐分量展开就是标准的 2D 旋转。设 `x₁ = x[i]`、`x₂ = x[i+48]`、`θ = pos · freqs[i]`：

$$\text{out}_i = x_1\cos\theta - x_2\sin\theta, \qquad \text{out}_{i+48} = x_1\sin\theta + x_2\cos\theta$$

也就是旋转矩阵 $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$ 作用在 $(x_1, x_2)$ 上。第 `i` 对通道是 **`(i, i+D/2)`**——隔了半个向量，不是相邻的两个下标。这也回答了上一章留的那个问题：正因为配对是 `(i, i+48)`，`cos` 表才必须满足「第 `i` 位 == 第 `i+48` 位」，`cat([cos, cos])` 才是对的。

为什么写成 `cat` 而不是写个循环转 48 次？因为这样整个旋转就是**两次逐元素乘法加一次加法**，没有循环、没有 reshape、可以完全融进 GPU kernel。代价是可读性：`rotate_half` 这个名字只描述了搬运动作，看不出它和 2D 旋转的关系。实验里用 `D=8` 的小向量把搬运结果打出来，再把手写的逐对旋转和仓库实现对一对。

{{lab:rope_rotate_half}}

{{quiz:q9}}

> [!KEY]
> `rotate_half` 把 `(x[i], x[i+D/2])` 配成一对做 2D 旋转，一次 `cat` 代替 48 次循环；配对方式和 `cat([cos,cos])` 是一套约定的两面。

# RoPE 的核心性质：相对位置

前两章把「怎么算」讲完了，这一章回答「为什么这么算有用」：旋转带来的是点积只依赖相对距离。第二节是选学，讲另一种同样合法、但权重不能互换的配对约定。

## 点积只依赖 m − n {#rope-relative}

RoPE 之所以叫「旋转式**相对**位置编码」，靠的是这条性质：把同一个内容向量 `q` 放在位置 `m`、`k` 放在位置 `n`，它们旋转之后的点积 $\langle R_m q, R_n k\rangle$ 只是 `m - n` 的函数，和 `m`、`n` 各自的绝对值无关。

{{source:model/model_minimind.py#L80-L84}}

推导只要一行：旋转矩阵是正交的，$R_m^\top = R_{-m}$，于是 $\langle R_m q, R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k$。而 `rotate_half` 对每一对通道做的正是 2D 旋转，48 对各转各的，整体仍然是一个正交变换，所以这条性质对整个 `D` 维向量成立。

这件事对 attention 的意义很直接：`Attention.forward` 里算分数的 `xq @ xk.transpose(-2,-1)` 拿到的每一个格子，都只反映「这两个 token 隔了多远」，而不是「它们各自排第几」。模型因此不需要为每个绝对位置单独学一套行为，长度外推也才有讨论的余地。

下面的实验把一个**固定的**内容向量摆在 0~47 每一个位置上，算出 `48 × 48` 的点积矩阵画成热力图。如果性质成立，图上应该出现沿对角线方向的条纹——同一条对角线上 `m - n` 相同，颜色就该一样。注意实验里 `q` 用的是 `expand`，每个位置放的是同一份内容；换成每个位置一份随机内容，条纹立刻消失，因为那时变的不只是位置。

{{lab:rope_relative}}

实测 `dots[5,20]` 和 `dots[25,40]`（相对距离都是 −15）相差 `7.75e-07`，而隔壁一格的 `dots[5,21]` 就差了 1.59。

{{quiz:q10}}

> [!KEY]
> 旋转是正交变换，$R_m^\top R_n = R_{n-m}$，所以 attention 分数只感知相对距离；热力图上表现为沿对角线方向的条纹。

> [!MORE] 「只依赖相对距离」不等于「能外推」
> 位置 5000 和 5010 之间的相对旋转，和位置 5 和 15 之间完全一样——公式上两者不可区分。但模型在训练时只见过有限范围内的相对距离组合（MiniMind 的 SFT 只训到 768），一旦推理时的距离跨到训练没覆盖过的量级，低频分量会转到训练中从未出现过的角度区间，attention 的行为就没有保障。这是第四章 YaRN 要处理的问题：公式合法 ≠ 模型学过。

## 另一种配对约定与那个 permutation {#rope-conventions .side}

`rotate_half` 不是实现 RoPE 的唯一写法。Meta 原版 `llama`（fairscale 版）用的是 `torch.view_as_complex`：把**相邻的两个下标** `(x[2i], x[2i+1])` 当成一个复数，乘以 $e^{i\theta}$ 完成旋转。GPT-J 也是这套。两种写法数学上都是「`D/2` 个 2D 旋转」，但**配成对的通道不是同一批**。

后果是：同一份 `q_proj` / `k_proj` 权重，在两种实现下会把同一个频率分配给不同的输入通道，算出来的 attention 分数不一样——**权重不能直接互换**。但两者之间差的只是一次固定的通道重排：把偶数下标搬到前半、奇数下标搬到后半（`perm = [0, 2, 4, …, 1, 3, 5, …]`），在重排后的向量上用 `rotate_half` 的公式算，结果换回原顺序就和复数写法完全一致。

这正是 HF 的 `convert_llama_weights_to_hf.py` 里那个 `permute()` 函数在做的事：把 Meta 原版 checkpoint 转成 HF 格式时，`q_proj` / `k_proj` 的权重矩阵要按输出通道重排一遍，之后才能喂给 `rotate_half` 版本的实现。反过来说，如果你从别处拿一份 RoPE 权重接进 MiniMind，第一件要确认的事就是它属于哪种约定——两种都「正确」，但混用会静默地算错，不报任何错。

{{lab:rope_conventions}}

{{quiz:q11}}

> [!KEY]
> `rotate_half`（HF / MiniMind）和复数相邻配对（Meta 原版 / GPT-J）只差一次通道 permutation：数学等价，但权重混用会静默算错。

# YaRN：把频率改窄的长文本外推

最后一章看 `precompute_freqs_cis` 里那段平时不执行的分支。YaRN 的思路很直接：既然低频分量负责远距离，那就只把低频压慢，让原本要走 32768 个位置才转完的一圈，摊到更长的范围上。三节分别讲「压多少」「什么时候压」和「压了到底有没有用」。

## low / high / ramp：只压低频那一段 {#yarn-ramp}

YaRN 不像线性插值那样把所有频率一刀切地除以 `factor`，而是按分量下标分三段处理：高频原封不动，低频整体压慢，中间线性过渡。分界点由 `beta_fast` / `beta_slow` 算出来。

{{source:model/model_minimind.py#L70-L73}}

`inv_dim(b)` 是一个反解：给定「波长恰好等于 `2π·b` 个位置」，反推这是第几个频率分量。`beta_fast=32` 对应波长约 201 个位置，`beta_slow=1` 对应约 6.3 个位置——注意 `b` 越大波长越长、分量下标越小，所以 `low = floor(inv_dim(beta_fast))`、`high = ceil(inv_dim(beta_slow))`，`low < high`。代入默认配置（`dim=96`、`rope_base=1e6`、`orig_max=2048`）得到 **`low=8`、`high=21`**。

`ramp = clamp((i - low) / (high - low), 0, 1)` 就是那条 0→1 的斜坡（注释里的 γ），最后一行 `freqs = freqs * (1 - ramp + ramp / factor)` 把它翻译成缩放系数：γ=0 时系数是 1（频率不变），γ=1 时系数是 `1/factor = 1/16`（频率除以 16，波长拉长 16 倍）。48 个分量里，**9 个完全不动、12 个线性过渡、27 个被完整压缩**。

这个分段的物理含义：`i ≤ 8` 的分量波长在 200 个位置以内，是靠它们区分「上一个词」和「上上个词」的，动了会直接破坏局部语序；`i ≥ 21` 的分量波长本来就上千，压慢 16 倍也不会让相邻位置糊在一起，却能把量程扩到 16 倍。

{{lab:yarn_ramp}}

{{quiz:q12,q13}}

> [!KEY]
> `low=8`、`high=21`：下标 ≤ low 的高频分量频率不变，≥ high 的低频分量除以 `factor=16`，中间按 ramp 线性过渡（默认 9 / 12 / 27 个）。

## 触发条件看的是 end，不是输入长度 {#yarn-always-on}

`inference_rope_scaling` 这个名字很容易让人以为它是个「检测到长输入就切换」的动态开关。实际上整段 YaRN 逻辑在 `MiniMindModel.__init__` 里只跑一次，而它跑不跑，只取决于 `precompute_freqs_cis` 的形参 `end`。

{{source:model/model_minimind.py#L31-L39}}

链路是这样的：`inference_rope_scaling=True` 才会把 `rope_scaling` 构造成上面那个 dict（否则是 `None`）；`MiniMindModel.__init__` L205 调用 `precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, …)`；函数里 L69 判断的 `end / orig_max`，分子就是 `config.max_position_embeddings`，分母是 dict 里的 `2048`。默认配置下比值是 16，恒大于 1——只要开了开关，**整张表从位置 0 开始就已经是缩放过的**，喂 10 个 token 用的也是缩放后的频率。

反过来，这个比值取决于你给 `max_position_embeddings` 填了多少。下面的实验拿三组配置各算一张表：前两组用默认的 32768、分别关 / 开开关，第三组把它调到 1024、开关仍然打开。先押注：第三组的表和完全不开 YaRN 的表相比会是什么关系？

{{predict:p2}}

{{lab:yarn_always_on}}

第三行的 `torch.equal` 是 `True`：`1024 / 2048 = 0.5`，L69 的条件不成立，YaRN 分支被整段跳过，`rope_scaling` 这个 dict 白构造了。位置 1000、第 24 个分量上的 `cos` 值也回到了不开时的 `0.5403`。

{{quiz:q14}}

> [!KEY]
> `if end / orig_max > 1.0` 里的 `end` 是 `config.max_position_embeddings`，和实际输入长度无关；条件不成立时 YaRN 被整段跳过，开关形同虚设。

## 超参数要和 checkpoint 的训练长度对齐 {#yarn-reality}

README 里有一张图，说明同一段长文本上开启 YaRN 后 PPL 更低。这是不是对任何 checkpoint、任何长度都成立？仓库里正好有 `out/full_sft_768.pth` 可以直接试。

先看一个错配：`trainer/train_full_sft.py` 里 `--max_seq_len` 的默认值是 **768**，也就是这份权重训练时从没见过超过 768 的位置；而 YaRN 默认假设「原本训练覆盖的范围」是 `original_max_position_embeddings = 2048`。`low` / `high` 这两个分界完全是按 2048 这个假设算出来的，和这份 checkpoint 的实际经历对不上。

下面的实验用同一段文本、同一份权重，只切换 `inference_rope_scaling`，比较 `T=512`（训练范围内）和 `T=3072`（远超训练范围）两种长度下的整段平均 loss。在跑之前先押注：`T=3072` 这一组，开启 YaRN 之后 loss 会更低还是更高？

{{predict:p3}}

{{lab:yarn_reality}}

实测：`T=512` 时 4.8806（关）vs 4.8886（开），几乎持平但开启的略高——这正好印证上一节，短输入也照样用的是缩放后的表；`T=3072` 时 4.1141（关）vs **4.3824（开）**，开启 YaRN 明显更差。

这不是说 YaRN 方法本身有问题。它是**免训练**的手段，前提是「模型在 `original_max_position_embeddings` 以内已经学得很稳」，再把这份能力靠压慢低频摊到更长的范围上。这份只训到 768 的 SFT 权重连前提都不满足，压缩低频只是把它没学过的一批角度换成了另一批没学过的角度。所以结论不是「别用 YaRN」，而是 `inference_rope_scaling` 是一个**需要在自己的 checkpoint 上实测**的旋钮，不是「开了一定更好」的开关——README 里那张 PPL 图也未必是用默认参数、默认权重跑出来的。

{{quiz:q15}}

> [!KEY]
> `full_sft_768` 上开启默认 YaRN 反而让 loss 变高（T=3072：4.11 → 4.38）——这是个要在自己权重上实测的旋钮，不是「开了一定更好」的开关。

> [!MORE] 那把 orig_max 改成 768 不就对齐了吗
> 这是最自然的下一步，实验 task 2 就是干这个：在 `build` 里加一句 `cfg.rope_scaling["original_max_position_embeddings"] = 768`。结果是 `T=3072` 的 loss 变成 **4.7397**，比默认的 4.3824 **还差**。
> 原因在 `inv_dim`：`orig_max` 变小会让 `low` 从 8 降到 4、`high` 从 21 降到 17，被完整压缩的分量从 27 个涨到 31 个，连中频都被动了——离模型真正学过的分布更远。
> 换句话说，「把超参数对齐到真实训练长度」是个必要的检查，但不是充分的修复：一个从头到尾只在 768 以内训练过的模型，本来就没有可供外推的长程能力。真想要长上下文，得在长数据上继续训（或者至少做一轮 YaRN-aware 的微调）。
