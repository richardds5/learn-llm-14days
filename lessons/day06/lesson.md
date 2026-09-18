---
day: 6
format: points
title: "MoE：路由、top-k、aux loss 与激活参数"
subtitle: "一张 [B*T, E] 的选票表，决定每个 token 走哪个 FFN"
minutes: 85
mainline: "x [B,T,C] 被拉平成 [B*T,C]，gate 给出一张 [B*T,E] 的选票表，top-k 选人，按 expert 分桶算完再 index_add_ 写回，最后变回 [B,T,C]"
files:
  - model/model_minimind.py#L40-L45
  - model/model_minimind.py#L148-L176
  - trainer/trainer_utils.py#L18-L28
goals:
  - 能背出 MOEFeedForward.forward 每一行之后 x_flat / scores / topk_idx / topk_weight / mask / token_idx / y 的 shape
  - 能说清 topk_weight 那一行的归一化在 k=1 和 k≥2 下是两件完全不同的事，以及它对 gate 梯度的影响
  - 能写出 aux_loss = E·Σ f_i·P_i·coef，并说明它惩罚的是「f 和 P 一起集中」而不是单纯的负载不均
  - 能从 total / base / active 三个数手算出 minimind-3-moe 的 198M-A64M
  - 能解释为什么 4 experts / top-1 的 FLOPs 和 dense 几乎一样，实测却更慢
---

Day 5 拆完了 dense 版的 `FeedForward`：所有 token 排队过同一组 SwiGLU 矩阵。今天换掉的就是这一个零件——`config.use_moe=True` 时 `MiniMindBlock.__init__` 把 `self.mlp` 换成 `MOEFeedForward`（`model/model_minimind.py:184`）。对上层是纯黑盒替换：进出都是 `[B,T,C]`，残差流那条宽 `C` 的总线一个字都不用改。

变的全在盒子里面：MoE 先给每个 token 发一张选票，按票把它分给 `E` 个 FFN 中的 `k` 个，算完再塞回原来那一行。今天跟着这张选票走一个来回。

四章分工：第一章看票怎么打出来——分数表、top-k，以及只有 k=1 时才显形的归一化陷阱；第二章看被选中的 token 怎么被捞出来、又怎么写回去；第三章讲 aux loss，它在默认的 top-1 配置下几乎是 router 唯一的学习信号；第四章算账：198M 的模型为什么只激活 64M。

{{flow: x [B,T,C] | *view* | x_flat [B*T,C] | *gate+softmax* | scores [B*T,E] | *topk* | topk_idx [B*T,k] | *分发 + index_add_* | y [B*T,C] | *view* | [B,T,C]}}

# 路由：从 [B,T,C] 到每个 token 的一张选票

`MOEFeedForward.forward` 的前五行（L157-L161）干的全是「决定谁去哪」这一件事，一个 expert 的参数都还没碰。这一章逐行拆它：先把三维的 batch 结构拍扁，再用一个极小的线性层打分，最后用 `torch.topk` 选人、归一化权重。

## 展平：[B,T,C] → [B*T,C] {#flatten}

进来的 `x` 和 dense FFN 拿到的是同一个东西：`post_attention_layernorm` 之后的 `[B, T, C]`。函数做的第一件事是把它拆成三个 python int 存起来，然后立刻拉平。

{{source:model/model_minimind.py#L156-L158}}

为什么要拉平？因为**路由是 token 级别的决策**。「这个 token 该由谁处理」只取决于它自己的那条 `C` 维向量，和它属于第几个样本、排在第几个位置毫无关系——这一点和 attention 完全相反（attention 必须知道位置才能做因果掩码）。拉平之后 `x_flat` 的每一行就是一个独立 token，接下来的 `topk`、布尔 `mask`、`nonzero()` 取下标、`index_add_` 写回，全都变成第 0 维上的一维索引，写起来最省事。

常见误解要先排掉：`nn.Linear` 和 `F.softmax(dim=-1)` 都能直接吃三维输入，所以拉平**不是**为了迁就算子。真正被简化的是「按下标捞出一批行」——`x_flat[token_idx]` 在二维上是一句话，在 `[B,T,C]` 上得先把一维下标拆回 `(b, t)` 两个坐标。

`view` 本身不搬数据：`x` 在这里是连续的，`view(-1, hidden_dim)` 只是换一副解释内存的眼镜，代价为 0。最后一行 `y.view(batch_size, seq_len, hidden_dim)`（L176）再用开头存下的三个 int 还原形状——这就是 L157 存在的意义。

{{lab:flatten}}

trace 里盯住两行：`x: [B=3, T=7, C=768]` 变成 `x_flat: [B*T=21, C=768]`，21 个 token 从此各走各的。

{{quiz:q1}}

> [!KEY]
> MoE 路由是 token 级决策，所以第一步就把 `[B,T,C]` 拉平成 `[B*T,C]`；`view` 零开销，结尾再用开头存下的三个 int 还原回去。

## gate：一个 [E,C] 的线性层加一次 softmax {#gate-scores}

router 本体小得惊人：一个 `nn.Linear(C, E, bias=False)`，默认 `E=4`，整个 8 层模型加起来只有 24,576 个参数（占 198M 的 0.012%）。

{{source:model/model_minimind.py#L152}}

它的 `weight` 是 `[E, C] = [4, 768]`，可以理解成 4 个「专家画像向量」；`gate(x_flat)` 就是拿每个 token 和这 4 个画像做内积，得到 `[B*T, E]` 的 logits。没有 bias，所以打分完全由方向决定。

{{source:model/model_minimind.py#L159}}

`softmax(dim=-1)` 沿 `E` 这一维归一化——**每个 token 各自**在 4 个 expert 上得到一个和为 1 的概率分布，token 之间互不影响。这里有个容易写反的地方：如果写成 `dim=0`，归一化就变成「沿 token 维」，每一列和为 1、每一行不再是概率分布，路由语义直接崩掉。实验标题里那个「每行和与 1 的最大偏差 1.2e-07」就是在验证 `dim=-1` 这件事。

`scores` 这个变量后面会被用两次：一次喂给 `topk` 做**离散选择**（选完就断了梯度），一次原样喂给 aux loss（`scores.mean(0)`，可导）。记住这个分叉，第三章会反复用到。

维度符号上有个坑：默认配置里 `num_experts=4` 和 `num_key_value_heads=4` 数值相同。learnkit 按「当前在哪个函数里」消歧，在 `MOEFeedForward.forward` 里把 4 标成 `E`，在 `Attention.forward` 里标成 `KV`，所以你在 trace 里看到的 `[B*T, E=4]` 是可信的。

{{lab:gate_scores}}

{{quiz:q2}}

> [!KEY]
> `scores = softmax(gate(x_flat), dim=-1)` 是 `[B*T, E]`：每行一个 token 在 E 个 expert 上的概率分布；gate 只有 `[E, C]` 这一个无 bias 的小矩阵。

> [!MORE] MiniMindConfig 里的五个 MoE 字段
> `model/model_minimind.py:41-45`，`use_moe=False` 时全部被忽略：
> `num_experts=4`（记作 `E`）、`num_experts_per_tok=1`（记作 `k`）、`moe_intermediate_size` 默认直接复用 dense 的 `intermediate_size=2432`（注意这是**每个 expert 自己的** `I`）、`norm_topk_prob=True`、`router_aux_loss_coef=5e-4`。
> `minimind-3-moe` 用的就是这套默认值：README 里管 `4 experts / top-1` 叫「甜点配置」。

## torch.topk：切成权重表和身份证表 {#topk}

有了 `[B*T, E]` 的分数表，选人只要一行。

{{source:model/model_minimind.py#L160}}

`torch.topk` 一次返回两个张量，形状都是 `[B*T, k]`，但含义完全不同：`topk_weight` 是选中的那 `k` 个概率值（float32），`topk_idx` 是它们的**列号**，也就是 expert 的编号（int64）。前者后面要当加权系数，后者后面要当分桶依据——一张权重表、一张身份证表。

`k` 来自 `config.num_experts_per_tok`，默认 **1**。k=1 时两个返回值都是 `[B*T, 1]`，注意那个尾巴上的 1 不会被自动压掉，后面 `mask = (topk_idx == i)` 得到的也是 `[B*T, 1]` 而不是 `[B*T]`——L166 那句 `mask.any(dim=-1)` 就是为了把这条长度为 1 的尾巴收掉。

`sorted=False` 是一句「我不在乎 k 个里谁排前面」的声明：反正后面按 expert 编号分桶，顺序无所谓，让底层实现省掉一次排序。但它只是**不保证**有序，不是**保证**无序——CPU 上实测返回的仍然是从大到小，别把这当成契约来依赖。另外 `k` 必须 ≤ `E`，传 5 会直接 `RuntimeError: selected index k out of range`。

{{lab:topk_shapes}}

表里第 0 个 token 选中的是 expert 2，概率 0.53；k=4 时四个权重是 0.53 / 0.215 / 0.129 / 0.125，加起来正好是 1——因为 `scores` 本来就是一个 4 维的概率分布。

{{quiz:q3}}

> [!KEY]
> `torch.topk(scores, k, dim=-1)` 返回两个 `[B*T, k]`：`topk_weight` 是权重（float32），`topk_idx` 是 expert 编号（int64）；k=1 时尾巴上那个 1 不会消失。

## norm_topk_prob 与那个 1e-20 {#norm-topk}

选出 k 个之后还有一行归一化，`norm_topk_prob` 默认为 `True`：

{{source:model/model_minimind.py#L161}}

意图很清楚：k≥2 时，选中的 k 个概率加起来一般不到 1（比如 0.53 + 0.215 = 0.745），直接拿去加权会让这个 token 的输出整体缩水；除以它们的和之后变成 0.712 / 0.288，加权和就是一个真正的凸组合。

`+ 1e-20` 是防除零：理论上 softmax 的输出恒为正，和不可能是 0，但 fp16/bf16 下几个极小的数相加确实可能下溢成 0。加一个远小于任何正常取值的常数，比写 `if` 分支便宜得多。

问题出在 **k=1**。此时 `topk_weight.sum(dim=-1)` 就是它自己，这一行变成 `w / (w + 1e-20)`。float32 的有效位只有约 7 位十进制，`w` 在 0.2~0.9 这个量级时，`w + 1e-20` 被舍入成 `w` **本身**，于是结果是精确的 `1.0`——实验里「与 1 的偏差」那一列打出来是 `0.000e+00`，不是「约等于 1」，是位模式完全相同的 1.0。

后果是：k=1 时这一行把一个携带了 gate 信息的概率（0.53、0.317……）碾成了常数 1，`y` 的数值从此和 gate 的输出**完全脱钩**，只剩「选了谁」这个离散信息还在。第三章会量一量这件事对梯度的影响。顺带一提，`norm_topk_prob=False` 时这一行整个不执行，`topk_weight` 保持原始概率值。

{{lab:norm_topk}}

{{quiz:q4,q5}}

> [!KEY]
> k≥2 时这一行是真归一化（0.53/0.215 → 0.712/0.288）；k=1 时它恒等于 `1.0`（float32 下 `w+1e-20` 舍入成 `w`），把 gate 的数值信息彻底抹掉。

# 分发与写回：只让被选中的 token 过 expert

选票发完了，接下来是 MoE 实现里最有特色的一段：`for i, expert in enumerate(self.experts)` 这个纯 Python 循环。它不是「每个 token 挑一个 expert」，而是反过来——**每个 expert 挑自己的那批 token**。这一章看 mask / token_idx 怎么产生、结果又怎么被加回去。

## mask 与 token_idx：长度由数据决定 {#dispatch}

循环体只有四行，但每一行的 shape 都值得盯一眼。

{{source:model/model_minimind.py#L163-L167}}

- `mask = (topk_idx == i)`：`[B*T, k]` 的布尔矩阵，标记「哪个 token 的哪个 slot 选中了 expert i」。
- `mask.any()`：整块有没有 `True`。没有就说明这个 expert 这一步闲置，跳到 `elif`（下一节讲）。
- `token_idx = mask.any(dim=-1).nonzero().flatten()`：先把 `[B*T, k]` 沿 k 压成 `[B*T]` 的一维布尔，`nonzero()` 得到 `[n_i, 1]` 的下标，`flatten()` 收成 `[n_i]`。这个 `n_i` 是**运行时才知道**的——路由结果数据依赖，同一个 expert 这一批拿 11 个 token，换一批数据可能只拿 1 个。
- `weight = topk_weight[mask].view(-1, 1)`：布尔索引把选中的权重抽成一维，再补一个尾巴变成 `[n_i, 1]`，为的是下一行能和 `[n_i, C]` 的 expert 输出广播相乘。

代价也在这里：循环次数等于 `E`，是 Python 层面的串行，`E` 越大启停开销越高（[见分桶调度的代价](#moe-cost)）。工业实现会把它换成一次 gather + 分段 matmul。

{{lab:dispatch}}

实验里 `E=4`，所以循环体被执行 4 次（网页上每行右边标着 `×4`，用分页器可以逐次翻看）。4 次的 `token_idx` 长度分别是 2 / 1 / 11 / 7，加起来正好 21——k=1 时每个 token 恰好被分配一次。

{{quiz:q6}}

> [!KEY]
> 循环是「每个 expert 捞自己的 token」：`mask` 是 `[B*T,k]` 布尔，`token_idx` 是 `[n_i]`，`n_i` 由路由结果决定，k=1 时所有 `n_i` 之和等于 `B*T`。

## index_add_：为什么是加而不是赋值 {#index-add}

一个 expert 算完之后，结果要回到 `y` 里它原来的那些行。

{{source:model/model_minimind.py#L168}}

从里往外读：`x_flat[token_idx]` 抽出 `[n_i, C]` 的子批，过这个 expert 的 SwiGLU 得到 `[n_i, C]`，乘上 `[n_i, 1]` 的权重（广播），`.to(y.dtype)` 对齐 dtype，最后 `y.index_add_(0, token_idx, ...)` 沿第 0 维把这 `n_i` 行**加**到 `y` 的对应行上。`y` 在 L162 被初始化成 `torch.zeros_like(x_flat)`，正是为了当这个累加器。

为什么必须是「加」？因为 k≥2 时同一个 token 会出现在多个 expert 的 `token_idx` 里。写成 `y[token_idx] = ...` 的话，后一个 expert 会直接覆盖前一个的结果，这个 token 就只剩最后一个 expert 的贡献、而且权重还不是 1——输出直接错掉。用 `index_add_` 写回，`y` 的每一行自然就是 `Σ_j w_j · expert_j(x)`，正是 MoE 的定义式。k=1 时两种写法数值等价，但代码要对任意 `k` 都成立。

`index_add_` 是**原地**操作（带下划线），不会新分配 `[B*T, C]` 的缓冲区。这也意味着这一行不适合用 trace 的 `expand=` 去拆——`expand` 会在该行执行前把子表达式预先求值一遍，对有副作用的行会重复写入。

{{lab:index_add}}

实验用 k=2：「真实 y vs 手工 k 个 expert 加权和」的最大逐元素差是 `0.00e+00`（完全一致），而把 `index_add_` 换成赋值之后差到了 `3.08e-01`；每个 token 被写回的次数是 2 次。

{{quiz:q7}}

> [!KEY]
> `y.index_add_(0, token_idx, expert(x)*weight)` 把结果按行加回 `y`；「加」而不是「赋值」是因为 k≥2 时同一行会被多个 expert 写，`y` 的每行是 k 个 expert 输出的加权和。

## 闲置 expert 的假梯度路径 {#ddp-unused .side}

`if mask.any()` 还有一条 `elif` 分支，是全文最费解的一行：

{{source:model/model_minimind.py#L169-L170}}

`0 * sum(...)` 数值上恒等于 0，加到 `y[0,0]` 上不改变任何结果。但它在**计算图**上是实打实的一条边：`sum(p.sum() for p in expert.parameters())` 把这个 expert 的每一个参数都连进了图里，于是 `backward()` 之后它们的 `.grad` 是全 0 的张量，而不是 `None`。

这是给 `DistributedDataParallel` 擦的屁股。DDP 在构造时会登记所有参数，反向传播时靠「每个参数的梯度都到齐了」来触发 all-reduce。如果某个 rank 这一步的 batch 里恰好没有 token 路由到某个 expert，那个 expert 的参数就掉出了计算图，DDP 等不到它的梯度，要么直接报错，要么必须打开 `find_unused_parameters=True`——那会在每次反向前多扫一遍整张图，很慢。造一条值为 0 的路径，就把这个问题在模型侧解决了。

注意 `elif self.training` 这个条件：只有训练时才做。推理时没有 backward，这一行纯属浪费。单卡训练其实也不需要它，但留着无害。

{{lab:ddp_unused}}

实验用 32 个 expert 配 21 个 token，必然喂不满：这一步只有 11 个 expert 拿到了 token，21 个闲置；但 backward 之后 **32 个 expert 全都有 `.grad`**，闲置那 21 个的梯度绝对值最大是 `0.0e+00`。把 `model.train()` 改成 `eval()` 再跑，有 `.grad` 的就只剩 11 个了。

{{quiz:q8}}

> [!KEY]
> `y[0,0] += 0 * sum(p.sum() ...)` 造了一条数值为 0 的计算图边，让这一步没分到 token 的 expert 也能有 `.grad`，避免 DDP 的「未使用参数」问题。

# aux loss：router 的学习信号

`topk_idx` 是 `argmax` 类的离散选择，不可导；`topk_weight` 在默认的 k=1 下又被归一化碾成了常数 1。那 `gate.weight` 靠什么学习？答案是 `forward` 末尾那三行——它们算出的 `aux_loss` 会一路加到主 loss 上。这一章先看公式，再看它怎么接线，最后用梯度量级验收。

## aux_loss = E · Σ f_i·P_i · coef {#aux-formula}

{{source:model/model_minimind.py#L171-L173}}

两行代码对应两个向量。`load`（记作 `f`）来自 `F.one_hot(topk_idx, E).float().mean(0)`：把每个 token 的选择摊成 one-hot 再沿 token 维求平均，得到「每个 expert 实际分到的 token 比例」——它来自离散选择，**不可导**。`scores.mean(0)`（记作 `P`）是 gate 给每个 expert 的平均概率，没过 topk、没被归一化污染，是一条干净的可导路径。两者点乘、乘 `E`、再乘 `coef`：

$$\text{aux\_loss} = E \cdot \sum_{i=1}^{E} f_i P_i \cdot \text{coef}$$

`f` 和 `P` 都是和为 1 的分布。两边都均匀时 `Σ f_i P_i = E·(1/E)(1/E) = 1/E`，乘 `E` 之后恰好是 1——**完全均匀路由时 aux_loss 正好等于 `coef` 本身**。两边都坍缩到同一个 expert 上时 `Σ f_i P_i` 逼近 1，乘 `E` 之后逼近 `E × coef`。这是 Switch Transformer 那一支的标准写法。

那么只坍缩一边呢？比如 token 全被路由到 expert 0（`f` 完全坍缩），但 gate 的平均概率 `P` 仍是完全均匀的 0.25——先下注再看。

{{predict:p1}}

{{lab:aux_formula}}

四行结果是 1.00 / 1.14 / **1.00** / 3.88 倍 `coef`。只要 `P` 均匀，`Σ f_i P_i = (1/E)·Σ f_i = 1/E` 就与 `f` 完全无关，aux_loss 钉在 `coef` 这个值上——但这**不是**全局最小值：让一半 expert 完全闲置、token 以微弱优势只在另一半里均分、剩余概率摊给闲置的那一半，`E=4` 时能压到约 `0.67×coef`（`f` 全押一个 expert 反而不行：被选中的概率至少是 `1/E`，此时 aux_loss ≥ `coef`）。可见它惩罚的从来不是「负载不均」本身，而是**「router 的选择和 router 的信心一起集中在少数 expert 上」**——也合理，梯度只能通过 `P` 回传。

{{quiz:q9}}

> [!KEY]
> `aux_loss = E·Σ f_i·P_i·coef`：两边都均匀时等于 `coef`，一起坍缩时逼近 `E·coef`；只要 `P` 均匀这个值恒为 `coef`（不是全局最小值，f 与 P 错开时还能更低）——它罚的是 f 与 P 的对齐。

> [!MORE] load 的 shape 其实是 [k, E]，不是 [E]
> `F.one_hot(topk_idx, E)` 把 `[B*T, k]` 变成 `[B*T, k, E]`，`.mean(0)` 只压掉第 0 维，所以 `load` 是 `[k, E]`——k=1 时是 `[1, 4]`，靠广播和 `[E]` 的 `scores.mean(0)` 相乘也能算对。
> 但 k≥2 时每个 slot 各自是一个和为 1 的分布，`load.sum()` 等于 `k`，于是整个 aux_loss 大约被放大 `k` 倍（实测 2 层模型 k=1 是 1.1e-3、k=2 是 2.1e-3）。换句话说，`router_aux_loss_coef` 的有效强度是跟着 `k` 走的，调 `k` 时值得留意。

## 只在 train() 下计算，再逐层求和 {#aux-plumbing}

`aux_loss` 的接线比公式本身更容易出错，值得单独走一遍。

{{source:model/model_minimind.py#L174-L175}}

第一处开关是 L171 的 `if self.training and self.config.router_aux_loss_coef > 0`。`model.eval()` 把 `self.training` 置为 `False`，于是走 `else`：`scores.new_zeros(1).squeeze()` 造一个**同 device、同 dtype 的标量 0**，不带梯度（用 `new_zeros` 而不是 `torch.tensor(0.)` 是为了在 cuda/mps 上不引入设备不匹配）。所以推理时 `res.aux_loss` 恒为 0，别拿 eval 下的值判断路由是否均衡。

第二处是存放方式：`self.aux_loss = ...` 挂在模块实例上，是个**副作用**，不走 `return`；`MiniMindModel.forward` 结束前再统一收集：

{{source:model/model_minimind.py#L231}}

第二个参数是 `sum` 的初始值，保证纯 dense 模型（列表为空）也能拿到标量 tensor 而不是 python int `0`——下游 `res.loss + res.aux_loss` 才不会类型出岔子。

最后一处在训练脚本里：`trainer/train_pretrain.py:37` 的 `loss = res.loss + res.aux_loss`，DPO / LoRA / GRPO 等所有训练循环都有同一行。

{{lab:aux_plumbing}}

实验里 4 层 MoE 模型 `train()` 下总 aux_loss 是 0.002237，每层各自 5.3e-4 ~ 6.0e-4（都在 `coef=5e-4` 附近）；`eval()` 下每层都是 0；dense 模型两种模式都是 0，shape 全是 `()` 的标量。

{{quiz:q10,q11}}

> [!KEY]
> `aux_loss` 只在 `train()` 且 `coef>0` 时计算，以副作用形式挂在每个 MoE 层上，由 `MiniMindModel.forward` 求和（初始值是标量 0），最后在训练脚本里加到主 loss 上。

## gate.weight 的梯度来源 {#router-grad}

把前面几节串起来问一个问题：一次 `loss.backward()` 之后，`gate.weight` 到底从哪里拿到梯度？

从 `y` 往回看只有两条路能到 gate。第一条是 `topk_idx`——离散的整数选择，`torch.topk` 的 indices 输出不在计算图里，这条路是断的。第二条是 `topk_weight` 的**数值**，它乘在 expert 输出上；但在默认的 k=1 + `norm_topk_prob=True` 下，[第一章的归一化那一节](#norm-topk)已经看到这个数被碾成了常数 1：解析导数 `1e-20/(w+1e-20)²` 只有 1e-20 量级，实际 autograd 里分子那条 `1/s` 和分母那条 `-w/s²` 又几乎完全相消，只剩舍入噪声。

所以默认配置下主 loss 给 router 的梯度约等于零，`aux_loss` 里那条 `scores.mean(0)` 就成了唯一的有效信号——`router_aux_loss_coef=5e-4` 不是可选的正则项，是命根子。

那如果保持 k=1、也不开 aux loss，只把 `norm_topk_prob` 关掉呢？此时 `topk_weight` 保留原始概率值。先下注：

{{predict:p2}}

{{lab:router_grad}}

四行的 `gate.weight.grad` 绝对值最大依次是 `7.1e-10`（k=1，归一化，无 aux）、`1.0e-4`（k=1，开 aux）、`1.2e-2`（k=1，**关掉归一化**，无 aux）、`2.8e-2`（k=2，归一化，无 aux）。第一行换几个种子都稳定停在 1e-9 附近，那是 float32 的噪声地板。结论是：**杀死梯度的不是 top-1，是那一行归一化**——关掉它，或者把 k 提到 2（归一化变成几个数之间的真除法，会产生耦合梯度），主 loss 立刻能推动 router。

{{quiz:q12}}

> [!KEY]
> k=1 且 `norm_topk_prob=True` 时主 loss 给 gate 的梯度只有 1e-9 量级的噪声，router 全靠 aux loss 学习；关掉归一化或把 k 提到 ≥2，主 loss 的梯度立刻回到 1e-2 量级。

# 参数账与训练好的 router

前面都在看一次 forward 的内部。这一章退一步看两件整体的事：`198M-A64M` 这个记法是怎么算出来的，以及一份真正训练过的 MoE 权重，router 学出来的负载长什么样。

## total、base、active 三个数 {#param-split}

`trainer_utils.get_model_params` 就是 README 里 `minimind-3-moe | 198M-A64M` 的来源。

{{source:trainer/trainer_utils.py#L18-L26}}

`total` 是全部参数，决定权重文件大小和显存占用。`base = total − expert × n_routed` 减掉每层全部 `E` 个 expert，剩下 embedding、attention、norm 和 gate——每个 token 无论走哪条路都要用到的「基座」。`active = base + expert × n_active` 就是基座加上这次前向真正走过的 `k` 个 expert。

最容易看错的是中间那个 `expert` 变量：过滤条件 `'mlp.experts.0.' in n` 在**每一层**都会命中（`model.layers.0.mlp.experts.0.…`、`model.layers.1.mlp.experts.0.…`……），所以它算的是「**一个 expert 槽位，跨全部 L 层加总**」，不是单层单个 expert；正因为已经跨层加总，后面两行才不需要再乘一次 `num_hidden_layers`。

源码里这四个量都除过 `1e6`（单位是 M），下表按原始个数代入默认配置（C=768, L=8, E=4, k=1）：

| 量 | 算法 | 默认配置下 |
|---|---|---|
| 单层单个 expert | `3·C·moe_I = 3×768×2432` | 5,603,328 |
| `expert`（0 号槽位 × L 层） | 上一行 × 8 | 44,826,624 |
| `base` | `total(198,416,640) − expert × 4` | 19,110,144 |
| `active` | `base + expert × 1` | 63,936,768 |

{{lab:param_split}}

实验表格最后一行是个漂亮的巧合：同配置的 dense 模型是 63,912,192，比 `active` 只少 24,576——正好是 8 层 gate 的参数量。因为 `moe_intermediate_size` 默认等于 `intermediate_size`，top-1 时一个 token 走过的 FFN 和 dense 一样宽，多出来的只有 router。

{{quiz:q13,q14}}

> [!KEY]
> `total` 决定存储/显存，`active = base + expert × k` 决定单 token 的计算量；`expert` 这个变量已经跨全部层加总，不要再乘 `L`。

> [!MORE] 那两个恒等于 0 的 shared expert 变量
> `get_model_params` 里还有 `n_shared = getattr(config, 'n_shared_experts', 0)` 和按 `'mlp.shared_experts.0.'` 过滤的 `shared_expert`。DeepSeek-MoE 风格的实现会留几个「共享 expert」让每个 token 无条件都过一遍，这两个变量就是为那种配置准备的。MiniMind 明确移除了这个设计，`MiniMindConfig` 根本没有 `n_shared_experts` 字段，`getattr` 取不到就回落成 0，所以 `base` / `active` 里那两项永远是 0。看这段代码时不必被它们干扰。

## 训练好的 router 的逐层负载 {#real-routing}

前面都是随机初始化的 gate——它的 `scores` 几乎均匀，路由基本等于抛硬币。换上 `out/full_sft_768_moe.pth` 这份真正训练过的权重，看看 8 层 router 各自学出了什么。

统计方法很直接：给每层的 `mlp.gate` 挂一个 forward hook 抓 logits，自己复现 `softmax` + `argmax`（k=1 时 top-1 就是 argmax），再 `bincount` 数每个 expert 接到多少 token。一段 44 token 的中英混合文本，8 层一共产生 352 次路由决策。

先说一个已知结果：8 层加总后，4 个 expert 拿到的 token 数是 **92 / 93 / 79 / 88**，相当接近均衡——`router_aux_loss_coef` 在训练全程确实起了作用。如果放任不管，router 很容易在训练早期「贫富分化」：一个 expert 见得多、训得好，下次更容易被选中，最后坍缩成伪 dense 模型。

那么逐层看呢？每一层都同样均衡，还是会有偏科的层？先下注：

{{predict:p3}}

{{lab:real_routing}}

热力图给出答案：第 0 层是 13/11/9/11，几乎完美均衡；但第 6 层是 **32/9/0/3**——expert 2 一个 token 都没拿到，expert 0 独吞了 73%。第 5 层（15/6/2/21）和第 7 层（1/18/14/11）也明显偏科。所以「全局均衡」是各层偏科方向互相抵消的结果，不代表每一层都均衡。这也正好解释了[第二章那条 `elif` 分支](#ddp-unused)为什么不是杞人忧天：在真实模型、真实文本上，某层某个 expert 完全闲置是会真实发生的。

{{quiz:q15}}

> [!KEY]
> 训练好的 router 在 8 层加总上很均衡（92/93/79/88），但单看某一层可以严重偏科（L6 是 32/9/0/3，有 expert 拿到 0 个 token）。

## 分桶调度的代价 {#moe-cost .side}

top-1 路由下，一个 token 走过的 FFN 参数和 dense 一模一样（[参数账那一节](#param-split)算过：`active` 只比 dense 多一个 router）。那 MoE 的前向应该和 dense 一样快才对——实际并不是。

瓶颈在 `for i, expert in enumerate(self.experts)` 这个 Python 循环。dense FFN 是一次 `[B*T, C] @ [C, I]` 的大矩阵乘法，BLAS 一把梭；MoE 把同样的 token 切成 `E` 份，每份单独做「布尔索引取子批 → 3 个 Linear → 乘权重 → `index_add_` 写回」。每一步都要过一遍 Python 解释器、分配临时张量、启动一批很小的 kernel。**多出来的是调度的固定成本，不是浮点运算量。**

验证这件事最干净的办法是把 `E` 调大而保持 `k=1`：激活参数和 FLOPs 一个字节都不变，变的只有循环次数。

{{lab:moe_cost}}

本机这次测到 dense 4.8ms、`E=4` 是 1.29×、`E=16` 是 2.99×——第四列「一个 token 实际过的参数」三行都是 5.6M，慢出来的完全是调度。CPU 上的计时噪声很大（同一台机器多跑几次就能差 20%），别记具体倍数，记方向。把 `T` 从 53 加到 400 再跑，倍数会掉回 1.00× / 1.34×：token 越多，每次循环的固定开销被摊得越薄。

工业实现靠 sort + 分段 matmul、Triton 自定义 kernel（Megatron-LM、DeepSpeed-MoE、vLLM 的 fused MoE）把这段开销压掉。MiniMind 这版是刻意写成最可读的形式。

{{quiz:q16}}

> [!KEY]
> top-1 MoE 的 FLOPs 和 dense 几乎相同，慢出来的是「逐 expert 分桶」的 Python 调度成本：`E` 越大越慢、token 越多摊得越薄。

> [!MORE] 和 Mixtral / DeepSeek-MoE 的三点差异
> **shared expert**：DeepSeek-MoE 留几个共享 expert，每个 token 无条件都过一遍，再叠加 top-k 个路由 expert，把「通用知识」和「专精知识」分开学。MiniMind 的 README 明确写了移除这个设计（`trainer_utils.py` 里那两个恒为 0 的变量就是它的遗迹）。
> **capacity factor / token dropping**：GShard、Switch Transformer 这类要把 expert 摊到不同设备上的实现，会给每个 expert 设容量上限（如「最多 1.25 倍平均值」），超出的 token 直接跳过 FFN 只走残差，以保证各卡计算量和通信量可预测。MiniMind 的 `token_idx` 长度完全数据依赖，来多少算多少——单进程没问题，要做 expert parallelism 就必须补上。
> **k 的取值**：Mixtral 是 8 选 2，MiniMind 默认 4 选 1。k 越大路由越平滑（输出是多个 expert 的加权和），但激活参数和计算量也跟着涨。
