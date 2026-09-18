---
day: 5
format: points
title: "FFN(SwiGLU)、Block 与完整前向 + Loss"
subtitle: "装上最后一块算子，再把一整块 [B,T,V] 压成一个能 backward 的标量"
minutes: 85
mainline: "hidden_states [B,T,C] 进 FeedForward 被撑到 [B,T,I] 再压回来，两次残差写回总线，过 lm_head 变成 logits [B,T,V]，shift 之后拍平成 [B*(T-1), V] 交给 cross_entropy"
files:
  - model/model_minimind.py#L136-L146
  - model/model_minimind.py#L178-L194
  - model/model_minimind.py#L245-L253
goals:
  - 能说出 gate_proj / up_proj / down_proj 在 FeedForward.forward 那一行里的四步 shape 变化，并算出 SwiGLU 比两矩阵 MLP 贵多少
  - 能解释 MiniMindBlock 里两处残差相加为什么一处能写 in-place、另一处写了就在 backward 报错
  - 能推导 logits_to_keep 两条分支各自切出什么，并指出它省下的到底是哪一步的算力和显存
  - 能用一个 T=6 的小例子说清 logits[t] 对 labels[t+1]，并说明 ignore_index=-100 改的是分子还是分母
  - 能用 logit lens 读出中间层的预测，并解释残差流的哪条性质让它读得出来
---

Day 2 把残差流这条宽 `C` 的总线搭好了，Day 3、Day 4 把 RoPE 和 Attention 装了上去。今天装最后一块算子——`FeedForward`（SwiGLU），合上 `MiniMindBlock` 的盖子，再顺着 `MiniMindForCausalLM.forward` 走到一个标量 `loss`。这是前向链路上最后一批全新代码：从 Day 6 起你看到的都是它的变体（MoE 换掉 FFN，DPO / PPO / GRPO 换掉 loss）。

今天还是跟 `hidden_states` 走：它进 `FeedForward` 被撑到 `[B,T,I]` 再压回 `[B,T,C]`，被两次残差写回总线，过 `lm_head` 变成 `logits [B,T,V]`，最后被 shift、拍平，塌成一个标量。

五章的分工：一拆 `FeedForward` 那一行流；二合上 Block，抠一个容易被当成风格差异的细节；三看收尾那行 `logits_to_keep`；四是全天最容易晕的地方——loss 的 shift、拍平和 `-100`；五反过来把标量拆开读模型。

{{flow: hidden_states [B,T,C] | *FeedForward* [B,T,I] | *MiniMindBlock × L* | norm | *lm_head* | logits [B,T,V] | *shift + cross_entropy* | loss}}

# FeedForward：SwiGLU 的三块矩阵

Attention 负责在位置之间搬信息，FFN 负责在每个位置内部做变换——它是模型里参数最多、也最宽的一块。这一章拆开 `FeedForward.forward` 那一行流：先看 shape 怎么从 `C` 撑到 `I` 再压回来，再看 `gate` 和 `up` 两条支路为什么不对称，最后算一笔账——多出来的那个矩阵到底贵在哪。

## 一行流里的四步 shape {#ffn-oneliner}

`FeedForward.__init__` 只有三个 `nn.Linear`，全部 `bias=False`（和仓库里其他投影层一样）：`gate_proj` 和 `up_proj` 都是 `C → I`，`down_proj` 是 `I → C`。注意这三行的参数顺序——`down_proj` 的 `in_features` 才是 `intermediate_size`，和另外两个正好反过来，这是抄这段代码时最常写错的地方。构造函数还留了一个 `intermediate_size` 参数覆盖 config（`intermediate_size or config.intermediate_size`）：稠密路径从不用它，Day 6 的 MoE 专家才会传。

{{source:model/model_minimind.py#L140-L146}}

`forward` 只有一行，但它是**两条并联的支路**，不是串联：`gate_proj(x)` 和 `up_proj(x)` 吃的是同一个 `x`，各自把 `[B,T,C]` 投到 `[B,T,I]`，其中只有 gate 那一支过 `act_fn`，然后两者**逐元素相乘**，最后由 `down_proj` 压回 `[B,T,C]`。整行的求值顺序正好是四步：`gate_proj(x)` → `act_fn(...)` → `up_proj(x)` → 相乘 → `down_proj(...)`。

默认配置下 `I = 2432`，是 `C = 768` 的 3.17 倍。这是整个模型里最宽的地方，也是 `hidden_states` 唯一一次离开总线宽度——出了 `down_proj` 立刻回到 `[B,T,C]`，否则下一步的残差相加就加不上了。下面的实验用 `expand=True` 把这一行拆成子表达式，逐个标 shape。

{{lab:ffn_oneliner}}

{{quiz:q1}}

> [!KEY]
> `down(act(gate(x)) * up(x))`：两条并联支路把 `[B,T,C]` 各自撑到 `[B,T,I]`，逐元素相乘后由 `down_proj` 压回 `[B,T,C]`。

## SiLU 的软门控：gate 和 up 的分工 {#ffn-gate}

`self.act_fn = ACT2FN[config.hidden_act]`，而 `hidden_act` 默认就是 `'silu'`（`model/model_minimind.py:25`）。`SiLU(x) = x · sigmoid(x)`：正半轴几乎等于 `x` 本身，负半轴被压向 0，但**压不到 0**——它有一个全局下界 `-0.2785`（在 `x ≈ -1.2785` 处取到），而且除了 `x = 0` 这一点，输出永远不会恰好等于 0。

{{source:model/model_minimind.py#L143}}

这正是它和 ReLU 的关键差别。ReLU 把负半轴硬切成 0，被切掉的通道梯度也是 0，容易出现「死神经元」；SiLU 处处可导、处处有梯度，代价是没有真正的稀疏——所以叫「软门控」。

两条支路的分工也不对称：只有 gate 过激活，up 是纯线性。相乘之后，gate 那一路的每个数就是对应通道的「开度」。有意思的是这个开度**可以是负的**——负门会把 up 那一路的符号整个翻过来，这是 ReLU 版 GLU 做不到的事。所以别把 SwiGLU 想成「开关」，它更像一个可学习的、带符号的逐通道增益。

实验拿真实 SFT 权重跑一句话，把两条支路的分布摆在一起对照。盯住「恰好 = 0」和「最小值」这两列——它们是「软」和「硬」的分界线。

{{lab:ffn_gate}}

{{quiz:q2}}

> [!KEY]
> 只有 gate 支路过 SiLU：它有下界 `-0.2785`、从不恰好等于 0，所以是软门控；门还可以为负，能翻转 up 支路的符号。

## 三个矩阵的参数代价 {#ffn-params}

三个矩阵每个都是 `C × I`，一层 FFN 就是 `3·C·I = 3 × 768 × 2432 = 5,603,328` 个参数，是同层 attention 四个投影（`1,769,472`）的 3.17 倍。Day 2 数过全模型的账：FFN 占 70%，attention 22%，embedding 7.7%。

{{source:model/model_minimind.py#L140-L142}}

「SwiGLU 要三个矩阵」这件事本身就是代价。拿最经典的两矩阵 MLP 当基准——`up: C → 4C` 加 `down: 4C → C`，合计 `8C² = 4,718,592`——MiniMind 的 FFN 比它多 **18.75%**。想让两者严格相等也不难：解 `3·C·I = 8C²` 得到 `I = 8C/3`，`C=768` 时正好是 2048。这就是 LLaMA 系那个 `8/3` 系数的来历，它本来就是一道「换成 SwiGLU 但参数量不变」的换算题。MiniMind 把系数换成了 `π`，多出来的容量就是 `π / (8/3) ≈ 1.178` 这个比值（Day 2 的 `intermediate_size` 一节推过这条公式）。

一句话：SwiGLU 的门控不是白来的表达力，是拿参数换的；而「换多少」由 `intermediate_size` 一个数字说了算。实验最后一行就是这个比值，把 `intermediate_size` 改成 2048 再跑一次，看它落到哪。

{{lab:ffn_params}}

{{quiz:q3}}

> [!KEY]
> 一层 FFN = `3·C·I`，比同宽的两矩阵 4C MLP 多 18.75%；把 `I` 设成 `8C/3 = 2048` 时两者正好相等，这就是 LLaMA 那个 8/3 系数的来历。

> [!MORE] 这个类在 Day 6 会被复用
> `MOEFeedForward.__init__` 里写的是 `nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])`（`model_minimind.py:153`）：同一个 `FeedForward` 类被原样实例化 `num_experts` 次，每个专家都是独立权重的「迷你 FFN」，只是中间宽度换成了 `moe_intermediate_size`（默认等于 `intermediate_size`）。今天把 `FeedForward.forward` 吃透，Day 6 只剩「怎么选专家、怎么加权」这一件新事。

# MiniMindBlock：两次 pre-norm 残差

FFN 和 Attention 都是「读一份、算一点、加回去」，真正把它们接到残差流上的是 `MiniMindBlock.forward` 的那五行。这一章先逐行走一遍，看清 pre-norm 的节奏；再抠一个很容易被当成风格差异的细节——为什么两次残差相加，一次写 `+=`、一次写 `=`。

## 一个 Block 的两组三拍 {#block-prenorm}

`MiniMindBlock.forward` 的函数体只有五行，节奏是两组「存一份 → 归一化一份 → 加回去」。

{{source:model/model_minimind.py#L186-L194}}

第一组：`residual = hidden_states` 先把原值存下来（L187），把 `input_layernorm(hidden_states)` 这份**归一化后的副本**喂给 `self_attn`，算出来的结果在 L192 加回 `residual`。第二组写得更紧凑，没有显式的 `residual` 变量：L193 右边那个 `hidden_states` 同时充当两个角色——既是 `post_attention_layernorm` 的输入，也是被加回去的那个残差。

这就是 pre-norm：`x + sublayer(norm(x))`，而不是 post-norm 的 `norm(x + sublayer(x))`。关键在于**总线本身从来没被归一化过**，被归一化的只是喂给子层的副本，所以残差流的范数会随深度一路增长（Day 2 用 pretrain 权重实测过 1.3 → 36）。

另外两个细节：`position_embeddings` 是 `MiniMindModel.forward` 算好后原样传进来的，L 层共用同一对 `cos/sin`；`present_key_value` 也只是把 `self_attn` 返回的 `(xk, xv)` 原样往上传，Block 自己不产生它（Day 7 讲 KV cache）。整个函数进出都是 `[B,T,C]`——这正是「任何子层都必须 `[B,T,C]` 进、`[B,T,C]` 出」这条约束的来源。

{{lab:block_prenorm}}

{{quiz:q4}}

> [!KEY]
> 两组 pre-norm 残差：归一化的是喂给子层的副本，加回去的是没归一化的原值，总线本身全程 `[B,T,C]` 且从未被重新缩放。

## 一个 `+=` 和一个 `=` {#block-inplace}

对照源码会发现两次残差相加的写法不一样：L192 是 `hidden_states += residual`，L193 是 `hidden_states = hidden_states + self.mlp(...)`。前者是**原地加**，在原来那块内存上累加，不新分配 `[B,T,C]`；后者会产生一个全新的张量。上一节的逐行追踪里，网页会给 L192 那一行标一个 `in-place` 小旗子，L193 则只是普通赋值。

{{source:model/model_minimind.py#L192-L193}}

in-place 对 autograd 是有风险的：反向传播时，有些算子会把自己的**输入张量**保存下来，因为算梯度要用到输入的值。PyTorch 给每个张量维护一个版本号 `_version`，每次原地修改就加一；`backward()` 时如果发现某个被保存的张量版本号变了，就直接抛 `RuntimeError`，而不是默默算出错误的梯度。

所以问题很具体：这两行里，哪一行的 `hidden_states` 是「已经被某个算子存起来准备反向」的那一个？实验把三种写法各跑一次 `loss.backward()`。先下注，再看结果。

{{predict:p1}}

{{lab:block_inplace}}

只有第三行报错。原因在 L193 的右边：`post_attention_layernorm(hidden_states)` 是 RMSNorm，它的反向要用**输入本身**算梯度（`model_minimind.py:56-60`），于是这个 `hidden_states` 被 autograd 存了下来；紧接着在同一个对象上 `+=`，存起来的那份就被就地改掉了。而 L192 改的是 `self_attn` 刚返回的全新张量，此刻还没人存过它，所以原地加安全。

{{quiz:q5}}

> [!KEY]
> in-place 只有在「这个张量还没被任何算子存起来做反向」时才安全：L192 满足，L193 因为 `post_attention_layernorm` 存了输入而不满足。

> [!MORE] 怎么确认「是 RMSNorm 存了它」
> 实验的第一条 task：把 `normed = self.post_attention_layernorm(hidden_states)` 换成 `normed = hidden_states.clone()`，其余不动，第三行的 `RuntimeError` 就消失了——因为 `clone()` 的反向只需要梯度本身，不需要保存输入。第二条 task 则说明报错只发生在 `backward()`：光跑 forward（不传 `labels`、不 `backward`）三种写法都正常，`_version` 检查是在反向时才做的。

# MiniMindForCausalLM.forward：总线的终点

`MiniMindModel` 交出来的还是 `[B,T,C]`。`MiniMindForCausalLM.forward` 只做两件事：过 `lm_head` 变成 `logits`，以及在给了 `labels` 时算一个标量 loss。这一章先看前半段——`logits_to_keep` 那一行三元表达式在切什么、为什么值得单独写一行，以及它真正省下来的是哪一步。

## logits_to_keep 的两条分支 {#head-slice}

`forward` 的第一行把活全交给 `self.model`，拿回 `hidden_states [B,T,C]`。接下来 L247 这一行是重点，它按传进来的类型分两条路：

{{source:model/model_minimind.py#L245-L248}}

- **是 `int`**：包成一个切片对象 `slice(-k, None)`，语义是「最后 k 个位置」。`hidden_states[:, slice(-k, None), :]` 就是普通的连续切片。
- **不是 `int`**（一般是一个 `torch.Tensor` 下标）：`slice_indices` 就是那个张量本身，`hidden_states[:, 下标张量, :]` 变成**花式索引**，能抠出任意几个不连续的位置——Day 11~13 那些只对部分位置算 log-prob 的训练代码会用到这一支。

这不是 MiniMind 自己发明的写法：`transformers` 里 Llama4、Dots1、Olmo3 等模型的 `forward` 写的是一模一样的一行。它的用意是让同一份 `forward` 既能给训练用（要完整的 `[B,T,V]` 去和 labels 对齐），也能给 `generate()` 用（只要最后一步的分布去采样）。认出这个模式，以后读别的开源模型会快很多。

那么默认值 `logits_to_keep=0` 时，这一行切出来的是几个位置？先下注，再看实验表格的第一行。

{{predict:p2}}

{{lab:head_slice}}

答案在表格第一行：`slice(-0, None)` 打印出来是 `slice(0, None, None)`——Python 里 `-0 == 0`，负号在整数 0 上不起作用，于是「最后 0 个」变成了「从第 0 个到最后」，也就是**全部 T 个位置**。这是这一行最容易看错的地方。最后一行的 `tensor([0, 2, 4])` 走的是另一支：`isinstance` 为 `False`，直接当下标用，抠出三个不连续的位置。

{{quiz:q6}}

> [!KEY]
> `logits_to_keep` 是 `int` 走 `slice(-k, None)`、是张量走花式索引；默认的 `0` 因为 `-0 == 0` 而等于「全部位置」，不是空张量。

## 省下来的只有 lm_head 那一步 {#head-cost}

既然 `logits_to_keep=1` 只让 `lm_head` 看一个位置，是不是能省掉一大块算力？要回答这个，得先把 forward 分成两段。

{{source:model/model_minimind.py#L248}}

第一段是 `self.model(...)`：L 层 attention + FFN，必须把**全部 T 个位置**的 `hidden_states` 都算出来——attention 里后面的位置要读前面的 key/value，少算一个都不行。`logits_to_keep` 在这一段跑完之后才出场，完全动不了它。

第二段才是 `lm_head(hidden_states[:, slice_indices, :])`：一个 `[B, n, C] × [C, V]` 的矩阵乘，`n` 是切出来的位置数。FLOPs 正比于 `n`，输出 `[B, n, V]` 的显存也正比于 `n`。`V = 6400`、`T = 300` 时，全量 logits 是 21.97 MB，只留最后一个是 0.07 MB。

实验把两段分别计时，看清这个比例很重要：`lm_head` 在这里只占总耗时的百分之几，所以别指望 `logits_to_keep=1` 让 forward 快 `T` 倍。它真正值钱的场景是 **prefill**——把一整段长 prompt 一次性送进去、却只要最后一个位置的分布，那一步 `[B, T, V]` 既费算力又费显存。`V` 越大、`T` 越长，这一刀的收益越明显。

{{lab:head_cost}}

{{quiz:q7}}

> [!KEY]
> `logits_to_keep` 只改变喂进 `lm_head` 的位置数，transformer 主体照算全部 T 个位置；收益集中在长 prompt 的 prefill 那一步。

> [!MORE] `generate()` 自己不传，但训练侧有人真的用它
> `generate()` 走的是 `outputs.logits[:, -1, :]`（`model_minimind.py:268`），也就是把 `[B,T,V]` 全物化出来再取最后一行，没有用上 `logits_to_keep`。但全仓 grep 会发现 `trainer/rollout_engine.py` 的 `compute_per_token_logps`（第 29 行）确实传了 `logits_to_keep=n_keep + 1`——这是 Day 12 GRPO/PPO rollout 阶段算每个 token log-prob 的地方，只要最后 `n_keep` 个位置的 logits，正好是这一节说的「省下 lm_head 那一刀」的真实应用场景。
> 不过实际损失比想象中小：`generate()` 配合 KV cache 时，从第二步起送进 forward 的是 `input_ids[:, past_len:]`，只有 1 个 token，此时 `T=1`，传不传 `logits_to_keep` 切出来的都是同一个位置。真正浪费的只有 prefill 那一步。

# 从 logits 到一个标量 loss

到这里 `logits` 还是 `[B,T,V]` 的一大块数，而训练需要的是一个能 `backward()` 的标量。`forward` 的最后三行做了三件事：shift、拍平、按 `ignore_index` 求平均。这一章一件一件拆——这是全天最容易晕的地方，所以每一节都配一个很小的例子。

## shift：logits[t] 对 labels[t+1] {#loss-shift}

`x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()`——一行里做了两次切片，方向相反。

{{source:model/model_minimind.py#L250-L252}}

理由是因果语言模型的定义：位置 `t` 的 `hidden_states` 是模型**看过 `input_ids[0..t]` 之后**的状态，所以 `logits[t]` 表达的是「下一个 token 是什么」，它的监督信号只能是 `labels[t+1]`。于是 `logits` 去掉最后一个位置（它的「下一个」不在这段序列里），`labels` 去掉第一个位置（没有任何 logits 负责预测它）。`T` 个位置最终只产生 `T-1` 项 loss，这个「少一项」是很多手算对不上的根源。

**这个 shift 是在模型内部做的**，Dataset 那边不需要错位：`PretrainDataset.__getitem__` 直接 `labels = input_ids.clone()`，再把 padding 位置设成 `-100`（`dataset/lm_dataset.py`）。Day 8 看各种 Dataset 时这条规则还会反复用到。

实验拿真实 pretrain 权重跑一句 6 个 token 的话，把每一格摆出来。盯住「`logits[t]` 的 top-1」和「监督目标 `labels[t+1]`」两列：`t=3` 的输入是 `'喜欢'`，模型猜的是 `'吃'`，而 `'吃'` 正是 `labels[4]`——模型确实在预测**下一个**，而不是复述当前这个。

{{lab:loss_shift}}

{{quiz:q8}}

> [!KEY]
> `logits[t]` 对 `labels[t+1]`，`T` 个位置只产生 `T-1` 项 loss；shift 在模型内部完成，Dataset 给的 `labels` 和 `input_ids` 是逐位对齐的。

## contiguous 与拍平成 [B*(T-1), V] {#loss-flatten}

`F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ...)`：两个 `.view(-1, ...)` 把 `[B, T-1, V]` 和 `[B, T-1]` 拍成 `[B·(T-1), V]` 和 `[B·(T-1)]`。为什么要拍？因为 `cross_entropy` 的接口就是「N 个独立样本，每个样本 C 个类别的分数」，它不关心这 N 个样本原本是怎么按 batch 和序列排的。拍平之后，每个位置都变成一道独立的 `V = 6400` 类分类题。

{{source:model/model_minimind.py#L251-L252}}

上一节那两个 `.contiguous()` 就是为这一步准备的，而且删不得。`logits[..., :-1, :]` 是在**中间那一维**（`T`）上切片：数据没搬家，只是把每条序列的最后一个位置留在原地不看，于是相邻两条序列的数据之间留下了空洞，`is_contiguous()` 变成 `False`。而 `view` 的语义是「按行优先顺序重新解释同一块内存」，遇到空洞就只能报 `view size is not compatible with input tensor's size and stride`。两种修法：像源码这样先 `.contiguous()` 拷一份连续内存，或者把 `view` 换成 `reshape`（它需要时会自己拷贝）。`labels[..., 1:]` 同理，所以源码给它也加了一个 `.contiguous()`。

实验把这几种写法并排跑一遍。注意最上面一行：没切片的 `logits` 本身是连续的，`view` 毫无问题——问题永远是切片带来的。

{{lab:loss_flatten}}

{{quiz:q9}}

> [!KEY]
> 在中间维上切片会让张量不连续，`view` 随即报 stride 错；源码用 `.contiguous()` 拷一份，等价写法是换成 `.reshape`。

## ignore_index=-100 改的是分母 {#loss-ignore}

`ignore_index=-100` 是 `cross_entropy` 的参数，语义是：**label 等于这个值的位置，当它不存在**。注意这和「贡献一个 0」完全是两回事——后者会把这些位置算进分母，把平均值稀释掉；`ignore_index` 是同时从分子（loss 累加）和分母（有效位置数）里把它们剔除。

{{source:model/model_minimind.py#L252}}

`-100` 这个数字是 PyTorch 的默认值，所以所有数据集只要把不该算 loss 的位置填成 `-100` 就能直接对上。MiniMind 用到它的地方有两类：`PretrainDataset` 把 padding 位置设成 `-100`；SFT / DPO 的 Dataset 还会把 prompt 段整段设成 `-100`，让 loss 只落在模型该学的那段回答上（Day 8 细讲）。顺带一提，`labels` 里的 `-100` 和 `ignore_index` 必须对得上：把 `ignore_index` 改成 `-1` 而 labels 里仍是 `-100`，会直接抛 `IndexError: Target -100 is out of bounds`。

既然分母是「有效位置数」，那把一整个 batch 的 labels 全填成 `-100`、让分母变成 0，会发生什么？这在真实训练里不是假想：一条全是 padding 的样本，或者一段被 mask 光的对话，就能让某一步的分母掉到 0。先下注，再跑实验。

{{predict:p3}}

{{lab:loss_ignore}}

表格最后一行：loss 是 **`nan`**，不是 0，也不抛异常。`mean` reduction 算的是 `sum / count`，`0 / 0` 在浮点里就是 `nan`，它会顺着 `backward()` 把整个模型的梯度污染掉——所以「整条样本被 mask 光」是必须在 Dataset 侧挡掉的坑。中间两行则说明分母确实变了：mask 掉 6 个位置之后 loss 仍在 8.7 附近（在剩下 12 个位置上平均），而不是被稀释成最后一列那个 5.86。

{{quiz:q10}}

> [!KEY]
> `ignore_index` 把这些位置从分子和分母里同时剔除；一旦有效位置数为 0，loss 就是 `nan` 而不是 0，且不报错。

# 把标量拆开看：逐 token loss 与 logit lens

loss 是个标量，最后一层的 logits 是一大块数——这两样都可以反过来拆开，用来「看」模型在想什么。这一章两节：先关掉 `reduction`，看同一句话里每个 token 各自贡献了多少 loss；再把最终的 `norm + lm_head` 提前套到中间层的 `hidden_states` 上，看预测是怎么一层层收敛的。

## reduction='none'：逐 token 的惊讶度 {#lens-per-token}

`F.cross_entropy(..., reduction='none')` 会跳过最后那次平均，直接返回 `[B·(T-1)]` 的逐位置 loss。数值上它就是 `-log p(正确 token)`：完全猜不到时等于 `ln(V) = ln(6400) ≈ 8.76`，猜得十拿九稳时接近 0。所以这一串数可以直接读成模型对每个 token 的「惊讶度」。

对齐关系还是上一章那条：`per[t]` 是「看过 `toks[0..t]` 之后预测 `toks[t+1]` 有多意外」，所以画彩带时要把它标在 `toks[1:]` 上，标错一位整张图就全偏了。

实验用 pretrain 权重跑一句大白话，结果很符合直觉：**词的第一个 token 贵，后面的几乎免费**——`'聊'` 是 4.51，紧跟的 `'天'` 只有 0.0007；`'跑'` 2.93，`'步'` 0.08。全句最贵的是开头的 `'今天'`（7.04），因为它前面只有一个 BOS，没有任何上文可依。这也解释了预训练 loss 的构成：大量的下降来自「把词内部补全学会」，真正难的是每个词的第一个 token。

{{lab:lens_per_token}}

{{quiz:q11}}

> [!KEY]
> `reduction='none'` 给出逐位置的 `-log p`，上限是 `ln(V) ≈ 8.76`；它标在 `toks[1:]` 上，词内部的后续 token 几乎零 loss。

## logit lens：中间层也能被 lm_head 读懂 {#lens-layers}

`MiniMindModel.forward` 只把**最后一层**的 `hidden_states` 交给 `self.norm`，再由 `lm_head` 变成 logits。但每一层的 `hidden_states` 都活在同一条残差流里，shape 完全一样——那把中间层的也拿去过一遍 `norm + lm_head`，能读出什么？这个技巧叫 **logit lens**。

{{source:model/model_minimind.py#L230}}

它能读出东西，靠的正是 Day 2 那条性质：每层都是 `hidden_states = hidden_states + 子层输出`，**只加不替换**，从不整体改写总线，所以中间层和最终层活在同一个坐标系里，`lm_head` 那份 `[V, C]` 权重对它们都部分「认识」。

实验对 `'小猫喜欢吃鱼，小狗喜欢吃'` 逐层解码最后一个位置，读出来的故事很清楚：layer0 的 top-1 是 `'吃'`——还在复读当前 token；layer1~6 一直在猜泛泛的词（`'过'`、`'各种'`、`'一些'`）；直到 layer7 才突然锁定 `'鱼' '肉' '水果'`，和模型真正的输出完全一致（这不是巧合——最后一行本来就是 `forward` 干的事）。顺便留意 `‖hidden_states‖` 那一列：7.2 → 52.0，Pre-Norm 的范数增长在这里又露了一次面。

{{lab:lens_layers}}

{{quiz:q12}}

> [!KEY]
> logit lens 靠的是残差流「只加不替换」：中间层和最终层共用同一个坐标系，所以同一份 `lm_head` 权重能提前把它们解读出来。

> [!MORE] 为什么要套上 `model.norm` 才好读
> 把 `mm.norm(h)` 换成 `h` 直接喂 `lm_head`（实验的第一条 task），top-1 大致还是那几个词，但 top-1 概率会塌成 0.1% ~ 5%（最后一层仍有 23.8%）。原因是浅层 `hidden_states` 的范数还很小（layer0 只有 7.2），乘出来的 logits 也就很小，softmax 之后接近均匀分布。
> `model.norm` 先把每个 token 的 RMS 拉到 1、再乘上学到的逐通道缩放，等于把各层拉到同一个尺度上——logit lens 里这一步不是可有可无的装饰，而是让不同深度的层「可比」的前提。
