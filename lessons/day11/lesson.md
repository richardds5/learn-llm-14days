---
day: 11
format: points
title: "DPO：把偏好对压成一个标量 loss"
subtitle: "一对 (chosen, rejected) → 拼成 [2B,T] → 两次前向 → 一个数"
minutes: 85
mainline: "一对 (chosen, rejected) 怎么一路塌成一个标量：cat 成 [2B,T] → policy/ref 两次前向 → logits [2B,T,V] → gather 成 [2B,T] → mask 求和成 [2B] → 切两半作差得到 margin [B] → logsigmoid → 标量"
files:
  - trainer/train_dpo.py#L25-L50
  - trainer/train_dpo.py#L57-L87
  - dataset/lm_dataset.py#L148-L174
goals:
  - 能背出降维链上每一步的 shape：[2B,T,V] → [2B,T] → [2B] → 两个 [B] → margin [B] → 标量，并说出每一步是哪一行代码
  - 能解释 DPO 的 y 为什么不能用 -100、mask 为什么必须是独立的 0/1 张量
  - 能说清 `[:batch_size // 2]` 这个切分约定唯一依赖什么，写反了会出现什么（以及为什么看 loss 发现不了）
  - 能推出「policy == ref ⇒ loss ≡ ln 2」，并算出此时每个 mask=1 的 token 拿到的梯度是 ∓β/(2n)
  - 能说出 ref_model 的三道冻结各自防什么、默认 lr 为什么只有 4e-8、用 sum 聚合带来的长度偏置从哪来
---

前十天的 loss 都长一个样：一个序列、一组 label、一次 `F.cross_entropy`。今天换成一**对**序列——同一个 prompt 下一个被选中（chosen）、一个被拒绝（rejected），模型要学的不是「把 chosen 背下来」，而是「给 chosen 比 rejected 更高的概率」。

DPO 的全部新代码只有 `trainer/train_dpo.py` 开头两个纯函数，不到 30 行；训练循环的其余部分和 Day 9 几乎逐字相同。所以今天不跟着某一层网络走，而是跟着**一对样本**走：它怎么被 Dataset 变成六个张量、被 `torch.cat` 摊成 `[2B, T]`、经过 policy 和 ref **两次**前向变成 `[2B, T, V]` 的 logits，再一步步塌成一个能 `.backward()` 的标量。

四章就是这条链的四段：数据侧（六个张量、0/1 mask、`cat` 定下的约定）→ 两步降维（gather、mask 求和）→ `dpo_loss` 那七行本身 → 回到训练循环（两个模型、小到离谱的 lr、求和聚合留下的长度偏置）。

{{flow: 一对 (chosen, rejected) | *cat* | x [2B,T] | *policy / ref 两次前向* | logits [2B,T,V] | *gather* | logp [2B,T] | *mask·sum* | [2B] | *切两半 → margin* | loss 标量}}

# 偏好对的数据形状：从一对样本到 [2B, T]

`DPODataset` 已经在 Day 8 出现过，这一章只挑今天必须用到的三件事：它返回的到底是什么、mask 为什么是独立的 0/1 张量、以及 `train_epoch` 开头那三行 `torch.cat` 为什么决定了后面所有代码的正确性。

## DPODataset 返回的六个张量 {#pair-tensors}

`SFTDataset.__getitem__` 返回两个张量（`input_ids`、`labels`），`DPODataset` 返回**六个**：chosen 和 rejected 各有一套 `x / y / mask`。

{{source:dataset/lm_dataset.py#L160-L165}}

两件事在这六行里同时发生了。

**第一，错位（shift）在 Dataset 里就做完了。** `x = input_ids[:-1]`、`y = input_ids[1:]`、`mask = loss_mask[1:]`。也就是说 `y[t]` 就是「模型看完 `x[0..t]` 之后应该吐出来的那个 token」，训练脚本拿到手可以直接用。

为什么 SFT 不用手动 shift？因为 SFT 调的是 `MiniMindForCausalLM.forward(labels=...)`，错位写在模型里——`logits[..., :-1, :]` 对 `labels[..., 1:]`（`model/model_minimind.py:251`）。而 `train_dpo.py` 调的是 `model(x)`，**不传 `labels`**，它只要 `outputs.logits` 然后自己算 log prob。模型这条路没走，错位就得 Dataset 自己来。

**第二，长度被 `padding='max_length'` 钉死了。** `dataset/lm_dataset.py:148` 起那两段对 chosen 和 rejected **各自独立**调用 tokenizer 并 padding 到 `max_length`，所以六个张量的 `T` 完全一样（`max_length` 经过 `[:-1]` / `[1:]` 之后是 `max_length - 1`），但**有效内容的长度可以差很多**——这是第 4 章长度偏置的伏笔。

实验用真实的 `dataset/dpo.jsonl` 取 3 对样本，`max_length=256`。盯住两处：六个张量的 shape 是不是都一样，以及最后那行 `x[:, 1:] == y[:, :-1]` 的验证。

{{lab:pair_tensors}}

{{quiz:q1,q2}}

> [!KEY]
> `DPODataset` 一次返回 chosen / rejected 各一套 `x / y / mask`，并且已经错开一位——因为 DPO 只取 `outputs.logits`、不传 `labels`，模型内部那次 shift 走不到。

> [!MORE] 同一个 index 取两次可能不一样
> `__getitem__` 里的 `post_processing_chat` 会以 80% 的概率删掉 `<think>\n\n</think>\n\n`，这是一次 `random.random()` 调用。所以**同一个 index 取两次，mask 的长度可能差 6 个 token**。今天所有碰真实数据的实验开头都写了 `random.seed(0)`，否则数字不可复现。

## 框住 assistant 段的 0/1 mask {#loss-mask}

`mask` 不是 padding mask，而是 **loss mask**：标记「这个位置算不算进序列的 log prob」。它由 `generate_loss_mask` 生成，逻辑是一个朴素的扫描。

{{source:dataset/lm_dataset.py#L176-L192}}

`self.bos_id` 是 `<|im_start|>assistant\n` 的 token 序列、`self.eos_id` 是 `<|im_end|>\n`（`dataset/lm_dataset.py:128`）。扫描器一旦匹配上 `bos_id`，就把它**后面**到 `eos_id`（含 eos）之间的位置全部置 1，然后跳到下一段继续找。于是 prompt、chat 模板标记、以及尾部所有 padding 一律是 0，只有助手真正说的话是 1。`mask.sum(1)` 因此就等于「这条回复有多少个 token 参与 loss」。

真正要记住的是：**这个 mask 是一个独立的 0/1 `int64` 张量，而不是塞进 `y` 里的 `-100`。** SFT 可以用 `-100`，因为它的 `y` 只会被 `F.cross_entropy(..., ignore_index=-100)` 消费，负数是约定好的哨兵值。DPO 的 `y` 要被送进 `torch.gather` 当**索引**，索引必须落在 `[0, V)` 里，一个 `-100` 就会当场炸：`RuntimeError: index -100 is out of bounds for dimension 2 with size 6400`。

所以两条信息只能分两路走：`y` 走 gather 取值，`mask` 走乘法做加权。实验把第 0 条样本 mask 翻转处的那一小段画成彩带，你能看到 1 是从 `<|im_start|>assistant\n` 之后紧接着开始的；最后一行则故意把 `-100` 塞进 `y`，把那条报错原样打出来。

{{lab:loss_mask}}

{{quiz:q3}}

> [!KEY]
> `mask` 是独立的 0/1 张量而不是 `-100`：DPO 的 `y` 要进 `torch.gather` 当索引，任何负数都会直接 `index out of bounds`。

## torch.cat 定下的 chosen/rejected 约定 {#cat-2b}

`train_epoch` 的开头把六个张量还原成三个：

{{source:trainer/train_dpo.py#L59-L67}}

`x_chosen` 和 `x_rejected` 各是 `[B, T]`，`torch.cat(..., dim=0)` 之后是 `[2B, T]`——**chosen 在上半边，rejected 在下半边**。`y` 和 `mask` 同样处理，三者行序严格对齐。

这么写图的是「一次前向」：chosen 和 rejected 走**同一次** `model(x)`，共享同一份 dropout 随机性、同一次 autocast、同一次 kernel launch，省掉一半调度开销，也避免两次前向之间出现任何不对称。代价是激活显存直接翻倍——同样的 `--batch_size`，DPO 实际进模型的行数是 SFT 的两倍。

关键在于这个顺序不是随手写的。`dpo_loss` 里用 `[:batch_size // 2]` 当 chosen、`[batch_size // 2:]` 当 rejected（`trainer/train_dpo.py:41-44`），而这个约定的**唯一依据**就是这里 `cat` 的参数顺序。两处相隔几十行、没有任何断言把它们绑在一起，谁改了顺序而没改另一边，代码照跑不误、loss 看起来也正常，但学到的东西整个反过来。顺带一提：`B` 本身不要求是偶数，`cat` 之后的行数是 `2B`，所以 `// 2` 永远切得整齐。

实验把 6 行逐行对回它的来源，并数出每行的有效 token 数——注意 chosen 和 rejected 的长度差有多大。

{{lab:cat_2b}}

{{quiz:q4}}

> [!KEY]
> `torch.cat([x_chosen, x_rejected], dim=0)` 把一对样本摊成 `[2B, T]`，上半边 chosen、下半边 rejected；`dpo_loss` 里 `// 2` 的切分完全依赖这个顺序。

# 两次前向：从 logits 到序列级 log prob

`[2B, T]` 的 `x` 会被 policy 和 ref 各前向一次，得到两份 `[2B, T, V]` 的 logits。这一章把它们压成 `dpo_loss` 真正要的东西：每条序列一个数。两步走——先 `[2B,T,V] → [2B,T]`，再 `[2B,T] → [2B]`。

## log_softmax 与 gather 的两步降维 {#logprob-gather}

`logits_to_log_probs` 是今天的第一个纯函数，三行，每一行都值得停一下。

{{source:trainer/train_dpo.py#L25-L31}}

**`F.log_softmax(logits, dim=2)`**：`dim=2` 是 vocab 维。结果和 `logits` 一样大，仍然是 `[2B, T, V]`——它把每个位置上 6400 个 token 的分数变成了合法的对数概率分布。注意这里不是 `softmax` 再取 `log`：`log_softmax` 内部做了 log-sum-exp 稳定化，数值上安全得多。

**`torch.gather(log_probs, dim=2, index=labels.unsqueeze(2))`**：我们只关心每个位置上**实际那个** `y_t` 的对数概率，所以沿 vocab 维按 `y` 取值。`gather` 有一条硬性要求——`index` 和 `input` 的**维数必须相同**（只有被 gather 的那一维长度可以不同），它不做 broadcast。`labels` 是 `[2B, T]` 的二维，所以必须 `unsqueeze(2)` 成 `[2B, T, 1]`；忘了就是 `RuntimeError: Index tensor must have the same number of dimensions as input tensor`。gather 的结果是 `[2B, T, 1]`，`.squeeze(-1)` 收回 `[2B, T]`。

返回的每个元素就是 $\log p_\theta(y_t \mid x_{\le t})$，**还没有乘 mask**——prompt 和 padding 位置也都有值，而且是有限值（padding 位置模型照样给出了一个概率）。

实验用 `expand=True` 把 gather 那一行拆开，你能亲眼看到中间那个 `[N, T, 1]`：只看逐行 trace 是看不到它的。

{{lab:logprob_gather}}

{{quiz:q5}}

> [!KEY]
> `log_softmax`（不改 shape，`[2B,T,V]`）→ `gather`（沿 vocab 取出 `y_t` 那一格，`[2B,T,1]`）→ `squeeze`（`[2B,T]`）；`unsqueeze(2)` 是 gather 对维数的硬性要求。

## mask 加权求和得到的序列级 log prob {#seq-logprob}

`dpo_loss` 的头两行就是第二步降维，也是 `mask` 在整个 DPO 里**唯一**出场的地方。

{{source:trainer/train_dpo.py#L36-L37}}

`log_probs` 是 `[2B, T]` 的 float32，`mask` 是 `[2B, T]` 的 int64。相乘时 PyTorch 会自动做类型提升（float × int64 → float），所以源码不用显式写 `.float()`。乘完之后 mask=0 的位置被清成 0，`sum(dim=1)` 沿 `T` 一收，就得到 `[2B]`——每条序列一个数，也就是 $\log \pi(y \mid x) = \sum_t \text{mask}_t \cdot \log p(y_t \mid x_{<t})$。从这一行往后，函数里再也不出现 `T` 这个维度，全是一维运算。

注意它是 **`sum` 而不是 `mean`**。这和 DPO 原论文一致：$\pi_\theta(y\mid x)$ 是整条序列的联合概率，取 log 本来就是逐 token 求和。但代价是：**序列级 log prob 的绝对值正比于有效长度**。实验里 26 个 token 的那条 chosen 是 −52.56，95 个 token 的那条是 −147.30，平均到每个 token 反而是前者更低（−2.02 vs −1.55）。这个「长的天然更负」的性质会在第 4 章变成一个真实的偏置。

另外，不乘 mask 会怎样？实验的第一条任务让你把 `* mask` 去掉——序列 log prob 会从 −50 量级掉到 −3000 量级，因为 200 多个 padding 位置的 log prob 全被算了进去。

{{lab:seq_logprob}}

{{quiz:q6}}

> [!KEY]
> `(log_probs * mask).sum(dim=1)` 是 mask 唯一出场的地方，把 `[2B,T]` 收成 `[2B]`；用的是求和不是平均，所以序列 log prob 的绝对值正比于有效长度。

## [2B, T, V] 那一份的显存代价 {#logprob-memory .side}

DPO 的显存峰值不在模型参数上，而在 `log_softmax` 的输出上。看 `train_epoch` 里真正跑前向的这一段：

{{source:trainer/train_dpo.py#L74-L81}}

一次迭代里至少同时存在三份 `[2B, T, V]`：ref 的 `ref_logits`、policy 的 `logits`，以及 `log_softmax` 在 policy 那条路上产生的中间结果。前者算完即弃（在 `no_grad` 里，`ref_log_probs` 是 `[2B, T]` 的小张量），后两份却要一直撑到 `backward()` 结束——`log_softmax` 的反向需要它自己的输出。

代入默认参数算一下：`--batch_size 4` → 实际 `2B = 8` 行，`max_seq_len=1024`，`vocab_size=6400`，fp32 下一份就是 210 MB，三份 629 MB。而同样 `--batch_size` 的 SFT，logits 只有 `[B, T, V]` 一份 105 MB。差距既来自行数翻倍，也来自多出来的那次 ref 前向——这就是 `train_dpo.py` 默认 `--batch_size 4`（SFT 通常给到 32 以上）的直接原因。

顺带一提，这也解释了为什么 DPO 对长序列格外敏感：这三份张量都随 `T` 线性增长，而 `V` 在大词表模型上还要再乘 5 倍。

{{lab:logprob_memory}}

{{quiz:q7}}

> [!KEY]
> 显存峰值是 `[2B, T, V]` 那几份 logits / log_softmax，而不是参数；默认 `batch_size=4` 就是被它逼出来的。

# dpo_loss：七行代码压出一个标量

到这里输入已经是 `[2B]` 的序列级 log prob（policy 一份、ref 一份）。这一章把 `dpo_loss` 剩下的七行走完：切两半、两次作差得到 margin、`logsigmoid`、取平均。三节分别看形状怎么变、margin 为 0 时会发生什么、以及 `beta` 在里面扮演几个角色。

## 切两半、两次作差、一个 margin {#split-margin}

{{source:trainer/train_dpo.py#L40-L50}}

先看一个容易读错的地方：**L40 的 `batch_size` 其实是 `2B`**，不是命令行里那个 `--batch_size`。它读的是 `ref_log_probs.shape[0]`，而此刻 `ref_log_probs` 已经被 L36 就地重新赋值成 `[2B]` 了。所以 `batch_size // 2` 切出来的两半各是 `[B]`。

L41-L44 一共切出四个 `[B]`：policy 和 ref 各自的 chosen / rejected。L46-L47 各自作一次差得到两个 log ratio，L48 再作一次差：

$$m = \underbrace{\big[\log\pi_\theta(y_w) - \log\pi_\theta(y_l)\big]}_{\texttt{pi\_logratios}} - \underbrace{\big[\log\pi_{ref}(y_w) - \log\pi_{ref}(y_l)\big]}_{\texttt{ref\_logratios}}$$

**L48 这个变量也叫 `logits`，但它和模型的 logits 毫无关系**——它是 `[B]` 的 margin，读源码时千万别串了。最后 L49 对每一对算 $-\log\sigma(\beta m)$（仍是 `[B]`），L50 `.mean()` 收成标量。

实验用 `expand=True` 逐行追踪一遍：注意两个形参在 L36-L37 之后同名但换了 shape，也注意实验里 policy 被人为抬高了 0.5，所以每对的 margin 恰好是 `0.5 × 8 = 4`，与随机初值无关。

{{lab:split_margin}}

{{quiz:q8}}

> [!KEY]
> `[2B]` → 切两半 → 两次作差得到 margin `[B]` → `-logsigmoid(beta·margin)` → `.mean()` 标量；L48 那个 `logits` 是 margin，不是模型 logits。

> [!MORE] margin 为什么是「reward 之差」
> DPO 的推导里，带 KL 约束的 RLHF 目标存在解析最优解，把它反解出来就得到一个重参数化的 reward：
> $$\hat r(x, y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)} + \text{const}(x)$$
> 常数项只和 prompt 有关，在同一对样本里作差时被消掉。于是 `beta * logits`（L49 里乘进去的那一步）恰好等于 $\hat r(x,y_w) - \hat r(x,y_l)$，也就是常说的 **reward margin**；整个 loss 就是「用 logistic 回归把 chosen 的隐式 reward 顶到 rejected 之上」。reward model 就是这样被消掉的——这也是 DPO 可以像普通监督学习一样反复跑 epoch（off-policy）的原因。

## policy == ref 时的 0.6931 {#ln2-sanity}

训练的第一步，policy 和 ref 是同一份权重（`train_dpo.py` 用同一个 `--from_weight` 加载两遍）。这意味着 `pi_logratios` 和 `ref_logratios` 逐元素相等，于是：

{{source:trainer/train_dpo.py#L46-L50}}

$$m = 0 \;\Rightarrow\; \mathcal{L} = -\log\sigma(\beta \cdot 0) = -\log\tfrac12 = \ln 2 \approx 0.6931$$

值得反复强调的是这个结论**有多不依赖别的东西**：`beta` 被乘在 0 上，所以 `beta` 取多少都一样；两条序列的长度、mask 框住了多少 token、数据内容是什么，全都在「policy 减 ref」这一步严格抵消掉了。哪怕 mask 恰好全是 0（一条 assistant 内容都没框到），`sum` 出来的是全 0 的 `[2B]`，margin 依然是 0，loss 依然是 ln 2——而且不会出现 `nan`，因为源码是 `sum` 不是 `mean`，压根没有除法。

所以 **0.6931 是调试 DPO 最好用的 sanity check**：第一步不是它，说明 ref 权重加载错了、或者 `cat` 和切分对不上、或者 mask 没对齐。

反过来说，这个检查也有它照不到的角落：如果只是把 `cat` 的顺序写反了，margin 依然恒为 0，第一步 loss 依然精确等于 0.6931——**看不出任何异常**，但梯度方向整个反了。实验里最后一行换成一个真正偏离了 ref 的 policy（`pretrain` 权重），loss 立刻跳到 2.85。

{{lab:ln2_sanity}}

{{quiz:q9,q10}}

> [!KEY]
> policy == ref ⇒ margin ≡ 0 ⇒ loss ≡ ln 2 ≈ 0.6931，与 beta、mask、长度、数据全都无关；第一步不是它就说明配置有问题。

## beta 的两个身份 {#beta-grad}

`--beta` 默认 0.15（`trainer/train_dpo.py:152`）。概念上它是隐式 KL 约束的强度：在 $\hat r = \beta\log\frac{\pi_\theta}{\pi_{ref}}$ 这个重参数化里，$\beta$ 越大，同样的 reward 差只需要更小的 $\log\frac{\pi_\theta}{\pi_{ref}}$ 就够，也就越不允许 policy 偏离 ref。

但它还有一个纯数值的身份，直接写在这一行里：

{{source:trainer/train_dpo.py#L49}}

$-\log\sigma(\beta m)$ 对 margin 求导是

$$\frac{\partial \mathcal{L}}{\partial m} = -\beta\,\sigma(-\beta m) \;\xrightarrow{\;m = 0\;}\; -\frac{\beta}{2}$$

也就是说 `beta` 同时是这条曲线在原点的斜率：beta 越大，曲线在 margin > 0 一侧塌得越快（梯度指数衰减，很快「学饱」），在 margin < 0 一侧涨得越陡。调 `beta` 等于同时在调「允许偏离多远」和「学习率」。

那么再往回传一层：`dpo_loss` 的入参 `policy_log_probs` 是 `[2B, T]` 的逐 token log prob，在 policy == ref、margin 全为 0 的那一刻，它每个位置上的梯度长什么样？先下注，再跑实验。

{{predict:p1}}

{{lab:beta_grad}}

实测（`beta=0.1`、3 对、每条 8 个有效 token）：chosen 行每个 mask=1 的位置都是 `-0.016667`、rejected 行都是 `+0.016667`，正好是 $\beta/(2n) = 0.1/6$（$n$ 是这个 batch 里的**对**数 3，来自 L50 的 `.mean()`），mask=0 的位置是 0。梯度下降朝梯度的**反**方向走，于是 chosen 被抬高、rejected 被压低。另外注意：因为 L36 是 `sum`，这份梯度会**原样均分**给该序列每一个有效 token，一个不多一个不少。

{{quiz:q11}}

> [!KEY]
> `beta` 既是隐式 KL 强度，也直接就是梯度尺度（margin=0 处 $|\partial\mathcal{L}/\partial m| = \beta/2$）；loss 在 ln 2 时梯度**不是 0**，每个有效 token 都拿到 ∓β/(2n)。

# 训练循环：两个模型、一个极小的学习率

`dpo_loss` 讲完了，剩下的都在 `train_epoch` 和 `__main__` 里。这一章看四件事：ref_model 是怎么被冻死的、一步训练里 loss 和梯度实际怎么动、默认学习率为什么小到 4e-8，以及第 2 章埋下的「求和聚合」最后带来了什么。

## ref_model 的三道冻结 {#ref-frozen}

policy 和 ref 用**同一个 `--from_weight` 加载两遍**（默认 `full_sft`），所以第一步两者完全一致——上一章那个 ln 2 就是这么来的。

{{source:trainer/train_dpo.py#L183-L189}}

然后对 ref 做了两件事，再加上调用点的 `with torch.no_grad():`（`trainer/train_dpo.py:74`），一共三道冻结；此外 `optimizer = AdamW(model.parameters())`（`trainer/train_dpo.py:194`）压根看不见 ref 的参数：

| 做法 | 防什么 |
|---|---|
| `ref_model.eval()` | 关 dropout；让 MoE 的 aux_loss 走 `else` 分支恒为 0。**不可替代** |
| `ref_model.requires_grad_(False)` | 参数不再是叶子节点，前向不建图、不保存激活 |
| 调用点的 `torch.no_grad()` | 同样阻止建图，和上一条**功能重复**，是防御性写法 |
| （额外）`AdamW(model.parameters())` | 优化器根本看不见 ref 的参数，所以它永远不会被更新 |

中间两道确实是重复保险：autograd 只在**有输入需要梯度**时才建图，而 ref 的参数全是 `requires_grad=False`、`input_ids` 又是不可微的 int64。所以把 `torch.no_grad()` 删掉，`ref_outputs.logits.requires_grad` 依然是 `False`、`grad_fn` 依然是 `None`。实验把这几项逐条验一遍（故意不套 `no_grad`），并顺手量一下 MoE 模型在 `train()` / `eval()` 下 aux_loss 的差别。

{{lab:ref_frozen}}

{{quiz:q12}}

> [!KEY]
> ref 的三道冻结里只有 `.eval()` 不可替代（dropout + MoE aux_loss）；`requires_grad_(False)` 和 `torch.no_grad()` 在这里互为冗余，任一道都够。

> [!MORE] `loss = dpo_loss_val + outputs.aux_loss` 那一行
> `trainer/train_dpo.py:84` 把 `outputs.aux_loss` 加进 loss。非 MoE 模型下它是一个值为 0 的标量（`model/model_minimind.py:231` 用 `new_zeros(1).squeeze()` 当 `sum` 的初值，列表为空时原样返回），加上去是 no-op；MoE 模型下才是真正的负载均衡 loss。而 `MOEFeedForward` 里 aux_loss 有 `if self.training` 的门（`model/model_minimind.py:171`），所以 **`ref_model.eval()` 的一个副作用就是它的 aux_loss 恒为 0**——反正 ref 那一路的 aux_loss 也没人用。

## 十六步迷你 DPO 的 loss 与 log prob 曲线 {#tiny-dpo}

一步训练的骨架和 SFT 只差「多跑一次 ref 前向 + 换个 loss 函数」：

{{source:trainer/train_dpo.py#L83-L94}}

`scaler`、`accumulation_steps`、`clip_grad_norm_`、余弦 lr、checkpoint 保存，全部和 Day 9 逐字相同。所以真正值得亲眼看的是**数值**怎么动。

实验拿 `full_sft` 当 policy 和 ref，8 对真实样本、`batch_size=2`、跑 4 个 epoch 共 16 步，并在每一步之后用一个**固定的探针 batch** 重新测一次序列级 log prob（如果直接打印训练 batch 自己的数值，不同 step 是不同样本，曲线没有可比性——这是自己写 RL 训练监控时最容易踩的坑）。学习率故意调到 `2e-6`，是仓库默认值的 50 倍，好让 16 步就看出趋势。

loss 会从 0.6932 出发，很快掉到 0 附近再也不动（$-\log\sigma(\beta m)$ 的梯度 $\propto \sigma(-\beta m)$，margin 拉开之后指数衰减，模型就「学饱」了）。真正值得先下注的是另一条：**chosen 的序列 log prob 在这 16 步里会往哪走？** loss 的目标是把它顶上去，但 loss 只看 chosen 和 rejected 的**差**。

{{predict:p2}}

{{lab:tiny_dpo}}

实测（Mac / MPS，约 8 秒）：chosen 从 −96.33 涨到第 7 步的 −74.45，之后掉头向下，第 16 步回到 −88.23；而 rejected 一路狂跌到 −260.79。也就是说**后半程 loss 已经是 0 了，模型却还在把 chosen 的绝对概率一起往下压**——因为「两个一起压、rejected 压得更狠」和「抬高 chosen」对 loss 完全等价，而前者在参数空间里往往更便宜。这就是 DPO 文献里被反复报告的退化现象。顺带验证上一节：第 1 步 loss 恰好是 0.6932，而裁剪前的 grad norm 是 160.60——鞍点，但梯度一点都不小。

{{quiz:q13}}

> [!KEY]
> DPO 的 loss 掉到 0 只说明这批数据的 chosen / rejected 被分开了，不代表模型变好了；真正要盯的是 chosen 的**绝对** log prob 有没有跟着一起跌。

## 默认学习率 4e-8 的来历 {#lr-4e-8}

`train_dpo.py` 的默认学习率是整个仓库里最小的一个，注释还专门叮嘱了一句：

{{source:trainer/train_dpo.py#L136-L137}}

`4e-8` 比 SFT 常用的 `5e-5` 小了三个数量级。原因就藏在前面推的梯度里：DPO 的梯度有一项是**无条件压低 rejected 序列上的每一个 token**，力度 $+\beta/(2n)$，一视同仁。而 rejected 和 chosen 是同一个 prompt 下的两个回答，它们共用大量最普通的 token——中文的「的、是、在」、英文的 `the`、标点、换行。压低 rejected 里的这些 token，等于同时压低模型在**所有**场景下生成它们的概率。

这就是 DPO 臭名昭著的退化路径：lr 稍大，模型发现「把常用词的概率整体压下去」是降 loss 最便宜的方向，于是几十步之内输出就崩成乱码，而 loss 曲线看起来漂亮极了（一路掉到 0）。上一节实验里 `LR=2e-6` 已经能在 16 步内看到 chosen 掉头；把它改成 `1e-5` 会崩得更快，改成 `2e-8` 则 16 步几乎看不出变化——这正是仓库默认值的量级。

另外两个配套设定也是同一个逻辑：`--epochs 1`（DPO 是 off-policy 的监督式 loss，可以反复跑 epoch，但跑多了必然过拟合这批偏好对），以及 `--from_weight full_sft`（必须从一个已经会说话的模型出发，DPO 只负责微调偏好，不负责教模型说话）。

{{quiz:q14,q15}}

> [!KEY]
> `lr=4e-8` 是因为 DPO 的梯度会无条件压低 rejected 里的每个 token，而这些 token 和 chosen 高度重合；lr 一大，模型就靠「整体压低常用词」作弊，输出崩掉而 loss 反而更好看。

## 求和聚合带来的长度偏置 {#length-bias}

回到第 2 章那两行。用 `sum` 忠于 $\log\pi_\theta(y\mid x)$ 的定义，也和原论文、TRL 的 `DPOTrainer` 默认一致。但它有一个直接后果：**每个有效 token 拿到的梯度一样大，所以越长的序列拿到的梯度总量越多**。

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)        # 仓库实现：求和
# 长度归一化版本（IPO / SimPO 走这条路）：
ref_log_probs = (ref_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
```

这个偏置在 policy == ref 时看不出来（margin 恒为 0，长度项严格抵消），一旦 policy 动起来就会显形。实验设计了一个干净的对照：让 policy 把**每一个** token 的 log prob 都统一压低 δ。这是一个完全不含偏好信息的扰动——模型只是整体变得不那么自信，没有学到任何「chosen 比 rejected 好」。在长度归一化的版本下它几乎没有效果；在求和版本下，每一对白赚（或白亏）的 margin 恰好是 $\delta \cdot (\text{len}_r - \text{len}_c)$。

{{predict:p3}}

{{lab:length_bias}}

实测（δ=0.3，β=0.1，4 对真实样本）：第 0 对（chosen 26 token、rejected 81 token）的 loss 从 0.6931 白白掉到 **0.1757**，第 3 对（33 vs 100）掉到 **0.1257**；而 chosen 更长的第 1、2 对反过来升到 1.7260 / 1.2412。长度归一化版本则四对全是 0.6931，纹丝不动。换句话说，只要数据里 rejected 系统性地更长，模型就能靠「整体调低长回复的概率」来降 loss，而不是真的学会了偏好。

那到底该用哪个？两边都有道理：求和忠于定义、是 DPO/TRL 默认；长度归一化能消掉这个白赚的 margin，但优化的已经不是原来那个 $\log\pi(y\mid x)$ 了。实践中更常见的做法是**在数据侧把长度配平**，而不是改 loss。

{{quiz:q16}}

> [!KEY]
> 用 `sum` 聚合时每个有效 token 梯度等大，于是长序列拿到更多梯度总量；一个纯粹「整体压低 log prob」的扰动就能凭空造出 $\delta\cdot(\text{len}_r-\text{len}_c)$ 的 margin。
