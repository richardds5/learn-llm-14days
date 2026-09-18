---
day: 13
format: points
title: "GRPO 与 Rollout Engine"
subtitle: "扔掉 critic，让同一个问题的 G 个回答互相当裁判"
minutes: 90
mainline: "一个 prompt 采样出 G 条 completion [B*G, R]：每条一个标量 reward → 在 [B, G] 上组内归一化成 advantage → 配上 completion mask 和逐 token logp → 一个标量 loss"
files:
  - trainer/train_grpo.py#L71-L153
  - trainer/rollout_engine.py#L24-L96
goals:
  - 能报出 GRPO 一个 step 里每个 tensor 的 shape：[B] → [B,P] → [B*G,P+R] → [B*G] → [B,G] → [B*G,R] → 标量
  - 能解释 [B*G] 为什么必须是 repeat_interleave 的排布，advantage 在哪一维归一化，+1e-4 兜的是什么
  - 能默写 completion mask 那 4 行，说清 is_eos.any() 兜底和 <= 闭区间各自防的是什么
  - 能说清三份 logps 谁带梯度、k3 估计器好在哪、grpo 和 cispo 的梯度差别出现在什么条件下
  - 能画出 RolloutEngine 的边界：哪些事归引擎、哪些事归训练脚本，sglang 引擎为此多做了什么
---

Day 12 的 PPO 要养一个 critic：actor / critic / ref / reward 四个网络，advantage 靠 GAE 逐 token 反向递推。今天的 GRPO 把 **critic 和 GAE 一起删掉**，换成一个粗暴到近乎朴素的 baseline：同一个 prompt 采样 G 个回答，拿这一组的平均 reward 当基准，高于平均的被鼓励、低于的被压制。代价是每个 prompt 要多跑 G 倍的生成，收益是少了一个自己都还没收敛、就要给别人当尺子的价值网络。

今天跟着一条数据流走：B 个 prompt 采成 `[B*G, P+R]` 的 outputs，每条轨迹拿一个标量 reward，在 `[B, G]` 上组内归一化成 advantage，再配上 completion mask 和逐 token logp，收成一个标量 loss。

第一章走完「采样 → reward → advantage」这条**序列级**主线；第二章把它们对齐到 **token 级**（mask 和错一位的下标），是全天最容易写错的地方；第三章拼出 loss；第四章跳出 `train_grpo.py` 看 `rollout_engine.py`，最后把整条链路在 CPU 上跑一遍。

{{flow: prompts [B] | *rollout ×G* | outputs [B*G,P+R] | *reward* | rewards [B*G] | *组内归一化* | advantages [B*G] | × completion_mask | per_token_loss [B*G,R] | loss 标量}}

# 组采样：一个 prompt 变成 G 条轨迹

GRPO 的全部特别之处都在「组」这个字上：一个 prompt 要采 G 条回答，这 G 条在 batch 里怎么排、每条怎么拿到一个分数、组内怎么把分数变成相对优势。这一章走完的是**序列级**的主线，token 维要到第二章才出现。

## 组采样的行序：repeat_interleave 与 i*G+j {#group-layout}

`grpo_train_epoch` 拿到的 `batch['prompt']` 是 `list[str]`，长度 B（默认 `batch_size=2`）。tokenizer 用 `padding_side="left"` 把它们编码成 `[B, P]`——生成任务必须左 padding，否则 pad 会夹在 prompt 和续写之间；`add_special_tokens=False` 是因为 `RLAIFDataset` 已经用 chat template 渲染好了（Day 8）。截断写的是 `[:, -args.max_seq_len:]`，**左截断**，保住最靠近回答的那段上下文。

真正把 B 变成 B*G 的是引擎里这一行：

{{source:trainer/rollout_engine.py#L74-L77}}

`repeat_interleave(G, dim=0)` 给出的排布是 `[p0,p0,…,p0, p1,p1,…]`：**同一个 prompt 的 G 个回答连续占 G 行**。这不是随手挑的写法，它和打分函数里的下标算法死死绑在一起：

```python
response_idx = i * args.num_generations + j     # train_grpo.py:46
```

`k = i*G + j` 反过来就是 `i = k // G`。要是哪天把它换成 `repeat(G, 1)`（排布变成 `[p0,p1,p2,p0,p1,p2,…]`），reward、advantage、logps 三者的行对应关系会全错，而且**不会报错**——静默错到底。sglang 引擎那边写的是 `[ids for ids in input_ids_list for _ in range(num_generations)]`，双层列表推导的语义和 `repeat_interleave` 一致。

实验直接从 `output_ids[k, :P]` 反查第 k 行带的到底是哪个 prompt，和 `k // G` 对照着看。

{{lab:group_layout}}

{{quiz:q1}}

> [!KEY]
> `repeat_interleave(G, dim=0)` 让同一个 prompt 的 G 个回答连续占 G 行，下标 `k = i*G + j`；换成 `repeat` 会让 reward 和轨迹静默错位。

## 序列级的 reward：一条轨迹一个标量 {#reward-scalar}

`calculate_rewards` 和 `train_ppo.py` 的同名函数逐字对比，规则部分**完全一样**，唯一的差别是外层多套了一个 `for j in range(args.num_generations)`——PPO 一个 prompt 对一个 response，GRPO 一个 prompt 对 G 个。

{{source:trainer/train_grpo.py#L44-L60}}

四个分量：L54 的长度分，`20 <= len(response.strip()) <= 800` 给 +0.5，否则 −0.5（太短和太长一样罚）；L55-L59 的 `</think>` 分支，思考段长度落在 [20, 300] 给 +1.0 否则 −0.5，`</think>` 恰好出现一次再给 +0.25、出现多次 −0.25，同时把 `answer` 改成 `</think>` **后面**那一段；L60 的 `rep_penalty(answer)` 按 3-gram 重复率扣分，上限 0.5；紧跟在这段下面的 L62-L66 还会加上外部 reward model 的打分（默认 `internlm2-1_8b-reward`，约 1.8B）。本地 `out/` 里没有这个模型，所以今天所有实验都用规则替身，并在代码里显式标注。

关键在 shape：`rewards` 是 `[B*G]`，**一条轨迹一个标量，没有 token 维**。PPO 还要把这个标量摊到最后一个有效 token 上再做 GAE，GRPO 不需要——它直接进下一节的组内归一化。实验里 `grpo.args` 被临时塞了一个 `SimpleNamespace`，这样就能把仓库里真正的 `calculate_rewards` 原样跑起来。

{{lab:reward_scalar}}

{{quiz:q2}}

> [!KEY]
> `rewards` 是 `[B*G]` 的序列级标量，排布同样是 `k = i*G + j`；规则分量和 PPO 逐字相同，GRPO 只是多了一层 `for j` 循环。

## 组内归一化：把同组的平均分当 baseline {#group-advantage}

{{source:trainer/train_grpo.py#L122-L125}}

四行做完了 PPO 里 critic + GAE 二十多行的活。第一行 `rewards.view(-1, G)` → `[B, G]`：靠内存布局把「同组」变成「同一行」，这就是上一节那个排布的回报。第二、三行在 **dim=1（组内）** 求 `mean` 和 `std`，得到 `[B]`；`unbiased=False` 是除以 G 而不是 G−1——这一组只是一次采样，不是要估计总体方差。然后 `.repeat_interleave(G)` 把 `[B]` 摊回 `[B*G]` 和 `rewards` 对齐（这里同样不能用 `repeat`）。

advantage 仍然是**序列级的一个标量**，后面靠 `advantages.unsqueeze(1)` 广播到这条轨迹的每一个 token。也就是说 GRPO 认为「这条回答里所有 token 的功过是一样的」，没有 GAE 那种逐 token 的信用分配。

最后那个 `+ 1e-4` 看着像可有可无的数值噪音。考虑一种在小模型上**非常常见**的情况：同一组 G 条回答的 reward 完全相同（比如都是 1.0）。这一组的 advantage 会是什么？先下注，再跑实验。

{{predict:p1}}

{{lab:group_advantage}}

实验里 g1 组（4 条 reward 都是 1.0）的 advantage 是 **0.0**：分子 `1.0 − 1.0 = 0`，分母 `0 + 1e-4`，结果是 0 而不是 `nan`。这一组安安静静地不贡献梯度——README 里说的 **Degenerate Group（退化组）**。把 `1e-4` 改成 0 再跑，那一组立刻变成 `nan`，而 nan 会顺着 loss 污染整个 batch 的梯度。所以这个数是**退化组的保险丝**，不是配平用的小数。

{{quiz:q3}}

> [!KEY]
> advantage = 组内 z-score：`view(-1, G)` 让一行 = 一组，`mean/std` 在 dim=1 上算，`+1e-4` 让退化组得到 0 而不是 nan。

# token 级对齐：completion mask 与三份 logps

上一章的量都是序列级的 `[B*G]`，loss 却要在 `[B*G, R]` 上算。这一章解决三件对齐的事：哪些 token 该计入 loss（completion mask）、每个 completion token 的 logp 藏在 logits 的第几列（`logp_pos`）、以及同一批 outputs 上那三份 logps 各自从哪来、谁带梯度。

## 第一个 EOS 说了算的 completion mask {#completion-mask}

要理解这几行，得先知道 MiniMind 的 `generate` 有个特殊行为（`model/model_minimind.py:279`）：某条序列一旦吐出 EOS 就被标记成 `finished`，之后**继续往里塞 EOS**，把这一行凑到和别人一样长。所以 `completion_ids` 的一行里可能有一长串 EOS，只有第一个是真结束符，后面全是垃圾。

{{source:trainer/train_grpo.py#L127-L131}}

逐行看：`is_eos` 是 `[B*G, R]` 的 bool；`eos_idx` 先被整体填成默认值 `R-1`；`is_eos.int().argmax(dim=1)` 是「找第一个 True」的惯用写法（bool 转 int 之后 argmax 返回第一个 1 的下标）；最后 `arange(R) <= eos_idx.unsqueeze(1)` 生成 mask，再和 `completion_pad_mask` 取交。

两个容易写错的地方。一是 `<=` 是**闭区间**：第一个 EOS 自己也计入 loss，因为「在这里停下来」本身就是要学的行为；改成 `<` 模型就永远学不到该在哪停。二是 `eos_idx[is_eos.any(dim=1)] = ...[is_eos.any(dim=1)]` 这个花式索引，它只覆盖「确实有 EOS」的行。那么，一条被 `max_new_tokens` 截断、**一个 EOS 都没有**的轨迹，如果去掉这层保护、直接写 `eos_idx = is_eos.int().argmax(dim=1)`，最后会保留几个 token 计入 loss？先下注。

{{predict:p2}}

{{lab:completion_mask}}

实验表格第 4 列给出答案：`argmax` 对**全 0 的行返回 0**，于是这一行只留下第一个 token，后面 8 个 token 的梯度全被扔掉——一条完整的长回答被削成一个字，还不报错。`.any()` 兜底就是为了让这种行保留默认的 `R-1`（全部保留）。

{{quiz:q4}}

> [!KEY]
> mask 由第一个 EOS 的下标决定，`<=` 让 EOS 自己也计入 loss；没有 EOS 的行必须靠 `is_eos.any(dim=1)` 兜住，否则裸 `argmax` 会把它削成 1 个 token。

## logp_pos：错一位的取数下标与 full_mask 的覆写 {#logp-pos}

{{source:trainer/train_grpo.py#L90-L94}}

L93 是今天第一个需要停下来想一秒的地方。模型 forward 出来的 `logits[:, :-1, :]` 里，第 $t$ 个位置预测的是 token $t{+}1$。completion 的第 $j$ 个 token 在 `outputs` 里的下标是 $P{+}j$，所以预测它的那个位置是 $P{+}j{-}1$，于是

$$\texttt{logp\_pos}[:,j] = \texttt{prompt\_lens} - 1 + j$$

torch 引擎里 `prompt_lens` 恒等于 P，三行一模一样，值域是 $P{-}1 \dots P{+}R{-}2$；到了 sglang 引擎，每行的 prompt 长度都不同，这个式子才显出必要性。

L94 的 `scatter_(1, logp_pos + 1, ...)` 写的是 `logp_pos+1`，也就是 completion **自己**占的那些列 $P \dots P{+}R{-}1$。为什么要覆写一次？因为 L92 的 `full_mask = (outputs != tokenizer.pad_token_id)` 是按「等于 pad id 就当 padding」推出来的，而 pad 是 `0 = <|endoftext|>`——**模型真的生成出一个 id 等于 0 的 token 时，它会被误判成 padding**，那一格在 attention 里就凭空消失了。scatter 用引擎给的 `completion_mask` 把 completion 区间的 mask 强行改回真值，顺带让 sglang 引擎的右 padding 生效。

{{lab:logp_pos}}

{{quiz:q5,q6}}

> [!KEY]
> `logp_pos = prompt_lens - 1 + j` 是「预测第 j 个 completion token 的那个位置」；`scatter_` 写的是 `logp_pos+1`，把被 pad id 误伤的 completion 格子改回真值。

## 三份 per-token logps 与唯一的梯度入口 {#three-logps}

{{source:trainer/train_grpo.py#L99-L105}}

同一批 `outputs` 上有三份 `[B*G, R]` 的 per-token logps。`old_per_token_logps` 是 rollout 时算的（引擎内部走 `compute_per_token_logps`），拿到手立刻 `.detach()`，只当 importance ratio 的分母；`per_token_logps` 是训练 forward 在 `autocast_ctx` 里算的，是**整个 loss 唯一的梯度入口**（默认 `use_moe=0` 时 `aux_loss` 是常量 `torch.tensor(0.0)`）；`ref_per_token_logps` 在 `torch.no_grad()` 里算，`ref_model` 本身也 `.eval().requires_grad_(False)` 冻着，只当 KL 的锚点。

取数的写法三份一样：`log_softmax(logits[:, :-1, :])` → `gather(2, outputs[:, 1:])` 取出每个位置上真实 token 的 logp（`[B*G, P+R-1]`，还覆盖着 prompt 区间）→ 再 `gather(1, logp_pos)` 挑出 completion 那 R 个。

`ratio = exp(per_token_logps - old_per_token_logps)` 就是 PPO 那个重要性采样比。在 MiniMind 的 torch 引擎下，这一批 ratio 的 min 和 max 会是多少？先下注。

{{predict:p3}}

{{lab:three_logps}}

实验给出 `min = max = 1.000000`。`old` 是**同一份权重**在 rollout 里算出来的，训练 forward 只是换了个写法算同一个东西（逐元素相等），而且每个 batch 只更新一次策略。所以 torch 引擎下的 GRPO 是**纯 on-policy**，ratio 恒为 1，下一章的 clip 和 clamp 从不激活。同理第 1 步 `ref_model` 和 policy 同权重，`per_token_kl` 恒为 0。

{{quiz:q7}}

> [!KEY]
> 三份 logps 只有 `per_token_logps` 带梯度；torch 引擎下 `old` 和它逐元素相等，所以 `ratio ≡ 1`，GRPO 在这里是纯 on-policy 的。

> [!MORE] 两处算同一个 logp，为什么写法不一样
> `train_grpo.py:102` 对**全量** logits 做 `log_softmax`（`[B*G, P+R-1, V]`），而引擎里的 `compute_per_token_logps` 用 `logits_to_keep=n_keep+1` 让 `lm_head` 只在最后 R+1 个位置上算，得到 `[B*G, R+1, V]` 再 `[:, :-1, :]` 对齐。后者省掉了 prompt 区间那一大片 `lm_head`（P≈768 时是 768×6400 的浪费），前者换来的是一行写完、而且天然配合 `logp_pos` 处理 sglang 那种「每行 prompt 长度都不同」的情况。

# loss：KL 估计器与两个变体

advantage、mask、三份 logps 都齐了，这一章把它们拼成一个标量。三个小节正好对应源码的三段：KL 怎么估、策略项怎么写（两个变体）、逐 token 的 loss 怎么聚合成一个数。

## k3 估计器：非负且低方差的 KL {#kl-k3}

{{source:trainer/train_grpo.py#L133-L134}}

两行而已，但它是 Schulman 的 **k3 估计器**。记 $lr = \texttt{kl\_div} = \log \pi_{ref} - \log \pi_\theta$（在一个从 $\pi_\theta$ 采样出来的 token 上），有三种常见估法：$k_1 = -lr$ 最朴素，期望等于真 KL，但**单个样本上可以是负数**——负的 KL 惩罚等于在奖励模型跑远，方向直接反了；$k_2 = lr^2/2$ 恒非负、方差也不大，但**有偏**；$k_3 = e^{lr} - lr - 1$ 恒 $\ge 0$（等号只在 $lr=0$ 取），期望仍等于真 KL，方差远小于 $k_1$。

实验用「加了一点点噪声的 `full_sft`」当 ref（模拟 policy 训了几十步之后拉开的小差距），在 2016 个从 $\pi_\theta$ 采样的 token 上对比三者，并且用对全词表求和的方式算出**真 KL** 当标尺。

{{lab:kl_k3}}

实测：真 KL 是 0.04168，k3 给 0.04123、k2 给 0.04375、k1 给 0.03670。k1 的 std 是 0.2936，k3 只有 0.1024；而且 2016 个样本里有 910 个（45%）k1 是负数，k3 一个都没有。「非负 + 无偏 + 低方差」三条全占，所以 k3 成了 GRPO 系实现的标配。乘上 `args.beta`（默认 0.1）之后，它就是 loss 里那个把策略拽回 ref 附近的弹簧。

{{quiz:q8}}

> [!KEY]
> `per_token_kl = exp(kl_div) - kl_div - 1` 是 k3 估计器：恒非负、期望等于真 KL、方差远小于 k1；k1 会在近一半 token 上变成负的「惩罚」。

## grpo 的 clip 与 cispo 的 detach 权重 {#loss-variants}

{{source:trainer/train_grpo.py#L135-L143}}

两个分支对应两个公式：

$$\mathcal{L}_{GRPO} = -\big[\min(r_t A_t,\ \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\, A_t) - \beta\,\mathrm{KL}_t\big]$$

$$\mathcal{L}_{CISPO} = -\big[\min(r_t, \varepsilon_{\text{high}})_{\texttt{.detach()}} \cdot A_t \cdot \log\pi_\theta - \beta\,\mathrm{KL}_t\big]$$

关键在那个 `.detach()`：CISPO 把 ratio 降级成一个**纯权重**（不再是梯度路径），梯度只从末尾那个 `per_token_logps` 走，于是 $\partial \mathcal{L}/\partial \log\pi = -\min(r, \varepsilon_{high})A$ 永远不为 0，只是被 $\varepsilon_{high}$ 封顶。GRPO 则是一旦 `min` 选中被 `clamp` 住的那一支，而 `clamp` 在区间外的梯度是 0，**这个 token 的策略梯度直接归零**。

默认值：`epsilon=0.2`、`epsilon_high=5.0`、`beta=0.1`，`loss_type` 默认就是 **cispo**。实验不跑模型，直接用 autograd 把 $\partial \mathcal{L}/\partial \log\pi$ 随 ratio 的变化画出来。

{{lab:loss_variants}}

四条曲线的读法：`grpo A=+1` 在 ratio > 1.2 之后掉到 0，`grpo A=−1` 在 ratio < 0.8 之前是 0——**clip 是单边生效的，看 advantage 的符号**；cispo 的两条从不归零，只是在 ±5.0 处走平。反过来看 `grpo A=−1` 在大 ratio 处的权重是 $-A\cdot r$，一路涨到 6 都**不封顶**，那是 CISPO 想管住的另一头。

{{quiz:q9,q10}}

> [!KEY]
> GRPO 用 `min + clamp` 把越界 token 的策略梯度掐断；CISPO 把 `clamp(r, max=ε_high)` **detach** 成权重乘到 `logπ_θ` 上，梯度永不断、只被封顶。

## 两级平均：每条轨迹在 loss 里等权 {#loss-reduce}

{{source:trainer/train_grpo.py#L144-L146}}

L144 是**两级平均**，不是全局 token 平均：先在每条序列内部对有效 token 求均值得到 `[B*G]`，再对这 B*G 条序列求均值得到标量。后果是每条轨迹在 loss 里权重相等、**与长度无关**——一条 3 token 的短回答和一条 40 token 的长回答话语权一样。换成 `sum() / mask.sum()` 的全局平均，每条轨迹的权重就正比于它的有效长度，长回答会主导梯度（DAPO 之类的变体正是在这里做文章）。

`clamp(min=1)` 防的是「整行 mask 全 0」时的除零：真出现这种行，去掉 clamp 就是 `0/0 = nan`。`aux_loss` 只有 `use_moe=True` 时非零（Day 6）；除以 `args.accumulation_steps` 是梯度累积的老规矩（Day 9）。

实验把每个 token 的 loss 都设成 1，再用 `torch.autograd.grad` 把「每条轨迹到底拿到多少权重」直接读出来——梯度就是权重本身，不用推公式。

{{lab:loss_reduce}}

{{quiz:q11}}

> [!KEY]
> 「序列内 token 平均 → 序列间平均」让每条轨迹等权；全局 token 平均会让长回答按长度加权，这是两种聚合最实质的差别。

# Rollout Engine 与一次完整的 step

前三章都在 `train_grpo.py` 里打转。这一章跳出去看 `rollout_engine.py`：它把「采样」抽象成两个方法，PPO / GRPO / Agent RL 三个脚本共用；再看 sglang 引擎为了训推分离多做了什么；最后把整条链路在 CPU 上真跑一遍。

## rollout 与 update_policy 划出的边界 {#engine-contract}

{{source:trainer/rollout_engine.py#L40-L60}}

`RolloutResult` 的六个字段就是训练循环需要的全部信息，抽象基类只要求两个方法。这条边界划得很干净：**引擎只管「给我 prompt，还我 completion + logp」，advantage / mask / loss 全留在训练脚本里**，所以 PPO（`num_generations=1`）、GRPO、Agent RL 能共用同一层。

`TorchRolloutEngine.rollout` 的实现只有二十来行：`repeat_interleave` 把 batch 复制宽 → `model.generate(...)`（MiniMind 自己写的那个，Day 7）→ 切出 `completion_ids` → 拼 `full_mask`（completion 区间无条件全 1）→ 调 `compute_per_token_logps` 算 old logps。末尾那个 `.clone()` 省不得：`generate` 带 `@torch.inference_mode()`，返回的是 inference tensor，不 clone 就进不了后续的自动求导世界。

`update_policy` 在 torch 引擎里只有一行 `self.policy_model = model`——引擎和训练脚本共享同一个 `nn.Module` 对象，权重本来就是同步的，这行只是在 `torch.compile` / DDP 包装之后重新绑一下引用。

注意最后两个字段：torch 引擎给的 `prompt_lens` 是常量 P、`completion_mask` 是全 1，看着完全多余——它们是给下一节那个引擎准备的。

{{lab:engine_contract}}

{{quiz:q12}}

> [!KEY]
> 引擎的契约只有 `rollout`（prompt → completion + logp + 两个对齐字段）和 `update_policy`（同步权重）；advantage、mask、loss 一概不归它管。

## SGLang 引擎：变长结果与权重同步 {#engine-sglang .side}

把采样换成 SGLang 这样的专用推理引擎，吞吐能高一个量级；代价是它跑在**另一个进程 / 另一台机器**上，于是多出两个问题。

第一是结果形状。HTTP 接口按变长 list 传，所以先 `ids[mask.bool()]` 把左 padding 撕掉，采样完再自己 pad 回矩形（`max_comp_len` / `max_out_len`，用 `pad_token_id` 补右边）。于是 `prompt_lens` 变成每行**真实的**、互不相同的 prompt 长度，`completion_mask` 变成真的 0/1 mask——`RolloutResult` 里那两个字段就是为这个场景存在的，训练脚本里那个看似多余的 `gather(1, logp_pos)` 也因此才必要。

第二是权重同步：

{{source:trainer/rollout_engine.py#L175-L193}}

走的是「共享 checkpoint 目录」这条最土但最省事的路：只有 rank 0 落盘（`half().cpu()` + `save_pretrained(safe_serialization=False)`，连 tokenizer 一起），再 `POST /update_weights_from_disk` 让 server 自己从磁盘热加载，结果用 `dist.broadcast` + `barrier` 发给所有 rank，失败直接 `raise`——宁可炸也不能让一半 rank 用新权重、一半用旧的。

还有一件事很关键：`update_policy` 只在 `step % args.save_interval == 0` 时调用（`train_grpo.py:194`，默认 10 步一次）。对 torch 引擎无所谓（共享对象），对 sglang 引擎则意味着中间 9 步的 rollout 用的是**旧权重**——真正的 off-policy，`ratio ≠ 1`，clip / clamp 这才真的开始工作。这也是默认 `loss_type="cispo"` 的动机。

{{lab:sglang_pad}}

{{quiz:q13}}

> [!KEY]
> sglang 引擎把变长结果 pad 回矩形，靠 `prompt_lens` + `completion_mask` 告诉训练脚本真值；权重同步走「落盘 + `/update_weights_from_disk` + `flush_cache`」，且只在 `save_interval` 做一次。

> [!MORE] 另外两个细节
> **为什么值得换引擎**：rollout 是自回归解码，memory-bound，朴素的 for 循环 + HF KV cache 几乎跑不满 GPU；SGLang 有 continuous batching、RadixAttention 前缀复用、CUDA graph。
> **logprob 怎么拿**：payload 里带 `"return_logprob": True`，从 `meta_info.output_token_logprobs` 里取，长度和 `completion_ids` 对不上时在**左边**补 0（少了）或用 `logprobs[-N:]` 丢掉**左边**多余的（多了），见 rollout_engine.py:147-150——两种情况都以右端对齐，因为 logprob 列表可能不含第一个 token。
> **`flush_cache()` 为什么单独存在**：换完权重之后，RadixAttention 里缓存的 KV 前缀全是旧模型算的，必须清掉。

## 把整条链路跑一遍 {#mini-grpo}

{{source:trainer/train_grpo.py#L148-L153}}

loss 有了，剩下的是 Day 9 那套老规矩：`accumulation_steps` 默认 1，所以每个 step 都会走「裁剪 → `optimizer.step()` → `scheduler.step()` → `zero_grad()`」，注意 `grad_clip=1.0` 的裁剪发生在 `optimizer.step()` **之前**（顺序反了就等于没裁）。GRPO 的默认学习率是 `3e-7`，比 SFT 的默认值 `1e-5` 还小一个数量级还多（约 33 倍）——RL 阶段只想轻轻推一把，跑偏了就再也回不来。

实验用 `full_sft` 同时当 policy 和 ref，规则函数当 reward model 替身，B=3、G=4、`max_new_tokens=32`，跑 2 个完整的 step，日志格式和 `train_grpo.py:165-168` 打出来的那一行对齐。跑完对照两个和正文对得上的现象：第 1 步 `KL_ref` 恰好是 `0.000000`（policy 与 ref 同权重），第 2 步变成 `-0.337685`（policy 被更新过了，而这一步的 token 又是新 policy 采的，所以是负的）；`ratio` 两步都恒为 1，因为 rollout 用的就是当前权重。

只跑 2 步、每步还换了采样种子，`Reward` 的变化主要来自**采样随机性**，不代表模型学到了什么——真实训练要跑几千步、配真 reward model 才有意义，这里只验证数据流跑得通。明天 Day 14 会看到 `train_agent.py` 把今天这套组采样 + CISPO 搬到多轮 tool-call 轨迹上去。

{{lab:mini_grpo_step}}

{{quiz:q14}}

> [!KEY]
> 一个 GRPO step = rollout → reward → 组内 advantage → mask → 三份 logps → KL + 策略项 → 两级平均 → backward / 裁剪 / step；torch 引擎下它是纯 on-policy 的。
