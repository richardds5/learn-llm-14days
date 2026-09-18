---
day: 12
format: points
title: "PPO：四个模型、GAE 与 clip 目标"
subtitle: "一条 response 拿到的一个标量 reward，怎么变成 [B,R] 上每个 token 的梯度"
minutes: 95
mainline: "序列级的 rewards [B] 怎么一步步摊成 token_rewards [B,R] → advantages [B,R]，再乘上 ratio 变成一次带 clip 的梯度"
files:
  - trainer/train_ppo.py#L29-L307
  - trainer/rollout_engine.py#L24-L96
goals:
  - 说出 PPO 四个模型各自从哪初始化、谁要梯度、谁持有优化器状态
  - 默写 ppo_train_epoch 的数据流，并报出每个中间 tensor 的 shape
  - 说清 logp_pos 为什么取 P-1 … P+R-2，以及同一份下标被用在哪三处
  - 说清序列级 reward 怎么进入逐 token 的 GAE、两层 mask 各挡住了什么
  - 解释 ratio 的起点、clip 的单边性、以及 value clip 同样会把梯度削成 0
---

Day 11 的 DPO 只有 actor + ref 两个模型，数据是离线的偏好对，一次 forward 就能算出 loss。今天的 PPO 是另一个极端：**四个模型**、**在线采样**、**两套优化器**，`ppo_train_epoch` 是全仓库最长的一个函数（trainer/train_ppo.py:78-307）。难点不在数学——PPO 的公式你都会——而在**工程对齐**：prompt 是左 padding 的、response 长度不一、EOS 之后还填着东西、logits 要错一位、value 要取「动作发生之前」那个位置。所以今天的读法是：先把每个 tensor 的 shape 和含义钉死，再看公式怎么套上去。

今天只跟一条线走：一条 response 拿到的**一个标量 reward**，怎么摊到 `[B, R]` 的每个 token 上，再变成一次带 clip 的梯度。五章依次是：谁在产生这些张量（四个模型 + 一次 rollout）、逐 token 的 logp 和 mask 怎么对齐、reward 怎么变成 advantage、actor 的 loss 怎么写、critic 的 loss 和整轮更新循环。

{{flow: prompt [B,P] | *rollout* | gen_out [B,P+R] | rewards [B] | old_logp / ref_logp / values [B,R] | *GAE* | advantages [B,R] | ratio × adv | *clip 目标* | 两个 optimizer}}

# 四个模型与一次 rollout

一个 PPO step 里有四个网络在跑，但只有两个会被更新。这一章先把它们的来源、梯度归属和显存账认清楚，再看其中最特别的 `CriticModel` 长什么样，最后真跑一次 rollout，拿到后面每一章都要用的那几个张量。

## 四个模型的来源与梯度归属 {#four-models}

`actor` 和 `ref` 是同一份 `full_sft` 权重加载两遍，区别只在后者多了 `.eval().requires_grad_(False)`：ref 全程冻结，只负责给出参考策略的 logp。真正值得盯的是 critic 这两行——

{{source:trainer/train_ppo.py#L383-L392}}

`CriticModel(lm_config)` 是个全新的随机初始化模型，`load_state_dict(state_dict, strict=False)` 把 `full_sft` 的 backbone 灌进去。`strict=False` 不是随手写的：`full_sft_768.pth` 里**没有** `value_head.*` 这两个 key，用 `strict=True` 会直接抛 `RuntimeError`。代价是 value head 保持随机初始化——**critic 一开始给出的价值估计是纯噪声**，actor 拿到的 advantage 方向也就是噪声。README 里「PPO 前期 reward 涨得慢」就是这么来的。

reward model 是外部的 InternLM2-1.8B-Reward，`LMForRewardModel.get_score` 的最后一行把分数硬截断到 $[-3, 3]$（trainer/trainer_utils.py:177）。本地没有这个模型，所以今天所有实验里它都由规则函数或常数顶替。

学习率上 actor 是 `3e-7`、critic 是 `5e-7`，都比 SFT 的 `1e-5` 小一到两个数量级；critic 略大一点，因为它要从随机 head 学起。显存账也就清楚了：两份**带梯度**的 63.9M 模型（各自还要 AdamW 的 m/v 两份状态）+ 一份冻结的 ref + 一份 1.8B 的 fp16 reward model。

{{lab:four_models}}

{{quiz:q1}}

> [!KEY]
> 只有 actor 和 critic 要梯度、各有一个 AdamW；ref 和 reward model 全程 `no_grad`。critic 的 `value_head` 是被 `strict=False` 漏下来的**随机初始化**权重。

> [!MORE] 两个 scheduler 的 T_max 为什么不是 iters
> ```python
> mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
> total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
> ```
> （L410-L411）因为**每个数据 step 会产生多次 `optimizer.step()`**：`ppo_update_iters` 轮 × 每轮 `ceil(B / mini_batch_size)` 个 minibatch。按 `iters` 来算的话余弦退火会退得太慢，训练结束时 lr 还停在起点附近。

> [!MORE] 四个模型在 DPO / PPO / GRPO 里的账
> | | 模型数 | 数据 | advantage 从哪来 | KL 在哪 |
> |---|---|---|---|---|
> | DPO（Day 11） | 2：actor + ref | 离线偏好对，可反复用 | 不需要（有解析解） | 隐含在 $\beta$ 里 |
> | **PPO（今天）** | **4**：actor + ref + critic + RM | 在线 rollout，复用 `ppo_update_iters` 轮 | **critic + GAE** | loss 项 `kl_coef * kl_ref` |
> | GRPO（Day 13） | 3：actor + ref + RM | 在线 rollout，一个 prompt 采一组 | **组内归一化**，没有 critic | loss 项 |
>
> PPO 最重的地方就在 critic：它要和 actor 联合优化，前期估不准就拖慢整个训练。GRPO 干脆把 critic 删掉，用「同一个 prompt 的一组样本」的组内均值当 baseline。

## CriticModel：把 [B,T,C] 压成 [B,T] {#critic-head}

十三行代码，但有四个点值得停一下。

{{source:trainer/train_ppo.py#L36-L48}}

**第一，它继承的是 `MiniMindForCausalLM` 而不是 `MiniMindModel`**，所以 `self.lm_head` 还原封不动地挂在那里（注释说「替换 lm_head」，实际是「新增 value_head」）。好在 `tie_word_embeddings=True` 让 `lm_head.weight` 和 `embed_tokens.weight` 是同一块存储，不额外占显存。

**第二，`value_head = nn.Linear(hidden_size, 1)`**，`hidden_size=768` 时只有 769 个参数（含 bias）——整个 critic 比 actor 只多这 769 个数。

**第三，第 45 行的 `self.model.norm(...)` 是第二次归一化。** `MiniMindModel.forward` 的最后一行已经做过 `self.norm(hidden_states)`，这里又做了一次，模块调用树里你会看到 `model.norm` 出现两次。RMSNorm 不是幂等的：它会把 rms 重新拉回 1 再乘上可学习的 `weight`，而 `full_sft` 的 `model.norm.weight` 均值约 1.85，所以这是一次真实的额外变换，不是 no-op。

**第四，`self.value_head(hidden_states)` 得到 `[B, T, 1]`，`.squeeze(-1)` 之后是 `[B, T]`**——覆盖 prompt + response 的**每一个位置**，每个位置一个标量价值，而不是整条序列一个。

{{lab:critic_head}}

{{quiz:q2}}

> [!KEY]
> `CriticModel.forward` 输出 `[B, T]`：prompt 和 response 的每个位置都有一个标量价值；后面只会 `gather` 出需要的那一段。

## rollout 一次产出的六个字段 {#rollout-fields}

`ppo_train_epoch` 的第一步是把 prompt 用 `padding_side="left"` 编码成 `[B, P]`——prompt 必须**左** padding，否则 `generate` 会从 pad token 后面接着写。注意 padding 用的是 `pad_token_id = 0`（`<|endoftext|>`），和 `eos_token_id = 2`（`<|im_end|>`）不是一个 token，这个区别下一章会变得很重要。

接着交给 `TorchRolloutEngine.rollout`，它整段跑在 `with torch.no_grad(), ctx:` 里：

{{source:trainer/rollout_engine.py#L85-L92}}

六个返回字段里，`output_ids` 是 `[B, P+R]` 的完整序列，`completion_ids = output_ids[:, P:]` 是 `[B, R]` 的 response 段，`per_token_logps` 就是后面要用的 `old_logp`，同样是 `[B, R]`。剩下三个是给下游对齐用的：`completions` 是 `skip_special_tokens=True` 解码出来的纯文本（喂给 reward），`prompt_lens` 在 torch 引擎下**恒等于 `P`**，`completion_mask` 在 torch 引擎下**恒为全 1**——最后两个字段只有换成 SGLang 引擎（逐样本长度不同、右侧真补 pad）时才有非平凡的值。

还有两点：`num_generations=1`，PPO 每个 prompt 只采一条（Day 13 的 GRPO 会采一组，靠组内比较算 advantage）；`R` 也不一定等于 `max_new_tokens`，`generate` 在**所有**序列都 finished 时会提前 `break`。

{{lab:rollout_fields}}

{{quiz:q3}}

> [!KEY]
> rollout 返回的六个字段全部在 `no_grad` 里算出；torch 引擎下 `prompt_lens ≡ P`、`completion_mask ≡ 1`，右 padding 的信息要换 SGLang 引擎才有。

# 逐 token 对齐：logp 与两层 mask

PPO 要的不是 loss，而是**每个 response token 的 $\log \pi(a\mid s)$**。这一章看两件事：怎么只对需要的那一段过 `lm_head` 而又不错位；以及 EOS 之后那一堆填充，靠哪一个 mask 才真正被剔出 loss。

## logits_to_keep 与末尾切一刀 {#logits-keep}

`compute_per_token_logps` 是 rollout engine 里最短也最容易读错的一段：

{{source:trainer/rollout_engine.py#L24-L36}}

关键是第 29 行。回忆 `MiniMindForCausalLM.forward` 里的这两行（model/model_minimind.py:247-248）：`slice_indices = slice(-logits_to_keep, None)`，`logits = self.lm_head(hidden_states[:, slice_indices, :])`——传 `logits_to_keep=n` 就只对最后 `n` 个位置过 `lm_head`，省掉 `(T-n) × V` 的大矩阵乘。`V=6400` 时这块 logits 是整个 batch 里最大的中间结果，能省则省。

那为什么传 `n_keep + 1` 再用 `[:, :-1, :]` 砍掉最后一个？因为**位置 $i$ 的 logits 预测的是 token $i+1$**。要给最后 `n_keep` 个 token（绝对位置 $T-n_{keep} \dots T-1$）打分，需要的是位置 $T-n_{keep}-1 \dots T-2$ 的 logits。直接留 `n_keep` 个会少一个在前、多一个在后，正好整体错一位。

第 31 行那个 `for` 循环等价于一次批量 `gather`，写成逐行是为了**不一次性把 `[B, n_keep, V]` 的 `log_softmax` 全部物化**。而 `ppo_train_epoch` 内部**没有**调用这个函数，它用的是等价的另一种写法（全量 forward + `logits[:, :-1]` + 两次 `gather`），因为它同时还需要 critic/ref 在整条序列上的输出——实验里对这两种写法做了逐元素对拍。

{{lab:logits_keep}}

{{quiz:q4}}

> [!KEY]
> `logits_to_keep=n_keep+1` 省掉 prompt 段的 `lm_head`，`[:, :-1, :]` 把「位置 $i$ 预测 token $i+1$」错回来；少 `+1` 会在 `gather` 处报 `RuntimeError`，切片方向错了（`[:, 1:, :]`）则静默算错、不报错。

## full_mask、logp_pos 与 resp_policy_mask {#mask-eos}

这十三行是整个函数最绕的地方。

{{source:trainer/train_ppo.py#L116-L128}}

**L116** `full_mask = (gen_out != pad_token_id)` 是 `[B, P+R]`，喂给 critic / ref 当 attention mask；**L117** `labels = gen_out[:, 1:]` 是 `[B, P+R-1]`，左移一位好和 `logits[:, :-1]` 对齐。

核心是 **L120-L121** 的 `logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx`，`[B, R]`，取值 $P-1 \dots P+R-2$——「第 $j$ 个 response token 在**错位后坐标系**里的下标」。gather logp、gather ref logp、gather value 用的都是同一份它。注意 value 也用它而不是 $P \dots P+R-1$：第 $j$ 个 token 是在「已读完前 $P-1+j$ 个 token」这个状态上做的决策，$V(s_j)$ 自然要取那个位置。

**L124-L126** 找出**第一个** EOS，`resp_lengths = where(has_eos, eos_pos + 1, resp_lengths)`——那个 `+1` 意味着 **EOS 本身算进有效长度**。

现在有个值得先下注的问题：`generate` 在某条序列 finished 之后仍会继续往后填 token（model/model_minimind.py:279）。那么 EOS **之后**的那些位置，在 `full_mask` 的 response 段里是 0 还是 1？

{{predict:p1}}

{{lab:mask_eos}}

实验表格的最后两列给出了答案：`full_mask` 在 response 段**恒为全 1**（原因见下面的折叠块），而 **L127-L128** 的 `resp_policy_mask` 才数出真实长度 4 / 11 / 8——所有 loss 的分母都来自它。换句话说，critic 和 ref 一定会 attend 到那一堆多余的 `<|im_end|>`。

{{quiz:q5}}

> [!KEY]
> 两层 mask 分工不同：`full_mask` 只挡左 padding（response 段被 `scatter_` 覆盖成全 1），`resp_policy_mask` 才把第一个 EOS **之后**的位置剔出 loss，EOS 本身保留。

> [!MORE] 为什么 full_mask 的 response 段挡不住任何东西
> 两层原因叠在一起。第一层：`generate` 在 finished 之后填的是 `eos_token_id`（2）而不是 `pad_token_id`（0），`(gen_out != pad_token_id)` 根本抓不到它们。
> 第二层更彻底：**L123 的 `full_mask.scatter_(1, logp_pos + 1, resp_pad_mask)` 把整个 response 段直接覆盖成了 rollout 给的 `completion_mask`**——torch 引擎下就是全 1，就算第一层能抓到也会被这一步冲掉（`logp_pos + 1` 才是绝对位置 $P+j$）。
> 所以这一行在 torch 引擎下是恒等操作，只有换成 SGLang 引擎（各样本长度不同、右侧真补 pad）时才真正起作用。实验的第 2 个 task 就是手动把 `resp_pad_mask` 的尾部改成 `False` 来模拟这种情况。

# 从序列级 reward 到逐 token advantage

reward model 给的是一条 response 一个标量，而 PPO 的 loss 是逐 token 的。这一章走完这条转换链：`rewards [B]` → `token_rewards [B,R]` → GAE 倒序递推 → `advantages [B,R]` → 归一化。

## calculate_rewards：四个分量凑一个标量 {#reward-scalar}

`calculate_rewards` 整段跑在 `torch.no_grad()` 里，逐条样本累加四个分量：

{{source:trainer/train_ppo.py#L57-L67}}

**长度合格**：`20 <= len(response.strip()) <= 800` 给 `+0.5`，否则 `-0.5`——注意 `len()` 数的是**字符**不是 token。**thinking 长度**：仅当 response 里有 `</think>` 时才判，合格 `+1.0`、不合格 `-0.5`。**`</think>` 个数**：恰好一个给 `+0.25`，否则 `-0.25`。**重复惩罚**：`-rep_penalty(answer)`，取值 $[-0.5, 0]$。最后再加上 RM 的分数（已被 clamp 到 $[-3,3]$）。带 think 时总分范围是 $[-4.75, +4.75]$。

两个容易看漏的细节。一是 `answer`：有 `</think>` 时只取它**之后**的部分，所以重复惩罚和 RM 打分都只看最终答案、不看思考过程。二是 `messages`：它是用正则 `r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"` 从 prompt 文本里**反解**出来的；`RLAIFDataset` 在 `open_thinking=True` 时给出的 prompt 以 `<|im_start|>assistant\n<think>\n` 结尾，这一段不匹配 `<|im_end|>`，不会被解析进 `messages`，正合适。

无论多少分量，`rewards` 最终是 `[B]`：**一条序列一个标量，没有任何逐 token 结构**。

{{lab:reward_scalar}}

{{quiz:q6}}

> [!KEY]
> `calculate_rewards` 返回 `[B]`：长度 ±0.5、think 长度 +1/-0.5、`</think>` 计数 ±0.25、重复惩罚 -0.5~0，再加上截断到 $[-3,3]$ 的 RM 分数。

> [!MORE] rep_penalty 的一个中文坑
> `rep_penalty(text, n=3, cap=0.5)` 用 `re.findall(r"\w+|[^\w\s]", text.lower())` 切词，取全部 3-gram，再用「重复的 gram 数 / 总 gram 数 × cap × 2」线性映射到 $[0, 0.5]$——重复率到 50% 就打满。
> 坑在 `\w+` 会把**一整串连续中文**当成 1 个 token：一句没有标点的中文长句可能只切出 1 个 token，`len(grams) == 0`，函数直接 `return 0.0`，惩罚失效。实验第 3 行用的是英文单词，才触发到 cap。

## reward 落位：只加在最后一个有效 token 上 {#token-rewards}

这三行是今天最需要记住的：

{{source:trainer/train_ppo.py#L136-L138}}

`token_rewards` 先被开成和 `old_resp_logp` 同形状的全 0 张量 `[B, R]`，然后只在 `last_idx = resp_lengths - 1` 这一列上 `+= rewards`。也就是说——**序列级 reward 只落在每条 response 最后一个有效 token 上，其余位置全是 0**。`valid_resp` 是长度大于 0 的过滤器，避免空 response 索引越界。

为什么不平均分摊？因为 reward 评价的是**整条 response**，中间某个 token 好不好，应该由 critic 通过价值函数去判断，而不是由外部奖励硬摊。把 reward 放在末位、再让 GAE 沿时间往回传播，正是「把序列级信号翻译成逐 token 信号」的标准做法——下一节会看到这个传播过程。

这里还有一个和主流实现的**实质差异**：TRL / DeepSpeed-Chat 的经典写法会先写 `token_rewards = -kl_coef * (logp - ref_logp)`，再在末位加上 RM 分数，也就是把**逐 token 的 KL 惩罚折进 reward**、让它参与 GAE 递推。MiniMind **没有**这么做：它的 KL 是以 `args.kl_coef * kl_ref_penalty` 的形式直接加在 loss 上的（L208-L213）。后果是 KL 约束只影响梯度，不影响 advantage 的形状。读 RL 代码时一定要以手上这份源码为准。

{{lab:token_rewards}}

{{quiz:q7}}

> [!KEY]
> `token_rewards` 只有每条序列最后一个有效位非零；MiniMind **没有**把逐 token KL 折进 reward，KL 是 loss 里的一项。

## GAE 的倒序递推 {#gae-recur}

{{source:trainer/train_ppo.py#L140-L147}}

标准 GAE 是 $\delta_t = r_t + \gamma V_{t+1} - V_t$，$\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$。右边依赖 $t+1$，所以只能**倒序**扫一遍（$O(R)$）；`advs_rev` 是从后往前 append 的，最后要 `[::-1]` 翻回来再 `stack`。

默认 `gamma=1.0, lam=0.95`。`gamma=1` 在 LLM 场景很常见：一条 response 就是一个 episode，没有理由对早期 token 和晚期 token 的贡献做时间折扣。此时 $\lambda=1$ 的极限就是蒙特卡洛：$\hat A_t = G - V_t$（实验表格最后一列就是这个对照），$\lambda=0$ 的另一端则退化成单步 TD 误差 $\delta_t$；0.95 是在方差和偏差之间取一个折中。

两个实现上的取舍。第一，`nv = ... if t < gen_len - 1 else 0.0`——**末位一律按 terminal 处理**，不做 bootstrap；即使这条 response 是被 `max_new_tokens` 硬截断的，也当作「到此结束、未来收益为 0」。第二，递推在**全部 `R` 个位置**上跑，包括 EOS 之后的无效位置：它们的 `old_resp_values` 早已被 mask 乘成 0，但 `lastgaelam` 依然会把它们串进来，mask 要到之后的归一化和 loss 里才真正生效。

{{lab:gae_recur}}

{{quiz:q8}}

> [!KEY]
> GAE 必须倒序递推，末位不 bootstrap（`nv=0`）；`gamma=1, lam=0.95` 下 reward 从末位沿时间往回衰减，越靠后（离末位越近）的 token 拿到的 advantage 越接近 $G - V_t$，越靠前的则被衰减得越多。

## 全局归一化与 returns 的算账顺序 {#adv-norm}

{{source:trainer/train_ppo.py#L147-L151}}

归一化这三行有两个容易看漏的点。第一，它是**跨整个 batch 的所有有效 token** 算一个标量均值和方差，**不是** per-sequence：分母是 `resp_policy_mask.sum()`（整个 `[B, R]` 上的有效位总数），不是每行各算各的。所以归一化之后，全局均值是 0，但**每一条序列自己的均值并不为 0**——长序列、短序列之间的相对高低被保留了下来，这正是 PPO 想要的（GRPO 才会做组内归一化，Day 13 讲）。

第二，末尾又乘了一次 `resp_policy_mask`：减去 `adv_mean` 会让原本为 0 的无效位变成非零，必须再清一遍零。

现在看 L147 那一行 `returns = advantages + old_resp_values`。它写在归一化**之前**。那么 critic 拟合的 target `returns`，用的是归一化前还是归一化后的 advantage？

{{predict:p2}}

{{lab:adv_norm}}

实验表格的最后两列给出了答案：`returns` 用的是**未归一化**的 advantage，所以 `returns - old_resp_values` 和归一化后的 `advantages` 并不相等（`allclose` 为 `False`）。这是有道理的：advantage 归一化是为了稳定 policy 梯度的尺度，而 critic 要拟合的是真实的价值尺度，把归一化后的值当 target 会让 critic 学到一个被压缩过的、和 reward 量纲无关的东西。

{{quiz:q9}}

> [!KEY]
> advantage 的均值方差是**跨 batch 全局**算的；而 `returns = advantages + old_resp_values` 发生在归一化**之前**，critic 拟合的是未归一化的 target。

# actor 的更新目标：ratio 与 clip

`no_grad` 块结束，四个 `[B, R]` 的量（`old_resp_logp` / `ref_resp_logp` / `old_resp_values` / `advantages`）都已备好。这一章看 actor 这条梯度通路：`ratio` 从哪来、clip 到底拦住了什么。

## ratio 的起点 {#ratio-one}

进入更新循环后，actor 要在**同一批 rollout 数据**上重新算一遍 logp：

{{source:trainer/train_ppo.py#L174-L181}}

注意这里的写法和 rollout 阶段**不一样**：rollout 走的是 `compute_per_token_logps`（`logits_to_keep=n_keep+1`，只对末段过 `lm_head`），更新走的是「全量 forward → `logits[:, :-1]` → `gather(2, labels)` → `gather(1, logp_pos)`」。之所以要全量 forward，是因为 `logp_pos` 按整条序列的绝对位置算，而且同一份 `full_mask` 还要喂给 critic。两条路数学上等价，但代码路径完全不同。

`log_ratio = mb_resp_logp - old_resp_logp[inds]` 就是 PPO 里的 $\log\frac{\pi_\theta}{\pi_{old}}$，`ratio = exp(log_ratio)` 是下一节 clip 目标的主角。在**第一个 ppo_epoch 的第一个 minibatch**，`actor_optimizer.step()` 还一次都没被调用过，参数完全没变。那么这两条不同代码路径算出来的 logp，逐元素之差会是浮点噪声量级（约 `1e-7`），还是严格的 `0.0`？

{{predict:p3}}

{{lab:ratio_one}}

实测是**逐元素严格等于 0.0**：39 个元素无一例外，`ratio` 的 min 和 max 都是 `1.00000000`，`approx_kl` 和 `clipfrac` 都是 0。两条路径虽然写法不同，但对同一组位置做的浮点运算序列是一致的，所以结果按位相同。源码里 `--debug_log_ratio`（L184-L194）这个开关正是拿来验证这件事的：一旦它打印出明显非 0 的值，就说明训练配置有问题。

{{quiz:q10}}

> [!KEY]
> 第一个 minibatch 里 `ratio ≡ 1`、`approx_kl ≡ 0`，因为 old 和 new 的 logp 来自同一份还没被更新过的参数；偏离 1 只可能来自 dropout、别的后端或已经 step 过。

> [!MORE] 混合精度只作用在 actor 上
> `autocast_ctx` 只包住了 L174-L179 的 actor forward，而 L171-L172 的 critic forward 在它**外面**——也就是说 critic 始终走参数本身的精度。源码没有解释原因（value head 只输出一个标量、用 fp32 更稳是一种合理猜测），但这是读这段时最容易忽略的事实差异。
> 另外 L177-L178 的注释交代了 `log_softmax` 为什么也写在 `autocast` 内：直接对 fp16/bf16 的 logits 做 `log_softmax` 会引入额外的数值偏差。

## clip 的单边性与 clipfrac {#clip-onesided}

policy loss 这一段把三件事写在了一起：

{{source:trainer/train_ppo.py#L205-L213}}

**L210-L212** 是 clip 目标：`max(-A·ratio, -A·clamp(ratio, 1-ε, 1+ε))`，也就是 $-\min(rA, \mathrm{clip}(r)A)$ 的等价写法（源码写的是 loss，所以取 `max` 而不是 `min`）。关键是它**单边**生效：$A>0$ 时只有 `ratio > 1+ε` 那一侧被削平，$A<0$ 时只有 `ratio < 1-ε` 那一侧被削平。换句话说，clip 只拦「往已经走过头的方向继续走」，不拦「往回走」。

**L206** 的 `clipfrac` 用的却是**对称判据** `|ratio - 1| > clip_epsilon`。它统计的是「ratio 跑出 $[0.8, 1.2]$ 的比例」，**不等于**「梯度真被截断的比例」——实验的梯度表里两列正好错开：`ratio=0.5` 时 `A=+1` 的梯度仍是 `-1.0`（没被削），但这一行的 `clipfrac` 是 `True`。它只是个粗糙的监控指标。

**L208-L209** 的 `kl_ref_penalty = mean(exp(d) - d - 1)`（其中 $d = \log\pi_{ref} - \log\pi_\theta$）是 KL 的 **k3 估计量**：无偏、非负、方差比朴素的 $-d$ 小得多，用来估计 $\mathrm{KL}(\pi_\theta \Vert \pi_{ref})$，以 `kl_coef=0.02` 的权重直接加进 `policy_loss`。它和 L195 那个 `approx_kl = mean(0.5·log_ratio²)`（k2 估计量，只做 early stop）是两个不同的量：前者量的是「离 ref 多远」，后者量的是「离本轮的 old policy 多远」。

{{lab:clip_onesided}}

{{quiz:q11}}

> [!KEY]
> clip 是单边的：$A>0$ 只削 `ratio>1+ε`、$A<0$ 只削 `ratio<1-ε`；而 `clipfrac` 用的是对称判据，只是监控指标。KL 以 k3 估计量的形式加在 `policy_loss` 上。

# critic 的 value loss 与整轮更新循环

最后一章收尾：critic 那条梯度通路长什么样，以及 `ppo_update_iters` × minibatch 这个双层循环里，梯度、早停和两个 optimizer 是怎么配合的。

## value loss 的双重裁剪 {#value-clip}

{{source:trainer/train_ppo.py#L214-L217}}

value loss 的结构和 policy 端是对称的：先算未裁剪版本 `(mb_resp_values - returns)²`，再算一个「把新估值钳在 `old_resp_values ± cliprange_value` 之内」的裁剪版本，然后取 `max`——**取更悲观的那一个**。前面乘 `0.5`，进 loss 时再乘 `vf_coef=0.5`。

裁剪的中心是 `old_resp_values`，也就是 rollout 阶段那次 `no_grad` 的 critic 输出。含义是：同一批数据要被重复用 `ppo_update_iters` 轮，不能让 critic 在这几轮里一口气跑太远。

这里有个和 policy clip 一样、但更容易被忽略的后果：**裁剪支在 $|v - v_{old}| > \text{cliprange}$ 时是个常数**，梯度为 0；一旦它比未裁剪支更大而被 `max` 选中，整项的梯度就被削成 0。实验里 `old_value=0.5, cliprange=0.2, returns=1.5`，`v` 一旦超过 0.7，loss 就被钉在常数 0.32 上、梯度归零，一直到 `v` 跑出 2.3 之外未裁剪支才重新胜出——也就是说 critic 每一步最多只能往 target 方向挪 `cliprange_value`。

还有一处细节：`mb_resp_values` 在 L172 `gather` 出来之后**没有**乘 mask（不像 `old_resp_values` 在 L133 乘过），mask 是在 loss 求和这一步才乘上去的，效果一样。

{{lab:value_clip}}

{{quiz:q12}}

> [!KEY]
> value loss 也取「更悲观的那一个」：裁剪中心是 `old_resp_values`，一旦新估值偏离超过 `cliprange_value` 且裁剪支胜出，梯度就被削成 0。

## 双层更新循环、早停与两个 optimizer {#update-loop}

`no_grad` 块之外的全部代码，就是这个双层循环：

{{source:trainer/train_ppo.py#L164-L169}}

外层 `ppo_update_iters=2` 表示**同一批 rollout 数据重复用两轮**——这就是 PPO「有限 off-policy 复用」的来源，也是正常配置下 `ratio` 会偏离 1 的唯一来源。内层用 `b_inds = torch.randperm(B)` 在 **batch 维**（不是时间维）上切 minibatch。

一次 `loss.backward()` 同时喂两张计算图：梯度经 `ratio` 流回 actor，经 `mb_resp_values` 流回 critic。两个 optimizer 共用同一个 `grad_accum_step` 计数器，在 L240-L248 一起 `step()`；L250-L258 还有一段收尾，最后一次累积没凑满 `accumulation_steps` 时补一次 step，否则这批梯度就白算了。

早停的写法很特别：`approx_kl` 超过 `early_stop_kl=0.25` 时并**不**当场 `break`，而是把 loss 乘 0——

{{source:trainer/train_ppo.py#L222-L228}}

因为 DDP 下必须保证**每张卡都走完一次 forward-backward**，否则梯度 all-reduce 会对不上（`approx_kl` 本身也在 L199-L200 做了 `all_reduce(AVG)`，防止各卡判断不一致）。真正的 `break` 要等到下一轮 ppo_epoch 开头的 L165-L166。

{{lab:update_loop}}

{{quiz:q13,q14}}

> [!KEY]
> `ppo_update_iters × minibatch` 双层循环里，一次 `backward()` 同时更新 actor 和 critic；早停靠把 loss 乘 0 实现，为的是不破坏 DDP 的 forward-backward 闭环。

> [!MORE] 日志里那六个指标怎么读
> | 指标 | 含义 | 健康的样子 |
> |---|---|---|
> | `Reward` | 本 step 的 `rewards.mean()` | 缓慢上升（README 说 PPO 涨得慢） |
> | `KL_ref` | `kl_ref_penalty`，与参考策略的距离 | 缓慢增大但不爆炸 |
> | `Approx KL` | `0.5·log_ratio²` | 远小于 `early_stop_kl=0.25` |
> | `ClipFrac` | ratio 跑出 `[0.8, 1.2]` 的比例 | 几个百分点 |
> | `Critic Loss` | value loss | 从随机 head 开始下降 |
> | `Avg Response Len` | `resp_lengths.float().mean()` | 盯着它防 reward hacking |
>
> 另外 L260 的 `rollout_engine.update_policy(actor_model)` 每 `save_interval` 步调一次：torch 引擎下只是重新绑定引用（几乎是 no-op），SGLang 引擎下才是真正把权重落盘并推给推理服务。Day 13 会细讲 rollout engine 的整体设计。
