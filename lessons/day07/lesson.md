---
day: 7
format: points
title: "推理：generate 循环、KV cache 与采样"
subtitle: "prefill 一次 → 每步只喂 1 个 token → logits[:, -1] 被四步依次削 → 抽一个拼回去"
minutes: 90
mainline: "一个 prompt [B,T] 怎么变成一串 token：prefill 一次建起 KV cache，之后每步只喂 [B,1]，取 logits[:, -1] 过一遍 temperature/top-k/top-p，抽一个拼回 input_ids"
files:
  - model/model_minimind.py#L255-L288
  - model/model_minimind.py#L111-L134
  - model/model_minimind.py#L209-L219
  - eval_llm.py#L62-L91
goals:
  - 能说出 generate 的 for 循环里每一步喂进 forward 的 input_ids 是什么 shape，以及 past_len 是怎么算出来的
  - 能指出 KV cache 在 Attention.forward 的哪一行被拼长、存的是哪个阶段的形态、shape 为什么是 [B, T, KV, D]
  - 能解释 start_pos 为什么必须存在，以及它切的是 RoPE 表的哪一段
  - 能把同一条分布交给 temperature / top-k / top-p，预测各自留下几个候选、排序会不会变
  - 能读懂 top-p 那四行里的「右移一位 + scatter」和 repetition_penalty 里的 torch.where，并说清去掉会怎样
---

Day 5 那次 forward 是**训练**时的用法：`labels` 在手边，模型不用自己「决定」下一个词。推理反过来——只有一个 prompt，下一个 token 得模型算出来、选出来，再喂回去算下一个。

今天跟着走的是一整条链路：`input_ids [B,T]` 进 `generate`，第 0 步一次吃下整个 prompt（prefill）并把每层的 K/V 存进 cache；之后每步只喂 1 个新 token（decode），出来的 `[B,1,C]` 过 `lm_head` 取 `logits[:, -1, :]` → `[B,V]`，被 temperature、repetition_penalty、top-k、top-p 依次改造，最后抽一个拼回 `input_ids`，直到所有行都吐出 eos。

五章的分工：第一章走通循环骨架；第二章钻进 KV cache——在哪一行被拼长、`start_pos` 怎么让 RoPE 接得上、关掉它要多算多少；第三章把同一条真实分布交给 temperature / top-k / top-p 依次削一遍；第四章拆两段最容易读错的代码；第五章接回命令行。

今天没有新的网络结构，变的只有一件事：**同一份代码被用两种方式调用**。

{{flow: prompt [B,T] | *prefill 一次* | KV cache [B,T,KV,D] | *decode 每步 [B,1]* | logits[:, -1] [B,V] | *temperature / top-k / top-p* | next_token [B,1]}}

# generate 的主循环

`generate` 一共 34 行（L255-L288），整个身体是一个 `for _ in range(max_new_tokens)`。这一章先不碰 cache 和采样，只看循环骨架：每一步到底喂进去多长的 `input_ids`、batch 里有人先说完了怎么办、以及 `num_return_sequences` 那行 `.repeat` 把行排成了什么样。

## 一行切片分出的 prefill 与 decode {#gen-loop}

循环体的前两行就把 prefill 和 decode 的区别定死了，而且它们长得一模一样——没有 `if is_prefill` 这种分叉。

{{source:model/model_minimind.py#L263-L266}}

先看 `past_len`。MiniMind 的 cache 格式是最朴素的 `list[tuple(k, v)]`，每层一个元组。取第 0 层的 K，它的 shape 是 `[B, T_past, KV, D]`，所以 `shape[1]` 就是「前面已经算过多少个位置」。`past_key_values` 在第 0 次循环还是 `None`（falsy），于是 `past_len = 0`。

再看 `input_ids[:, past_len:]`：**只喂还没算过的那一段**。第 0 次 `past_len=0`，切出来就是整个 prompt `[B, T]`——这就是 prefill，它顺带产出第 1 个新 token。这一步结束后 `past_key_values` 被填上，`past_len` 变成 prompt 长度；而循环末尾 L280 刚把新采的 token `cat` 进 `input_ids`，让它比 `past_len` 长了恰好 1。于是从第 1 次循环起，切片结果永远是 `[B, 1]`——这就是 decode。

所以循环一共跑 `max_new_tokens` 次，其中**第 0 次是 prefill、后面 `max_new_tokens - 1` 次是 decode**，不是 `max_new_tokens + 1` 次。这个 off-by-one 是读这段代码最常见的第一个坑。

L266 那行是给 `attention_mask` 补位：每步在末尾接一个 1，让它的长度始终跟 `input_ids` 的**总长度**对齐——哪怕这一步只喂了 1 个 token，mask 记的是「全部历史」的可见性。`attention_mask` 传 `None` 时（下面的实验就是）整行短路。

实验在 `MiniMindModel.forward` 上挂了个 pre-hook，把每一步真正喂进去的 shape 和 `past_len` 记下来：

{{lab:gen_loop}}

{{quiz:q1,q2}}

> [!KEY]
> prefill 和 decode 不是两段代码，只是同一行 `input_ids[:, past_len:]` 在 `past_len=0` 和 `past_len=已算长度` 下的两种结果：一次 `[B,T]`，之后每次 `[B,1]`。

> [!MORE] prefill 那一步算了多少用不上的 logits
> `generate` 调 `self.forward(...)` 时没有传 `logits_to_keep`，用的是默认值 `0`，于是 `slice(-0, None)` 等价于 `slice(0, None)`——`lm_head` 会对 prompt 的**每一个位置**都算一遍 `[C] → [V]`，得到 `[B, T, 6400]`，然后 L267 只取最后一个位置，其余全扔。22 个 token 的 prompt 就浪费了 21/22 的 `lm_head` 计算。`forward` 本身是支持 `logits_to_keep=1` 的（Day 5 讲过这个参数），`generate` 只是没用上；HF 的 `generate` 在 prefill 时会传这个参数。小模型上无所谓，`V` 上十万之后这一项就不能忽略了。

## finished 向量与 batch 的收尾 {#gen-eos}

`generate` 支持一次跑一个 batch，但 batch 里各行的回答长度必然不同：有的十几个 token 就说完了，有的要几十个。麻烦在于 `input_ids` 是一个稠密张量——**每一步所有行必须一起拼上等长的新列**，没办法让某一行「原地停止增长」。

{{source:model/model_minimind.py#L278-L285}}

机制有三处。L261 起了一个 `finished = torch.zeros(B, dtype=torch.bool)`；L284 每步末尾 `finished |= next_token.squeeze(-1).eq(eos_token_id)`，记下谁吐过 eos；L285 `if finished.all(): break`——**所有行都结束了才退出循环**，只要还有一行在说，其余行都得陪着跑完。

那些已经结束、却还在被拼列的行怎么办？答案在 L279：`next_token` 在 `cat` 进去之前，先被 `torch.where(finished.unsqueeze(-1), ...)` 过了一手。注意执行顺序——先 `where` 覆盖（L279）、再 `cat`（L280）、最后才更新 `finished`（L284）。所以某一行**第一次**采到 eos 的那一步，eos 是模型真采出来的；从下一步起，这一行拼进去的就是覆盖的结果了。

下面的实验用同一个 prompt 复制成两行、开随机采样，让两行在不同的步数结束。先猜：先说完的那一行被标成 `finished` 之后，接下来几步拼进它 `input_ids` 的是什么？

{{predict:p1}}

{{lab:gen_eos}}

实验里第 1 行在第 16 步就采到了 eos，但循环一直跑到第 27 步（第 0 行结束）才停。第 1 行那 28 个位置里 `id=2` 出现了 **12 次**，尾部 5 个 token 全是 `[2, 2, 2, 2, 2]`——已完成的行被**重复的 eos 一路填平**到和最长的那行等长，而不是填 pad、也不是停止增长。所以读结果时必须 `skip_special_tokens=True` 或者按第一个 eos 截断，不能拿行长度当作真实生成长度。

{{quiz:q3}}

> [!KEY]
> `finished` 让已结束的行每一步都被强制写入 `eos_token_id`，直到全 batch 都结束；`input_ids` 的行长度是「最长那一行」的长度，不是各自的真实长度。

## num_return_sequences 的行排列 {#gen-repeat .side}

想为同一个 prompt 拿多个候选时，`generate` 没有另开一套逻辑，而是在入口处把 batch 复制几份，让它们在同一个 `[B·n, ...]` 的张量里并行跑完。

{{source:model/model_minimind.py#L258-L259}}

关键是 `Tensor.repeat` 的语义：`x.repeat(3, 1)` 是把整个 `[B, T]` 张量沿 dim0 **首尾接 3 遍**（tile），得到 `[p0, p1, p0, p1, p0, p1]`；而不是逐行重复成 `[p0, p0, p0, p1, p1, p1]`——后者是 `repeat_interleave` 的语义。所以 `B=2`、`num_return_sequences=3` 时，prompt1 的三个候选落在第 **1、3、5** 行，步长等于原始 batch size。取错行是个静默错误：形状对、数值也像模像样，只是候选配错了 prompt。

L259 把 `attention_mask` 也照同样的方式 repeat——两者必须同步，否则 L266 每步 `cat` 出来的 mask 行数和 `input_ids` 对不上。

多个候选之所以会走出不同的结果，是因为采样是**逐行独立**的：`torch.multinomial` 作用在 `[B·n, V]` 上，每一行各抽各的。这正是上一节实验的做法——同一个 prompt 复制两行，靠采样的随机性造出两条长度不同的回答。`do_sample=False` 时就没有意义了：贪心下 n 份复制会生成完全一样的东西。

{{lab:gen_repeat}}

{{quiz:q4}}

> [!KEY]
> `.repeat(n, 1)` 是整块 tile 不是逐行重复，同一个 prompt 的 n 个候选在第 `i, i+B, i+2B, …` 行；要连续排列得用 `repeat_interleave`。

# KV cache：只喂一个 token，却看得见全部历史

decode 每步只喂 1 个 token，模型却要让它对**全部历史**做 attention。补上这个落差的就是 KV cache。这一章分三步：先在 `Attention.forward` 里找到它被拼长的那一行（顺便看清它存的是哪个阶段的形态），再看 `start_pos` 怎么让 RoPE 接得上，最后把它关掉，量一量它到底省下了什么。

## xk 在 dim=1 上被接长 {#kv-cat}

`Attention.forward` 里和 cache 有关的只有 5 行，位置卡得非常讲究：在 RoPE 之后、`repeat_kv` 之前。

{{source:model/model_minimind.py#L119-L124}}

走到 L119 之后，`xk` 已经过完 `k_proj` → `view` → `k_norm` → RoPE，shape 是 `[B, T_new, KV, D]`——**这就是要进 cache 的形态**。L120-L122 把上一步存下来的 `past_key_value[0]` 和它在 `dim=1`（T 维）拼起来：prefill 时 `past_key_value is None`，整个 if 跳过；decode 时每次拼上 1 个，T 维就这样一步一格地长。

为什么存 RoPE **之后**的？因为 RoPE 编的是绝对位置：第 t 个 token 的 K 一旦转到位就永远是那个角度，后面的步不需要重转。要是存 RoPE 之前的，每一步都得把整段历史重新转一遍，cache 就白存了。

L123 `past_kv = (xk, xv) if use_cache else None` 是另一个关键位置：它在 L124 的 `repeat_kv` **之前**。所以 cache 里存的是按 KV 头数算的 `[B, T_total, KV=4, D]`，而不是撑开之后按 Q 头数算的 `[B, H=8, T_total, D]`。这两行的顺序一旦调换，cache 的体积直接翻 `n_rep` 倍（默认配置下就是 2 倍）——`repeat_kv` 的结果只在这一次 attention 里活着，算完就扔。

实验只留 1 层，用调用切换器可以在 prefill 和两次 decode 之间翻页，盯住 `xk` 的第 1 维：

{{lab:kv_cat}}

第 1 次调用（prefill）：L119 之后 `xk` 是 `[B=3, T=11, KV=4, D=96]`，L121 那行根本没执行。第 2 次调用：L119 之后是 `[3, 1, 4, 96]`，L121 拼完变成 `[3, 12, 4, 96]`。第 3 次：`[3, 13, 4, 96]`。

{{quiz:q5,q6}}

> [!KEY]
> cache 存的是 RoPE 之后、`repeat_kv` 之前的 `[B, T_total, KV, D]`，在 `Attention.forward` 里按 `dim=1` 拼长；先存再 repeat，是 GQA 能省一半 cache 的前提。

> [!MORE] 一份 KV cache 要多少显存
> 每层 K、V 各一份 `[B, T, KV, D]`，所以 $\text{bytes} = 2 \times L \times B \times T \times \text{KV} \times D \times \text{bytes\_per\_elem}$。
> 代入 MiniMind 默认配置（`L=8`, `KV=4`, `D=96`）、`B=1`、`T=4096`、fp16：$2 \times 8 \times 1 \times 4096 \times 4 \times 96 \times 2 = 50{,}331{,}648$ 字节 ≈ **48 MB**。
> 如果不做 GQA（`num_key_value_heads = num_attention_heads = 8`），同样的条件是 96 MB——**GQA 省显存省的主要就是这一块**，注意力本身的计算量并没有减少（`repeat_kv` 之后照样是 H 个头在算）。极端一点把 `num_key_value_heads=1`（MQA）就只剩 12 MB。cache 对 `T` 是线性增长的，而且和模型参数量无关：长上下文服务里它经常比权重还占地方。

## start_pos 决定 RoPE 从哪一格切起 {#kv-startpos}

`Attention.forward` 收到的 `position_embeddings` 是上游切好的，它自己不知道当前 token 在第几位。真正做决定的是 `MiniMindModel.forward` 里的两行。

{{source:model/model_minimind.py#L213}}

先看条件：L212 已经把 `None` 换成了 `[None] * len(self.layers)`，所以这里判断的是 `past_key_values[0] is not None`（第 0 层有没有 cache），而不是整个列表是否为空。取第 0 层 K 的 `shape[1]` 就是「cache 里已经有多少个位置」——所有层的 cache 长度必然一致，看一层就够。

{{source:model/model_minimind.py#L219}}

RoPE 表 `freqs_cos` 是 `[max_position_embeddings, D] = [32768, 96]` 的常量表（Day 3 讲过它怎么算出来的），第 `i` 行对应绝对位置 `i`。prefill 时 `start_pos=0`、`seq_length=T`，切出 `[T, 96]`；decode 时 `seq_length=1`，切出的是 `freqs_cos[start_pos:start_pos+1]` 这**一行**——恰好是这个新 token 的真实位置。

这就是 `start_pos` 非存在不可的原因：decode 阶段喂进来的 `input_ids` 只有 1 个 token，如果按老办法从 0 开始切，每个新 token 都会被当成句首来旋转，它和 cache 里那些按真实位置转过的 K 做内积，得到的分数完全是错的。切片切错一格不会报错，只会让模型开始胡说。

另外，这一次切片的结果会**原样传给全部 L 层**（Day 2 提过 RoPE 表不是逐层参数），所以每步只算一次。

{{lab:kv_startpos}}

三次调用里 `start_pos` 依次是 `0 / 11 / 12`，`position_embeddings` 里两个 tensor 的第 0 维是 `11 / 1 / 1`，`input_ids` 是 `[3,11] / [3,1] / [3,1]`——三者严丝合缝。

{{quiz:q7}}

> [!KEY]
> `start_pos = past_key_values[0][0].shape[1]`，RoPE 表从这一格往后切 `seq_length` 行；decode 时切的是 1 行，正好对上这个新 token 的绝对位置。

> [!MORE] L211 那行防御在防谁
> `if hasattr(past_key_values, 'layers'): past_key_values = None`。`MiniMindForCausalLM` 仍然继承着 `transformers.GenerationMixin`（L234），万一有人通过 HF 的标准路径调用、或者手动塞进来一个 `DynamicCache`（这类对象有 `.layers` 属性），MiniMind 的 `Attention.forward` 只会按 `past_key_value[0]` / `[1]` 这种元组下标取值，根本读不懂它。所以这里认出不是自己家的格式就直接丢弃、退化成不用 cache，而不是让后面崩在一个莫名其妙的地方。这一行本身就是「两套 cache 格式对不上」的直接证据。

## 关掉 cache 之后多算了多少 {#kv-win}

`use_cache` 是 `generate` 的参数，默认 `True`。关掉它只影响一行：

{{source:model/model_minimind.py#L280-L281}}

`use_cache=False` 时 `past_key_values` 每步都被设回 `None`，于是下一轮 `past_len` 恒等于 0，`input_ids[:, 0:]` 就是**当前为止的完整序列**。结果依然是对的——因果 mask 保证第 t 个位置的输出只依赖前 t 个 token，重算一遍得到的还是同一个值——只是把前面所有 token 的 K/V 又算了一遍。

算笔账。有 cache 时送进 `embed_tokens` 的 token 总数是 `P + (N-1)`（prefill 的 P 个，加 N-1 步各 1 个）；无 cache 时第 k 步要喂 `P+k` 个，总数是 $\sum_{k=0}^{N-1}(P+k) = N\cdot P + \frac{N(N-1)}{2}$。`P=22, N=64` 代进去是 85 和 3424，差 **40 倍**。

那么墙上时间会慢多少？先下注，再跑。

{{predict:p2}}

{{lab:kv_win}}

实测只慢 3~4 倍，和 40 倍的计算量差距对不上。原因在于 decode 的每一步都是一个 `[B, 1, C]` 的极瘦矩阵乘：算力根本喂不饱，时间花在逐层的 kernel 启动、权重搬运上，属于访存受限。而无 cache 那条路每步喂的是 `[B, P+k, C]` 的胖矩阵，单位算力的效率高得多。换句话说，**KV cache 省掉的那 40 倍计算量，本来就是最便宜的那种计算量**。真实推理系统靠 batch / continuous batching 把 decode 的矩阵重新变胖，正是冲着同一个问题去的。

顺带确认了正确性：贪心模式下两条路径生成的 token 序列逐位相同。

{{quiz:q8}}

> [!KEY]
> 关掉 cache 结果照样对，只是每步把整段历史重算一遍；计算量差 40 倍，但 decode 是访存受限的瘦矩阵乘，墙上时间只差 3~4 倍。

# 把一条分布削成能抽签的样子

`logits[:, -1, :]` 出来之后、`multinomial` 之前，源码里排着一行除法和三个 `if`。这一章用同一条真实分布——`full_sft` 已经写出「秋」，正在想下一个字——依次过一遍 temperature、top-k、top-p，看它们各自把这条曲线改成什么形状、留下几个候选。

## temperature：除在 logits 上的一个数 {#sample-temperature}

采样链路的第一行同时干了两件事，而且两件都值得看清楚。

{{source:model/model_minimind.py#L267}}

`outputs.logits` 的 shape 是 `[B, T_step, V]`，`[:, -1, :]` 只取**最后一个位置** → `[B, V]`。prefill 那一步 `T_step` 是整个 prompt 长度，取最后一个；decode 那一步 `T_step=1`，取的还是它自己。从这里开始，后面所有采样代码面对的都是 `[B, V]`。

然后是 `/ temperature`。它的作用要放进 softmax 里看：$p_i \propto \exp(z_i/\tau)$。$\tau < 1$ 把 logits 之间的差距**放大**，指数一放大，分布就变尖；$\tau > 1$ 压缩差距，分布变平。$\tau \to 0$ 等价于 argmax，$\tau \to \infty$ 等价于均匀分布。

有一条性质要记住：**除以一个正数是单调变换，不改变排序**。第 2 名永远不会因为调温度就超过第 1 名。所以下一节的 top-k（按名次截断）完全不受 temperature 影响，而 top-p（按概率质量截断）会被它剧烈影响——同样 `top_p=0.85`，$\tau=0.5$ 时只留 1 个候选，$\tau=1$ 留 2 个，$\tau=2$ 留 993 个。

还有个坑：`temperature=0` 会除零，源码没有任何防护。想要贪心请用 `do_sample=False`（L278 走 `argmax` 分支），不要试图用 `temperature=0` 去模拟。

{{lab:sample_temperature}}

实测同一条分布：`τ=1.0` 时 top1 概率 0.752、前 8 个候选合计 0.982；`τ=0.5` 时 top1 涨到 0.946；`τ=2.0` 时 top1 掉到 0.191，而且前 8 个合计只剩 0.395——被摊平的概率跑到长尾的几千个词上去了。

{{quiz:q9}}

> [!KEY]
> `temperature` 除在 logits 上，只改分布的尖锐程度、不改排序；`τ<1` 变尖、`τ>1` 变平，`τ=0` 是除零错误而不是贪心。

## top-k：固定留 k 个，其余置 -inf {#sample-topk}

top-k 只有一行，但这一行把 `topk`、广播比较和布尔索引赋值叠在了一起。

{{source:model/model_minimind.py#L271-L272}}

拆开看：`torch.topk(logits, top_k)[0]` 取出 values，shape 是 `[B, k]`；`[..., -1, None]` 取每行第 `k` 大的那个值、并补回一个长度 1 的维度 → `[B, 1]`；`logits < 阈值` 广播成 `[B, V]` 的布尔 mask；最后用它做索引赋值。

为什么是置 `-inf` 而不是把这些词删掉？因为词表大小不能变——`multinomial` 要的是一个长度为 `V` 的概率向量，`next_token` 拿到的下标必须能直接当 token id 用。`softmax(-inf) = 0` 正好让被削掉的词概率精确归零，剩下的自动重新归一化，不需要另外除一次。

两个细节。一是比较用的是 `<` 而不是 `<=`：并列第 k 名会一起留下，实际候选数可能略多于 `k`（连续 logits 上几乎不会发生）。二是开关：`if top_k > 0` 才执行，所以关掉 top-k 要传 `top_k=0`。`generate` 的默认值是 `top_k=50`，而 `eval_llm.py` 没有开放这个命令行参数——也就是说命令行推理时 top-k=50 **一直在生效**，只是你看不见它。

{{lab:sample_topk}}

实测 `top_k=3`：6400 个位置里 6397 个变成 `-inf`，剩下三个重新归一化，`0.752 → 0.785`、`0.178 → 0.186`、`0.028 → 0.029`。

top-k 的毛病也正在这个「固定」上：这条分布前两个词就占了 93%，默认的 `k=50` 等于白留了 48 个几乎不可能的候选；而分布很平时它又可能把真正合理的词砍掉。下一节的 top-p 就是冲着这一点来的。

{{quiz:q10}}

> [!KEY]
> top-k 按名次截断：把小于「第 k 大」的位置全置 `-inf`，softmax 之后它们的概率精确为 0；`k` 是固定数量，不随分布形状变。

## top-p：固定留多少概率质量 {#sample-topp}

top-p（nucleus sampling）换了一个截断标准：不管留几个词，只管留下的这些词加起来占多少概率。

{{source:model/model_minimind.py#L274-L275}}

两行做的事：先按 logits 降序排（`sorted_indices` 记住每个名次原来是词表里的哪个 id），对排序后的 logits 做 softmax 再 `cumsum`，得到「前 j 名累计占多少概率」，然后标出所有 `> top_p` 的位置。核心区别就在这里——**k 是数量，p 是概率质量**。分布尖锐时几个词就凑够了 `top_p`，核很小；分布平坦时要很多词才凑得够，核很大。它自己会随分布调节。

还有一个容易漏的顺序问题：top-k 在前、top-p 在后。被 top-k 打成 `-inf` 的词，在 top-p 这一步 softmax 出来是 0，不占概率预算。所以两者叠加的实际效果是「先砍到最多 50 个，再在这 50 个里按概率质量砍」。

下面的实验把同一个 `top_p=0.85` 用在两条尖锐程度完全不同的分布上：一条是已经写出「秋」之后（top1 概率 0.752），一条是已经写出「天空是蓝色的，因为」之后（top1 只有 0.208）。各会留下几个候选？

{{predict:p3}}

{{lab:sample_topp}}

尖锐的那条只留下 **2** 个（`天` 和 `风`，因为 0.752 + 0.178 = 0.930 就已经越过 0.85），平坦的那条留下 **16** 个。同一个参数、同一个模型，差了 8 倍——这就是「自适应」的含义，也是 top-p 比 top-k 更常用的原因。

{{quiz:q11}}

> [!KEY]
> top-p 按概率质量截断：排序 → 累加 → 越过 `top_p` 就停。同一个 `top_p` 在尖锐分布上只留几个、在平坦分布上留几十个，候选数随分布自适应。

# 两段容易读错的代码

top-p 的意图上面讲清楚了，但它那四行里还压着两处极紧的写法——右移一位和 `scatter`，第一次读几乎必错。另一处是 `repetition_penalty` 里的 `torch.where`：看起来只是个可有可无的小分支，去掉却会让惩罚变成奖励。这一章各拆一节，都配一个能直接对着数字看的小实验。

## 右移一位与 scatter 回原顺序 {#topp-scatter}

`mask = cumsum > top_p` 标出来的是「算上它，累计已经超过阈值」的位置。问题是：那个**刚好把累计推过阈值的词自己**也在里面，而它必须保留——不然剩下的加起来凑不够 `top_p` 的概率质量，语义就不对了。

{{source:model/model_minimind.py#L276-L277}}

L276 干了两件事。`mask[..., 1:] = mask[..., :-1]` 把整条 mask **右移一位**，判定标准从「它自己越过了」变成「它前面那个已经越过了」，于是越界的那一个被放行。`.clone()` 不能省——这是同一个张量的自赋值，不 clone 会边写边读到刚改过的值。`mask[..., 0] = 0` 是**保底**：当 `top_p` 比第一名自己的概率还小时（比如 `top_p=0.3` 而 top1 有 0.5），右移之后第 0 位依然是 `True`，那就一个候选都不剩，`softmax` 全 `-inf` 会产出 `nan`，`multinomial` 直接抛 `RuntimeError: probability tensor contains either inf, nan or element < 0`。强制第 0 位为 `False` 保证至少留 1 个。顺带一提，这一整行是元组赋值：右边两个表达式先全部求值，再从左到右赋值——所以「先右移、后清零」的顺序不能反。

L277 的 `scatter` 是坐标系转换。`mask` 是按**排序后的名次**排的，而 `logits` 是按**原始词表顺序**排的，不散回去就索引错位。`mask.scatter(1, sorted_indices, mask)` 的语义正是 `out[i][sorted_indices[i][j]] = mask[i][j]`——把第 j 名的判定写回它原本的词表位置。

{{lab:topp_scatter}}

6 个词、`top_p=0.9` 的表：排序后 `cat(累计 0.500) → dog(0.804) → the(0.916，越过阈值但被右移放行) → sat / on / mat 被 -inf`，最终留下 3 个。右半边就是 `scatter` 之后按原始词表顺序重排的同一份 mask。

{{quiz:q12}}

> [!KEY]
> 右移一位是为了保留「刚好越过阈值」的那个词，`mask[..., 0] = 0` 是为了保证至少留 1 个候选；`scatter` 负责把按名次算出来的 mask 送回原始词表顺序。

## repetition_penalty 的正负分叉 {#rep-penalty}

`repetition_penalty` 三行代码，核心是一个 `torch.where`。这个分支不是优化，是必需品。

{{source:model/model_minimind.py#L268-L270}}

先看范围：`seen = torch.unique(input_ids[i])` 是**这一行从 prompt 到目前为止出现过的所有 token id**，去重后一视同仁。prompt 里的词也算，chat 模板的特殊 token 也算；出现 1 次和出现 10 次罚得一样重，也没有距离衰减或者窗口。

再看 `torch.where(score > 0, score / rp, score * rp)`：正 logit 除以 `rp`、负 logit 乘以 `rp`。为什么不能一律相除？因为负数除以一个大于 1 的数，绝对值会变小、往 0 靠——softmax 之后这个词的概率反而**升高**了，和「惩罚重复」完全相反。分正负两条路，才能保证不管原始 logit 是什么符号，`rp > 1` 都让它更负、概率更低。

还有个位置问题：这段在 `/ temperature` **之后**、top-k 之前执行，罚的是已经被温度缩放过的 logits。所以同一个 `repetition_penalty` 配不同的 `temperature`，实际力度并不相同。

{{lab:rep_penalty}}

实测 `rp=1.5`：`</think>` 的 logit 是 `-7.08`，源码写法变成 `-10.62`、概率降到 **0.10 倍**；一律相除会变成 `-4.72`、概率涨到 **35.36 倍**。正 logit 那两行两种写法完全一样（`+18.68 → +12.45`，概率 0.01 倍）——错误只发生在负 logit 上，而且是「越罚越容易被选中」这种最难察觉的方向。

顺带说一句，`if repetition_penalty != 1.0` 让这段默认整体短路，而 `eval_llm.py` 把它硬编码成了 `1`（L86，命令行也没开放这个参数）。这就是为什么实验里那段贪心生成会原样复读一遍「北京是中国的首都，也是中国的经济、文化、科技中心之一。」

{{quiz:q13}}

> [!KEY]
> `torch.where(score > 0, score / rp, score * rp)`：正 logit 除、负 logit 乘。统一相除会让负 logit 的词概率不降反升（实测 ×35），惩罚变成奖励。

# 接回命令行：streamer 与 eval_llm.py

到这里 `generate` 的 34 行全部拆完了。最后两节把它接回真实用法：一个是让字一个一个往外蹦的 `streamer`，一个是 `eval_llm.py` 这个命令行入口——它决定了喂给模型的到底是一段等着被续写的文本，还是一轮对话。

## streamer.put 的两处时机 {#streamer-put}

命令行里「字一个个往外蹦」不是 `generate` 自己打印的，它只负责在三个位置回调一个外部对象。

{{source:model/model_minimind.py#L282-L286}}

三处调用：L262 在**进入循环之前**把整个 `input_ids` 推一次；L282 每步推一次新采的 `next_token`；L286 循环结束调 `streamer.end()`。

第一次为什么要把 prompt 也推过去？因为 `TextStreamer(skip_prompt=True)` 是靠「第一次收到的那一批不打印」来跳过 prompt 的。如果不推这一次，它会把第 1 个生成的 token 当成 prompt 吞掉，输出少一个字。`.cpu()` 则是因为 streamer 内部要调 tokenizer 解码，而 tokenizer 只认 CPU 上的 id。

接口薄得惊人：只要有 `put(tensor)` 和 `end()` 两个方法就能塞进来，不需要继承任何基类——下面实验里那个五行的 `Recorder` 就是证据。

{{lab:streamer_put}}

prompt 22 个 token、`max_new_tokens=6`，结果是 **7 次 put + 1 次 end**：第 0 次收到 `[1, 22]`（整个 prompt），之后 6 次每次 `[1, 1]`。所以 put 的次数是 `1 + 实际循环步数`，第一次的 shape 和后面都不一样——自己写 streamer 时这是最容易踩的地方。

{{quiz:q14}}

> [!KEY]
> `streamer` 的接口只有 `put` / `end`；`generate` 在循环前 put 一次整个 prompt（`[B, P]`），之后每步 put 一个 `[B, 1]`，`TextStreamer(skip_prompt=True)` 正是靠第一次那批来跳过 prompt。

> [!MORE] 为什么 eval_llm.py 每次只喂一条 prompt
> `TextStreamer.put` 的第一行就是 `if len(value.shape) > 1 and value.shape[0] > 1: raise ValueError("TextStreamer only supports batch size 1")`。流式输出本质上是往一个终端里按顺序写字，多行同时流根本没法在一个屏幕上表达。所以 `eval_llm.py` 的 `for prompt in prompt_iter` 一次只处理一条。想批量推理就别传 `streamer`，`generate` 里三处 `if streamer:` 会全部短路。

## pretrain 拼 bos，full_sft 套 chat 模板 {#eval-prompt}

`eval_llm.py` 在调 `generate` 之前有一个分叉，决定了喂给模型的文本长什么样——选错了模型就不会「回答」。

{{source:eval_llm.py#L73-L76}}

判断依据是**权重文件名**：`args.weight` 里带 `pretrain` 就只拼 `tokenizer.bos_token + prompt`（bos 就是 `<|im_start|>`，id=1），否则走 `apply_chat_template(..., add_generation_prompt=True, open_thinking=...)`。

理由很直接：pretrain 权重的训练目标只有「续写下一个 token」，它从没见过 `<|im_start|>user … <|im_start|>assistant` 这套结构，套上去等于喂一堆没有意义的特殊 token。而 `full_sft` 是在这个格式上微调出来的，不套模板它就不知道「该轮到我回答了」。

`add_generation_prompt=True` 除了拼上 `<|im_start|>assistant\n`，在 `open_thinking=False`（默认）时还会补一个**空的** `<think>\n\n</think>\n\n`——注意这是**输入**的一部分，不是模型生成的。

{{lab:eval_prompt}}

同一个「请用一句话介绍北京」：pretrain 续写出「的文化。北京是中国的首都，有着悠久的历史和丰富的文化遗产，如故宫、天坛、颐和园等。」——语法通顺，但它在**接话**；full_sft 直接给出一句回答。这也是为什么 pretrain 权重不能拿去做问答式评测。

{{quiz:q15}}

> [!KEY]
> 权重名里带 `pretrain` 就只拼 `bos_token + prompt` 当续写用，否则套 chat 模板；`add_generation_prompt=True` 补出来的空 `<think></think>` 属于输入，不是模型写的。

> [!MORE] 这个脚本里另外三行
> `conversation = conversation[-args.historys:] if args.historys else []`（L71）：`--historys` 默认 0，**每轮都清空历史**，模型只看当前这一句；传 N>0 时保留最近 N 条消息（用户和助手合起来算，所以帮助文字里写「需为偶数」）。
> `open_thinking=1` 时模板改成只插一个未闭合的 `<think>\n`，推理内容和 `</think>` 留给模型自己接着生成。
> `tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)`（L88）：按 prompt 长度切掉输入部分再 decode，`skip_special_tokens=True` 顺手丢掉结尾的 eos。
