---
day: 2
format: points
title: "骨架：Config、Embedding、RMSNorm 与残差流"
subtitle: "几个数字 → 一条宽 C 的总线 → 谁在读写它 → 一份共享的 [V, C] 表"
minutes: 75
mainline: "config 里的七个数字怎么撑起一条宽度为 C 的总线：input_ids [B,T] 上车，L 层轮流读写，最后靠一份共享的 [V, C] 表下车"
files:
  - model/model_minimind.py#L10-L60
  - model/model_minimind.py#L196-L243
goals:
  - 只看 config 就能报出模型里任意一个 tensor 的 shape，包括 head_dim / intermediate_size 这两个派生值
  - 能说清 hidden_states 为什么全程保持 [B, T, C]，以及每个子层是怎么读写这条总线的
  - 能默写 RMSNorm 的那两行：mean 沿哪一维、keepdim 为什么必须是 True、.float() 省掉会怎样
  - 能解释 embed_tokens 和 lm_head 为什么是同一份 [V, C] 权重，关掉共享会多出多少参数
  - 能把 64M 参数拆成 FFN / attention / embedding 三块，并预测改 config 之后占比怎么变
---

Day 1 已经完整跑过一次 forward。今天把镜头推近到**骨架**：`MiniMindConfig` 里的一组数字怎么定死全模型的 shape，`input_ids` 怎么变成 `hidden_states`，L 层怎么轮流读写它。Attention 和 FFN 内部一律当黑盒——那是 Day 4/5 的事，今天只关心它们「进去什么形状、出来什么形状」。

今天只跟一个张量走：`hidden_states`。它在 `embed_tokens` 那一步上车，宽度被 `hidden_size` 定成 `C`，之后穿过多少层都不再改变，直到靠一份和输入端共享的 `[V, C]` 表下车。

四章的分工：第一章把 config 收敛成七个符号，后面三章报 shape 全靠它们；第二章看这条总线怎么被建立、被谁读写；第三章钻进唯一的归一化 RMSNorm，并用真实权重量一量总线上的能量；第四章数参数——64M 里有七成花在同一个地方。

{{flow: input_ids [B,T] | *embed_tokens* | hidden_states [B,T,C] | MiniMindBlock × L | *norm* | *lm_head* | logits [B,T,V]}}

# 模型的配置：MiniMindConfig

一个模型的「形状」全部写在 `MiniMindConfig.__init__` 这三十来行里。这一章把它收敛成七个符号，弄清其中哪两个是从 `hidden_size` 派生出来的，最后再看 `kwargs.get(...)` 这种啰嗦写法为什么是必要的。

## 决定 shape 的七个数字 {#cfg-dims}

`MiniMindConfig` 有二十来个字段，但决定 tensor shape 的只有七个符号，后面每一天报 shape 都靠它们：`C`=`hidden_size`、`L`=`num_hidden_layers`、`H`=`num_attention_heads`、`KV`=`num_key_value_heads`、`D`=`head_dim`、`I`=`intermediate_size`、`V`=`vocab_size`，默认值依次是 768 / 8 / 8 / 4 / 96 / 2432 / 6400。构造函数的具名参数只有 `hidden_size`、`num_hidden_layers`、`use_moe` 三个，其余一律从 `**kwargs` 里掏（写法见 [kwargs.get 一节](#cfg-kwargs)）。

{{source:model/model_minimind.py#L12-L26}}

七个数字里有两个不是你直接给的，而是**派生**出来的——盯住 L24 和 L26，只有这两行的默认值表达式右边出现了 `hidden_size`：`head_dim` 默认 `768 // 8 = 96`，`intermediate_size` 默认 `ceil(C·π/64)·64 = 2432`（下一节专讲）。

注意 L24 读的是**上一行刚取出来的** `self.num_attention_heads`，所以传 `num_attention_heads=16` 时 `D` 变成 48。写成 `kwargs.get("head_dim", ...)` 还意味着 `D` 能和 `C`、`H` 彻底解耦——Qwen3 就这么用，见折叠块。

实验把七个符号列成一张表，第二列是改了 `hidden_size=512` 的结果——跟着变的有几行？

{{lab:cfg_dims}}

{{quiz:q9,q1}}

> [!KEY]
> 七个数字（C/L/H/KV/D/I/V）定死了全模型的 shape；其中 `head_dim` 和 `intermediate_size` 默认由 `hidden_size` 派生，改 `hidden_size` 会连带改掉它们。

> [!MORE] head_dim 为什么可以和 C、H 解耦
> 显式传 `head_dim=128` 时 `H·D = 1024` 就不等于 `C = 768` 了，模型照样跑得通：`q_proj` 是 `Linear(C, H·D)`、`o_proj` 是 `Linear(H·D, C)`，一进一出把 attention 内部的宽度和残差流的宽度隔开了，`H·D` 只在 attention 内部存在。Qwen3 正是靠这一点把 `head_dim` 固定在 128，而不管 `hidden_size` 怎么变。

> [!MORE] 剩下那些字段归哪一天管
> `rope_theta` / `max_position_embeddings` / `inference_rope_scaling` / `rope_scaling`（YaRN）→ Day 3 讲 RoPE 和长文本外推；
> `flash_attn` / `num_key_value_heads` 的用法 → Day 4 讲 GQA 和两条 attention 分支；
> `num_experts` / `num_experts_per_tok` / `moe_intermediate_size` / `norm_topk_prob` / `router_aux_loss_coef` → Day 6，`use_moe=False` 时它们全部被忽略；
> `dropout` 在 `.eval()` 下恒等于没有，训练时才起作用。

## intermediate_size 的来历：π 与 64 对齐 {#cfg-intermediate}

`2432` 这个数字既不是拍脑袋定的，也不是常见的 `4·C`。它是一条公式的结果，而公式里有两个值得追问的地方：为什么用 `π` 当系数？为什么算完还要乘回 64？

{{source:model/model_minimind.py#L26}}

代入默认值：`768·π/64 ≈ 37.70`，`ceil` 到 38，`38×64 = 2432`。

先说 `·64`。`ceil(...)·64` 的效果是把 `I` 强行钉在 64 的倍数上。FFN 的三个矩阵都有一条边是 `I`，而 GPU 上的矩阵乘法是按 tile 切块算的（典型 tile 边长 16/32/64/128）；`I` 带零头时最后一块 tile 只能算半格，算力白白浪费，张量并行按卡切 `I` 时也会切不齐。小模型上这点差别不明显，但这是整个业界的惯例写法。顺带一提，L26 用的是签名里的局部变量 `hidden_size`，而 L24 用的是 `self.hidden_size`——此刻两者相同，因为 L14 刚把前者赋给了后者。

再说 `π`。LLaMA 系的惯例系数是 `8/3 ≈ 2.667`，它的来历是一道等参数量的换算题：普通两层 MLP 是 `up: C→4C` 加 `down: 4C→C`，共 `8C²` 个参数；SwiGLU 有 `gate` / `up` / `down` 三个矩阵，参数量是 `3·C·I`。想让换成 SwiGLU 之后参数量不变，就要 `3·C·I = 8C²`，解出 `I = 8/3·C`。MiniMind 把系数换成 `π ≈ 3.14159`，等于主动给 FFN 多加了约 18% 的容量——`π / (8/3) ≈ 1.178`。代价很直接：FFN 本来就是参数量最大的一块（[见「64M 参数的分布」一节](#params)），这 18% 全部加在最大的那一块上。

{{lab:cfg_intermediate}}

{{quiz:q10}}

> [!KEY]
> `intermediate_size = ceil(C·π/64)·64`：π 是作者选的容量系数，`·64` 是为了矩阵乘法对齐；同样的 `C` 下 MiniMind 的 FFN 比 LLaMA 的 8/3 惯例宽约 18%。

> [!MORE] 这条公式还牵动谁
> `moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)`（L43）：MoE 每个专家的中间层宽度默认跟着这条公式走，所以 Day 6 里「4 个专家」的默认配置每个专家都和稠密 FFN 一样宽。
> `FeedForward.__init__(self, config, intermediate_size=None)`（L137-L139）还留了一个显式覆盖的口子：`intermediate_size = intermediate_size or config.intermediate_size`，给的话就不读 config。稠密路径从来不用它，MoE 路径才会传。

## kwargs.get 的覆盖顺序 {#cfg-kwargs .side}

`__init__` 里几乎每一行都长成 `self.xxx = kwargs.get("xxx", 默认值)`，比直接写具名参数啰嗦得多。要理解为什么，得先看第一行 `super().__init__(**kwargs)` 之后发生了什么。

{{source:model/model_minimind.py#L12-L20}}

继承 `PretrainedConfig` 换来的是整个 `transformers` 生态的兼容性：`from_pretrained` / `save_pretrained` / `AutoConfig` 全部白拿。代价是父类 `__init__` 也定义了一批同名关键字参数——`bos_token_id`、`eos_token_id`、`tie_word_embeddings`——默认值和 MiniMind 想要的不一样（两个 token id 在父类里都是 `None`）。

关键在于 `super().__init__(**kwargs)` 是**重新展开传参**，不会消耗调用方那个 `kwargs` 字典。父类先把 `self.bos_token_id` 设成 `None`，紧接着 L19 的 `kwargs.get("bos_token_id", 1)` 又从**原封不动的** `kwargs` 里读一遍，用 `1` 覆盖掉——「后写覆盖前写」。调用方显式传了 `bos_token_id=99` 的话读到的就是 99，覆盖不会吃掉它。

对于父类根本不认识的字段（`vocab_size`、`dropout`、`flash_attn`……），`kwargs.get` 是唯一的赋默认值途径：父类末尾那个 `setattr(self, key, value)` 循环只处理**调用方显式传了**的键，不传就连属性都不会存在。

{{lab:cfg_kwargs}}

{{quiz:q11}}

> [!KEY]
> `kwargs.get(name, default)` 干两件事：给父类不认识的字段（`vocab_size` 等）兜底默认值；给父类也认识的字段（`bos_token_id` 等）用 MiniMind 自己的默认值覆盖父类的。

> [!MORE] 父类那边到底发生了什么
> ```python
> class PretrainedConfig:
>     def __init__(self, *, tie_word_embeddings=True, bos_token_id=None, eos_token_id=None, **kwargs):
>         self.tie_word_embeddings = tie_word_embeddings   # 具名参数吃掉
>         self.bos_token_id = bos_token_id                 # 不传就是 None
>         for key, value in kwargs.items():                # 没被具名参数吃掉的
>             setattr(self, key, value)                     # 才走这里
> ```
> 注意最后那个 `setattr` 循环：它只处理调用方显式传了的键。所以 `vocab_size` 这种父类不认识的字段，如果不写 `kwargs.get("vocab_size", 6400)`，`MiniMindConfig()` 上压根不会有这个属性（实验表格第 2 列就是这么显示的）。

# 从 token id 到残差流：Embedding 与骨架 forward

七个数字有了，现在看它们怎么变成真正的张量。这一章从 `embed_tokens` 的那一次查表开始，走完 `MiniMindModel.forward` 的主干循环，顺带看一眼挂在 `__init__` 末尾、却不进 checkpoint 的两个 RoPE 表。

## Embedding：查表，而不是矩阵乘 {#embed-lookup}

`nn.Embedding(V, C)` 常被解释成「把 token id 变成 one-hot，再乘一个 `[V, C]` 的矩阵」。数学上确实等价，但实现上完全不是这么做的——没有 one-hot，也没有矩阵乘法。

{{source:model/model_minimind.py#L201}}

`embed_tokens.weight` 就是一张 `[V, C] = [6400, 768]` 的表，每一行对应词表里的一个 token；前向 `F.embedding(ids, weight)` 干的事就是高级索引 `weight[ids]`，按行号把对应的行取出来堆好。用矩阵乘法实现的话要做 `B·T·V·C` 次乘加，而查表只是 `B·T` 次访存——差了三四个数量级。

既然是查表，输出形状就遵循高级索引的规则：**下标张量的形状 + 被索引那一行的形状**。`[B, T]` 的整数下标配上 `[C]` 的行，得到 `[B, T, C]`。这条规则对任意形状的下标都成立：喂一个一维的 `[7]`，出来就是 `[7, C]`。这也解释了越界为什么必然炸在这里——超出 `V` 的 id 根本没有对应的行，CPU 上直接 `IndexError: index out of range in self`。

从这一步开始，`hidden_states` 就上车了，总线的宽度 `C` 也在这一步被定死。注意两件 MiniMind **没做**的事：没有乘 `sqrt(d_model)`，也没有传 `padding_idx`（详见下面的折叠块）。另外这张表还会在输出端以 `lm_head` 的身份再用一次，那是[「一份 [V, C] 权重的两个身份」一节](#tying)的内容。

{{lab:embed_lookup}}

{{quiz:q12}}

> [!KEY]
> Embedding 就是 `weight[input_ids]`：`[B,T]` 的整数下标取出 `[B,T,C]` 的浮点。总线从这一步开始，宽度 `C` 从这一步定死。

> [!MORE] 和原始 Transformer 的两点不同
> 1. 「Attention Is All You Need」里 embedding 之后要乘 `sqrt(d_model)`，用来和 sinusoidal position encoding 的幅度配平。MiniMind（和 LLaMA/Qwen）用 RoPE，位置信息在 attention 内部注入，所以**没有这个缩放**。
> 2. 没有传 `padding_idx`。MiniMind 的 padding 靠 `attention_mask` 和 loss mask 处理（Day 8 训练循环里会看到），不依赖 embedding 层把 pad 行冻成 0。

## 残差流：一条宽度恒为 C 的总线 {#bus}

把 `MiniMindBlock` 内部当黑盒之后，`MiniMindModel.forward` 的主干只剩三步：查表一次（L214）、循环 L 次、最后归一化一次（L230）。

{{source:model/model_minimind.py#L221-L230}}

循环体里 `hidden_states` 被同名覆盖了 L 次——这个反复被读、被写、名字始终不变的张量就是**残差流**。每一层的读写模式是固定的三拍（L187、L192、L193）：先 `residual = hidden_states` 把当前值存下来，再把它的一份**归一化后的副本**喂给子层，最后把子层的输出加回 `residual`。

「加回去」这个动作本身就锁死了宽度：张量加法要求两边 shape 一致，所以任何想接上这条总线的子层，都必须 `[B,T,C]` 进、`[B,T,C]` 出。子层内部爱用什么形状都行——`mlp.gate_proj` 的输出是 `[B,T,I=2432]`，`q_norm` 看到的是 `[B,T,H,D]`——但那些都是「下车之后」的事，对总线不可见。这条约束是后面三件事的共同前提：`lm_head` 敢用 `Linear(C, V)`、参数量能按模块相加、范数能一层层量下去。

还有个细节：`position_embeddings` 在 L219 切一次，然后在循环里**原样传给每一层**，L 层共用同一对 `cos/sin`。RoPE 表不是逐层的参数，这一点 Day 3 会用到。

下面的实验把整棵模块调用树打出来，每个节点标着「输入 → 输出」的 shape，你可以直接数一数总线上出现过几种形状。

{{lab:bus}}

{{quiz:q13}}

> [!KEY]
> `hidden_states` 全程是 `[B, T, C]`：每个子层从这条总线上读一份、算完再加回去，谁都不改变它的宽度。这就是「残差流是一条宽 C 的总线」。

> [!MORE] 骨架里另外两行
> `hidden_states = self.dropout(self.embed_tokens(input_ids))`（L214）：`config.dropout` 默认 0.0，而且 `build_model` / `load_model` 返回的都是 `.eval()` 模型，所以这层 dropout 在本课所有实验里等于恒等映射。
> `start_pos = past_key_values[0][0].shape[1] if ... else 0`（L213）：推理时用 KV cache，这一轮只喂新 token，需要知道「前面已经有多少个位置」才能切对 RoPE 表。Day 7 讲 KV cache 时再看。

## freqs_cos：不进 checkpoint 的常量表 {#rope-buffer .side}

`MiniMindModel.__init__` 的最后两行注册了两个 buffer，参数表里那个 `persistent=False` 很少见。它到底关掉了什么？

{{source:model/model_minimind.py#L205-L207}}

先分清 buffer 和 parameter：`nn.Parameter` 会被 `parameters()` 收走、参与梯度更新；buffer 不参与反向传播，但同样是模块状态的一部分——跟着 `.to(device)` 搬家，出现在 `named_buffers()` 里，默认也会被写进 `state_dict()`。BatchNorm 的 `running_mean` 就是典型的 buffer。

`persistent=False` 只关掉最后那一件事：**不写进 `state_dict()`**，其他一概不变，`forward` 里照样 `self.freqs_cos[start_pos:start_pos+seq_length]` 地切片用。为什么要关？因为这两张表是纯粹由 config 派生的常量——只要 `head_dim`、`max_position_embeddings`、`rope_theta` 不变，算出来就一模一样，`__init__` 里现算一遍只要几毫秒。它的 shape 是 `[32768, 96]`，fp32 下 12.6 MB，两张就是 25 MB，而整个模型才 63.9M 参数；塞进 checkpoint 等于每次存盘多背 40% 的体积去存一堆能现算的常数。顺带还有个好处：`load_state_dict(strict=True)` 不会因为权重文件里缺这两个 key 而报错。

代价也有一处，见下面的折叠块。

{{lab:rope_buffer}}

{{quiz:q17}}

> [!KEY]
> `persistent=False` = 是 buffer、参与 forward，但不写进 `state_dict()`；纯粹由 config 派生的常量表都该这么注册。

> [!MORE] L215-L218 那段 workaround 在防什么
> ```python
> if self.freqs_cos[0, 0] == 0:
>     freqs_cos, freqs_sin = precompute_freqs_cis(...)
>     self.freqs_cos, self.freqs_sin = freqs_cos.to(...), freqs_sin.to(...)
> ```
> `transformers>=5.x` 的某些加载路径会先在 `meta` 设备上构造模型骨架（不分配真实内存），再把 `state_dict` 里的权重搬进来。而 `persistent=False` 的 buffer **不在 `state_dict` 里**，搬运时就被漏掉了，留下一个全 0 的占位张量。
> `freqs_cos[0, 0]` 正常恒等于 `cos(0) = 1`（实验里 print 了这个值），一旦发现它变成 0，就说明 buffer 丢了，于是在第一次 forward 时补算一遍。这是「不进 checkpoint」这个设计选择的代价。

# RMSNorm 与残差流的尺度

MiniMind 里唯一的归一化只有四行代码，但每个细节都有讲究：沿哪一维、`keepdim`、`.float()`、`eps`、以及 `dim` 到底决定了什么。最后一节用真实权重把上一章那条总线量一遍，看看这些归一化管住了什么、又漏掉了什么。

## mean(-1) 与 keepdim：per-token 的归一化 {#rmsnorm-dim}

`RMSNorm` 的全部实现就是两个方法四行代码，核心是 `norm` 里那一行流。问题很具体：`mean(-1)` 的 `-1` 是哪一维？是「每个 token 各归各的」，还是「整句话一起归」？

{{source:model/model_minimind.py#L56-L60}}

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{C}\sum_i x_i^2 + \epsilon}} \odot \gamma$$

`-1` 是最后一维，对残差流来说就是 `C`。所以每个 token 的 768 个通道**独立**求自己的均方根，和 batch 里的其他句子、其他位置都无关——归一化不会在序列维度上串信息，逐 token 生成也能算出和训练时一致的结果。「沿 T 维归一化」是最常见的误解。

`keepdim=True` 把被压掉的那一维保留成长度 1：`[B,T,C]` 被 `pow(2).mean(-1, keepdim=True)` 压成 `[B,T,1]`，`+eps`、`rsqrt` 之后仍是 `[B,T,1]`，最后 `x * rsqrt(...)` 靠广播回到 `[B,T,C]`。改成 `False` 会得到 `[B,T]`，广播从右往左对齐，拿 `768` 去对 `7`，于是 `norm` 那一行直接抛 `RuntimeError`：768 和 7 在第 2 维上对不齐。

和 `LayerNorm` 的差别只有两条：**不减均值**、**没有 bias `β`**——少一次求均值、少一组参数，效果基本持平，这就是它取代 LayerNorm 的原因。另外 `RMSNorm(dim)` 的 `dim` 只决定 `self.weight` 的长度，沿哪一维归一化则写死是 `-1`——下一节会看到这两者分开的好处。

{{lab:rmsnorm_dim}}

{{quiz:q14}}

> [!KEY]
> `x.pow(2).mean(-1, keepdim=True)` 把 `[B,T,C]` 压成 `[B,T,1]`：归一化是 per-token 的；`keepdim=True` 是为了下一步的广播，改成 `False` 会直接报 shape 不匹配。

> [!MORE] 和 torch.nn.RMSNorm 的关系
> PyTorch 2.4+ 自带 `torch.nn.RMSNorm`，公式和这里完全一样（同样的输入下逐元素差为 0），还带 fused kernel。MiniMind 自己写一遍是为了教学可读，顺便把 `.float()` 的时机攥在自己手里——下一节就讲这个。

## .float() 的作用：防 fp16 平方溢出 {#rmsnorm-float}

`forward` 只有一行，却夹了一升一降：`x.float()` 把输入提到 fp32，算完 `.type_as(x)` 降回原 dtype。混合精度训练时 `hidden_states` 是 fp16/bf16，这一升一降多出一份 fp32 临时张量和两次转换，看着像可以优化掉的开销。

{{source:model/model_minimind.py#L60}}

删不得的原因在 `x.pow(2)`——整个网络里最容易溢出的一步。fp16 的最大值是 65504，而 `256² = 65536` 就已越界：**只要某个通道的绝对值达到 256，它的平方在 fp16 里就是 `inf`**。fp32 的上限是 3.4e38，够都够不着。

这个阈值并不遥远：残差流的范数会随深度增长（[见范数增长一节](#norm-growth)），越靠后的层越容易撞上。而 `mean` 是沿 `C` 求的，768 个通道里**只要有一个** `inf`，整行均值就是 `inf`——这个 token 的归一化会被单个异常通道绑架。

那么，一个 `inf` 掉进 `rsqrt` 之后，这个 token 的输出会变成什么？这决定了这个 bug 长什么样。先下注，再跑实验。

{{predict:p7}}

{{lab:rmsnorm_float}}

看实验的四行结果：`x.pow(2)` 里有 6390 个 `inf`；`rsqrt(inf)` 等于 **0.0 而不是 `nan`**（IEEE 754 规定 `1/√(+∞) = +0`）；于是 `x * 0`，21 个 token 的表示全被抹平成 0，「输出里有 nan 吗」是 `False`。程序正常跑完，没有异常、也没有 `nan` 能被抓住，只是信息没了——这类静默失败比崩溃难查得多。真实实现先升 fp32，全 0 行数是 0。

至于降回去的 `.type_as(x)`：`norm(x)` 之后 RMS 已是 1 量级，转回 fp16 安全。

> [!KEY]
> 不做 `.float()` 时 fp16 溢出的后果是**静默清零**（`rsqrt(inf)=0`），不是报错也不是 nan——这类 bug 比崩溃难查得多。

> [!MORE] bf16 和 eps 各自救了什么
> **bf16** 的指数位和 fp32 一样宽（动态范围到 ~3.4e38），同样的输入一个 `inf` 都不会出现，全 0 行数是 0。所以这个坑本质上是 fp16 专属的；但 `.float()` 写在这里对两种 dtype 都安全。
> **eps** 救的是另一头：输入全 0 时 `mean` 是 0，没有 eps 的话 `rsqrt(0) = inf`，`0 * inf = nan`，整个 token 变成 nan。加了 `+ eps` 之后 `rsqrt(1e-6) = 1000`，输出仍是全 0，不会污染。

## 五类 RMSNorm：C 和 D 两种宽度 {#rmsnorm-five}

同一个 `RMSNorm` 类在 MiniMind 里被实例化成了五类角色，作用的宽度却不是同一个数——这正是上一节说的「`dim` 决定 `weight` 长度、不决定归一化维度」带来的灵活性。

{{source:model/model_minimind.py#L104-L105}}

三类在残差流上，`dim` 都是 `config.hidden_size = 768`：`input_layernorm`（L182，喂给 `self_attn` 之前）、`post_attention_layernorm`（L183，喂给 `mlp` 之前）、`model.norm`（L204，`lm_head` 之前的最后一次）。

另外两类藏在 `Attention.__init__` 里，构造参数是 `self.head_dim` 而不是 `config.hidden_size`，所以 `weight` 只有 96 长。它们作用在 `[B,T,H,D]` 和 `[B,T,KV,D]` 上，最后一维正好是 `D=96`，于是同一行 `mean(-1)` 归的就变成了「每个 head 内部的 96 个分量」——这是 Qwen3 风格的 QK-Norm，放在 RoPE 之前稳定注意力分数的尺度，Day 4 细讲。

数实例个数要按 `4L+1` 算：每层 4 个（两个在 Block 上、两个在 Attention 里），加最后的 `model.norm`。`L=2` 的小模型有 9 个，默认的 `L=8` 有 33 个。注意「5」是**种类**数不是实例数，这是最容易数错的地方。参数上它们完全无足轻重：每层 `2C + 2D = 1728`，全模型 14,592 个，占总量的 0.02%（[见「64M 参数的分布」一节](#params)）。

{{lab:rmsnorm_five}}

{{quiz:q15,q2}}

> [!KEY]
> 五类 RMSNorm：`input_layernorm` / `post_attention_layernorm` / `model.norm` 作用在 `C=768` 上，`q_norm` / `k_norm` 作用在每个 head 内部的 `D=96` 上；实例总数是 `4L+1`。

> [!MORE] 一个从没被用到的默认值
> `RMSNorm.__init__(self, dim, eps=1e-5)` 的类默认 eps 是 `1e-5`，但实验表格里 9 个实例的 `eps` 全是 `1e-6`——因为每一处实例化都显式传了 `eps=config.rms_norm_eps`（L104、L105、L182、L183、L204）。类自己写的 `1e-5` 在整个模型里从未被触发过。
> 想验证的话，改 `MiniMindConfig(rms_norm_eps=...)` 会让 9 行全变，而改 `RMSNorm` 的类签名一行都不会变。

## 真实权重下的范数增长 {#norm-growth}

上一章那句「每个子层读一份、算完加回去」，对残差流的**尺度**意味着什么？答案写在 `MiniMindBlock.forward` 里。

{{source:model/model_minimind.py#L186-L193}}

盯住 L187 和 L192：`residual = hidden_states` 存的是**没归一化**的原值，`input_layernorm` 只归一化了喂给 `self_attn` 的那份临时副本，加回来的是 `residual + 子层输出`。子层**读到**的尺度受控（RMS 被拉到 1 再乘 `weight`），但**写回**总线的增量没有任何人管，总线本身从头到尾没被重新缩放过。

这正是 Pre-Norm 和 Post-Norm 的分水岭。Post-Norm 写的是 `norm(x + sublayer(x))`，每层结束都把总线重新归一化一次；Pre-Norm 写的是 `x + sublayer(norm(x))`，残差连接成了一条不经过 norm 的直通路径，梯度好走得多，代价就是总线尺度没人兜底。MiniMind、LLaMA、Qwen 全是 Pre-Norm。

那么在真实权重下，每层之后 `hidden_states` 的 L2 范数会画出什么形状的曲线？先下注，再用 `pretrain_768` 量一遍——随机初始化的尺度完全是两回事。

{{predict:p11}}

{{lab:norm_growth}}

实测：`embed` 之后 1.28，`layer0` 就跳到 6.12，随后 8.09 → 10.57 → 11.82 → 14.58 → 17.16 → 22.62 → **36.42**，几乎单调，最后两层涨得最快，全程 ×28.4。这条曲线也解释了上一节那个 fp16 坑为什么在深层和长输入上发作。最后一行 `model.norm` 之后是 42.14，比 `√C = 27.7` 还大——原因见折叠块。

> [!KEY]
> Pre-Norm 下子层读到的是归一化后的副本，但写回残差流的增量没有被归一化，所以范数随深度近似单调增长（实测 1.3 → 36）。

> [!MORE] 过了 model.norm 之后，为什么反而更大了
> `model.norm` 之后是 42.14，比 `√C = 27.7` 还大。因为 RMSNorm 只保证 `norm(x)` 那一步的 RMS 等于 1，紧接着还要乘可学习的 `self.weight`。这份 pretrain 权重里 `model.norm.weight` 的均值本身就有 1.5 左右，最终范数约等于 `RMS(weight)·√C`。
> 换句话说，最后那一层 norm 并不是「把总线拉回某个固定尺度」，而是「先归一化、再由模型自己学一个每通道的缩放」——它决定了 `lm_head` 看到的输入分布。

# 权重共享与参数量

最后一章数数字。先看输出端那份 `[V, C]` 和输入端的关系，再把全部参数按模块归类，看看 64M 花在哪、以及加宽 / 加深 / 换词表分别会动到谁。

## 一份 [V, C] 权重的两个身份 {#tying}

`nn.Embedding(V, C).weight` 的形状是 `[V, C]`；`nn.Linear(C, V, bias=False).weight` 的形状是 `[out_features, in_features] = [V, C]`。两者恰好一样，于是 L242 用一行赋值把它们接上了。

{{source:model/model_minimind.py#L240-L243}}

形状相同不是巧合。输入端做的是「token → 向量」，输出端做的是「向量 → 每个 token 的分数」，后者本质上是拿隐状态和词表里每一行做内积——同一张「token ↔ 向量」对应表，只是用的方向相反。既然如此就没有理由学两份：一个 token 作为输入时的表示，和它作为预测目标时的「原型向量」应该是一回事——这就是 weight tying（Press & Wolf, 2017）。

那么问题来了：执行完这一行，模型里到底有几份 `[V, C]` 的数据？`sum(p.numel() for p in model.parameters())` 会把它数几次？先下注，再跑实验。

{{predict:p9}}

{{lab:tying}}

实验的前四行给出了答案：`e is h` 为 `True`、`data_ptr()` 相同——两个属性名指向**同一个** `nn.Parameter`，不是数值相同的两份拷贝。`named_parameters()` 按对象去重，只保留先注册的那个名字，所以 `'lm_head.weight'` 压根不在参数列表里，只被计入一次：`tie=True` 19,665,024，`tie=False` 24,580,224，差 4,915,200 = `6400 × 768`。

既然是同一个对象，autograd 里它就是同一个叶子节点，而且在一次 forward 里被用了两次——那反向传播时它的 `.grad` 该怎么算？

{{quiz:q3}}

> [!KEY]
> `tie_word_embeddings=True` 让输入端的查表和输出端的打分共用同一个 `[V, C]` 张量；关掉之后凭空多出 `V·C = 4,915,200` 个参数。

> [!MORE] 类里那两行配套设施
> `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`（L236）是给 `transformers` 看的元信息：`save_pretrained` / `from_pretrained` / 量化逻辑据此知道「这两个 key 本来就是一份权重」，不会存两遍、也不会在加载时报 missing key。
> `post_init()`（L243）是 `PreTrainedModel` 的收尾钩子，负责按 `transformers` 的规则做权重初始化；因为共享在它之前已经完成，不会被初始化成两份不同的值。

## 64M 参数的分布 {#params}

「minimind-3 是 64M 模型」这句话里的 64M 分布得非常不均。所有投影层都是 `bias=False`，所以每一块都可以用一个乘法算出来：

| 模块 | 公式 | 默认配置（C=768, H=8, KV=4, D=96, I=2432） |
|---|---|---|
| `q_proj` / `o_proj` | `C·H·D` | 589,824 各一个 |
| `k_proj` / `v_proj` | `C·KV·D` | 294,912 各一个（GQA 只有一半宽） |
| `gate/up/down_proj` | `C·I` | 1,867,776 各一个 |
| 每层 4 个 norm | `2C + 2D` | 1,728 |
| `embed_tokens`（tied） | `V·C` | 4,915,200，只有一份 |

{{source:model/model_minimind.py#L140-L142}}

每层 FFN 是 `3·C·I = 5,603,328`，attention 四个投影合起来是 `2·C·H·D + 2·C·KV·D = 1,769,472`——FFN 比 attention 大 3.2 倍，而这还是在 GQA 已经把 k/v 砍掉一半的前提下。norm 的参数量是 `C` 的一次方项，在 `C²` 量级面前可以直接忽略。乘上 `L=8` 再加上 embedding，总数是 63,912,192，和 README 里的「64M」对得上。

这个比例决定了改 config 的代价：加宽 `hidden_size` 是**超线性**的（FFN 和 attention 的投影都是 `O(C²)`），加深 `num_hidden_layers` 近似线性且完全不动 embedding，换大词表则只动 embedding 那一项。想省参数，永远先动 FFN。

{{lab:params}}

{{quiz:q16,q4}}

> [!KEY]
> 默认配置下 FFN 约 70%、attention 约 22%、embedding（tied）约 7.7%、所有 norm 加起来约 0.02%——想省参数，先动 FFN。

> [!MORE] 小模型为什么倾向于用小词表
> `embed_tokens` 是 `[V, C]`，只和词表大小有关，不随层数变。MiniMind 用 6400 的词表时它占 7.7%；换成 Qwen 量级的 32000，光这一项就吃掉总参数量的 **29.4%**（总量从 63.9M 涨到 83.6M），而且这部分参数对「模型有多聪明」的贡献远不如 FFN。
> 反过来，模型越大这个比例越不重要——所以大模型敢用 15 万级别的词表换更短的序列长度。
