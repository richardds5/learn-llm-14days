# MiniMind 14 天学习计划

> 读者画像：Python 精通、LLM 原理了解、没手写过 LLM。方式：读源码 → 跑实验看每个 tensor 的维度 → 改代码看 diff → 选择题。每天 1~1.5 小时。
> 打开方式：`./start.sh` → http://127.0.0.1:8877 （进度自动保存在 `progress.json`）

**总量**：14 天 · 约 20 小时 · 175 个可运行实验 · 305 道选择题

## 第一阶段 · 模型与推理（Day 1-7）

*把 model/model_minimind.py 这 287 行逐行读透*

### Day 1 · 全景与 Tokenizer

仓库地图 → 一次 forward 的鸟瞰 → 文字怎么变成 token

- ⏱ 80 分钟 · 🧪 12 个实验 · ✍️ 20 道选择题
- 📄 精读：`model/model_minimind.py#L186-L253`、`trainer/train_tokenizer.py`、`model/tokenizer_config.json`
- 🎯 学完能做到：
  - 说得出 model / dataset / trainer / scripts 各自的边界，以及 out/{前缀}_{hidden_size}{_moe}.pth 这条全仓库统一的命名规则
  - 看得懂模块调用树：形参名、`tuple( … )` 里的元素名、被折叠的重复层，各自是什么意思
  - 说得清 logits [B,T,V] 里每个位置在预测什么，以及为什么「要生成的下一个 token」只能看 logits[0, -1]
  - 把 6400 拆成 256 + 6108 + 36，并解释 ByteLevel 为什么让 token 看起来像乱码却永远不会编码失败
  - 逐段讲得出 chat_template 的回合结构、assistant 分支怎么处理 `<think>`、tools 分支与 tool 角色合并
- 🧵 主线：跟着一句「天空是什么颜色的？」走完全程：chat_template 把它摊平成一个字符串，tokenizer 切成 input_ids [B,T]，一次 forward 出 logits [B,T,V]，最后一个位置决定下一个 token
- 📍 4 章 · 12 个小节（一次只看一个）：
  - **仓库与训练流水线的地图**
    - 1.1 四个目录与一个手测脚本
    - 1.2 训练流水线与权重命名的死规则
  - **一次 forward 的鸟瞰**
    - 2.1 模块调用树的读法
    - 2.2 logits [B,T,V]：T 个位置同时在预测
    - 2.3 logits[0, -1] 到底是谁的概率
  - **Tokenizer：6400 个格子怎么来的**
    - 3.1 6400 的分账：256 + 6108 + 36
    - 3.2 ByteLevel：像乱码，但永远不会失败
    - 3.3 被改回 special=False 的那 15 个
    - 3.4 现场训练一个迷你 BPE
  - **chat_template：把对话摊平成一个字符串**
    - 4.1 回合结构与 system 段
    - 4.2 assistant 分支与 open_thinking 开关
    - 4.3 tools 分支与 tool 角色的合并（选学）

### Day 2 · 骨架：Config、Embedding、RMSNorm 与残差流

几个数字 → 一条宽 C 的总线 → 谁在读写它 → 一份共享的 [V, C] 表

- ⏱ 75 分钟 · 🧪 12 个实验 · ✍️ 20 道选择题
- 📄 精读：`model/model_minimind.py#L10-L60`、`model/model_minimind.py#L196-L243`
- 🎯 学完能做到：
  - 只看 config 就能报出模型里任意一个 tensor 的 shape，包括 head_dim / intermediate_size 这两个派生值
  - 能说清 hidden_states 为什么全程保持 [B, T, C]，以及每个子层是怎么读写这条总线的
  - 能默写 RMSNorm 的那两行：mean 沿哪一维、keepdim 为什么必须是 True、.float() 省掉会怎样
  - 能解释 embed_tokens 和 lm_head 为什么是同一份 [V, C] 权重，关掉共享会多出多少参数
  - 能把 64M 参数拆成 FFN / attention / embedding 三块，并预测改 config 之后占比怎么变
- 🧵 主线：config 里的七个数字怎么撑起一条宽度为 C 的总线：input_ids [B,T] 上车，L 层轮流读写，最后靠一份共享的 [V, C] 表下车
- 📍 4 章 · 12 个小节（一次只看一个）：
  - **模型的配置：MiniMindConfig**
    - 1.1 决定 shape 的七个数字
    - 1.2 intermediate_size 的来历：π 与 64 对齐
    - 1.3 kwargs.get 的覆盖顺序（选学）
  - **从 token id 到残差流：Embedding 与骨架 forward**
    - 2.1 Embedding：查表，而不是矩阵乘
    - 2.2 残差流：一条宽度恒为 C 的总线
    - 2.3 freqs_cos：不进 checkpoint 的常量表（选学）
  - **RMSNorm 与残差流的尺度**
    - 3.1 mean(-1) 与 keepdim：per-token 的归一化
    - 3.2 .float() 的作用：防 fp16 平方溢出
    - 3.3 五类 RMSNorm：C 和 D 两种宽度
    - 3.4 真实权重下的范数增长
  - **权重共享与参数量**
    - 4.1 一份 [V, C] 权重的两个身份
    - 4.2 64M 参数的分布

### Day 3 · 位置编码：RoPE 与 YaRN

一张 [32768, 96] 的常量表：怎么算出来、怎么切、怎么乘进 q/k、长文本时怎么改

- ⏱ 85 分钟 · 🧪 11 个实验 · ✍️ 22 道选择题
- 📄 精读：`model/model_minimind.py#L62-L84`、`model/model_minimind.py#L209-L219`、`model/model_minimind.py#L27-L39`
- 🎯 学完能做到：
  - 能默写 precompute_freqs_cis 里 freqs → outer → cat 三步之后的 shape，并说清为什么是 cat([cos, cos]) 而不是交错重复
  - 能解释 cos.unsqueeze(1) 在 MiniMind 的 [B,T,H,D] 布局下一次广播掉了哪两个维度，和 HF Llama 的同名参数差在哪
  - 能说出 start_pos 从哪来、为什么读的是 shape[1]，以及 KV cache 解码时切出来的 cos 有多长
  - 能用一张热力图证明 ⟨R_m q, R_n k⟩ 只依赖 m − n，并说清 rotate_half 约定和 GPT-J 交错约定之间差的那个 permutation
  - 能算出 YaRN 的 low / high，说清哪些频率分量被压、压多少，以及 `if end / orig_max > 1.0` 判断的到底是什么
- 🧵 主线：跟着 freqs_cos / freqs_sin 这张表走：precompute_freqs_cis 算出 [32768, D] → 按 start_pos 切成 [T, D] → 广播乘到 [B, T, H, D] 的 q/k 上 → 长文本时 YaRN 改写它的频率
- 📍 4 章 · 11 个小节（一次只看一个）：
  - **频率表：precompute_freqs_cis**
    - 1.1 D/2 个频率：每两个通道共用一个转速
    - 1.2 波长：rope_theta 定下能走多远
    - 1.3 outer 与 cat：从 [48] 到 [32768, 96]
  - **把表用到 q、k 上：切片与广播**
    - 2.1 start_pos：这次 forward 要表里的哪一段
    - 2.2 cos.unsqueeze(1)：一次广播掉 B 和 H
    - 2.3 rotate_half：一行做掉 48 个 2D 旋转
  - **RoPE 的核心性质：相对位置**
    - 3.1 点积只依赖 m − n
    - 3.2 另一种配对约定与那个 permutation（选学）
  - **YaRN：把频率改窄的长文本外推**
    - 4.1 low / high / ramp：只压低频那一段
    - 4.2 触发条件看的是 end，不是输入长度
    - 4.3 超参数要和 checkpoint 的训练长度对齐

### Day 4 · Attention：GQA、QK-Norm 与因果掩码

一份 [B,T,C] 拆成 8 个头、和 4 组 KV 对齐、被掩码切成三角形，再拼回 [B,T,C]

- ⏱ 90 分钟 · 🧪 13 个实验 · ✍️ 21 道选择题
- 📄 精读：`model/model_minimind.py#L86-L134`
- 🎯 学完能做到：
  - 能报出 xq/xk/xv 在 Attention.forward 每一行之后的 shape，并说清 view / QK-Norm / repeat_kv / transpose 各自动的是哪一维
  - 能解释 GQA 为什么需要 repeat_kv、n_rep 怎么算、KV cache 里存的是 repeat 之前还是之后的版本
  - 能说出第 125 行那四个条件各自在挡什么，以及为什么自回归单步 decode 永远走不到 SDPA
  - 能读懂因果掩码那一行的 `-seq_len:` 切片，和 padding mask 的 `[B,1,1,T_kv]` 广播
  - 能在真实权重上认出 attention sink，并说清它是训练涌现的现象而不是结构规定
- 🧵 主线：一份 x [B,T,C] 走完 Attention.forward 的 24 行：拆成 [B,T,H,D] 和 [B,T,KV,D]，对齐成 [B,H,T,D]，算出一张 [B,H,T,T_kv] 的分数表，掩码 + softmax 之后再合并回 [B,T,C]
- 📍 5 章 · 13 个小节（一次只看一个）：
  - **进门：从一条总线拆成多头**
    - 1.1 三个投影：Q 宽 768，K/V 只有 384
    - 1.2 view 拆头：两个头数字段，两条分岔
    - 1.3 QK-Norm：归一化落在 head_dim 上
  - **对齐到 [B,H,T,D]：cache、repeat_kv、transpose**
    - 2.1 cache 拼接：在 dim=1 上接上旧的 KV
    - 2.2 repeat_kv：expand 不拷贝，reshape 才拷贝
    - 2.3 第 124 行：一行里做完三件事
  - **两条分支：SDPA 与手写实现**
    - 3.1 第 125 行的四个开关
    - 3.2 两条分支的数值差异
  - **手写分支的三行**
    - 4.1 scores：D 被消掉，换来一张 T×T_kv 的表
    - 4.2 因果掩码只加在最后 seq_len 列
    - 4.3 padding mask：按列屏蔽，广播到所有 head
  - **出门：合并多头，以及真实权重下的样子**
    - 5.1 合并多头，投影回残差流
    - 5.2 真实权重下的 attention：从对角线到 attention sink

### Day 5 · FFN(SwiGLU)、Block 与完整前向 + Loss

装上最后一块算子，再把一整块 [B,T,V] 压成一个能 backward 的标量

- ⏱ 85 分钟 · 🧪 12 个实验 · ✍️ 19 道选择题
- 📄 精读：`model/model_minimind.py#L136-L146`、`model/model_minimind.py#L178-L194`、`model/model_minimind.py#L245-L253`
- 🎯 学完能做到：
  - 能说出 gate_proj / up_proj / down_proj 在 FeedForward.forward 那一行里的四步 shape 变化，并算出 SwiGLU 比两矩阵 MLP 贵多少
  - 能解释 MiniMindBlock 里两处残差相加为什么一处能写 in-place、另一处写了就在 backward 报错
  - 能推导 logits_to_keep 两条分支各自切出什么，并指出它省下的到底是哪一步的算力和显存
  - 能用一个 T=6 的小例子说清 logits[t] 对 labels[t+1]，并说明 ignore_index=-100 改的是分子还是分母
  - 能用 logit lens 读出中间层的预测，并解释残差流的哪条性质让它读得出来
- 🧵 主线：hidden_states [B,T,C] 进 FeedForward 被撑到 [B,T,I] 再压回来，两次残差写回总线，过 lm_head 变成 logits [B,T,V]，shift 之后拍平成 [B*(T-1), V] 交给 cross_entropy
- 📍 5 章 · 12 个小节（一次只看一个）：
  - **FeedForward：SwiGLU 的三块矩阵**
    - 1.1 一行流里的四步 shape
    - 1.2 SiLU 的软门控：gate 和 up 的分工
    - 1.3 三个矩阵的参数代价
  - **MiniMindBlock：两次 pre-norm 残差**
    - 2.1 一个 Block 的两组三拍
    - 2.2 一个 `+=` 和一个 `=`
  - **MiniMindForCausalLM.forward：总线的终点**
    - 3.1 logits_to_keep 的两条分支
    - 3.2 省下来的只有 lm_head 那一步
  - **从 logits 到一个标量 loss**
    - 4.1 shift：logits[t] 对 labels[t+1]
    - 4.2 contiguous 与拍平成 [B*(T-1), V]
    - 4.3 ignore_index=-100 改的是分母
  - **把标量拆开看：逐 token loss 与 logit lens**
    - 5.1 reduction='none'：逐 token 的惊讶度
    - 5.2 logit lens：中间层也能被 lm_head 读懂

### Day 6 · MoE：路由、top-k、aux loss 与激活参数

一张 [B*T, E] 的选票表，决定每个 token 走哪个 FFN

- ⏱ 85 分钟 · 🧪 13 个实验 · ✍️ 23 道选择题
- 📄 精读：`model/model_minimind.py#L40-L45`、`model/model_minimind.py#L148-L176`、`trainer/trainer_utils.py#L18-L28`
- 🎯 学完能做到：
  - 能背出 MOEFeedForward.forward 每一行之后 x_flat / scores / topk_idx / topk_weight / mask / token_idx / y 的 shape
  - 能说清 topk_weight 那一行的归一化在 k=1 和 k≥2 下是两件完全不同的事，以及它对 gate 梯度的影响
  - 能写出 aux_loss = E·Σ f_i·P_i·coef，并说明它惩罚的是「f 和 P 一起集中」而不是单纯的负载不均
  - 能从 total / base / active 三个数手算出 minimind-3-moe 的 198M-A64M
  - 能解释为什么 4 experts / top-1 的 FLOPs 和 dense 几乎一样，实测却更慢
- 🧵 主线：x [B,T,C] 被拉平成 [B*T,C]，gate 给出一张 [B*T,E] 的选票表，top-k 选人，按 expert 分桶算完再 index_add_ 写回，最后变回 [B,T,C]
- 📍 4 章 · 13 个小节（一次只看一个）：
  - **路由：从 [B,T,C] 到每个 token 的一张选票**
    - 1.1 展平：[B,T,C] → [B*T,C]
    - 1.2 gate：一个 [E,C] 的线性层加一次 softmax
    - 1.3 torch.topk：切成权重表和身份证表
    - 1.4 norm_topk_prob 与那个 1e-20
  - **分发与写回：只让被选中的 token 过 expert**
    - 2.1 mask 与 token_idx：长度由数据决定
    - 2.2 index_add_：为什么是加而不是赋值
    - 2.3 闲置 expert 的假梯度路径（选学）
  - **aux loss：router 的学习信号**
    - 3.1 aux_loss = E · Σ f_i·P_i · coef
    - 3.2 只在 train() 下计算，再逐层求和
    - 3.3 gate.weight 的梯度来源
  - **参数账与训练好的 router**
    - 4.1 total、base、active 三个数
    - 4.2 训练好的 router 的逐层负载
    - 4.3 分桶调度的代价（选学）

### Day 7 · 推理：generate 循环、KV cache 与采样

prefill 一次 → 每步只喂 1 个 token → logits[:, -1] 被四步依次削 → 抽一个拼回去

- ⏱ 90 分钟 · 🧪 13 个实验 · ✍️ 22 道选择题
- 📄 精读：`model/model_minimind.py#L255-L288`、`model/model_minimind.py#L111-L134`、`model/model_minimind.py#L209-L219`、`eval_llm.py#L62-L91`
- 🎯 学完能做到：
  - 能说出 generate 的 for 循环里每一步喂进 forward 的 input_ids 是什么 shape，以及 past_len 是怎么算出来的
  - 能指出 KV cache 在 Attention.forward 的哪一行被拼长、存的是哪个阶段的形态、shape 为什么是 [B, T, KV, D]
  - 能解释 start_pos 为什么必须存在，以及它切的是 RoPE 表的哪一段
  - 能把同一条分布交给 temperature / top-k / top-p，预测各自留下几个候选、排序会不会变
  - 能读懂 top-p 那四行里的「右移一位 + scatter」和 repetition_penalty 里的 torch.where，并说清去掉会怎样
- 🧵 主线：一个 prompt [B,T] 怎么变成一串 token：prefill 一次建起 KV cache，之后每步只喂 [B,1]，取 logits[:, -1] 过一遍 temperature/top-k/top-p，抽一个拼回 input_ids
- 📍 5 章 · 13 个小节（一次只看一个）：
  - **generate 的主循环**
    - 1.1 一行切片分出的 prefill 与 decode
    - 1.2 finished 向量与 batch 的收尾
    - 1.3 num_return_sequences 的行排列（选学）
  - **KV cache：只喂一个 token，却看得见全部历史**
    - 2.1 xk 在 dim=1 上被接长
    - 2.2 start_pos 决定 RoPE 从哪一格切起
    - 2.3 关掉 cache 之后多算了多少
  - **把一条分布削成能抽签的样子**
    - 3.1 temperature：除在 logits 上的一个数
    - 3.2 top-k：固定留 k 个，其余置 -inf
    - 3.3 top-p：固定留多少概率质量
  - **两段容易读错的代码**
    - 4.1 右移一位与 scatter 回原顺序
    - 4.2 repetition_penalty 的正负分叉
  - **接回命令行：streamer 与 eval_llm.py**
    - 5.1 streamer.put 的两处时机
    - 5.2 pretrain 拼 bos，full_sft 套 chat 模板

## 第二阶段 · 数据与训练（Day 8-10）

*从 jsonl 到 loss 下降：Pretrain / SFT / LoRA / 蒸馏*

### Day 8 · 数据管线：从 jsonl 到 (input_ids, labels)

五种 Dataset 共用一套 chat 模板，真正的教材是 loss mask

- ⏱ 85 分钟 · 🧪 12 个实验 · ✍️ 22 道选择题
- 📄 精读：`dataset/lm_dataset.py#L37-L119`、`dataset/lm_dataset.py#L122-L252`
- 🎯 学完能做到：
  - 能背出 PretrainDataset 的长度账：max_length-2、手动拼的 BOS/EOS、右 padding、-100 各占哪几格
  - 能手动跑一遍 generate_labels 的滑动匹配，说清多轮对话和截断边界分别怎么处理
  - 能解释 DPODataset 为什么在 Dataset 层就把 x/y/mask 错开，而 Pretrain/SFT 不用
  - 知道 RLAIFDataset/AgentRLDataset 为什么不 tokenize，AgentRL 为什么必须配自定义 collate_fn
  - 能用「截断率 vs padding 占比」两个数为一份数据挑 max_length
- 🧵 主线：一行 jsonl 怎么变成模型吃的张量：apply_chat_template 渲染成文本 → tokenizer → 定长 input_ids → generate_labels 打出 loss mask → DataLoader 拼成 batch
- 📍 4 章 · 12 个小节（一次只看一个）：
  - **PretrainDataset：定长张量的最短流水线**
    - 1.1 手动拼的 BOS/EOS 与 max_length-2 的预算
    - 1.2 labels 的两个约定：-100 与不 shift
  - **SFTDataset：把 conversations 渲染成一段文本**
    - 2.1 apply_chat_template 渲染出来的 ChatML 文本
    - 2.2 tools 与 tool_calls 的反序列化
    - 2.3 两处随机扰动：system 注入与空 think 块
  - **loss mask：generate_labels 的滑动匹配**
    - 3.1 两段锚点 token：bos_id 与 eos_id
    - 3.2 双层 while 扫描：标注区间与跳转
    - 3.3 找不到 eos_id 时的收尾
  - **另外三种 Dataset 与 max_length 的预算**
    - 4.1 DPODataset：在 Dataset 层就把 x / y 错开
    - 4.2 RLAIFDataset：只渲染到「轮到 assistant 说话」为止
    - 4.3 AgentRLDataset：不渲染、不 tokenize 的原生对象
    - 4.4 定长方案的账：截断率与 padding 占比（选学）

### Day 9 · Pretrain 与 SFT 的训练循环

一个 batch 在 train_epoch 里走完的七拍：lr、autocast、累积、裁剪、日志、续训

- ⏱ 90 分钟 · 🧪 13 个实验 · ✍️ 22 道选择题
- 📄 精读：`trainer/train_pretrain.py#L24-L81`、`trainer/trainer_utils.py#L40-L157`、`trainer/train_full_sft.py#L84-L110`
- 🎯 学完能做到：
  - 能按顺序说出 train_epoch 一次迭代的七拍，以及每一拍读写了哪个对象
  - 能写出 get_lr 的 cosine 公式、算出起点和终点，并说清楚喂进去的是全局 step 而不是 epoch 内的步号
  - 能解释 loss 为什么要除以 accumulation_steps、日志里为什么又乘回去，以及「累积 N 步 = 大 batch」在什么条件下不成立
  - 能说出 unscale_ → clip → step → zero_grad 这个顺序里每一步的必要性，以及 GradScaler 在 bfloat16 / 无 CUDA 时是个空壳
  - 能对着 lm_checkpoint 和 SkipBatchSampler 说出续训存了什么、恢复时靠什么做到不多训也不漏训
  - 能列出 train_pretrain.py 与 train_full_sft.py 的全部差异，并解释学习率差两个数量级意味着什么
- 🧵 主线：跟着一个 (input_ids, labels) 走完 train_epoch 的一次迭代：算 lr → autocast 前向出 loss → 除以 N → backward 攒梯度 → 每 N 步 unscale/clip/step/zero → 日志与 checkpoint
- 📍 5 章 · 13 个小节（一次只看一个）：
  - **一次迭代的七拍**
    - 1.1 train_epoch 的七拍与它依赖的全局变量
    - 1.2 get_lr：一条没有 warmup 的余弦
    - 1.3 喂给 get_lr 的是全局 step
  - **前向那三行**
    - 2.1 autocast_ctx 的那行 device 判断
    - 2.2 res.loss + res.aux_loss 与那次除法
    - 2.3 累积 N 步等于大 batch 的条件
  - **真正改权重的五行**
    - 3.1 GradScaler：只有 fp16 才真的在缩放
    - 3.2 clip_grad_norm_ 与它的返回值
    - 3.3 更新的顺序，与 epoch 末尾的尾巴
  - **存盘与续训**
    - 4.1 lm_checkpoint 存的两个文件
    - 4.2 SkipBatchSampler：跳过已经训过的 batch
  - **Pretrain、SFT 与多卡**
    - 5.1 两个脚本的差异清单
    - 5.2 DDP 动过的那几处

### Day 10 · LoRA 与知识蒸馏：手写实现，不套 peft

给方阵 Linear 挂一条 768→16→768 的旁路；再用 teacher 的软分布当标签

- ⏱ 90 分钟 · 🧪 13 个实验 · ✍️ 21 道选择题
- 📄 精读：`model/model_lora.py#L1-L65`、`trainer/train_lora.py#L128-L152`、`trainer/train_distillation.py#L25-L93`
- 🎯 学完能做到：
  - 能报出 LoRA 旁路上每一步的 shape，并用 r·(in+out) 算出任意 rank 下的参数占比
  - 能说清 apply_lora 的筛选条件为什么只命中 q_proj/o_proj，以及把这个条件去掉会撞上什么
  - 能解释 forward_with_lora 的两个默认参数在防什么，以及去掉之后为什么不报错却算错
  - 能说出 save_lora / load_lora / merge_lora 各自存了什么、靠什么 key 对上、误差从哪来
  - 能写出 distillation_loss 的公式，说清 kl_div 的参数顺序、batchmean 的分母、T² 在补偿什么
- 🧵 主线：跟着 q_proj 旁边那条旁路走：A/B 怎么让 ΔW 从 0 开始、apply_lora 怎么把它挂上去、只训练它、存成 195 KB 再合并回 W；最后换一条路，用 teacher 的 [M, V] 软分布喂 student
- 📍 4 章 · 13 个小节（一次只看一个）：
  - **LoRA 模块：一条 768 → 16 → 768 的旁路**
    - 1.1 A 与 B 的两条边：rank 卡在中间
    - 1.2 B 的全零初始化：挂上去的那一刻等于没挂
  - **apply_lora：一道筛子、一次 monkey-patch、一对默认参数**
    - 2.1 in_features == out_features 这道筛子
    - 2.2 被换掉的 forward 与被自动收编的子模块
    - 2.3 layer1= 和 layer2= 这两个默认参数
  - **训练、存盘、合并与热插拔**
    - 3.1 冻结之后真正在动的那 0.6%
    - 3.2 save_lora 存下来的 195 KB
    - 3.3 merge_lora：把 B @ A 加回 W
    - 3.4 同一个基座热插拔两份 LoRA
  - **知识蒸馏：把 teacher 的软分布当标签**
    - 4.1 distillation_loss 的四行与 kl_div 的参数顺序
    - 4.2 温度 T 与那个 T²
    - 4.3 从 [B,T] 的 labels 到 [M,V] 的蒸馏输入
    - 4.4 alpha 混合，以及整个 student 都在训练

## 第三阶段 · 对齐与 Agent（Day 11-14）

*DPO / PPO / GRPO / Agent RL 与部署*

### Day 11 · DPO：把偏好对压成一个标量 loss

一对 (chosen, rejected) → 拼成 [2B,T] → 两次前向 → 一个数

- ⏱ 85 分钟 · 🧪 12 个实验 · ✍️ 23 道选择题
- 📄 精读：`trainer/train_dpo.py#L25-L50`、`trainer/train_dpo.py#L57-L87`、`dataset/lm_dataset.py#L148-L174`
- 🎯 学完能做到：
  - 能背出降维链上每一步的 shape：[2B,T,V] → [2B,T] → [2B] → 两个 [B] → margin [B] → 标量，并说出每一步是哪一行代码
  - 能解释 DPO 的 y 为什么不能用 -100、mask 为什么必须是独立的 0/1 张量
  - 能说清 `[:batch_size // 2]` 这个切分约定唯一依赖什么，写反了会出现什么（以及为什么看 loss 发现不了）
  - 能推出「policy == ref ⇒ loss ≡ ln 2」，并算出此时每个 mask=1 的 token 拿到的梯度是 ∓β/(2n)
  - 能说出 ref_model 的三道冻结各自防什么、默认 lr 为什么只有 4e-8、用 sum 聚合带来的长度偏置从哪来
- 🧵 主线：一对 (chosen, rejected) 怎么一路塌成一个标量：cat 成 [2B,T] → policy/ref 两次前向 → logits [2B,T,V] → gather 成 [2B,T] → mask 求和成 [2B] → 切两半作差得到 margin [B] → logsigmoid → 标量
- 📍 4 章 · 13 个小节（一次只看一个）：
  - **偏好对的数据形状：从一对样本到 [2B, T]**
    - 1.1 DPODataset 返回的六个张量
    - 1.2 框住 assistant 段的 0/1 mask
    - 1.3 torch.cat 定下的 chosen/rejected 约定
  - **两次前向：从 logits 到序列级 log prob**
    - 2.1 log_softmax 与 gather 的两步降维
    - 2.2 mask 加权求和得到的序列级 log prob
    - 2.3 [2B, T, V] 那一份的显存代价（选学）
  - **dpo_loss：七行代码压出一个标量**
    - 3.1 切两半、两次作差、一个 margin
    - 3.2 policy == ref 时的 0.6931
    - 3.3 beta 的两个身份
  - **训练循环：两个模型、一个极小的学习率**
    - 4.1 ref_model 的三道冻结
    - 4.2 十六步迷你 DPO 的 loss 与 log prob 曲线
    - 4.3 默认学习率 4e-8 的来历
    - 4.4 求和聚合带来的长度偏置

### Day 12 · PPO：四个模型、GAE 与 clip 目标

一条 response 拿到的一个标量 reward，怎么变成 [B,R] 上每个 token 的梯度

- ⏱ 95 分钟 · 🧪 13 个实验 · ✍️ 21 道选择题
- 📄 精读：`trainer/train_ppo.py#L29-L307`、`trainer/rollout_engine.py#L24-L96`
- 🎯 学完能做到：
  - 说出 PPO 四个模型各自从哪初始化、谁要梯度、谁持有优化器状态
  - 默写 ppo_train_epoch 的数据流，并报出每个中间 tensor 的 shape
  - 说清 logp_pos 为什么取 P-1 … P+R-2，以及同一份下标被用在哪三处
  - 说清序列级 reward 怎么进入逐 token 的 GAE、两层 mask 各挡住了什么
  - 解释 ratio 的起点、clip 的单边性、以及 value clip 同样会把梯度削成 0
- 🧵 主线：序列级的 rewards [B] 怎么一步步摊成 token_rewards [B,R] → advantages [B,R]，再乘上 ratio 变成一次带 clip 的梯度
- 📍 5 章 · 13 个小节（一次只看一个）：
  - **四个模型与一次 rollout**
    - 1.1 四个模型的来源与梯度归属
    - 1.2 CriticModel：把 [B,T,C] 压成 [B,T]
    - 1.3 rollout 一次产出的六个字段
  - **逐 token 对齐：logp 与两层 mask**
    - 2.1 logits_to_keep 与末尾切一刀
    - 2.2 full_mask、logp_pos 与 resp_policy_mask
  - **从序列级 reward 到逐 token advantage**
    - 3.1 calculate_rewards：四个分量凑一个标量
    - 3.2 reward 落位：只加在最后一个有效 token 上
    - 3.3 GAE 的倒序递推
    - 3.4 全局归一化与 returns 的算账顺序
  - **actor 的更新目标：ratio 与 clip**
    - 4.1 ratio 的起点
    - 4.2 clip 的单边性与 clipfrac
  - **critic 的 value loss 与整轮更新循环**
    - 5.1 value loss 的双重裁剪
    - 5.2 双层更新循环、早停与两个 optimizer

### Day 13 · GRPO 与 Rollout Engine

扔掉 critic，让同一个问题的 G 个回答互相当裁判

- ⏱ 90 分钟 · 🧪 12 个实验 · ✍️ 21 道选择题
- 📄 精读：`trainer/train_grpo.py#L71-L153`、`trainer/rollout_engine.py#L24-L96`
- 🎯 学完能做到：
  - 能报出 GRPO 一个 step 里每个 tensor 的 shape：[B] → [B,P] → [B*G,P+R] → [B*G] → [B,G] → [B*G,R] → 标量
  - 能解释 [B*G] 为什么必须是 repeat_interleave 的排布，advantage 在哪一维归一化，+1e-4 兜的是什么
  - 能默写 completion mask 那 4 行，说清 is_eos.any() 兜底和 <= 闭区间各自防的是什么
  - 能说清三份 logps 谁带梯度、k3 估计器好在哪、grpo 和 cispo 的梯度差别出现在什么条件下
  - 能画出 RolloutEngine 的边界：哪些事归引擎、哪些事归训练脚本，sglang 引擎为此多做了什么
- 🧵 主线：一个 prompt 采样出 G 条 completion [B*G, R]：每条一个标量 reward → 在 [B, G] 上组内归一化成 advantage → 配上 completion mask 和逐 token logp → 一个标量 loss
- 📍 4 章 · 12 个小节（一次只看一个）：
  - **组采样：一个 prompt 变成 G 条轨迹**
    - 1.1 组采样的行序：repeat_interleave 与 i*G+j
    - 1.2 序列级的 reward：一条轨迹一个标量
    - 1.3 组内归一化：把同组的平均分当 baseline
  - **token 级对齐：completion mask 与三份 logps**
    - 2.1 第一个 EOS 说了算的 completion mask
    - 2.2 logp_pos：错一位的取数下标与 full_mask 的覆写
    - 2.3 三份 per-token logps 与唯一的梯度入口
  - **loss：KL 估计器与两个变体**
    - 3.1 k3 估计器：非负且低方差的 KL
    - 3.2 grpo 的 clip 与 cispo 的 detach 权重
    - 3.3 两级平均：每条轨迹在 loss 里等权
  - **Rollout Engine 与一次完整的 step**
    - 4.1 rollout 与 update_policy 划出的边界
    - 4.2 SGLang 引擎：变长结果与权重同步（选学）
    - 4.3 把整条链路跑一遍

### Day 14 · Agent RL、部署与总复习

多轮 tool-call 轨迹上的 0/1 mask → reward → 把模型端出去 → 14 天串成一条线

- ⏱ 95 分钟 · 🧪 14 个实验 · ✍️ 28 道选择题
- 📄 精读：`trainer/train_agent.py#L76-L186`、`trainer/train_agent.py#L189-L245`、`trainer/train_agent.py#L248-L338`、`scripts/serve_openai_api.py#L83-L102`、`scripts/convert_model.py#L16-L62`
- 🎯 学完能做到：
  - 能逐行说清 rollout_single 的多轮循环，以及 response_mask 上每一个 0/1 是怎么来的
  - 能说出 marker 哨兵解决的是什么问题，以及不用它会在哪几种输出上切错
  - 能默写 packing 那五行，解释 old_logps 为什么从 len(p)-1 开始填、completion_mask 为什么要左移一位
  - 能列出 calculate_rewards 的全部规则和分值，手算一条轨迹的总分，并说出组内 advantage 什么时候全是 0
  - 能把 out/*.pth 转成 transformers 目录或 Qwen3 结构，并说清 trust_remote_code 到底在信任什么
- 🧵 主线：一条多轮 tool-call 轨迹：response_ids 里模型写的 token 和环境喂进来的 token 交错排列，靠一条等长的 0/1 response_mask 决定谁算 loss
- 📍 5 章 · 14 个小节（一次只看一个）：
  - **多轮 rollout：一条轨迹是怎么滚出来的**
    - 1.1 工具的三张表与 parse_tool_calls 的容错
    - 1.2 工具结果回填之后的那一段 prompt
    - 1.3 rollout_single 的多轮循环与全量重放
    - 1.4 marker 哨兵：切出模板新增的那一截
  - **response_mask：哪些 token 参与 loss**
    - 2.1 三个列表同步 extend 与被置 0 的 eos
    - 2.2 把 prompt 和 response 打包成一条右 padding 的序列
    - 2.3 左移一位的 completion_mask 与 eos 截断
  - **reward：一条轨迹值多少分**
    - 3.1 两个分支与它们的分值表
    - 3.2 组内归一化：G 条轨迹互相当基准
  - **把模型端出去：服务与格式转换**
    - 4.1 parse_response：一段文本拆成 OpenAI 的三个字段
    - 4.2 register_for_auto_class 与 trust_remote_code
    - 4.3 同一份权重灌进 Qwen3ForCausalLM
  - **14 天总复习**
    - 5.1 从 jsonl 到能调工具的模型
    - 5.2 五种 loss mask 的对照
