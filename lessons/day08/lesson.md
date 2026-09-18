---
day: 8
format: points
title: "数据管线：从 jsonl 到 (input_ids, labels)"
subtitle: "五种 Dataset 共用一套 chat 模板，真正的教材是 loss mask"
minutes: 85
mainline: "一行 jsonl 怎么变成模型吃的张量：apply_chat_template 渲染成文本 → tokenizer → 定长 input_ids → generate_labels 打出 loss mask → DataLoader 拼成 batch"
files:
  - dataset/lm_dataset.py#L37-L119
  - dataset/lm_dataset.py#L122-L252
goals:
  - 能背出 PretrainDataset 的长度账：max_length-2、手动拼的 BOS/EOS、右 padding、-100 各占哪几格
  - 能手动跑一遍 generate_labels 的滑动匹配，说清多轮对话和截断边界分别怎么处理
  - 能解释 DPODataset 为什么在 Dataset 层就把 x/y/mask 错开，而 Pretrain/SFT 不用
  - 知道 RLAIFDataset/AgentRLDataset 为什么不 tokenize，AgentRL 为什么必须配自定义 collate_fn
  - 能用「截断率 vs padding 占比」两个数为一份数据挑 max_length
---

Day 7 看完 `generate` 怎么把 `input_ids` 采样成文字。今天往回退一步，看训练开始前的那半步：**一行 jsonl 怎么变成模型 `forward` 吃的两个张量**。这一步常被当成「不就是 tokenize 一下嘛」，但 MiniMind 从 pretrain 到 agentic RL 共五个阶段，数据格式、要不要 tokenize、labels 怎么打、shift 谁来做，每个阶段都不一样——`dataset/lm_dataset.py` 这 255 行决定了 Day 9~14 每个训练脚本长什么样。

今天只跟一条数据流走，全天的核心证据也只有一个：**哪些位置参与 loss**，我们会反复把它画成 token 彩带。

四章分工：第一章用最简单的 `PretrainDataset` 跑通「定长张量」这条流水线，建立长度账和 `-100` 的语义；第二章进入对话数据，看 `apply_chat_template` 怎么渲染 `conversations`；第三章是全天最重的一章——`generate_labels` 的滑动匹配；第四章横向对比另外三个 Dataset，最后算一笔定长方案的账。

{{flow: 一行 jsonl | *apply_chat_template* | 一段 ChatML 文本 | *tokenizer* | input_ids [max_length] | *generate_labels* | labels / loss mask | DataLoader | batch [B, max_length]}}

# PretrainDataset：定长张量的最短流水线

`PretrainDataset` 只有 19 行，是五个 Dataset 里唯一能一眼看完的。这一章用它把「一条样本占满一个定长窗口」这件事讲透：第一节数长度，第二节看 `labels` 的两个约定。后面三章全部是在这条骨架上加东西。

## 手动拼的 BOS/EOS 与 max_length-2 的预算 {#pretrain-ids}

pretrain 数据是最简单的一行 `{"text": "..."}`，没有角色、没有轮次。`__getitem__` 里真正干活的就是下面这三行（第四行只是转成 tensor）：

{{source:dataset/lm_dataset.py#L49-L52}}

第一行的三个参数各管一件事。`add_special_tokens=False` 关掉了 tokenizer 自动插特殊 token 的行为——因为下一行要**手动**拼 `[bos_token_id] + tokens + [eos_token_id]`，自动和手动只能留一个。`max_length=self.max_length - 2` 给这两个手动 token 各预留一格：正文最多占 `max_length-2`，加上头尾正好填满 `max_length`。`truncation=True` 让超长文本在这一步被砍掉尾巴。

注意截断发生的**位置**：它只作用在正文上，BOS/EOS 是截断之后才拼上去的，所以**再长的样本也一定有 `<|im_start|>` 开头、`<|im_end|>` 结尾**——这和「先拼好再整体截断」结果完全不同，后者会把结束符切掉。实验表格里 `idx=1` 那一行（原文 35 个 token、`max_length=24`）就是被截断的样本。

第三行 `tokens + [pad_token_id] * (max_length - len(tokens))` 是右 padding，短样本用 `<|endoftext|>` 把右边补满。三段加起来恒等于 `max_length`，所以 `DataLoader` 直接 stack 就能得到 `[B, max_length]`，不需要任何 `collate_fn`。顺带记住三个特殊 token 的 id：`<|endoftext|>`=0（pad）、`<|im_start|>`=1（bos）、`<|im_end|>`=2（eos）。

{{lab:pretrain_ids}}

{{quiz:q1,q2}}

> [!KEY]
> `max_length-2` 是给手动拼的 BOS/EOS 预留的两格；截断只砍正文，所以任何一条 pretrain 样本都以 `<|im_start|>` 开头、`<|im_end|>` 收尾，右边用 `<|endoftext|>` 补满定长。

> [!MORE] MiniMind 没有做 packing
> 主流预训练框架通常会把多条短文本首尾相连塞满一个定长窗口（packing），把 padding 浪费压到接近 0。MiniMind 选了更好读的「一条样本一个窗口」，代价是短文本会浪费大量 padding——[定长方案的账那一节](#len-budget)会用真实数据量化。

## labels 的两个约定：-100 与不 shift {#pretrain-labels}

有了 `input_ids`，`labels` 只用两行就做出来了。这两行看着平淡，但定下了后面所有 Dataset 都遵守的两个约定。

{{source:dataset/lm_dataset.py#L53-L55}}

第一个约定是 **`-100` = 这个位置不产生 loss**。它不是 MiniMind 自创的暗号，而是 `nn.CrossEntropyLoss(ignore_index=-100)` 的默认值（Day 5 讲 `forward` 里的 loss 时用到过）。

第二个约定是 **`labels` 与 `input_ids` 逐位对齐、Dataset 这一层不做 shift**：`labels = input_ids.clone()`，`labels[i]` 就是 `input_ids[i]` 本身，而不是下一个 token。错位（`shift_logits` / `shift_labels`）留给模型的 `forward` 去做——这正是 `trainer/train_pretrain.py:36` 能直接写 `res = model(input_ids, labels=labels)` 的前提。

现在盯住最后那一行 `labels[input_ids == self.tokenizer.pad_token_id] = -100`。它是一次**逐元素的值比较**：拿整条 `input_ids` 和 `pad_token_id`（也就是 0）比，相等的位置全部置 `-100`。这和「把我们补上去的那段 padding 置 `-100`」听起来是一回事，但判据不同——一个看值，一个看位置。真实语料里 `<|endoftext|>` 常被用作文档分隔符，正文中间完全可能出现它，下面这条实验样本就带了一个。

{{predict:p1}}

{{lab:pretrain_labels}}

彩带给出了答案：正文中间那个 `<|endoftext|>`（下标 6）也被打成了 0，20 格里只有 12 格参与 loss——句子中间被挖了一个洞。同时 `input_ids[:4]` 和 `labels[:4]` 完全相同，印证了「不 shift」这条约定。

{{quiz:q3}}

> [!KEY]
> `labels` 与 `input_ids` 逐位对齐（shift 交给模型 `forward`），padding 位置按**值**置 `-100`——正文里字面出现的 `<|endoftext|>` 会被一起误伤。

> [!MORE] `-100` 省掉的只是 loss，不是算力
> 被标成 `-100` 的位置不算 loss、也不产生梯度，但它们照样占着 attention 的 K/V 位置，显存分配和矩阵乘法的 FLOPs 全都是按 `max_length` 算的——[定长方案的账那一节](#len-budget)会把这笔浪费量出来。

# SFTDataset：把 conversations 渲染成一段文本

从这一章起数据变成多轮对话。`SFTDataset.__getitem__` 比 pretrain 多了一步：先把 `conversations` 这个结构化的 list 渲染成一段扁平的 ChatML 文本，再交给 tokenizer。这一章的三个小节分别看渲染本身、工具字段的反序列化，以及藏在渲染前后的两处随机扰动。

## apply_chat_template 渲染出来的 ChatML 文本 {#sft-template}

SFT 数据每行是 `{"conversations": [{role, content, ...}, ...]}`，每条 message 最多带五个字段：`role`、`content`、`reasoning_content`（思考过程）、`tools`（工具 schema，JSON 字符串）、`tool_calls`（工具调用，JSON 字符串）。`create_chat_prompt` 的结尾就是把它们交给 tokenizer 的模板引擎：

{{source:dataset/lm_dataset.py#L81-L86}}

`tokenize=False` 是关键——先只要文本，tokenize 留到 `__getitem__` 里统一做。模板（Day 1 读过的 `tokenizer_config.json` 里那段 Jinja）把每条 message 包成 `<|im_start|>{role}\n{content}<|im_end|>\n`，其中 assistant 那一条额外多一个 `<think>...</think>` 块：没有 `reasoning_content` 时它是空的 `<think>\n\n</think>\n\n`。

`add_generation_prompt=False` 的意思是**不在结尾追加 `<|im_start|>assistant\n`**。SFT 训练要的是完整对话，最后一条 assistant 回复已经躺在数据里了；只有推理（Day 7 的 `generate`）和 RLAIF 才需要把这个「开头」留给模型自己续写。

{{lab:sft_template}}

{{quiz:q4}}

> [!KEY]
> 渲染这一步只产出文本（`tokenize=False`）；`add_generation_prompt=False` 表示不给结尾留 `<|im_start|>assistant\n`——SFT 要的是完整对话。

> [!MORE] 为什么 tokenizer 没有重复插入特殊 token
> `<|im_start|>` / `<|im_end|>` 是以**字面文本**形式写进渲染结果的，而 `__getitem__` 里那句 `self.tokenizer(prompt).input_ids` 并没有传 `add_special_tokens=False`——之所以不会被重复插入，靠的是 `tokenizer_config.json` 里的 `add_bos_token=False` / `add_eos_token=False`。实验里第 0 个 token 就是模板自己写的那个 `<|im_start|>`。

> [!MORE] `Features` 为什么要显式写出五个字段
> `__init__` 里 `load_dataset('json', ..., features=Features({'conversations': [{...五个 Value('string')...}]}))`（L63）显式声明了 schema。不传 `features` 时 `datasets` 会分批扫数据自己推断；小文件整个落在一批里看不出问题，但大文件跨多个批次、而不同批次遇到的字段集合不一样（有的样本带 `tools`，有的不带）时，各批推断出的 schema 不一致，拼接时直接报 `DatasetGenerationError`。显式声明之后，缺失字段一律填 `None`，后面的 `.get('tools')` 才敢随便调。

## tools 与 tool_calls 的反序列化 {#sft-tools}

`create_chat_prompt` 的前半段是一个看起来很啰嗦的循环，其实只干两件事，而且两件都是**类型转换**。

{{source:dataset/lm_dataset.py#L74-L80}}

原因在上一节的 `Features`：`tools` 和 `tool_calls` 在 schema 里都是 `Value('string')`，所以从 jsonl 读出来是 **JSON 字符串**。而模板引擎要的是结构化对象——`{%- for tool in tools %}{{ tool | tojson }}` 要遍历一个 list of dict，`{{ tool_call.name }}` 要访问属性。类型不对就渲染不出来。

第一件事：扫一遍 messages，找到 `role == "system"` 且带 `tools` 的那条，把它的 `tools` 字符串 `json.loads` 成 list，存进局部变量 `tools`，最后作为 `apply_chat_template(..., tools=tools)` 的参数传出去。注意工具 schema **不是**跟着那条 system 消息走的，而是被抽出来当一个独立参数——模板里 `{%- if tools %}` 这个分支会把它渲染成一整段 `# Tools ... <tools>...</tools>` 的系统提示。

第二件事：把每条 assistant 消息的 `tool_calls` 从字符串反序列化成 list，让模板能拼出 `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`。还有 `message = dict(message)` 这行拷贝——`datasets` 返回的 row 不能就地改，拷一份才安全。

少做哪一次 `json.loads` 会怎样？两种漏法报的错完全不同，实验把它们并排列了出来。

{{lab:sft_tools}}

{{quiz:q5}}

> [!KEY]
> jsonl 里 `tools` / `tool_calls` 是 JSON 字符串，`create_chat_prompt` 必须先 `json.loads` 成对象；`tools` 还要从 system 消息里抽出来当独立参数传给模板。

## 两处随机扰动：system 注入与空 think 块 {#sft-randomness}

`SFTDataset.__getitem__` 在渲染前后各夹了一个函数：`pre_processing_chat(sample['conversations'])` 和 `post_processing_chat(prompt)`。它们都是**数据增强**，而且都带随机性——同一个 `index` 取两次，拿到的文本可能不一样。

{{source:dataset/lm_dataset.py#L26-L35}}

`pre_processing_chat`（渲染前，作用在 messages 上）：如果第一条不是 system 消息，就以 `add_system_ratio`（默认 0.2）的概率从 10 条候选里随机挑一条塞到最前面，让模型见过「有 system / 没 system」两种输入分布。

`post_processing_chat`（渲染后，作用在字符串上）：条件是 `'<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio`。注意这里是 `>` 而不是 `<`：`empty_think_ratio=0.2` 意味着有 **80% 的概率把空 think 块删掉**，只有 20% 的概率保留。这样模型既见过「先空想一下再回答」的格式，也见过「直接开口」的格式，推理时才能灵活选择。

这两处随机性有个很实际的后果：**同一条样本在不同 epoch 会被渲染成不同的文本，loss mask 的位置也跟着变**。下面的实验把同一条样本取 200 次，数一数两个事件各出现了多少次。

{{lab:sft_randomness}}

200 次里 system 出现 48 次（24%）、空 think 块保留 35 次（18%），都在 20% 附近抖动——没抽到正好 40 次是正常的采样波动。

{{quiz:q6}}

> [!KEY]
> `add_system_ratio=0.2` → 20% 概率注入 system；`empty_think_ratio=0.2` 配 `random() >` 号 → **80%** 概率删掉空 think 块。带 `tools` 的样本跳过前一处扰动。

> [!MORE] tool-use 数据为什么不做 system 注入
> `pre_processing_chat` 的第一行是 `if any(conv.get('tools') for conv in conversations): return conversations`——只要任何一条 message 带 `tools`，整条对话**原样返回**，一个字都不动。因为工具 schema 会被模板渲染成一整段自带的 system 提示（`# Tools ... <tools>...</tools>`），再随机塞一条进去只会打架。

# loss mask：generate_labels 的滑动匹配

渲染出的文本里，user 说的话、system 提示、工具返回值全都不该参与 loss——模型只需要学会「轮到我说话时说什么」。`generate_labels` 用一段纯 python 的滑动匹配把这件事做出来，是全天信息密度最高的 17 行。三个小节依次看：拿什么当锚点、怎么扫、扫不到收尾锚点时怎么办。

## 两段锚点 token：bos_id 与 eos_id {#mask-anchors}

滑动匹配要先有「找什么」。`SFTDataset.__init__` 的最后两行就是在准备这两把尺子：

{{source:dataset/lm_dataset.py#L65-L66}}

这两行是全篇最容易被误读的地方。`tokenizer.bos_token` / `eos_token` 在一般分词器里指「整段文本的开始 / 结束」，但 MiniMind 的 `tokenizer_config.json` 里 `bos_token = "<|im_start|>"`、`eos_token = "<|im_end|>"`——它们其实是 ChatML 的**角色分隔符**。所以这里拼出来的是字符串 `<|im_start|>assistant\n` 和 `<|im_end|>\n`，标志的是「一段 assistant 回复的开始 / 结束」，不是整个序列的首尾。（第一章里 pretrain 拿同样两个 token 当真正的 BOS/EOS 用，属于一词两用。）

关键在于：编码之后它们各是**一小段 token 序列**，不是单个 id。`<|im_start|>assistant\n` 是 5 个 token（`<|im_start|>` / `ass` / `ist` / `ant` / `\n`），`<|im_end|>\n` 是 2 个。为什么不能只匹配第一个 id？因为 `<|im_start|>user\n`、`<|im_start|>system\n`、`<|im_start|>tool\n` 的第 0 个 id 全都是 1——只看它分不出是谁在说话，角色信息藏在后面那几个普通 token 里。这就是 `generate_labels` 必须写成切片比较 `input_ids[i:i+len(self.bos_id)] == self.bos_id` 的原因，而 `len(self.bos_id) = 5` 马上还会被用来跳过这段模板脚手架。

{{lab:mask_anchors}}

{{quiz:q8}}

> [!KEY]
> `bos_id` = `<|im_start|>assistant\n` 的 5 个 token，`eos_id` = `<|im_end|>\n` 的 2 个 token；四种角色的第 0 个 id 都是 1，所以必须整段切片比较。

## 双层 while 扫描：标注区间与跳转 {#mask-scan}

有了两把尺子，`generate_labels` 就是一个双层 while 扫描：先把 `labels` 全部初始化成 `-100`，再在 `input_ids` 上从左往右走。

{{source:dataset/lm_dataset.py#L88-L104}}

外层 `while i < len(input_ids)`：在位置 `i` 处切片比对 `bos_id`，匹配上就说明进入了一段 assistant 回复，`start = i + len(self.bos_id)`——**跳过 `<|im_start|>assistant\n` 这 5 个 token 本身**，它们属于模板脚手架，不用模型学；没匹配上就 `i += 1` 往右挪一格。内层 `while end < len(input_ids)`：从 `start` 往后找 `eos_id`，找到就 `break`，`end` 停在 `<|im_end|>` 那一格。

然后 `for j in range(start, min(end + len(self.eos_id), self.max_length)): labels[j] = input_ids[j]`——把 `[start, end+2)` 这个左闭右开区间的 `labels` 填成 `input_ids` 本身。右端的 `end + len(eos_id)` 意味着**收尾的 `<|im_end|>\n` 也参与 loss**：模型必须学会什么时候停下来，否则永远不会主动结束生成。

标完这一段，`i = end + len(self.eos_id)` 跳到段尾继续外层循环——**所以多轮对话里每一轮 assistant 回复都会被单独找到、单独标注，不只是最后一轮**，这是对 SFT 最常见的误解。

{{lab:mask_scan}}

彩带里三段亮区一目了然，`参与 loss 的 token 数` 是 29 / 96（下一节的表格会把这三段的下标列出来：`[15,23]`、`[40,48]`、`[62,72]`）。注意每段都从 `<think>` 亮起：`start` 之后的第一个内容 token 正是模板渲染的 `<think>`，所以空 think 块（没被 `post_processing_chat` 删掉时）也计入 loss。

{{quiz:q9,q10}}

> [!KEY]
> `generate_labels` 从 `bos_id` 之后标到 `eos_id` 结束（含收尾的 `<|im_end|>\n`），扫完一段就跳过去接着扫——多轮对话里每一轮 assistant 回复都参与 loss。

## 找不到 eos_id 时的收尾 {#mask-truncate}

上一节默认每段 `<|im_start|>assistant\n` 后面都能找到配对的 `<|im_end|>\n`。但 `__getitem__` 里有一句 `input_ids = self.tokenizer(prompt).input_ids[:self.max_length]`：文本超长时会被硬切，切口完全可能落在某一轮 assistant 回复的正中间，这一段就永远等不到收尾锚点了。

{{source:dataset/lm_dataset.py#L95-L101}}

这种情况下内层 `while` 一路走到底，`end` 涨到 `len(input_ids)` 才退出循环（注意是 while 条件失败退出，不是 `break`）。接下来两行各有一个防御：`for` 的上界是 `min(end + len(self.eos_id), self.max_length)`，`i` 的更新是 `end + len(self.eos_id) if end < len(input_ids) else len(input_ids)`——后者保证外层循环一定能结束，不会因为 `i` 越界越算越远。

那么，那半段已经出现、却没有收尾符的 token，最终会得到什么 label？三种可能都说得通：被当成「没匹配成功」整段保持 `-100`；被 `end` 越界拖出一个 `IndexError`；或者照标不误。先下注，再看实验里 `max_length=64` 那三行。

{{predict:p2}}

{{lab:mask_truncate}}

答案是第三种：`max_length=64` 时第三段只剩 `[62,63]` 两个 token（`<think>` 和 `\n`），照样被标进了 loss。`min(..., self.max_length)` 只是把上界钳在窗口边界上，防止越界，并没有「匹配失败就整段作废」的逻辑。这带来一个很实际的后果：**长 system / 工具 schema 会实打实地挤压多轮对话里靠后轮次的训练信号**，被挤掉的部分悄无声息，不会报任何错。

{{quiz:q11}}

> [!KEY]
> 截断发生在 assistant 回复中间时，已出现的那部分 token **仍然计入 loss**，上界被 `min(end + len(eos_id), max_length)` 钳在窗口边界——没有报错，也没有整段作废。

# 另外三种 Dataset 与 max_length 的预算

`PretrainDataset` / `SFTDataset` 交出的都是对齐的 `(input_ids, labels)`。这一章看剩下三个 Dataset 各自在哪一步分道扬镳：DPO 提前做了 shift，RLAIF 只吐字符串，AgentRL 连模板都不渲染。最后一节算一笔所有定长 Dataset 共同的账。

## DPODataset：在 Dataset 层就把 x / y 错开 {#dpo-shift}

`DPODataset` 的前大半段和 SFT 几乎重合：`chosen` / `rejected` 两组对话各走一遍 `apply_chat_template` + `post_processing_chat` + `tokenizer(..., padding='max_length')`，再各算一遍 `generate_loss_mask`——那个函数和 `generate_labels` 是**同一套滑动匹配**，只是一个填 token id、一个填 0/1。真正的分叉在 `__getitem__` 最后六行：

{{source:dataset/lm_dataset.py#L160-L165}}

`x = ids[:-1]` 去掉最后一个 token，`y = ids[1:]` 去掉第一个——`y[i]` 就是「喂 `x[i]` 进去时应该预测出来的那个 token」。`mask` 跟着 `y` 走，所以是 `[1:]` 而不是 `[:-1]`。副作用是**长度变成 `max_length - 1`**：传 `max_length=128` 拿到的是 `(127,)`。

为什么 SFT 不做这一步、DPO 却必须做？看 `trainer/train_dpo.py`（Day 11 细讲）：它是 `outputs = model(x)` 拿 `logits`，**不传 `labels`**，模型内置的 shift + CrossEntropy 路径根本没被触发；接着 `logits_to_log_probs(logits, y)` 手动 `gather` 出每个位置对应 `y` 的 log-prob，再 `(log_probs * mask).sum(dim=1)` 算偏好对比 loss。整条链路没人替它错位，所以错位必须在 Dataset 做完。记一个直觉：**谁负责算 loss，shift 就出现在谁的输入准备阶段**。

{{lab:dpo_shift}}

实验里 `mask` 从 `i=14` 开始变 1：此处 `x[14]` 还是 `<|im_start|>assistant\n` 的最后那个 `\n`，而 `y[14]` 已经是回复正文的第一个 token——正好是「该模型开口」的那一格。

{{quiz:q12,q13}}

> [!KEY]
> DPO 训练自己算 loss（`model(x)` + `gather(y)`），所以 Dataset 就得交出错位好的 `x=ids[:-1]` / `y=ids[1:]` / `mask=loss_mask[1:]`，长度是 `max_length - 1`。

> [!MORE] 两份滑动匹配代码
> `DPODataset.generate_loss_mask`（L176-L192）和 `SFTDataset.generate_labels`（L88-L104）逐行几乎一模一样：同样的双层 while、同样的 `min(end + len(eos_id), self.max_length)` 钳制，区别只有 `loss_mask[j] = 1` 和 `labels[j] = input_ids[j]` 这一行。两个类各写一份，没有抽成公共函数。

## RLAIFDataset：只渲染到「轮到 assistant 说话」为止 {#rlaif-prompt}

从这一节开始，Dataset 不再产出张量。`RLAIFDataset` 服务于在线 RL（PPO / GRPO，Day 12~13）：**回答不是数据里现成的，而是训练时用当前策略模型现场采样出来的**，所以 Dataset 只负责把「问题」准备好。

{{source:dataset/lm_dataset.py#L208-L216}}

三个参数每一个都在为「留白」服务。`conversations[:-1]` 扔掉数据里最后那条 assistant 消息——`rlaif.jsonl` 里它的 `content` 本来就是空的或者一句占位符，留着反而会让模型抄答案。`add_generation_prompt=True` 与 SFT 恰好相反：在结尾补上 `<|im_start|>assistant\n`，把话筒交出去。`open_thinking` 是 MiniMind 给模板加的自定义变量，控制这个话筒后面接什么：`True` 时只写半个 `<think>\n`（模型自己续写思考过程再自己闭合），`False` 时直接给一个闭合的空 `<think>\n\n</think>\n\n`（让它跳过思考直接答）。`thinking_ratio` 就是在这两种模式间按概率切换。

`__getitem__` 返回的是 `{'prompt': <str>, 'answer': ""}`——两个字符串，没有 tokenize、没有 padding、没有 labels。`answer` 恒为空字符串，因为答案要等 rollout 采样出来才有。

{{lab:rlaif_prompt}}

两行结尾分别是 `...<|im_start|>assistant\n<think>\n` 和 `...<|im_start|>assistant\n<think>\n\n</think>\n\n`，正好对上 `open_thinking` 的两个分支。

{{quiz:q14}}

> [!KEY]
> RLAIF 的 prompt 停在「轮到 assistant 说话」处（`conversations[:-1]` + `add_generation_prompt=True`），返回的是字符串不是张量；`thinking_ratio` 决定结尾是不是半开的 `<think>\n`。

> [!MORE] 一处死代码
> `RLAIFDataset.__init__` 里也定义了 `self.bos_id` / `self.eos_id`（L202-L203，写法还和 SFT/DPO 不同，少了结尾的 `\n`），但 `create_chat_prompt` / `__getitem__` 从未引用过它们——RLAIF 阶段还不存在 assistant 回复，没有 loss mask 可切。同类还有 `DPODataset.padding`（L127）以及 `RLAIFDataset` / `AgentRLDataset` 的 `self.max_length`。

## AgentRLDataset：不渲染、不 tokenize 的原生对象 {#agent-collate}

`AgentRLDataset` 走得更远——连 `apply_chat_template` 都不调用。

{{source:dataset/lm_dataset.py#L249-L252}}

`parse_conversations` 只做两件事：把 system 消息里的 `tools` 字符串 `json.loads` 出来，以及丢掉最后一条 assistant 消息（还是 `messages[:-1]`）。返回的三个字段全是原生 python 对象：`messages` 是 `list[dict]`、`tools` 是 `list[dict]` 或 `None`、`gt` 则是这个 Dataset 独有的字段——最终校验目标（比如数学题的正确答案），训练时拿模型跑出来的完整轨迹和它比，算 reward。

为什么模板渲染要往后推？因为 agentic rollout 是**多轮的**：模型先吐一个 `<tool_call>`，训练脚本真的去执行那个工具，把结果作为新的 `tool` 消息追加进 `messages`，再渲染一次喂回去……模板必须在 rollout 循环里边生成边做。

代价是返回值结构不定长也不统一：不同样本的 `messages` 轮数不同，`tools` 里的函数名和参数字段更是各不相同。交给 `DataLoader` 的默认 `collate_fn` 会怎样？

{{predict:p3}}

{{lab:agent_collate}}

默认 collate 报的是 `KeyError: 'location'`——它对 `list` / `dict` 是**递归**处理的：先按位置把两条样本的 `tools` 一一配对，再逐层往里拼 dict，走到某一层发现第一条样本有 `location` 这个 key、第二条没有，于是抛 `KeyError`。报错信息完全不提「批内样本结构不一致」这个真正原因。`trainer/train_agent.py:462` 里那个自定义 `collate_fn` 干脆什么都不拼，三个字段各自收集成 list 就交给 rollout 自己按样本处理。

{{quiz:q15}}

> [!KEY]
> `AgentRLDataset` 返回 `messages` / `tools` / `gt` 三个原生 python 对象（含 RL 专用的校验目标 `gt`），结构不定长，**必须**配自定义 `collate_fn`，否则默认 collate 会抛一个牛头不对马嘴的 `KeyError`。

## 定长方案的账：截断率与 padding 占比 {#len-budget .side}

`PretrainDataset` / `SFTDataset` / `DPODataset` 都是定长方案，`max_length` 是一个纯粹的权衡：定小了截断训练信号，定大了浪费显存和算力。

{{source:dataset/lm_dataset.py#L111-L112}}

两头的代价不对称。**定小**的损失是隐性的：截断发生在 assistant 回复中间时不会报错，[上一章那一节](#mask-truncate)已经看到，靠后轮次的训练信号被悄悄削掉。**定大**的损失是显性的：padding 位置 `labels=-100` 不产生 loss，但 attention 矩阵是 `[B, H, max_length, max_length]`、矩阵乘法的 FLOPs 也是按 `max_length` 算的，纯粹白烧。

实验对 `lora_identity.jsonl` 的全部 91 条样本量了一遍：中位数 60 个 token，最长 114。`max_length=64` 时 25.3% 的样本被截断，但 padding 只占 10.4%；`max_length=128` 时一条都不截断，padding 已经吃掉 53.7%；套上 `train_full_sft.py` 的默认值 768，padding 占比高达 **92.3%**——十分之九的算力在算 `<|endoftext|>`。

{{lab:len_budget}}

那为什么真实脚本还敢用 768？因为 `--max_seq_len` 是按**主线数据**的分布 + 显存预算拍的（`train_pretrain.py` 340、`train_full_sft.py` / `train_grpo.py` 768、`train_dpo.py` / `train_agent.py` 1024），而 `sft_t2t.jsonl` 这类主线数据里有大量样本远比这份 91 条的玩具身份数据长。换句话说：这两个数要针对**你自己的数据**重新量一遍再定。

{{quiz:q16}}

> [!KEY]
> 定长方案的两个成本一显一隐：截断悄无声息地削掉训练信号，padding 明明白白地烧算力；`max_length` 要按自己数据的长度分布量出来，别直接抄默认值。
