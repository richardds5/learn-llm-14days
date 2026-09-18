---
day: 1
format: points
title: "全景与 Tokenizer"
subtitle: "仓库地图 → 一次 forward 的鸟瞰 → 文字怎么变成 token"
minutes: 80
mainline: "跟着一句「天空是什么颜色的？」走完全程：chat_template 把它摊平成一个字符串，tokenizer 切成 input_ids [B,T]，一次 forward 出 logits [B,T,V]，最后一个位置决定下一个 token"
files:
  - model/model_minimind.py#L186-L253
  - trainer/train_tokenizer.py
  - model/tokenizer_config.json
goals:
  - 说得出 model / dataset / trainer / scripts 各自的边界，以及 out/{前缀}_{hidden_size}{_moe}.pth 这条全仓库统一的命名规则
  - 看得懂模块调用树：形参名、`tuple( … )` 里的元素名、被折叠的重复层，各自是什么意思
  - 说得清 logits [B,T,V] 里每个位置在预测什么，以及为什么「要生成的下一个 token」只能看 logits[0, -1]
  - 把 6400 拆成 256 + 6108 + 36，并解释 ByteLevel 为什么让 token 看起来像乱码却永远不会编码失败
  - 逐段讲得出 chat_template 的回合结构、assistant 分支怎么处理 `<think>`、tools 分支与 tool 角色合并
---

接下来 14 天，你会把 MiniMind 从「一个 token id 进来」一直读到「Agent RL 与部署」。今天是第一天，先建立坐标系。

先认路：`model/` `dataset/` `trainer/` `scripts/` 四个目录加一个 `eval_llm.py`，不到 4700 行 python，每一天都落在这张地图的某一格上。再顺一条真实的数据流：「天空是什么颜色的？」先被 `chat_template` 摊平成一个长字符串 → tokenizer 切成 `input_ids [B,T]` → 一次 forward 出 `logits [B,T,V]` → `logits[0, -1]` 挑出下一个 token。

四章：画地图、认权重命名；鸟瞰一次 forward，把 `logits` 的语义讲透；再掉头看入口——tokenizer 和 chat_template 是整条流水线里唯一从不更换的组件。模型内部怎么算（Attention / RoPE / MoE）今天当黑盒，那是 Day 2~6 的事。

{{flow: messages | *chat_template* | prompt 字符串 | *tokenizer* | input_ids [B,T] | MiniMindForCausalLM | logits [B,T,V] | *logits[0,-1]* | 下一个 token}}

# 仓库与训练流水线的地图

先把「东西都放在哪」和「产出叫什么名字」这两件事解决掉。这一章不读模型代码，只认路：四个目录各管什么、八个训练阶段怎么串成一条链、每一步吐出来的 `.pth` 为什么叫那个名字。

## 四个目录与一个手测脚本 {#repo-map}

MiniMind 的目录切分是按**生命周期**来的，不是按技术栈：

`model/` 只放**结构定义**——`model_minimind.py` 里 `MiniMindConfig` + `RMSNorm` + `Attention` + `FeedForward` + `MOEFeedForward` + `MiniMindForCausalLM` + `generate`，一共 288 行；`model_lora.py` 65 行。tokenizer 的两份产物文件（`tokenizer.json` / `tokenizer_config.json`）也放在这里，因为它们和权重一样是「模型的一部分」。

`dataset/` 只放**怎么把 jsonl 变成 tensor**：`lm_dataset.py` 里 5 个 `Dataset` 子类（Pretrain / SFT / DPO / RLAIF / AgentRL）加几个小语料。注意仓库里**没有** pretrain / sft 的主数据（那是几个 GB），所以本课的实验要么自己造几条文本，要么从小 jsonl 里取前几十行。

`trainer/` 是最大的一块（2800 多行）：每个训练阶段一个 `train_*.py`，公共部分抽在 `trainer_utils.py`，RL 采样抽在 `rollout_engine.py`。八个训练脚本的骨架高度雷同——读懂 Day 9 的 `train_pretrain.py`，后面几个只要看差异。

`scripts/` 是**训练结束之后**的事：格式转换、OpenAI 协议服务端、web demo、工具调用评测。根目录的 `eval_llm.py`（94 行）则是唯一一个「随手加载某个阶段的权重聊两句」的脚本。

{{lab:repo_map}}

{{quiz:q1}}

> [!KEY]
> `model/` = 结构，`dataset/` = 数据变 tensor，`trainer/` = 训练，`scripts/` = 训练之后，`eval_llm.py` = 手测；整个本体不到 4700 行 python。

> [!MORE] 本地实际有哪些权重和语料
> `out/` 里只有 5 个权重：`pretrain_768.pth`、`full_sft_768.pth`、`full_sft_768_moe.pth`、`lora_identity_768.pth`、`lora_medical_768.pth`。没有 dpo / ppo / grpo / reward 权重，所以 Day 11~13 的 RL 实验只能拿 `full_sft` 同时当 policy 和 reference，演示算法逻辑而不是真实效果。
> `dataset/` 里能用的 jsonl：`dpo.jsonl`、`rlaif.jsonl`、`lora_identity.jsonl`、`lora_medical.jsonl`、`lora_exam.jsonl`、`agent_rl.jsonl`、`agent_rl_math.jsonl`。

## 训练流水线与权重命名的死规则 {#pipeline-weights}

八个训练阶段串成一条链，越往后越依赖前面的产出：

```text
pretrain ──▶ full_sft ──┬──▶ lora          (冻结主干，只训旁路)
                        ├──▶ distillation  (教师 → 学生)
                        ├──▶ dpo / ppo / grpo
                        └──▶ agent         (多轮 Tool-Use RL)
```

每一步的产出都写进 `out/`，文件名规则**全仓库统一**。这不是我总结的约定，而是每个 `train_*.py` 里都复制粘贴的同一段代码：

{{source:trainer/train_full_sft.py#L61-L68}}

三件事值得记住。第一，文件名由三部分拼成：`{save_weight 前缀}` + `_{hidden_size}` + `{_moe 或空}`，前缀来自 argparse 的 `default`（`pretrain` / `full_sft` / `full_dist` / `dpo` / `ppo_actor` / `grpo` / `agent`）。LoRA 是唯一的例外，它用的参数叫 `--lora_name`，默认 `lora_medical`。

第二，`hidden_size` 进了文件名，意味着改模型宽度就**换了一套权重文件**。`eval_llm.py:22` 用同一条 f-string 反过来拼路径，所以 `--hidden_size` 传错就直接找不到文件——这是新手最常见的一个坑。

第三，存的是 `v.half().cpu()`，也就是 fp16。这里有个容易漏看的细节：`tie_word_embeddings=True` 时 `embed_tokens.weight` 和 `lm_head.weight` 训练中共享同一份存储，但 `v.half()` 会给每个 key 各自拷出一份新张量，把这层共享拆开了——存盘的 state_dict 里因此比模型真实的 63.9M 参数多存了一份 `[V,C]` 嵌入表，一共 68.8M 个值。`full_sft_768.pth` 是 137 MB ≈ 68.8M × 2 字节，pickle 本身的开销其实只有几十 KB。而 `lm_checkpoint(...)` 写进 `../checkpoints` 的是另一套东西：带 optimizer 状态、用于断点续训，Day 9 再看。

{{lab:pipeline_weights}}

{{quiz:q2}}

> [!KEY]
> `out/{前缀}_{hidden_size}{_moe}.pth` 是全仓库统一的死规则，前缀就是各 `train_*.py` 里 `--save_weight`（LoRA 是 `--lora_name`）的默认值；存盘时统一转 fp16。

# 一次 forward 的鸟瞰

现在把整个模型当一个黑盒，只看「什么进去、什么出来」。这一章先学会读模块调用树这种输出，再把 `logits [B,T,V]` 这个三维张量的语义彻底讲清楚——它是后面十几天里出现频率最高、也最容易想歪的一个 tensor。

## 模块调用树的读法 {#module-tree}

`trace(model)` 给的是**模块级**的调用树：每一行是一次 `nn.Module.__call__`，缩进表示嵌套，格式是 `模块路径 <类名>  入参  →  返回`。入参前面的名字不是编造的，是那个模块 `forward` 的**形参名**——`MiniMindBlock` 的第一个参数叫 `hidden_states`，`Attention` 的叫 `x`，`nn.Linear` 的叫 `input`。

{{source:model/model_minimind.py#L186-L194}}

**`tuple( … )` 是什么意思**：它表示这个实参或返回值本身是一个 python tuple（是 list 就显示 `list( … )`），括号里列的是它的各个**元素**。比如 `position_embeddings=tuple(cos=[T=7, D=96], sin=[T=7, D=96])` 要读成：传给 `MiniMindBlock` 的第二个参数是一对张量 `(freqs_cos 的切片, freqs_sin 的切片)`——它在 `MiniMindModel.forward` 的 L219 被打包，L 层共用同一对。`cos` / `sin` 这两个元素名来自真正解包它的那一行：`Attention.forward` 的 L118 写的是 `cos, sin = position_embeddings`，工具照着回填。

返回侧同理：`→ tuple(hidden_states=[B,T,C], present_key_value=None)` 对应 L194 的 `return hidden_states, present_key_value`。最外层那行不是 tuple 而是 `MoeCausalLMOutputWithPast{…}`，花括号里是它的字段。

最后注意 `… 后面 7 个 MiniMindBlock 结构相同，省略` 这一行——树只展开第 0 层，别误以为模型只有一层。

{{lab:module_tree}}

{{quiz:q3,q4}}

> [!KEY]
> 调用树里 `名字=` 是模块 `forward` 的形参名；`tuple( … )` 表示这个值本身是个 python tuple，括号内列的是它的元素，元素名来自真正解包它的那一行代码。

> [!MORE] `?` 是什么意思
> 树上还有一个符号：`presents=?`、`past_key_values: ?`。`?` 表示「这个值里一个张量都没有，工具不描述它」——`presents` 是 `MiniMindModel.forward` 收集每层 KV cache 的 list，`use_cache=False` 时它装的是 8 个 `None`。注意和 `present_key_value=None` 区分：后者是真的 `None`。

> [!MORE] 从这棵树能直接读出的三件事
> 1. 只有 `embed_tokens` 一处把 `int64` 变成 `float32`，只有 `lm_head` 一处把宽度从 `C=768` 变成 `V=6400`；中间所有 Block 的进出都是 `[B,T,C]`。
> 2. `k_proj` / `v_proj` 的输出是 `[B,T,KV*D=384]`，只有 `q_proj` 的一半——GQA，Day 4 讲。
> 3. `mlp.gate_proj` 把宽度顶到 `I=2432` 再由 `down_proj` 压回 768，参数量的大头就在这里（Day 2 会算账）。

## logits [B,T,V]：T 个位置同时在预测 {#logits-grid}

`lm_head` 是 `nn.Linear(C, V, bias=False)`，把 `[B,T,C]` 映射成 `[B,T,V]`：

{{source:model/model_minimind.py#L245-L248}}

关键在于这个形状**不是**「跑一次 forward 得到一个 6400 维分布」，而是「序列里的每一个位置各得到一个 6400 维分布」。位置 `t` 的那一条，预测的是 `t+1` 位置该是什么 token。之所以能这样，是因果掩码保证了位置 `t` 的隐状态只看得到 `0..t`（Day 4 细讲），不会偷看答案。

这条对齐关系正是训练效率的来源：一次 forward 就拿到了 `T-1` 个监督信号。L251 那行 `logits[..., :-1, :]` 配 `labels[..., 1:]` 就是把它写成代码——砍掉最后一个位置的预测（没有答案可对）和第一个位置的标签（没有谁去预测它）。loss 的细节 Day 5 再展开。

推理时反过来看：这 `T` 个分布里，前 `T-1` 个的「标准答案」已经写在 prompt 里了，只有最后一个的答案还不存在。实验用真实的 `full_sft` 权重把 23 个位置逐个摊开对照，猜对 9 个——而且分布极不均匀：user 提问那几格几乎全错（模型从没被要求猜用户会说什么，SFT 的 loss mask 只盖住 assistant 段，Day 8 细讲），而 `<|im_start|>assistant` 之后那串固定骨架几乎全对。

{{lab:logits_grid}}

{{quiz:q5}}

> [!KEY]
> `logits` 是 `[B, T, V]`：每个位置 `t` 都输出一个完整的词表分布，预测的是 `t+1`；训练靠这个一次拿到 `T-1` 个监督信号，推理只有最后一个位置是新的。

> [!MORE] `logits_to_keep` 这个参数在省什么
> L247 的 `slice_indices = slice(-logits_to_keep, None)`：推理时其实只要最后一个位置的 logits，前面 `T-1` 个位置的 `[C] → [V]` 全是白算的（`V=6400` 不小）。传 `logits_to_keep=1` 就能让 `lm_head` 只算一行。默认 `0` 等价于 `slice(0, None)`，也就是全都要。Day 7 的 generate 和 Day 12 的 PPO 会用到它。

## logits[0, -1] 到底是谁的概率 {#next-token}

这三个下标各自的含义：`0` = batch 里第 0 条样本，`-1` = 序列的最后一个位置，剩下的 6400 维 = 词表上每个 token 的分数。合起来读：**读完整段 prompt 之后，第 `T+1` 个 token 应该是什么**。

为什么偏偏是最后一个位置？接着上一节：位置 `t` 预测 `t+1`。`t = 0 … T-2` 的答案都已经躺在 prompt 里，你不需要模型告诉你；只有 `t = T-1`（也就是 `-1`）预测的那个 `T` 位置还是空的——那才是「要生成的下一个 token」。写成 `logits[:, 0, :]` 拿到的是 prompt 第一个 token 之后的预测，写成 `logits[-1, :, :]` 则是把时间维的下标错用到了 batch 维上（`B=1` 时不报错，但语义全错）。

接下来的实验会暴露一个非常容易误读的现象。问「天空是什么颜色的？」，`logits[0, -1]` 的 top-1 概率高达 0.89，而那个 token 恰好是问题的第一个字。先别急着下结论，题目就问这个。

{{predict:p1}}

{{lab:next_token}}

答案在贪心续写那一行：`'天空的颜色因颜色的不同而有所差异，但通常可以分为以下几种：…'`。模型并没有复读输入，它是在**开始回答**，而这个回答碰巧以「天」字开头（复述主语是中文回答的常见开场）。换成「你叫什么名字？」，top-1 就变成 `'我'`，没有任何重合。

顺带看清自回归的动作：把这个 token 拼回 `input_ids`，再 forward 一次，再取 `logits[0, -1]`……贪心解码时，续写的第一个 token 必然等于第一次 forward 的 top-1，实验最后两行 print 就是在对这件事。

{{quiz:q6}}

> [!KEY]
> `logits[0, -1]` = 「读完 prompt 之后要生成的那一个 token」的分布；top-1 和问题首字重合只是回答本身以那个字开头，不是在复读输入。

> [!MORE] 这一步之后还会发生什么
> `generate`（L257 起）拿到 `logits[:, -1, :]` 之后依次做：除以 `temperature`、按 `repetition_penalty` 压低已出现过的 token、`top_k` / `top_p` 截断、再 `multinomial` 采样。所以「top-1 是什么」只在 `do_sample=False`（贪心）时等于实际输出。Day 7 会逐行拆这个循环。

# Tokenizer：6400 个格子怎么来的

镜头掉头到这条链的入口。tokenizer 是整个仓库里唯一**从 pretrain 用到部署、中途从不更换**的组件：它一旦定下来，`vocab_size`、`embed_tokens` 的行数、每一条训练数据的切法就全都定死了。这一章把 6400 这个数字拆开，再看 ByteLevel 这个设计到底换来了什么。

## 6400 的分账：256 + 6108 + 36 {#vocab-6400}

`vocab_size=6400` 是一个**总预算**，不是「学 6400 次合并」：

{{source:trainer/train_tokenizer.py#L43-L48}}

`initial_alphabet` 里的 256 个字节先占坑，`special_tokens` 里的 36 个符号再占坑，剩下 `6400 - 256 - 36 = 6108` 个名额才拿去学 BPE merge。实验直接数 `model/tokenizer.json` 就能验证这三个数。

36 这个数是人为凑出来的（`SPECIAL_TOKENS_NUM = 36`）：21 个结构性 / 多模态占位符（`<|endoftext|>`、`<|im_start|>`、`<|im_end|>`，以及一批为图像、视频、音频预留的 `<|vision_*|>` / `<|audio_*|>`）、6 个功能标记（`<tool_call>` / `<tool_response>` / `<think>` 三对），再用 9 个 `<|buffer1..9|>` 填满到 36。留 buffer 是为了以后加新标记时直接占一个位，不用改 `vocab_size`、不用重训 embedding。

训练语料是 `sft_t2t_mini.jsonl` 的前 10000 行（L15 那个 `if i >= 10000: break`），仓库里没带这个文件，所以脚本你跑不动——但产物 `model/tokenizer.json` 就在手边，数一遍即可。

至于为什么只给 6400：`embed_tokens` 是 `[V, C]` 的一张表，`V` 越大它越吃参数。6400 时这张表占总参数量的 7.7%，换成 Qwen 量级的 32000 就是 29.4%（Day 2 会把这笔账算完）。代价是压缩率低——同样一句话要切成更多 token。

{{lab:vocab_6400}}

{{quiz:q7}}

> [!KEY]
> `vocab_size` 是总预算：256 个字节 + 36 个特殊符号先占坑，剩下 6108 个名额才用来学 merge。

> [!MORE] 为什么 README 劝你别重训
> 换一套 tokenizer 等于换一套「id ↔ 子词」的对应关系。已训练好的 `embed_tokens` 每一行都会对错号，所有 `.pth` 权重、所有已经 tokenize 好的数据、社区里别人的模型全部失配。这个脚本的第一行注释写的就是这件事——它只供学习，不供使用。

## ByteLevel：像乱码，但永远不会失败 {#bytelevel}

预分词器只有一行：

{{source:trainer/train_tokenizer.py#L25-L26}}

`ByteLevel` 的做法是：先把文本当成**字节流**，再把 256 个可能的字节各自映射到一个**可打印**的 unicode 替身（空格 → `Ġ`，换行 → `Ċ`，汉字的三个字节各自变成 `æ` `Ī` `ĳ` 这类字符），BPE 在这些替身上学合并。所以 `convert_ids_to_tokens` 打印出来的 `'æĪĳ'` 不是 bug，是「我」这个字三个字节的替身被合并成了一个 token；`tokenizer.decoder = decoders.ByteLevel()` 负责在 decode 时精确逆映射回原始字节。

换来的好处是**没有 `<unk>`**：任何字节序列都编得出来、也 decode 得回去，encode → decode 永远无损，不存在「生僻字失败」。代价是：一个汉字占 3 个字节，如果这 3 个字节没被合并成一个 token，就要两三个 token 才装得下它。

下面这个实验就切三个汉字。先猜一下它会被切成几个 token：

{{predict:p3}}

{{lab:bytelevel}}

`'我'` 和 `'是'` 各占一个 token（语料里太常见，merge 早就学到了），`'谁'` 却被切成两个——`'è°'` 和 `'ģ'`，单独 decode 各是一个 `'�'`（字节不完整），两个拼起来才显出「谁」。这正是流式输出必须**缓冲字节**的原因：一个 token 一个 token 地往外吐，中途会吐出半个汉字。

{{quiz:q8}}

> [!KEY]
> ByteLevel 让 token 长得像乱码（`'æĪĳ'` 就是「我」的三个字节），但保证任何文本都能无损 encode/decode；低频汉字会被拆成多个字节 token。

> [!MORE] 压缩率与 add_prefix_space
> 实验里 `'我是谁'` 的压缩率只有 `3/4 = 0.75`——低频字吃亏。把 `TEXT` 换成一句普通中文长句就会回到 README 说的 1.5~1.7；换成英文能到 4 以上（实验 task 1 里 `'Large language models'` 是 21 字符 5 token），因为空格天然切词，而 `' language'`、`' models'` 这种带前导空格的整词早就被 merge 成了单个 token。
> `add_prefix_space=False` 则是说：不在整段文本开头补一个空格。英文 tokenizer 常补（让句首的词和句中的词共用 `Ġword` 这个 token），但 MiniMind 以中文为主，补了反而白白多一个 token。

## 被改回 special=False 的那 15 个 {#special-visible}

36 个符号在训练时全部作为 `special_tokens` 传给 `BpeTrainer`——这保证它们各占一个完整 id，不会被 BPE 拆开、也不会参与 merge。但训练完之后，脚本又把其中一部分改了回来：

{{source:trainer/train_tokenizer.py#L60-L64}}

判据是「在不在 `special_tokens_list` 里」。不在的那 15 个（6 个功能标记 + 9 个 buffer）被改成 `special=False`，同样的标记也写进了 `tokenizer_config.json` 的 `added_tokens_decoder`（L75 那个三元表达式）。

为什么要多此一举？因为 `tokenizer.decode(ids, skip_special_tokens=True)` 会把 `special=True` 的 token **直接吃掉**，而 `eval_llm.py` 和 `TextStreamer` 都是这么调的。如果 `<think>`、`<tool_call>` 也标成 `True`，用户在终端里就看不到模型的思考过程和工具调用内容了——而这恰恰是 MiniMind 最想展示给人看的东西。所以只有真正的**结构边界符**（`<|im_start|>` / `<|im_end|>` / `<|endoftext|>` 和多模态占位符）保持 `True`，功能性标记留着可见。

要分清的是：`special=False` 只影响 decode 时会不会被过滤，**不影响 encode**。`<think>` 仍然在 `added_tokens` 里，`tokenizer.encode("<think>")` 拿到的还是单个 id 25，不会被拆成 `'<'` + `'think'` + `'>'`。

{{lab:special_visible}}

{{quiz:q9}}

> [!KEY]
> `special=True` 的 token 会被 `skip_special_tokens=True` 吃掉，所以 `<think>` / `<tool_call>` 这 6 个功能标记被特意改回 `False`——为了让思考和工具调用对用户可见。

> [!MORE] 四个特殊角色分别指向谁
> `tokenizer_config.json` 里：`bos_token = <|im_start|>`（id 1）、`eos_token = <|im_end|>`（id 2）、`pad_token` 和 `unk_token` 都是 `<|endoftext|>`（id 0）。注意 `add_bos_token` / `add_eos_token` 都是 `false`——边界符不由 tokenizer 自动加，全部交给 chat_template 去拼（第四章的内容）。

## 现场训练一个迷你 BPE {#mini-bpe}

BPE 的训练过程可以一句话说完：把语料切成最小单位，反复找**出现次数最多的相邻 token 对**，把它合并成一个新 token 记进 merges 表，名额用完就停。所以最终词表 = 256 个初始字节 + 每次合并产出的新符号，而 merges 的**先后顺序**就是一部学习史：越靠后加入的，通常越长、越是这份语料里的高频片段。

官方脚本喂的是 10000 行通用对话：

{{source:trainer/train_tokenizer.py#L12-L22}}

实验用同一套 API，但只喂 `lora_identity.jsonl` 的前 40 行（全是「我是 MiniMind，由 Jingyao Gong 开发」这类自我介绍），`vocab_size` 只给 400。144 条 merge 学出来的最长子词是 `' MiniMind'`、`' Jingyao'`、`'的人工智能助手'`——完完全全是这份语料的指纹。

这件事解释了两个现象。一是官方为什么要在更宽的语料上学 6108 次 merge：merge 的覆盖面直接决定真实文本的压缩率，窄语料训出来的词表碰到别的主题就会退化到接近单字节粒度（但仍然 decode 无损，这是 ByteLevel 兜的底）。二是为什么换 tokenizer 等于换模型：merge 顺序一变，id 和子词的对应关系全变，`embed_tokens` 的每一行都对不上号了。

{{lab:mini_bpe}}

{{quiz:q10}}

> [!KEY]
> BPE = 反复合并语料里最高频的相邻符号对；词表的内容是语料的指纹，merge 次数和语料覆盖面共同决定压缩率。

# chat_template：把对话摊平成一个字符串

模型只认一维的 token 序列，不认「messages 列表」。`chat_template` 就是那段把结构化对话压扁成字符串的 Jinja 模板，存在 `model/tokenizer_config.json` 里（源头是 `trainer/train_tokenizer.py:100`）。这一章按分支读它。

## 回合结构与 system 段 {#template-turns}

主循环遍历 `messages`，把每条消息包成一个回合：

```jinja
{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
    {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
{%- endif %}
```

于是每条 message 都变成 `<|im_start|>{role}\n{content}<|im_end|>\n`。`<|im_start|>` 和 `<|im_end|>` 各是一个 id（1 和 2），正好也是 tokenizer 的 `bos` / `eos`——生成时遇到 `<|im_end|>` 就停，这是同一件事的两面。

system 的处理有个容易忽略的细节。模板开头单独判断了 `messages[0].role == 'system'`，只有**第一条**才会被渲染进开头那个 system 段；没有 system 就干脆不输出，不会补一个空的。而上面那行条件里的 `message.role == "system" and not loop.first` 是说：**不在第一条**的 system 消息不会被丢掉，它会退化成一个普通回合，和 user 长得一模一样（实验 task 1 就是试这个）。

最后一个参数决定这段字符串是拿来训练还是拿来推理：`add_generation_prompt=False`（默认）渲染出的是一段完整对话，可以直接当训练样本用——Day 8 的 `SFTDataset` 就是这么做的；`=True` 会在末尾再追加一个等待模型续写的 assistant 前缀，`eval_llm.py` 用的是这一种。

{{lab:template_turns}}

{{quiz:q11}}

> [!KEY]
> 每条 message → `<|im_start|>{role}\n{content}<|im_end|>\n`；只有 `messages[0]` 的 system 才进开头的 system 段，`add_generation_prompt` 决定末尾要不要留一个 assistant 开口。

## assistant 分支与 open_thinking 开关 {#think-skeleton}

assistant 分支是全篇最关键的一段。它先做一件解析的事：

```jinja
{%- set reasoning_content = '' %}
{%- if message.reasoning_content is string %}
    {%- set reasoning_content = message.reasoning_content %}
{%- else %}
    {%- if '</think>' in content %}
        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
    {%- endif %}
{%- endif %}
```

消息自带 `reasoning_content` 字段就直接用；否则从 `content` 里找 `</think>`，把它前后切成「思考」和「正文」两半。两半都准备好之后，才轮到真正往外写字符串的那一行。

模板末尾还有另一半管推理：`add_generation_prompt=True` 追加的 assistant 前缀里，`<think>` 是留一个开口还是当场闭合，由 `open_thinking` 决定——`eval_llm.py --open_thinking 1` 传的就是它。

那么，一条**完全没有**思考内容的普通 assistant 消息（`content` 里连 `<think>` 字样都没有），渲染出来会长什么样？

{{predict:p2}}

{{lab:think_skeleton}}

实验表格第一行给出答案：`<|im_start|>assistant\n<think>\n\n</think>\n\n等于2<|im_end|>\n`。原因就在渲染那一行外面套的条件上——它是恒真的 `{%- if true %}`：

```jinja
{%- if true %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
{%- endif %}
```

骨架**永远**在，没有思考内容时就是一对空标签。训练数据因此长成一个固定模式，模型学会的是「先给一对（可能是空的）think 标签，再给正文」。

推理端完全对称：不传 `open_thinking` 就把**闭合**的空骨架焊死在 prompt 末尾（表格第 3 行），模型的下一个待预测位置已经在 `</think>` 之后，只能直接写正文；传 `open_thinking=True` 则只给一个未闭合的 `<think>\n`（第 4 行），模型得自己写推理再闭合。回头看 [logits[0, -1] 那一节](#next-token)：那个 prompt 的结尾正是 `<think>\n\n</think>\n\n`。

{{quiz:q12}}

> [!KEY]
> 恒真的 `{%- if true %}` 让每条 assistant 消息都带上 `<think>…</think>` 骨架（没内容就是空的）；`open_thinking` 只决定推理时这个骨架是焊死的空壳还是留给模型的开口。

> [!MORE] 模板里那段没人用的死代码
> `{%- if true %}` 前面有一段反向遍历 `messages`、算出 `ns.last_query_index` 的逻辑，是从 Qwen3 的模板移植过来的：Qwen3 用它判断「只在最后一次真实提问之后的 assistant 回复里保留 `<think>` 内容，更早的历史轮次抹掉 reasoning」以省 context。
> MiniMind 精简版把条件换成了恒真的 `true`，`ns.last_query_index` **从头到尾没被用到**。读源码时很容易被这段计算误导，以为存在某种「只有最后一轮才带 think」的行为——并没有。

## tools 分支与 tool 角色的合并 {#tools-branch .side}

传了 `tools` 参数时，模板会**完全接管** system 段：先把用户自己的 system 内容原样带出（如果有），再无条件拼上一段固定的英文说明、把每个工具 `tojson` 成一行塞进 `<tools>…</tools>`，最后交代返回格式要用 `<tool_call>{...}</tool_call>` 包裹，再收一个 `<|im_end|>`。没传 `tools` 就走朴素分支，只渲染用户给的 system。

assistant 的 `tool_calls` 追加在正文之后，每个序列化成一个 `<tool_call>\n{"name": …, "arguments": …}\n</tool_call>` 块；模板里那句 `if tool_call.function: set tool_call = tool_call.function` 是为了兼容 OpenAI 那种多包一层的格式。

最反直觉的是 `tool` 角色——它**不是**一个独立的 role 段，而是被塞进 **user** 轮次里：

```jinja
{%- elif message.role == "tool" %}
    {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}{{- '<|im_start|>user' }}{%- endif %}
    {{- '\n<tool_response>\n' }}{{- content }}{{- '\n</tool_response>' }}
    {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}{{- '<|im_end|>\n' }}{%- endif %}
```

只有「上一条不是 tool」才开 `<|im_start|>user`，只有「下一条不是 tool」才收 `<|im_end|>`。所以模型一次并行调用 3 个工具、拿回 3 条结果时，它们会被裹进**同一个** user 轮次里的 3 段 `<tool_response>`，而不是 3 个独立轮次。Day 14 的 `train_agent.py` 和 `eval_toolcall.py` 全靠这个结构拼多轮对话。

{{lab:tools_branch}}

{{quiz:q13}}

> [!KEY]
> `tools` 参数会让模板接管整个 system 段；连续多条 `tool` 消息被合并进同一个 user 轮次，每条一段 `<tool_response>`。
