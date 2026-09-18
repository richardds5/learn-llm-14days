---
day: 14
format: points
title: "Agent RL、部署与总复习"
subtitle: "多轮 tool-call 轨迹上的 0/1 mask → reward → 把模型端出去 → 14 天串成一条线"
minutes: 95
mainline: "一条多轮 tool-call 轨迹：response_ids 里模型写的 token 和环境喂进来的 token 交错排列，靠一条等长的 0/1 response_mask 决定谁算 loss"
files:
  - trainer/train_agent.py#L76-L186
  - trainer/train_agent.py#L189-L245
  - trainer/train_agent.py#L248-L338
  - scripts/serve_openai_api.py#L83-L102
  - scripts/convert_model.py#L16-L62
goals:
  - 能逐行说清 rollout_single 的多轮循环，以及 response_mask 上每一个 0/1 是怎么来的
  - 能说出 marker 哨兵解决的是什么问题，以及不用它会在哪几种输出上切错
  - 能默写 packing 那五行，解释 old_logps 为什么从 len(p)-1 开始填、completion_mask 为什么要左移一位
  - 能列出 calculate_rewards 的全部规则和分值，手算一条轨迹的总分，并说出组内 advantage 什么时候全是 0
  - 能把 out/*.pth 转成 transformers 目录或 Qwen3 结构，并说清 trust_remote_code 到底在信任什么
---

Day 11~13 的 RL 都是**一次生成、一次打分**：模型吐一段话，reward 给一个数，结束。`trainer/train_agent.py` 打破了这一点——模型在一个 episode 里来回好几轮：生成 → 解析出 `<tool_call>` → 真去执行工具 → 把结果塞回对话 → 再生成。于是训练数据里第一次出现了「不是模型生成的 token」：工具返回的 JSON、模板补的 `<|im_start|>user`、下一轮的 generation prompt。它们必须待在 `input_ids` 里，否则模型看不到工具结果；又绝对不能进 loss，否则模型是在学着模仿环境。

今天跟着一条轨迹走：`prompt_ids` 只在第一轮算一次，之后每轮往 `response_ids` 后面追加两段——模型自己写的（mask=1）和环境喂进来的（mask=0）。这条等长的 0/1 列表就是 `response_mask`。

五章的分工：轨迹怎么滚出来；mask 一路跟到训练侧的 `completion_mask`；给整条轨迹打分；把模型端出去；最后用两屏把 14 天串成一条线。

{{flow: messages + tools | *apply_chat_template* | prompt_ids | *generate* | new_ids (mask=1) | *execute_tool* | obs_delta (mask=0) | … | *response_mask* | completion_mask [B·G, L-1]}}

# 多轮 rollout：一条轨迹是怎么滚出来的

`rollout_single` 是今天的第一块骨头：它把「模型说话」和「环境说话」交替拼成一条序列。这一章按执行顺序拆四步——先看工具那三张表和解析函数决定了分支怎么走，再看 chat_template 把工具结果渲染成什么，然后是多轮循环的骨架，最后是最精妙的那段 marker 哨兵。

## 工具的三张表与 parse_tool_calls 的容错 {#tool-parse}

`train_agent.py` 顶部放了三张同名不同用的表，最容易被当成一回事：`TOOLS` 是六个工具的 JSON Schema（格式和 OpenAI function calling 完全一致），`MOCK_RESULTS` 是「工具的实现」（全是假的，查表或 `eval`），`CHECK_ARGS` 是「参数合法性检查」。**这三张表互不相干**：`CHECK_ARGS` 只在算 reward 时用来判断「这次调用算不算有效」，`execute_tool` 执行时根本不查它——所以会出现「参数校验没过、工具却照样返回了结果」的情况。更意外的是 `TOOLS` 在训练路径里**一次都没被引用**：实际训练时 tools 全部从数据的 system 消息里读，这份定义留着是给 lab 和其它脚本当参考工具集用的。

{{source:trainer/train_agent.py#L76-L95}}

`parse_tool_calls` 只有 6 行，却决定了整条轨迹的分支走向：正则 `<tool_call>(.*?)</tool_call>` 加 `re.DOTALL`，**标签必须成对闭合**才匹配得上；`json.loads` 失败就 `except: pass` 直接丢弃——**解析不出来 = 没有调用**，轨迹就此结束（而不是报错）；返回的是 list，所以模型一次输出多个 `<tool_call>` 是被支持的。

`execute_tool` 则用 `signal.SIGALRM` + `alarm(1)` 给每次调用加了 1 秒超时（防 `2**99999999` 这种），任何异常都吞掉返回 `None`，调用方拿到 `None` 就写成 `{"error": "tool not found"}` 塞回对话。

{{lab:tool_parse}}

{{quiz:q1,q2}}

> [!KEY]
> 解析（`parse_tool_calls`）、校验（`CHECK_ARGS`）、执行（`MOCK_RESULTS`）是三张互不相干的表：解析不出来轨迹就结束，校验不过只影响 reward，执行失败一律返回 `None`。

> [!MORE] calculate_math 那一行 eval 的两道护栏
> `eval(expr, {"__builtins__": {}, "math": math})` 的第二个参数把 builtins 清空了，所以 `__import__("os").system("ls")` 这种会直接 `NameError`，被 `except` 吞掉变成 `None`（lab 的 case ⑨）。它前面还串了 6 个 `.replace()`，把全角符号（`×` `÷` `−` `（` `）`）和 `^` 翻译成 Python 语法——因为模型真的会吐全角（case ④ 的 `（771−242）+84×27` 能算出 2797 就是靠它）。这不是一套严肃的沙箱，只是让训练时的假工具别把进程搞崩。

## 工具结果回填之后的那一段 prompt {#tool-template}

工具算完了，结果怎么回到模型眼前？答案全在 `model/tokenizer_config.json` 的 `chat_template` 里，和工具有关的只有三个分支。**① 传了 `tools` 时 system 段被整段重写**：原来的 system content 拼在 `# Tools` 之前（数据集里它常常是空串，于是就剩两个换行），然后把每个 tool 的 `tojson` 直接铺进 `<tools>…</tools>`。也就是说，**工具描述是以纯文本 JSON 的形式出现在 prompt 里的**，不存在什么特殊通道。

**② `role: "tool"` 被渲染成 user 消息**：

```jinja
{%- elif message.role == "tool" %}
    {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}{{- '<|im_start|>user' }}{%- endif %}
    {{- '\n<tool_response>\n' }}{{- content }}{{- '\n</tool_response>' }}
    {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}{{- '<|im_end|>\n' }}{%- endif %}
```

模型眼里根本没有 `tool` 这个角色，工具结果就是一条被 `<tool_response>` 包起来的 **user 消息**；连续多条 tool 只包一层 `<|im_start|>user … <|im_end|>`。**③ generation prompt 二选一**：`add_generation_prompt=True` 时补 `<|im_start|>assistant\n`，再按 `open_thinking` 决定是留一个**开着的** `<think>\n`（让模型自己写思考并闭合），还是补一个**空的** `<think>\n\n</think>\n\n`（等于关掉思考）。`rollout_single` 的 `thinking_ratio` 控制的就是它。

实验把「第 1 轮的 prompt」和「工具结果回填后第 2 轮的 prompt」都 tokenize 一遍，重点看后者是不是前者的严格前缀——这个性质是下一节整个增量拼接的前提。

{{lab:tool_template}}

{{quiz:q3}}

> [!KEY]
> 工具结果在模型眼里是一条 `<tool_response>` 包起来的 user 消息；同一段历史渲染出的 token 序列严格前缀一致，所以 prompt 只需要 tokenize 一次。

## rollout_single 的多轮循环与全量重放 {#rollout-loop}

现在看主角的骨架。`open_thinking = random.random() < thinking_ratio` 写在循环**外面**——整条轨迹共用一个 thinking 开关，不是每轮重抽。循环体的第一件事是重新渲染 `context`，但 `prompt_ids` 只在 `is None` 时算一次；之后每轮送进模型的是 `torch.tensor([prompt_ids + response_ids])`，也就是**把整条历史全量重放一遍**，轮与轮之间没有复用 KV cache（`generate` 内部会自己建一份）。敢这么写，靠的正是上一节那个前缀一致的性质。

{{source:trainer/train_agent.py#L107-L118}}

调 `rollout_engine.rollout` 时 `num_generations=1`，为什么不是 `G`？因为组采样在外层的 `rollout_batch` 里做：每个样本独立跑 `num_gen` 次**完整的多轮 episode**。多轮轨迹会分叉（第一轮调了不同的工具，后面的上下文就完全不同），没法像 Day 13 的 GRPO 那样一次 `generate` 生成 `G` 条。`rollout_batch` 的双层循环顺序是「sample0 的 G 条、sample1 的 G 条……」，这个顺序决定了后面 advantage 归一化必须用 `repeat_interleave`。

拿回结果后，`valid_len` 取自 `completion_mask`——torch 引擎里它恒为全 1，只有 SGLang 引擎才真的有 padding；紧跟的 `len(new_ids) != len(new_logps)` 自检是在防引擎返回的 token 数和 logprob 数对不上。解析不出 tool_call 就 `break`，这是绝大多数轨迹的退出方式；`unfinished = turn == max_turns - 1` 写在 `break` **之后**，所以它的含义是「轮数用光了还在调工具」，reward 里会被狠狠惩罚。

{{lab:rollout_turns}}

{{quiz:q4,q5}}

> [!KEY]
> 每一轮都把 `prompt_ids + response_ids` 全量重放给 `generate`；`num_generations=1`，组采样由外层 `rollout_batch` 串行跑 G 遍完整 episode 来完成。

## marker 哨兵：切出模板新增的那一截 {#obs-marker}

工具结果追加进 `messages` 之后，要把「模板新增了哪些文本」变成 token 接到 `response_ids` 后面。最自然的想法是减前缀：渲染一遍新的 `messages`，砍掉「旧 context + 模型这一轮的 `new_text`」那一段，剩下的就是增量。**这个想法是错的**，因为模板对 assistant 内容做了重排：内容里有 `</think>` 时它会 `split('</think>')` 拆成 reasoning 和 content 两半，再分别 `rstrip('\n')` / `lstrip('\n')` 后按规范格式重拼；有两个 `</think>` 时中间那段甚至会被整个丢掉。只有模型的输出恰好长成模板期望的样子时，减前缀才成立。

{{source:trainer/train_agent.py#L145-L154}}

所以作者用了一个哨兵：`marker` 由 `id(messages)` 和 `len(response_ids)` 拼成，几乎不可能和正文撞车；把它 append 到 assistant 内容尾部，渲染一遍，`marked_context.partition(marker)` 拿到的后半段就是「本轮 assistant 内容结束之后模板新增的全部文本」——`<|im_end|>\n` + `<|im_start|>user\n<tool_response>…` + 下一轮的 generation prompt。拿完立刻把 content 还原，别把 marker 留在 `messages` 里；`if not found: raise` 是在防模板把 marker 吃掉。

最后那两行是一个容易漏的去重：`new_text` 是 `skip_special_tokens=True` 解码出来的，不含 `<|im_end|>`，但 `new_ids` 里很可能**真的**以 eos 结尾；而 `observation` 又是从 `<|im_end|>\n` 开头的。不去重就会出现两个 `<|im_end|>`。

{{lab:obs_marker}}

{{quiz:q6}}

> [!KEY]
> 模板会重排 assistant 内容，所以「渲染结果 = 旧 context + 原样 new_text」时灵时不灵；插一个哨兵再 `partition`，才能稳定切出观测增量。

# response_mask：哪些 token 参与 loss

轨迹滚完了，下面这三屏只做一件事：把「谁算 loss」这条信息从 `rollout_single` 一路带到 `loss.backward()`。它先是一个 Python list（`response_mask`），再被打包成 `[B·G, L]` 的 `full_response_masks`，最后左移一位变成 `completion_mask`。

## 三个列表同步 extend 与被置 0 的 eos {#response-mask}

`rollout_single` 里有三个长度必须永远相等的列表：`response_ids`、`response_mask`、`response_old_logps`。它们在每一轮被 extend 两次——一次是模型生成的 token，一次是观测增量。

{{source:trainer/train_agent.py#L126-L133}}

模型生成的那一批，mask 写的是 `int(t != tokenizer.eos_token_id)`：正文全是 1，**唯独 eos 被置 0**。这是个刻意的选择——Agent 轨迹里 eos 只表示「本轮说完了」，不是「整个任务结束」，让模型去学它没有意义。（对比 Day 8 的 SFT：`SFTDataset.generate_labels` 反而**把 `<|im_end|>\n` 也算进监督区间**，因为那里必须教会模型停下来。两者的取舍正好相反，最后一章会把这两条 mask 摆在一起看。）

观测增量那一批就简单粗暴了：`response_mask.extend([0] * len(obs_delta))`，`response_old_logps.extend([0.0] * len(obs_delta))`。**logps 填 0.0 不是「概率为 1」的意思**，只是个占位——这些 token 不是模型采样出来的，没有 logprob 可言；反正它们的 mask 是 0，后面乘进 loss 时会被整行抹掉。

实验用一个脚本化的 rollout 引擎跑出一条固定的两轮轨迹（省掉 `generate`，结果每次都一样），再把 `response_ids` 按 mask 逐 token 上色。盯住彩带里那三段 0：模型的 eos、工具结果、下一轮的 generation prompt。

{{lab:response_mask}}

{{quiz:q7}}

> [!KEY]
> `response_mask` 里 1 = 模型自己生成的正文，0 = eos + 工具结果 + 模板补的 generation prompt；`old_logps` 在观测位置填 0.0 只是占位。

## 把 prompt 和 response 打包成一条右 padding 的序列 {#packing}

Day 13 的 GRPO 里 prompt 和 completion 是**两段分开的 tensor**（prompt 左 padding、completion 右 padding）。Agent RL 不行——一条轨迹里模型 token 和环境 token 是交错的，切不开。于是打包方式整个换了：拼成一整条右 padding 的序列，靠逐 token 的 mask 区分谁算 loss。

{{source:trainer/train_agent.py#L260-L270}}

前三行是核心。`ids = p + r` 把 prompt 和 response 首尾相接；`mask = [0] * len(p) + m` 让整段 prompt 都不算 loss；最容易看错的是 `old_logps = [0.0] * max(len(p) - 1, 0) + old_lp`——**是 `len(p)-1` 不是 `len(p)`**。因为 per-token 量的下标 `j` 对应的是「用位置 `j` 预测 `input_ids[j+1]`」，response 第 `i` 个 token 在 `input_ids` 里的下标是 `len(p)+i`，它的 logprob 该落在 `len(p)+i-1`。差一格不会报错，只会让每个 token 配错自己的旧概率。

`if len(ids) > args.max_total_len` 那段是**从左边截断**：多轮轨迹会越滚越长，必须有个上界（默认 2500），保留最新的上下文比保留最老的 system prompt 更重要；`old_logps` 跟着截成 `len(ids)-1`。`prompt_len` 取的是 mask 里第一个 1 的位置，只用于 debug 打印。之后统一右 padding 到这一批里最长的 `max_len`，`full_mask` 用 `arange < seq_lens` 造出来标记哪些是真 token，`old_per_token_logps` 则只 pad 到 `max_len - 1`。

{{lab:packing}}

{{quiz:q8}}

> [!KEY]
> `ids = prompt + response` 拼成一条右 padding 的序列；`old_logps` 前面填 `len(p)-1` 个 0.0（不是 `len(p)`），因为 per-token 量天生比 token 少一格、且整体左移一位。

## 左移一位的 completion_mask 与 eos 截断 {#completion-mask}

有了 `[B·G, L]` 的 `full_response_masks`，训练侧要的却是 `[B·G, L-1]` 的 per-token mask——因为 `logits` 已经被 `res.logits[:, :-1, :]` 去了尾，`per_token_logps[j]` 说的是「位置 `j` 预测出 `input_ids[j+1]` 的对数概率」。所以只要把 mask 整体左移一位：`completion_mask = full_response_masks[:, 1:]`，下标就对齐了。

{{source:trainer/train_agent.py#L291-L299}}

左移之后紧跟着一段从 Day 13 GRPO 抄过来的代码：先用 `is_eos` 标出「既是 eos、mask 又非 0」的位置，`argmax` 取每行第一个，把它之后的全部乘成 0。在 GRPO 里这一步很关键——那边 `generate` 出来的 completion 在 eos 之后全是 padding，不截干净就会把 padding 也算进 loss。

那么在 Agent RL 这套 mask 约定下，中间这段截断会做些什么？实验里这条轨迹的序列中一共有 5 个 eos（prompt 里 2 个、response 里 3 个），先下注再跑。

{{predict:p1}}

{{lab:eos_scan}}

答案是 `is_eos.any() = False`：`response_mask` 在上上节已经把所有 eos 位置写成了 0，prompt 段又整段是 0，「是 eos」和「mask 非 0」永远不会同时成立。于是 `eos_idx` 保持为最后一列，`pos <= eos_idx` 恒真，`token_counts` 截断前后都是 54。这不是 bug，只是一段在新的 mask 约定下彻底失去意义的代码——知道它空转，读日志里的 `AvgLen` 时就不会去怀疑「是不是被 eos 截短了」。

> [!KEY]
> `completion_mask = full_response_masks[:, 1:]` 一步到位；后面那段 eos 截断因为 eos 早就被置 0，`is_eos` 恒为 False，从头到尾空转。

> [!MORE] 真正要用的是最后两行
> `token_counts = completion_mask.sum(dim=1)` 是每条轨迹参与 loss 的 token 数，它是后面 `policy_loss` 做**序列内平均**的分母：`(per_token_loss * completion_mask).sum(dim=1) / token_counts.clamp(min=1)`，先在每条轨迹内部按有效 token 数平均，再对 batch 取平均——这样长轨迹不会因为 token 多而占更大权重。
> `valid_rows = token_counts > 0` 则挡掉「一个 1 都没有」的空轨迹（比如第一轮就被 `max_total_len` 左截断到只剩 prompt）。整批都空时还有兜底：`per_token_loss.sum() * 0.0`，保证 `backward()` 有个合法的计算图。

# reward：一条轨迹值多少分

mask 决定了「哪些 token 算」，reward 决定了「往哪个方向算」。这一章两屏：先看 `calculate_rewards` 那一堆手写规则怎么给一条轨迹打分，再看这些分数怎么在组内互相比较、变成每条轨迹的 advantage。

## 两个分支与它们的分值表 {#reward-rules}

`calculate_rewards` 先做三件公共的事：把每轮输出 `</think>` 之前的部分砍掉得到 `turn_answers`（最后一轮就是 `answer`）；把所有轮的 tool_call 汇总；然后按 `<tool_call>` 和 `</tool_call>` 的数量差做**标签配对惩罚**，每差一个扣 0.5。接着看 `tool_calls` 是否为空，分两条完全不同的路——注意判据是「解析出几条」而不是「有没有写标签」，所以标签不闭合的轨迹会掉进**无工具**分支。

无工具分支（闲聊 / 直接回答）按格式给分：回答长度落在 `[5, 800]` 得 +0.5、否则 -0.5；有 `</think>` 时再加思考长度分（`[20, 300]` 得 +1.0、否则 -0.5）和闭合分（恰好一个 `</think>` 得 +0.25、否则 -0.25）；可选的 reward model 给 [-3, 3]；最后减去复读惩罚。

{{source:trainer/train_agent.py#L226-L244}}

有工具分支才是主线。`valid_call_count` 的判据是「名字在本条数据的 tools 里 **且** `CHECK_ARGS[name](args)` 通过」；`tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)` 同时惩罚「调少了/调多了」和「调了但参数不合法」——它拿 `len(gt)` 当作「应该调几次工具」的代理，这对 `agent_rl_math.jsonl` 这种「一个 gt 对应一次计算」的数据是成立的。`gap == 0` 得 +0.5，否则 `-0.5 × gap`。真正的大头是 GT 命中分 `2.5 × len(verified) / len(gt)`。`unfinished` 会把 `final_text` 直接置空（等于放弃 GT 分）再额外扣 0.5。最后一行两个分支都一样：`max(min(reward, 3.0), -3.0)`，**总分硬夹在 [-3, 3]**。

{{lab:reward_parts}}

{{quiz:q9,q10}}

> [!KEY]
> 分支由「解析出几条 tool_call」决定；有工具时 GT 命中占 2.5 分的大头、工具对齐只有 ±0.5 量级，总分最后硬夹在 [-3, 3]。

> [!MORE] validate_gt_in_text 的两条命中路径，以及一个假阳性
> `validate_gt_in_text` 判「gt 有没有出现在文本里」，两条路任一命中即可：**字符串包含**（大小写不敏感，`str(g).lower() in text.lower()`）；**数值相等**——如果 gt 本身是个数字，就把文本里所有数字用正则抠出来（先 `replace(',', '')` 去掉千分位），比较 `abs(gt - n) < 1e-6`。所以 `20,758,280`、`20758280`、`20758280.0` 都算命中，正则里的 `(?<![\w.])` / `(?![\w.])` 是在避免从 `abc123` 或 `1.2.3` 里抠出假数字。
> 但字符串包含这条路会带来假阳性：文本里写 `120758280` 时，`"20758280" in "120758280"` 为真，照样算命中。想验证的话，把 lab 里 case A 的答案改成「结果是 120758280」，GT 分依然是满分。

## 组内归一化：G 条轨迹互相当基准 {#group-adv}

拿到 `rewards`（shape `[B·G]`）之后的四行，和 Day 13 的 GRPO 一模一样：`view(-1, G)` 折成 `[B, G]`，组内求均值和标准差（`unbiased=False`），`repeat_interleave(G)` 摊回 `[B·G]`，相减再相除。这里没有 critic，**同一个 prompt 的 G 条轨迹互相当基准**——比组内平均好的往正方向推，差的往负方向推。

{{source:trainer/train_agent.py#L320-L323}}

`repeat_interleave` 不能写成 `repeat`：`rewards` 的排列来自 `rollout_batch` 的双层循环（外层 sample、内层 num_gen），是「sample0 的 G 条、sample1 的 G 条……」，所以 `[B]` 的均值必须**逐元素重复**成 `[m0,m0,…,m1,m1,…]`；`.repeat(G)` 给的是 `[m0,m1,…,m0,m1,…]`，shape 同样是 `[B·G]`、**不会报错**，只是每条轨迹减错了别人的均值。这正是它危险的地方。

现在问一个训练里天天遇到的问题：如果组内 G 条轨迹的 reward 完全一样（SFT 模型在数学工具调用上的分布非常尖，这太常见了），这一组的 advantage 会是什么？

{{predict:p2}}

{{lab:group_adv}}

分母上那个 `+ 1e-4` 保证了不会除零，但也就仅此而已：`std_r = 0` 时分子本来就是 0，整组 advantage 全是 0，`per_token_loss` 里除了 KL 项之外没有任何梯度信号——**这一步白跑了**。这就是训练日志里一定要盯 `GrpStd`（`grouped_rewards.std(dim=1).mean()`）的原因：它一路贴着 0，说明采样多样性不够或 reward 分辨率太粗，该调温度、调 `num_generations`，或者换更难的数据。

{{quiz:q11}}

> [!KEY]
> advantage = (自己的 reward − 组内均值) / (组内标准差 + 1e-4)；组内 reward 一致时它全是 0，这一步没有任何策略梯度，所以日志里要盯 `GrpStd`。

# 把模型端出去：服务与格式转换

训练完的 `out/agent_768.pth` 只是一个 torch state_dict，谁也认不出来。这一章看 MiniMind 提供的两条出口：一条是自己起 OpenAI 兼容服务（关键是把自回归文本拆成协议字段），另一条是转成 HuggingFace 目录——要么带上自己的建模代码，要么干脆伪装成 Qwen3。

## parse_response：一段文本拆成 OpenAI 的三个字段 {#serve-parse}

`serve_openai_api.py` 里真正有意思的纯函数只有一个。模型吐出来的是一整段自回归文本（`<think>…</think>` 加正文加可能的 `<tool_call>`），而 OpenAI 协议要的是三个分开的字段：`content`、`reasoning_content`、`tool_calls`。`parse_response` 负责这次翻译。

{{source:scripts/serve_openai_api.py#L83-L102}}

think 部分有两条路径：优先匹配**成对的** `<think>…</think>`；匹配不上就退而求其次，看有没有裸的 `</think>`。第二条路比想象中常用——模板用 `open_thinking=True` 时，`<think>` 是 prompt 的一部分、根本不在生成结果里，所以模型输出永远只有闭合标签。tool_call 部分把每段 JSON 解析成 OpenAI 格式，注意 `arguments` 要**再 `json.dumps` 成字符串**（协议就是这么规定的，客户端拿到后得再 `json.loads` 一次）。

最后那个 `if tool_calls:` 是全函数最该记住的一行：**只有在至少解析成功一条的前提下**，才把 `<tool_call>` 标签从 content 里 `re.sub` 掉。JSON 写错时这个清理不会执行，标签连同非法 JSON 原封不动留在 `content` 里返回给客户端——这是「解析失败时的降级行为」，客户端看到的是一段带标签的普通回答，而不是一个空 message。

{{lab:parse_response}}

{{quiz:q12}}

> [!KEY]
> `parse_response` 把一段文本拆成 content / reasoning_content / tool_calls；`arguments` 要二次 `json.dumps`，而清理标签只在至少解析成功一条时才做。

> [!MORE] 流式那半边：streamer + queue + 线程
> HF 的 `generate` 是阻塞的，但 SSE 要边生成边吐。解法是 `CustomStreamer(TextStreamer)` 把本该 print 的文本改成 `queue.put(text)`、结束时 `put(None)` 当哨兵，`model.generate(...)` 扔进 `Thread` 里跑，主协程在 `while True: text = queue.get()` 上消费。消费端还要把 `</think>` 之前的内容作为 `reasoning_content` 发、之后的作为 `content` 发，用 `emitted` 这个游标记录已经吐到哪了；等生成完成后再整体 `parse_response` 一次拿 tool_calls。这个 venv 里没装 fastapi / uvicorn，服务跑不起来，所以 lab 用 `ast` 把 `parse_response` 这一个函数单独抠出来执行。

## register_for_auto_class 与 trust_remote_code {#convert-auto}

`convert_torch2transformers_minimind` 才十几行，关键全在开头那两行 `register_for_auto_class`。

{{source:scripts/convert_model.py#L16-L28}}

它们让 `save_pretrained` 顺手把 `model/model_minimind.py` **拷进输出目录**，并在 `config.json` 里写下 `"auto_map": {"AutoConfig": "model_minimind.MiniMindConfig", "AutoModelForCausalLM": "model_minimind.MiniMindForCausalLM"}`。于是 `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` 就知道「去这个目录里的 `model_minimind.py` 找 `MiniMindForCausalLM`」——这就是 `trust_remote_code` 的全部含义：**执行模型目录里自带的建模代码**。实验里读回来的类，`__module__` 会是 `transformers_modules.<目录名>.model_minimind`，一眼能看出它是从目录里动态 import 的。

{{lab:convert_roundtrip}}

{{quiz:q13}}

> [!KEY]
> `register_for_auto_class` = 把建模代码拷进目录 + 在 config 里写 `auto_map`；`trust_remote_code=True` 就是允许 transformers 执行这份代码。

> [!MORE] 另外三个容易忽略的细节
> **`strict=False`**：这里加载权重用的是非严格模式，漏 key 不报错。不过官方权重其实一个 key 都不差（lab 里换成 `strict=True` 照样加载），Day 2 那两个 `persistent=False` 的 RoPE buffer 两边都没有，也不构成差异。
> **`safe_serialization=False`**：存出来的是 `pytorch_model.bin` 而不是 `model.safetensors`，lab 打印的目录清单里能看到。
> **反向转换**：`convert_transformers2torch` 只是 `from_pretrained(..., trust_remote_code=True)` 读回来再 `torch.save({k: v.cpu().half()})`——两种格式共用同一份 state_dict key，来回转换只损失精度。

## 同一份权重灌进 Qwen3ForCausalLM {#convert-qwen}

还有第二条出口，而且更实用：`convert_torch2transformers`（没有 `_minimind` 后缀的那个）**不拷任何代码**，它按字段把 `MiniMindConfig` 翻译成 `Qwen3Config`，`new Qwen3ForCausalLM(cfg)`，然后把 MiniMind 的 `state_dict` 直接 `load_state_dict(..., strict=True)` 灌进去。

{{source:scripts/convert_model.py#L43-L55}}

`strict=True` 意味着**两边的 key 必须一个不多一个不少**。能对上是因为 MiniMind 的结构就是 Qwen3 的子集：RMSNorm + GQA + **QK-Norm** + SwiGLU + RoPE，连参数名都一模一样（`model.layers.N.self_attn.q_norm.weight` 这种）。转完之后不需要 `trust_remote_code`，也就能直接喂给 vLLM / SGLang / ollama——这才是真正能上生产的那条路。

那么问题来了：同一份权重、同一段 `input_ids`，MiniMind 自己的 forward 和 Qwen3 的 forward 算出来的 logits 会差多少？两份实现是不同的人写的，attention 的算子选择、RoPE 的构造方式都不一定一样。先下注，再跑实验。

{{predict:p3}}

{{lab:convert_qwen}}

答案是 `maxdiff = 0.0`——fp32 下**逐位相同**，不是「误差很小」而是「一个 bit 都不差」。这说明两份实现的算子调用序列完全等价（同样的 `scaled_dot_product_attention`、同样的 cos/sin 表构造、同样的归一化顺序）。顺带一提，源码里只写了 Qwen3 和 Qwen3Moe 两个分支，**没有 Llama 分支**——Llama 没有 QK-Norm，`q_norm` / `k_norm` 这两个 key 在它那边根本不存在，`strict=True` 会当场报 unexpected key。

> [!KEY]
> MiniMind 的结构是 Qwen3 的严格子集，同一份 state_dict 能 `strict=True` 灌进 `Qwen3ForCausalLM`，fp32 下 logits 逐位相同；转完即可进 vLLM / ollama。

# 14 天总复习

最后两屏把前面 13 天串起来。第一屏用一个真跑起来的工具调用闭环，把整条流水线的终点摆在眼前；第二屏回到今天的主题——mask——把五个训练阶段的「哪些 token 算 loss」并排放一次，这是 14 天里最容易记混、也最值得记牢的一张对照表。

## 从 jsonl 到能调工具的模型 {#recap-line}

把 14 天压成一句话：**一堆 jsonl → 一个 6400 词的 tokenizer → 64M 参数的 Transformer → 在文本上预训练 → 在对话上 SFT → 用偏好/规则信号做 RL → 一个能在多轮里调工具的模型**。每一站都有一个今天还能回看的落点：Day 1 的 `apply_chat_template` 决定了模型眼里的对话长什么样；Day 2~6 的 config 七个数字定死了每个 tensor 的 shape；Day 7 的 `generate` 是所有 rollout 的底座；Day 8 的 Dataset 决定了监督区间；Day 9 的训练循环被 Day 10~14 反复复用；Day 11~13 从偏好信号一路走到组内归一化；今天把「环境」接了进来。

下面这个实验就是终点站：加载 `full_sft` 权重（没有 agent 权重，但 SFT 阶段已经学过工具调用格式），照着 `scripts/eval_toolcall.py` 的 `run_case` 循环跑一遍——生成 → `parse_tool_calls` → `execute_tool` → 以 `role=tool` 追加 → 再生成。它和 `rollout_single` 是同构的，只是不算 mask 也不算 reward。

{{lab:graduation}}

`eval_toolcall.py` 还有两件事值得知道：它内置 8 条 `TEST_CASES` 和**自己的一套** `MOCK_RESULTS`（和 `train_agent.py` 那套不是同一份——这里 `get_current_time` 返回真实时间，还多了 `random_number` / `text_length`）；`--backend` 可以在 `local`（直接加载权重）和 `api`（打 OpenAI 兼容接口）之间切，后者正好能测你自己起的 `serve_openai_api.py` 或转成 Qwen3 后灌进 ollama 的模型。

{{quiz:q14}}

> [!KEY]
> 14 天是一条线：jsonl → tokenizer → Transformer → pretrain → SFT → RL → 能调工具的模型；`eval_toolcall.run_case` 就是不带 mask 和 reward 的 `rollout_single`。

> [!MORE] 14 天知识地图（速查）
> | Day | 核心 tensor / shape | 关键函数 | 最容易错的点 |
> |---|---|---|---|
> | 1 全景 & Tokenizer | `input_ids [B, T]`，`vocab_size=6400` | `apply_chat_template` | chat_template 存在 `tokenizer_config.json` 里；assistant 段**永远**带 `<think>…</think>` 壳 |
> | 2 骨架 | `hidden_states [B, T, C=768]` | `RMSNorm`、`MiniMindModel.forward` | `tie_word_embeddings` 是把 `embed_tokens.weight` **指向** `lm_head.weight`；参数量大头在 FFN |
> | 3 RoPE | `freqs_cos/sin [max_pos, D=96]` | `precompute_freqs_cis`、`apply_rotary_pos_emb` | `rotate_half` 是**前后半切分**不是奇偶交错；`unsqueeze_dim=1` 是因为此时 q 还是 `[B, T, H, D]` |
> | 4 Attention | `xq [B,T,8,96] → [B,8,T,96]`；`xk [B,T,4,96] --repeat_kv--> [B,T,8,96]` | `repeat_kv`、`scaled_dot_product_attention` | QK-Norm 在 RoPE **之前**；因果 mask 只加到 `[..., -seq_len:]` 列 |
> | 5 FFN + loss | `logits [B, T, V=6400]` | `MiniMindForCausalLM.forward` | **shift 在 forward 里做**；`aux_loss` 不在 forward 里加进 loss |
> | 6 MoE | `x_flat [B*T, C]`，`topk_idx [B*T, k]` | `MOEFeedForward.forward`、`index_add_` | batch 和 seq 被拍平成 token 池；`aux_loss` 只在 `self.training` 时算 |
> | 7 推理 | `past_key_value` 每层 `(k, v)`，各 `[B, T_total, KV=4, D]` | `generate`、`start_pos` 切片 | cache 存的是 **RoPE 之后、repeat_kv 之前** |
> | 8 数据管线 | SFT `labels [max_len]`；DPO `x [B, max_len-1]` | `generate_labels`、`generate_loss_mask` | SFT 的监督区间**包含**结尾的结束符和换行；DPO 在 Dataset 里就错好位了 |
> | 9 Pretrain/SFT 循环 | — | `get_lr`、`lm_checkpoint`、`SkipBatchSampler` | `get_lr` **没有 warmup**；梯度累积下 loss 要除以 `accumulation_steps` |
> | 10 LoRA / 蒸馏 | `A.weight [r=16, C]`、`B.weight [C, r]` | `apply_lora`、`merge_lora` | 条件是 `in_features == out_features` → 每层只有 q_proj 和 o_proj，8 层共 16 个 |
> | 11 DPO | `x [2B, T-1]`，`log_probs [2B, T-1]` | `logits_to_log_probs`、`dpo_loss` | **chosen 在前一半** `[:B]`；policy / ref 各只跑一次 forward |
> | 12 PPO | `values [B, T]`，`advantages [B, gen_len]` | `CriticModel`、GAE 反向循环 | KL **不是**塞进 token reward，而是独立项加在 `policy_loss` 上 |
> | 13 GRPO | `rewards [B*G] → view(-1,G) [B,G]` | 组内归一化、k3 KL | `repeat_interleave` ≠ `repeat`；组内 reward 相同 → advantage 全 0 |
> | 14 Agent RL | `response_mask` 与 `response_ids` 等长；`completion_mask [B·G, L-1]` | `rollout_single`、`calculate_rewards` | 工具返回的 token mask=0；**eos 也 mask=0**（和 SFT 相反） |

> [!MORE] 接下来可以做什么
> **① 自己完整训一遍。** 按 README 的顺序 `train_pretrain → train_full_sft → train_dpo`，最小配置（`hidden_size=512, num_hidden_layers=8`）在单卡 3090 上是小时级的。先把 `--use_wandb` 打开，盯三条曲线：pretrain 的 loss、SFT 的 loss、DPO 的 `chosen - rejected` margin。
> **② 换数据。** 最便宜的实验是 LoRA：`dataset/lora_identity.jsonl` 只有 91 行，照着它的格式写 50 条自己的数据，`train_lora.py` 几分钟就能让模型改口。想验证 Agent RL，就照 `agent_rl_math.jsonl` 的格式造一批新工具（记得同时改 `TOOLS` / `MOCK_RESULTS` / `CHECK_ARGS` 三张表）。
> **③ 改架构。** 把 `num_key_value_heads` 从 4 改成 1（MQA）看显存和质量怎么变；把 `use_moe` 打开对比总参数 vs 激活参数；把 `rotate_half` 换成奇偶交错的实现（然后你会发现旧权重全废了——这正说明 RoPE 约定必须和训练时一致）。
> **④ 把它端出去。** `convert_model.py` 转成 Qwen3 结构 → vLLM / ollama 起服务 → `eval_toolcall.py --backend api` 跑一遍，这是最接近真实工程链路的一条路径。

## 五种 loss mask 的对照 {#recap-mask}

如果只能从 14 天里带走一件事，我会选这张表：**每个训练阶段都在回答同一个问题「哪些位置要算 loss」，而它们的答案全都不一样**。

| 阶段 | 谁算 loss | 写在哪 |
|---|---|---|
| Pretrain | 除 pad 之外全部（`labels[input_ids == pad] = -100`） | `PretrainDataset.__getitem__` |
| SFT | assistant 正文，**连结尾的结束符和换行一起算** | `SFTDataset.generate_labels` |
| DPO | assistant 正文，且 Dataset 里就已经错好位 | `DPODataset.generate_loss_mask` |
| GRPO | completion 段，到第一个 eos 为止 | `train_grpo.py` 的 `completion_mask` |
| Agent RL | 模型生成的正文，**eos 和工具结果全是 0** | `rollout_single` 的 `response_mask` |

最刺眼的一格是 SFT 和 Agent RL 在 `<|im_end|>` 上的取舍完全相反。SFT 必须把它算进去——模型不学会自己吐结束符就永远停不下来；而 Agent RL 里 eos 只是「本轮说完了」，把它算进 loss 等于在鼓励模型早点收场。同一个 token、同一个模型、两个阶段，一个 1 一个 0。

下面的实验把这两条 mask 摆在**同一条序列**上：先用脚本化引擎滚出一条 Agent 轨迹，拿到 `[0]*len(prompt) + response_mask`；再把这同一串 `input_ids` 丢给 `SFTDataset.generate_labels` 走一遍 SFT 的打标签逻辑，最后按「两者取值相同」折叠成连续段落对照着看。四处不一致的地方，正好是今天讲的四件事。

{{lab:recap_mask}}

{{quiz:q15}}

> [!KEY]
> 同一条序列上 SFT 监督 69 个 token、Agent RL 只算 54 个；差别集中在 think 壳和结尾的 `<|im_end|>` 这两处（工具结果两边都是 0，不算差异）——mask 才是区分各个训练阶段的真正分水岭。
