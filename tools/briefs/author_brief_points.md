# 给课程作者 (subagent) 的说明 —— 把某一天改写成「总览 → 章 → 小节，先讲后练」格式

仓库根目录：本仓库根目录（learn-llm-14days）。你负责 **Day NN**（由调用方指定）。不要再派 subagent，自己完成。

## 背景：读者的两轮反馈（原话）
读者：Python 精通、LLM 原理都懂、但没手写过 LLM。学习方式「读源码 + 改代码 + 跑实验」，最看重**直观看到每个 tensor 的维度**。
- 对第一版（长文）：「内容组织没有重点，偏向内容堆砌/平铺直叙。老师教学应该是一次只讲一个点，让用户任意时刻都能在一个很细节的点上互动学习。」
- 对第二版（一堆问句卡片）：「又走另一个极端，没有整体感（总览）；标题全是问号、太散，得有提炼——比如『模型的配置』是主题，下面才是子问题；只有问题没有讲解，应该有问题也有答案。」

## 必读（按顺序）
1. `AUTHORING.md`：第 0 节（教学法）、第 2 节（章节格式、quiz/predict、微实验要求）、第 3/4/5 节（lab 硬性要求、learnkit API、quiz.json）。
2. **样板**：`lessons/day02/`（lesson.md + quiz.json + labs/）。照着它的结构、讲解深度、语气来写。
3. 如果这一天已经有旧版内容（`lessons/dayNN/`），先通读：事实已经核实过的讲解和题目是主要素材来源；你交付的是整体替换后的新版。
4. 你这一天对应的源码，逐行读。

## 产出（覆盖写入 `lessons/dayNN/`）
- `lesson.md`：`format: points` + `mainline`；总览（200~400 字 + 一条 `{{flow}}`）；3~5 章；9~13 个小节；不写小结小节。
- `quiz.json`：每小节 1~2 道 quiz（解析写完整）；predict 全天最多 3~4 道，只用在反直觉的结果上；外加 3~5 道不被任何小节引用的综合题。旧版的好题改造复用。
- `labs/*.py`：微实验（≤ 30 行、单一焦点输出、`focus=`/`expand=` 收窄 trace）。**删掉旧的大实验文件**，有用的代码拆进微实验。
  训练 / rollout / generate 类实验做不到 5 秒和 30 行：放宽到 **≤ 40 行、≤ 90 秒**（头部写好 `timeout`），用极小的模型/极少的步数，并用 `live()` 实时画曲线；但仍然**一个实验只看一件事**。

## 怎么组织
1. 先定**主线**：今天跟着哪个 tensor / 哪条数据流走？一句话写进 `mainline`，并画成总览里的 `{{flow}}`。
2. 把内容**提炼成 3~5 个主题 = 章**（名词短语标题），每章 2~4 个小节（陈述式标题，不用问句）。
3. 每个小节**先讲后练**：讲解 300~700 字把这个点讲透（做什么、为什么这么写、shape 怎么变、坑在哪），源码插在讲解中间 → 微实验让读者亲眼验证 → 1~2 道小测 → 一句话 KEY。**只读讲解也应该能学会。**
4. 旧版里不在主线上的内容：进 `[!MORE]`，或做成 `.side` 选学小节，或删。
5. 每个小节先想清楚「实验输出里哪一个 shape / 数是这一节的证据」，再围绕它写微实验和题。

## 验证（必须做）
- 每个 lab：在仓库根目录（learn-llm-14days）执行 `venv/bin/python runner.py lessons/dayNN/labs/<id>.py`，退出码 0。
- tasks 里写的每一处改动都亲手试一遍，确认现象和你写的一致（试完改回去）。
- 每道 predict / quiz 的答案先跑代码验证再定。临时文件写到一个临时目录（如 `tempfile.mkdtemp()`）而不是仓库里。
- `venv/bin/python tools/check_lessons.py --days NN --run` 必须无 ✗，警告 (·) 尽量清零。
- 此刻有十几个作者在同一台机器上并行跑实验，计时会偏慢、偶尔超时：超时先单独重跑一次确认，不要为了抢时间把实验砍到失去意义。
- 讲解里出现的每个数字 / shape / 现象都必须和 lab 的实际输出、源码一致；`{{source}}` 行号用 `sed -n 'a,bp'` 核对。

## 约束
- **绝对不要编辑 minimind/ 源码文件（model/ dataset/ trainer/ scripts/ 等），哪怕是「临时改一下试完再改回去」也不行。**
  要验证「在源码标签页改一处」这类 task，用 runner 的 override 机制（和网页完全一致，不碰原文件）：
  把改过的副本写到临时目录，再写一个 `overrides.json`：`{"model/model_minimind.py": "/abs/path/to/你的副本.py"}`，
  然后 `venv/bin/python runner.py lessons/dayNN/labs/<id>.py --overrides overrides.json`。
只能写 `lessons/dayNN/` 和上面的临时目录。不要动 `lib/`、`web/`、`server.py`、`tools/`、`minimind/` 源码、其他天。不联网、不装包。
trace 里 B、T 避开 2/4/8/96/384/768/2432/6400，推荐 B=3, T=7。

## 14 天总大纲（知道上下文，避免重复讲别人的内容；需要时写「Day X 会细讲」）
- Day 1  全景与 Tokenizer：仓库地图、训练流水线总览、一次完整 forward 的鸟瞰；tokenizer.json / tokenizer_config.json / chat_template / train_tokenizer.py
- Day 2  骨架：MiniMindConfig、Embedding、RMSNorm、MiniMindModel.forward 骨架(残差流)、lm_head 权重共享、参数量
- Day 3  位置编码：precompute_freqs_cis、apply_rotary_pos_emb(rotate_half 约定)、start_pos 切片、YaRN 长文本外推
- Day 4  Attention：GQA、QK-Norm、repeat_kv、因果掩码、SDPA vs 手写分支、padding attention_mask
- Day 5  FFN(SwiGLU) + MiniMindBlock(pre-norm 残差) + MiniMindForCausalLM.forward 与 loss(shift、ignore_index、logits_to_keep) + logit lens
- Day 6  MoE：MOEFeedForward 路由/top-k/index_add_、aux loss、DDP 小技巧、总参数 vs 激活参数
- Day 7  推理：generate 循环、KV cache、temperature/top-k/top-p/repetition penalty、streamer、eval_llm.py
- Day 8  数据管线：lm_dataset.py 全部 Dataset（Pretrain/SFT 的 label mask、DPO、RLAIF、AgentRL）、chat 模板与 loss mask 的配合
- Day 9  Pretrain & SFT 训练循环：train_pretrain.py / train_full_sft.py / trainer_utils.py（lr 调度、梯度累积、混合精度、裁剪、checkpoint/续训、DDP）
- Day 10 LoRA 与蒸馏：model_lora.py、train_lora.py、train_distillation.py
- Day 11 DPO：train_dpo.py（logits_to_log_probs、dpo_loss、ref model、chosen/rejected 拼 batch）
- Day 12 PPO：train_ppo.py（CriticModel、reward、GAE、clip 目标、KL）、rollout_engine.compute_per_token_logps
- Day 13 GRPO 与 Rollout Engine：train_grpo.py（组采样、组内归一化 advantage、completion mask、KL 估计、loss 变体）、rollout_engine.py
- Day 14 Agent RL + 部署 + 总复习：train_agent.py、serve_openai_api.py、convert_model.py、eval_toolcall.py；毕业测验

## 最终回复（简短）
章节结构（章 → 小节 id + 标题）；每个 lab 耗时；哪些题是 predict；发现的 learnkit / 格式问题；需要人工复核的点。
