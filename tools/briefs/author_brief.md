# 给课程作者 (subagent) 的统一说明

你要为一个 MiniMind 学习网站编写**其中一天**的课程内容。仓库根目录：本仓库根目录（learn-llm-14days）

读者画像: Python 精通、LLM 原理比较了解、但从没手写过 LLM。他选择的学习方式是「读源码 + 改代码 + 跑实验」(不做从零实现的练习)，
最看重的是**能直观看到每个 tensor 的输入输出维度**。每天学习 1~1.5 小时。

## 工作步骤
1. 完整阅读 `AUTHORING.md`（课程格式、learnkit API、硬性要求），严格遵守。
2. 精读你负责的源码范围（逐行读，不要只看函数签名）。需要时读 README.md 相关章节获取作者的设计意图。
3. 编写 `lessons/dayNN/` 下的 `lesson.md`、`quiz.json`、`labs/*.py`（每天 4~6 个 lab）。
4. 逐个运行 lab 验证：在仓库根目录（learn-llm-14days）执行 `venv/bin/python runner.py lessons/dayNN/labs/<lab>.py`
   必须退出码 0、无 traceback。**正文里提到的 shape、数值、现象必须与 lab 的实际输出一致**——先跑，再写结论。
5. quiz 里每道涉及 shape / 数值 / 行为的题，写临时脚本验证后再定答案（临时文件写到一个临时目录，如 `tempfile.mkdtemp()`，而不是仓库里）。
6. 最后自检：
   - `{{source:path#Lx-Ly}}` 行号范围准确框住了你要展示的代码（用 sed -n 'x,yp' 核对）；
   - 每个 `{{lab:id}}` 都有对应的 `labs/id.py`；`{{quiz:...}}` 的 id 都存在；文末有 `{{quiz}}`；
   - `quiz.json` 是合法 JSON（python -c "import json;json.load(open(...))"）；lesson.md 的 frontmatter 是合法 YAML；
   - lab 头部的 `# ---` YAML 元信息合法（title / timeout / sources / tasks）。

## 约束
- 只能写 `lessons/dayNN/`（和上面的临时目录）。**不要**修改 `lib/`、`AUTHORING.md`、`minimind/` 源码或其他天的内容。
- 如果发现 learnkit 有 bug 或缺少某种能力：绕过它，并在最终回复里报告。
- 不要联网下载任何东西。不要安装包。
- lab 的 trace 里 B、T 取值避开 2/4/8/96/384/768/2432/6400，推荐 B=3, T=7（或 11、13）。

## 14 天总大纲（让你知道上下文，避免重复讲别人的内容；需要时可以写「Day X 会细讲」）
- Day 1  全景与 Tokenizer：仓库地图、训练流水线总览、一次完整 forward 的鸟瞰；tokenizer.json / tokenizer_config.json / chat_template / train_tokenizer.py
- Day 2  骨架：MiniMindConfig、Embedding、RMSNorm、MiniMindModel.forward 骨架(残差流)、lm_head 权重共享、参数量都花在哪
- Day 3  位置编码：precompute_freqs_cis、apply_rotary_pos_emb(rotate_half 约定)、start_pos 切片、YaRN 长文本外推
- Day 4  Attention：GQA、QK-Norm、repeat_kv、因果掩码、SDPA vs 手写分支、padding attention_mask
- Day 5  FFN(SwiGLU) + MiniMindBlock(pre-norm 残差) + MiniMindForCausalLM.forward 与 loss(shift、ignore_index、logits_to_keep) + logit lens
- Day 6  MoE 深入：MOEFeedForward 路由/top-k/index_add_、aux loss、DDP 小技巧、总参数 vs 激活参数
- Day 7  推理：generate 循环、KV cache、temperature/top-k/top-p/repetition penalty、streamer、eval_llm.py
- Day 8  数据管线：lm_dataset.py 全部 Dataset（Pretrain/SFT 的 label mask、DPO、RLAIF、AgentRL）、chat 模板与 loss mask 的配合
- Day 9  Pretrain & SFT 训练循环：train_pretrain.py / train_full_sft.py / trainer_utils.py（lr 调度、梯度累积、混合精度、裁剪、checkpoint/续训、DDP）
- Day 10 LoRA 与蒸馏：model_lora.py、train_lora.py、train_distillation.py
- Day 11 DPO：train_dpo.py（logits_to_log_probs、dpo_loss、ref model、chosen/rejected 拼 batch）
- Day 12 PPO：train_ppo.py（CriticModel、reward 计算、GAE、clip 目标、KL）、rollout_engine.compute_per_token_logps
- Day 13 GRPO 与 Rollout Engine：train_grpo.py（组采样、组内归一化 advantage、completion mask、KL 估计、loss 变体）、rollout_engine.py
- Day 14 Agent RL + 部署 + 总复习：train_agent.py（多轮 tool-call rollout、mask、reward）、serve_openai_api.py、convert_model.py、eval_toolcall.py；毕业测验

## 最终回复（保持简短）
文件清单；每个 lab 的运行耗时；发现的 learnkit 问题；你不确定、需要人工复核的点。
