# 课程编写规范 (AUTHORING)

本学习站面向的读者：**Python 精通、LLM 原理了解、但没手写过 LLM** 的工程师。
目标是让他通过 MiniMind 这个仓库把「原理」和「每一行实现、每一个 tensor 的维度」对上号。
所以：**不要科普原理（他都懂），要讲实现细节**——为什么这么写、这一行之后 shape 变成什么、
有哪些坑、和 Llama/Qwen 等主流实现有何异同、改掉会怎样。

语言：中文为主，技术术语保留英文 (attention, KV cache, logits, residual…)。

## 1. 目录结构

```
learn-llm-14days/            # 仓库根目录
  start.sh                   # 启动脚本：./start.sh
  server.py                  # 网页后端
  runner.py                  # 命令行跑实验：venv/bin/python runner.py lessons/dayNN/labs/x.py
  lib/learnkit/               # learnkit（labs 里 from learnkit import *）
  web/                        # 前端
  lessons/dayNN/
    lesson.md                 # 正文 (markdown + 少量指令)
    quiz.json                 # 选择题
    labs/<lab_id>.py          # 可运行实验，每个文件一个实验
  tools/
    check_lessons.py          # 静态检查：venv/bin/python tools/check_lessons.py [--run]
    make_plan.py               # 生成 PLAN.md
    briefs/                    # 给作者/审校 subagent 的说明
  scripts/download.py          # 下载 minimind/out、minimind/dataset
  minimind/                    # MiniMind 源码快照 (@a3c7b01, 2026-09-10；可用环境变量 MINIMIND_ROOT 覆盖)
    model/ trainer/ dataset/ scripts/ eval_llm.py ...
    out/                        # 权重，下载得到，已 gitignore
    dataset/*.jsonl              # 数据，下载得到，已 gitignore
  requirements.txt
  progress.json / workspace/ / .runs/   # 运行时产物，已 gitignore
```

## 0. 教学法：总览 → 章 → 小节，一次只讲一个点（最重要，先读这一节）

读者对前两版的反馈：
- 第一版（长文）：「内容堆砌、平铺直叙、没有重点」。
- 第二版（一堆问句卡片）：「又走另一个极端：没有整体感；标题全是问号、太散，没有提炼；只有问题没有讲解，应该有问题也有答案」。

所以现在的形态是**老师上课**：先给全景，再按主题分章，每章拆成几个小节；**每个小节先把一个点讲清楚，再让读者就这一个点动手**。

- **总览**：进来第一屏。今天这部分代码在整个模型/流水线里的位置、解决什么问题、分成哪几章、学完能得到什么。配一条 `{{flow}}` 数据流。
- **章**（3~5 个）：提炼过的主题，名词短语标题，如「模型的配置：MiniMindConfig」。章下面有 1~3 句导语。
- **小节**（每章 2~4 个，全天 9~13 个）：网页上的**一屏**、5~8 分钟、只讲一个点。标题是**陈述式的名词短语**（「决定 shape 的七个数字」），**不要用问句**——问题放到小节里面去。
- 每个小节的固定节奏——**先讲，后练**：
  1. **讲解**：这段代码在做什么、为什么这么写、shape 怎么变、坑在哪。要讲透，读者只读这一段也应该能学会（300~700 字，源码片段插在讲解中间）。
  2. **动手验证**：一个只针对这个点的微实验，让读者亲眼看到刚才讲的 shape / 数值；再给一处改动让他试。
  3. **小测**：1~2 道题检验是否真懂（题目自带解析 = 答案）。
  4. **结论**：一句话 `[!KEY]`。
- 不在主线上的东西（历史包袱、和 Llama/Qwen 的对比、边角坑、下游才用到的字段）：放进 `> [!MORE]` 折叠块，或做成 `.side` 选学小节。
- 判断标准：只看总览，能说出今天的结构；只看某一小节，10 秒内能说出「这一屏要我搞懂的是哪一件事」，并且这一屏里**既有问题也有答案**。

## 2. lesson.md —— 章节格式（`format: points`，新内容一律用这个）

```markdown
---
day: 2
format: points               # ← 必须有，网页据此进入「总览 → 章 → 小节」的步进模式
title: "骨架：Config、Embedding、RMSNorm 与残差流"
subtitle: "一句话副标题"
minutes: 75
mainline: "config 里的几个数字怎么撑起一条宽 C 的总线：input_ids 上车，L 层读写，最后靠共享的 [V,C] 表下车"
files:
  - model/model_minimind.py#L10-L60
goals:                        # 3~5 条
  - ...
---

总览正文（第一个 `# ` 之前）：200~400 字。今天这部分代码在全局的位置、解决什么问题、分哪几章、各章之间的关系。

{{flow: input_ids [B,T] | *embed_tokens* | hidden_states [B,T,C] | MiniMindBlock × L | *norm* | *lm_head* | logits [B,T,V]}}

# 模型的配置：MiniMindConfig

章导语：1~3 句。这一章要解决什么问题、包含哪几个小节、和上一章什么关系。

## 决定 shape 的七个数字 {#cfg-dims}

讲解（先讲清楚）……

{{source:model/model_minimind.py#L20-L31}}

讲解续：盯住哪一行、为什么……

{{lab:cfg_dims}}

{{quiz:q1}}

> [!KEY]
> 一句话结论（≤ 2 句）。

> [!MORE] 其余字段归哪一天管
> 折叠起来的延伸内容。

## kwargs.get 的写法与默认值覆盖 {#cfg-kwargs .side}
……（`.side` = 选学，进度条上是虚线，不计入「今天完成」）

# RMSNorm
……
```

`{#cfg-dims}` 是这个小节的 **id**（直达链接 `#/day/2/p/cfg-dims`、进度记录、`[见参数量一节](#params)` 这种站内引用都靠它），网页上不会显示。

规则（`tools/check_lessons.py` 会检查）：

| 项 | 要求 |
|---|---|
| 章 | `# 名词短语标题`；全天 3~5 章；每章 60~150 字导语（不计入小节字数）+ 2~4 个小节 |
| 小节标题 | `## 陈述式名词短语 {#kebab-id}`，**不要以问号结尾**；id 当天唯一；选学加 ` .side` |
| 讲解 | 每小节正文（不含 MORE、代码块、表格、指令）**300~700 字**：太短 = 没讲清楚，太长 = 其实是两个点。口径 = 去掉空白后的字符数，行内 `code` 也算、链接只算文字部分；表格只放速查数据（≤ 6 行），不要把讲解塞进表格 |
| 源码 | 每小节 ≤ 1 个 `{{source}}`（确有必要可 2 个），每个 ≤ 20 行，框最窄的范围 |
| 实验 | 每小节 ≤ 1 个 `{{lab}}`。主线小节至少 70% 要有实验 |
| 题目 | 每小节 1~2 道 `{{quiz:qN}}`（或 `{{predict:pN}}`，见下） |
| 结论 | 每小节**恰好一个** `> [!KEY]`，≤ 2 句。最后一屏「今日小结」由所有 KEY 按章自动汇总，**不要自己写小结小节** |
| 综合题 | quiz.json 里没被任何小节引用的题会出现在最后的「综合测验」里，留 3~5 道跨章的题 |
| 数据流 | `{{flow: a | *b* | c}}`：一条横向数据流，`*星号*` 包住的是今天/本节的主角（高亮）。总览必须有一条；小节里也可以用 |

两种题：
- **quiz**（默认用这个）：立即判分 + 解析。放在实验后面，考「刚才讲的东西换个条件会怎样」。解析要完整——它就是「答案」。
- **predict**（可选，少用）：放在实验**前面**，读者选完先不揭晓，同小节的实验跑成功后才揭晓（也可以点「直接看答案」）。
  只在结果**反直觉**、值得先下注时用（如「fp16 溢出得到的不是 nan 而是 0」），全天最多 3~4 道；用的时候，前面的讲解、**小节标题、lab 的 title/tasks 都不要提前说出答案**
  （标题只点主题不点结论：「.float() 的作用」可以，「fp16 下的静默清零」不行），把解读放在实验后面。
  题目必须能被实验输出直接验证；trace 类实验只问 shape / 调用次数 / 哪一行没执行 / 在哪一行报错，不要问精确数值；
  解析里**按内容定位**（「看 `x.pow(2).mean(...)` 那一行后面的色块」），不要按位置（「第 2 行」）。

### 微实验（小节里的 lab）额外要求

- 代码 **≤ 30 行**（不含头部 YAML），**只产出一个焦点输出**：一个 trace，或一张表 / 一张图 / 几行 print。不要一个实验里打印五样东西。
- 5 秒内跑完（加载真实权重的 ≤ 10 秒）。
- `tasks` 只写 **1~2 条**，每条是**一处**具体改动 + 让读者观察什么（「把 `keepdim=True` 改成 `False`：在哪一行报什么错？」）。
- 追踪函数时用 `focus=` 把视野收窄到这个小节关心的那几行（其余行网页上默认折叠）：
  `trace(model, fns=[Attention.forward], dims=dict(B=3, T=7), focus=(112, 116), tree=False)`；
  也可以按变量名：`focus=["xq", "xk", "xv"]`。只想看逐行追踪、不需要模块调用树时加 `tree=False`。
- **一行流要拆开看**：`trace(..., expand=True)` 会把 focus 的那几行按求值顺序拆成子表达式，逐个标出 shape——
  `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 里那个 `[B,T,1]` 就是这样才看得见的。也可以点名某几行：`expand="L57"` / `expand=[57, 60]`。
  它的实现是「在该行执行前把各子表达式先求值一遍」，所以**不要用在有副作用的行上**（原地修改 `+=`、KV cache `torch.cat`、随机采样）。
- lab 文件名 = 小节 id 的下划线形式（`cfg-dims` → `labs/cfg_dims.py`）。

### 2.1 正文里其他可用的 markdown

- 公式用 KaTeX：行内 `$...$`，独立 `$$...$$`。
- 提示框写在 blockquote 第一行：`> [!KEY]`（每节唯一的一句话结论）、`> [!MORE] 标题`（折叠的延伸）、`> [!TIP]`、`> [!WARNING]`。
- 表格、代码块 (```python) 照常使用。
- 引用源码位置写成 `model/model_minimind.py:124`（路径相对 `minimind/`），课文和 quiz 解析里都会自动变成可点击跳转。
- **所有关于代码的陈述必须对照源码核实**；`tools/check_lessons.py` 会检查 `{{source}}` 的行号范围是否存在，但查不出「引用对了行、说错了事」。

## 3. labs/*.py

每个 lab 是一个**可以独立运行的 python 脚本**，运行环境：
- cwd = `minimind/` 目录（即 `MINIMIND_ROOT`）；`sys.path` 里有 `minimind/` 目录和 `lib/`（`learnkit` 所在目录）
- 所以可以 `from model.model_minimind import ...`、`from dataset.lm_dataset import ...`、
  `from trainer.train_dpo import dpo_loss` (trainer 下的脚本都有 `if __name__ == "__main__"` 保护，import 是安全的)
- python 3.10，torch 2.x (CPU / MPS，无 CUDA)，transformers 4.57

文件头是一段 YAML 元信息（写在注释里）：

```python
# ---
# title: 逐行追踪 Attention.forward 的 tensor 维度
# timeout: 60                      # 秒，默认 60；训练类实验可以给到 300~600
# sources:                         # 可选：允许读者在网页里直接修改的仓库源码文件（改动只对本实验生效，不会写回仓库）
#   - model/model_minimind.py
# tasks:                           # 「动手改」任务，2~4 条，要具体到改哪个值、预期观察到什么
#   - "把 num_key_value_heads 改成 2：xk 的 shape 变成什么？repeat_kv 之后呢？"
#   - "把 flash_attn 改成 True，再跑一次：哪几行不再被执行？"
# ---
import torch
from learnkit import *
...
```

### 硬性要求

1. **必须实际跑通**：`cd 仓库根目录 && venv/bin/python runner.py lessons/dayNN/labs/xxx.py`，退出码 0，无 traceback。
2. **快**：看 shape 的实验用 `build_model(num_hidden_layers=2)`，整个脚本 < 10 秒；需要真实权重的 < 20 秒；
   训练类实验要用很小的模型/很少的步数，< 2 分钟，并用 `live()` 实时画 loss。
3. **代码要短而清晰**（20~60 行），变量名要和仓库源码保持一致，读者要在上面改东西。
   关键位置写中文注释，特别是「这里值得改一改」的地方用 `# 👉` 标出来。
4. **优先调用仓库里的真实函数/类**，不要自己重写一份玩具实现（除非是为了和仓库实现做对照）。
5. 不要读取不存在的数据：`dataset/` 下只有 `dpo.jsonl rlaif.jsonl lora_identity.jsonl lora_medical.jsonl lora_exam.jsonl agent_rl.jsonl agent_rl_math.jsonl`，
   **没有** pretrain / sft 主数据。需要预训练文本时自己在脚本里写几条，或者从上述 jsonl 里取前 N 条。
   大 jsonl 不要整个读进来，用 `itertools.islice` 取前几十行。
6. 可用权重（`minimind/out/`）：`full_sft_768.pth`、`pretrain_768.pth`、`full_sft_768_moe.pth`、`lora_identity_768.pth`、`lora_medical_768.pth`。
   没有 dpo/ppo/grpo/reward model 权重；RL 课用 `full_sft` 同时当 policy 和 ref，reward 用规则函数或随机数模拟。
7. 不要联网，不要写仓库里的任何文件；临时文件写到一个临时目录（如 `tempfile.mkdtemp()`）而不是仓库里。
8. 不要 `plt.show()` / matplotlib（没装）。所有可视化用 learnkit。

## 4. learnkit API（`from learnkit import *`）

```python
# ---- 模型 / tokenizer ----
model = build_model(num_hidden_layers=2, **任意 MiniMindConfig 参数)   # 随机初始化, eval, CPU, 固定 seed
model = load_model("full_sft")             # 官方权重；load_model("pretrain")；load_model("full_sft", use_moe=True)
model = load_model("full_sft", flash_attn=False)   # 走手写 attention 分支（才有 scores 这个局部变量）
tok   = get_tokenizer()                    # AutoTokenizer.from_pretrained("model")
dev   = best_device()                      # 'mps' on Mac

# ---- 核心：tensor 维度追踪 ----
with trace(model,                              # 可选。给了就记录「模块调用树」(每个子模块的输入→输出 shape)
           fns=[Attention.forward, apply_rotary_pos_emb],   # 逐行追踪这些函数：每行读/写了哪些 tensor、shape 是什么
           dims=dict(B=3, T=7),                # 你知道的维度符号；H/KV/D/C/I/V/E 会自动从 model.config 推出 (前提是传了 model；只传 fns 时要自己写全，如 dims=dict(B=3, T=7, C=768, I=2432))
           title="标题",
           capture=["scores"],                 # 可选：把这些局部变量的真实 tensor 抓出来
           max_calls=4,                        # 每个函数最多详细记录前几次调用
           focus=(112, 116),                   # 可选：这个知识点只关心哪几行 (行号范围 / 变量名列表 / {函数名: ...})，其余行网页上默认折叠
           expand=True,                        # 可选：把 focus 的那几行拆成子表达式逐个看 shape (不要用在有副作用的行上)
           tree=True) as tr:                   # False = 不要模块调用树
    model(input_ids)
tr.captured["Attention.forward"][0]["scores"]   # 第 0 次调用里 scores 的值 (list 下标 = 第几次调用)

# fns 里可以放：函数、方法、类(取其 forward)、或字符串 "trainer.train_dpo:dpo_loss"
# 也可以不给 model，只追踪普通函数： with trace(fns=[dpo_loss], dims=dict(B=2, T=16)): dpo_loss(...)
# ⚠️ 为了让维度标注不混淆，B、T 的取值要避开 config 里的数 (8, 4, 96, 768, 2432, 6400, 2)，推荐 B=3, T=7 或 T=11/13。
#    带 KV cache 的 decode 类实验 T 会逐步 +1：用 T=11 (11→12→13 都不撞)，不要用 T=7 (下一步 T+1=8 会被标成 H)。

# ---- 其他可视化 ----
show(xq=xq, xk=xk)                          # tensor 卡片：shape/dtype/统计量/数值预览
heatmap(t2d_or_3d, title=, labels=tokens)   # 热力图；3D [N,R,C] 会画成 N 张小图 (facet_titles=[...])；也可分别给 xlabels/ylabels
bars(labels, values, title=, highlight=[0]) # 条形图 (next-token 概率分布等)
plot({"lr": lrs}, title=, xlabel=, ylabel=, logy=False)   # 折线图；也可 plot(list)
live("loss 曲线", "train", step, loss)       # 训练循环里实时追加一个点
table(rows, headers=[...], title=)          # 表格
token_strip(tokens, values, title=, legend=)  # token 彩带：values 是 0/1 mask 或浮点数 (如逐 token loss)
note("**markdown** 说明")                    # 在输出区插入一段说明文字
print(...)                                  # 普通 stdout 也会显示
```

在终端直接运行时，上述函数会退化成文本打印，所以可以用 runner 在命令行里验证。

## 5. quiz.json

```json
[
  {
    "id": "q1",
    "type": "single",                       
    "question": "markdown 题干，可以含 `code` 和 $公式$",
    "options": ["选项 A", "选项 B", "选项 C", "选项 D"],
    "answer": [2],                          
    "explain": "markdown 解析：为什么对、其他为什么错，引用到具体源码行 model/model_minimind.py:124",
    "ref": "model/model_minimind.py#L124"
  }
]
```

- `type`: `single` 或 `multi`；`answer` 永远是下标数组（0 开始）。
- 章节格式：每小节 1~2 道 + 3~5 道综合题（全天 15~22 道）；旧长文格式是每天 6~8 题。题目要考**实现细节和维度推理**（"这一行之后 shape 是什么"、"如果把 X 改成 Y 会在哪一行报错/结果怎么变"），
  不要考背诵式的原理题。干扰项要有迷惑性，来自真实的常见误解。
- **答案必须通过读源码或跑代码核实**。涉及 shape/数值的题，务必写个小脚本验证后再定答案。
- 正确答案的位置要打散，不要总是同一个下标。
