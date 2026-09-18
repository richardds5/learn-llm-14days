# 给课程审校 (reviewer subagent) 的统一说明

仓库根目录：本仓库根目录（learn-llm-14days）。这是一个 MiniMind 源码学习站，`lessons/dayNN/` 下是每天的课程
（`lesson.md` 正文、`quiz.json` 选择题、`labs/*.py` 实验）。格式规范见 `AUTHORING.md`（第 0、2 节：总览 → 章 → 小节，每个小节先讲解、再微实验、再小测、最后一句 `[!KEY]` 结论）。

你的任务：**独立盲审**指定几天的选择题答案和正文里的事实性陈述。学习者会完全信任答案，所以错误的答案键危害极大。

## 步骤（对每一天）
1. 读 `quiz.json`。对每一道题：**先不要看 `answer` 和 `explain`**（用脚本只打印 id/question/options），
   自己去读相关源码、必要时写小脚本实际运行（涉及 shape / 数值 / 报错 / 行为的题必须跑代码验证），得出你自己的答案。
   运行方式：在仓库根目录（learn-llm-14days）执行 `venv/bin/python runner.py <脚本路径>`（cwd=minimind/ 目录，可 `from model.model_minimind import ...`、`from learnkit import *`）。
   临时脚本写到一个临时目录（如 `tempfile.mkdtemp()`）而不是仓库里。
2. 再对照 `answer`。不一致 → 深入查证谁对（以源码 + 实际运行结果为准）。同时检查：
   - 是否存在**多个选项都说得通**或题干有歧义的情况（single 题必须有且仅有一个最佳答案）；
   - `explain` 里的论述、引用的行号是否正确；`ref` 的行号是否指向相关代码；
   - 多选题 (`type: multi`) 的每个选项都要单独判断真假。
3. 抽查 `lesson.md`：**每个小节的 `[!KEY]` 结论都要核实**（它们会被汇总成「今日小结」，读者会当成定论记住）；另外再挑 10 条左右讲解里带具体数字 / 行号 / shape / 行为断言的陈述，对照源码或运行结果核实。
   `{{predict:pN}}` 题的答案必须能被**同一小节的 lab 输出**直接验证：把那个 lab 跑一遍（在仓库根目录执行 `venv/bin/python runner.py lessons/dayNN/labs/<id>.py`），确认输出确实支持答案键；lab 的 `tasks` 里断言了现象的，也顺手核实。
   注意 `{{source:path#Lx-Ly}}` 由工具保证行号不越界，但你要抽查 3 个看框住的代码是不是正文要讲的那段。
4. 发现问题就**直接修**（最小改动）：改 `quiz.json` 的 answer / 选项措辞 / explain / ref；改 `lesson.md` 里的事实错误。
   不要重写风格、不要增删题目、不要动 labs（除非 lab 的注释/任务里有事实错误）。改完确保 `quiz.json` 仍是合法 JSON，
   并运行 `venv/bin/python tools/check_lessons.py --days N` 确认静态检查通过。

## 约束
- 只能改你负责的那几天的 `quiz.json` / `lesson.md`（以及临时目录）。不要改 `lib/`、`web/`、`tools/`。
- **绝对不要编辑 minimind/ 源码文件（model/ dataset/ trainer/ scripts/ 等），临时改也不行。** 要验证「改源码后会怎样」，用 override：把改过的副本写到临时目录，写 `overrides.json`：`{"model/model_minimind.py": "/abs/path/副本.py"}`，再 `venv/bin/python runner.py <脚本> --overrides overrides.json`。
- 不要再派 subagent，自己完成。
- 不要联网，不要安装包。

## 最终回复（简短）
每天一行：检查了几道题、几条正文陈述；然后列出**每一处修改**（题号 / 位置、原来是什么、改成什么、证据）。没有问题就明说没有。
