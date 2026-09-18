# ---
# title: 仓库地图：五个入口各自有多少行 python
# timeout: 60
# tasks:
#   - "把 ENTRIES 里 'model/' 那一行的路径改成 'model/model_minimind.py'：整个模型定义只有多少行？和 trainer/ 那一行比一比"
#   - "在 ENTRIES 末尾追加 ('../lib/', '这个学习站的 trace / 可视化工具库', '—')：它比 model/ 那一行大还是小？"
# ---
import glob
from learnkit import *

# 每一项 = (路径, 这个目录/文件负责什么, 哪几天会读它)
ENTRIES = [
    ("model/", "模型结构：MiniMindConfig + MiniMindForCausalLM + LoRA，以及 tokenizer 的两份产物", "Day 1~6, 10"),
    ("dataset/", "lm_dataset.py 里的 5 个 Dataset 子类 + 配套 .jsonl 语料", "Day 8"),
    ("trainer/", "每个训练阶段一个 train_*.py + 公共工具 trainer_utils.py + rollout_engine.py", "Day 9~13"),
    ("scripts/", "训练完之后要用的：格式转换、OpenAI 协议服务端、web demo、工具调用评测", "Day 14"),
    ("eval_llm.py", "根目录下唯一的手测脚本，--weight 指定要加载哪个阶段的权重", "Day 7"),
]

rows = []
for path, duty, days in ENTRIES:
    files = [f for f in glob.glob(path + "**/*.py", recursive=True) if "__pycache__" not in f] \
        if path.endswith("/") else [path]                      # 👉 只数 .py，不数 .jsonl / .json
    n_lines = sum(len(open(f, encoding="utf-8").read().splitlines()) for f in files)
    rows.append([path, len(files), f"{n_lines:,}", duty, days])

total = sum(int(r[2].replace(",", "")) for r in rows)
rows.append(["合计", sum(r[1] for r in rows), f"{total:,}", "整个 MiniMind 本体（不含这个学习站自己的代码）", "14 天"])
table(rows, headers=[".py 所在位置", "文件数", "行数", "职责", "归哪几天"], title="MiniMind 仓库地图")
