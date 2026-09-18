# ---
# title: 八个训练阶段各自往 out/ 里写什么文件名
# timeout: 60
# tasks:
#   - "把 HIDDEN 改成 512：八行文件名全变了，out/ 里一个都对不上——这就是换 hidden_size 之后加载不到权重的原因"
#   - "把 MOE 改成 '_moe'：只有 full_sft 那一行在 out/ 里找得到（本地只存了这一个 MoE 权重）"
# ---
import re
from pathlib import Path
from learnkit import *

HIDDEN, MOE = 768, ""  # 👉 对应 MiniMindConfig(hidden_size=768, use_moe=False)

SCRIPTS = ["train_pretrain", "train_full_sft", "train_distillation", "train_lora",
           "train_dpo", "train_ppo", "train_grpo", "train_agent"]

rows = []
for name in SCRIPTS:
    src = Path(f"trainer/{name}.py").read_text(encoding="utf-8")
    # 前缀不是我总结的约定，就写在各脚本 argparse 的 default 里
    line = next(l for l in src.split("\n") if "--save_weight" in l or "--lora_name" in l)
    flag, prefix = re.search(r"--(\w+)", line).group(1), re.search(r"default=['\"]([^'\"]+)", line).group(1)
    ckp = f"out/{prefix}_{HIDDEN}{MOE}.pth"                      # 和 train_full_sft.py:64 同一条 f-string
    rows.append([f"{name}.py", f"--{flag}", prefix, ckp, "✅ 本地有" if Path(ckp).is_file() else "—"])

table(rows, headers=["训练脚本", "前缀参数", "默认前缀", "产出文件 out/{前缀}_{hidden_size}{_moe}.pth", "本地"],
      title=f"权重命名规则（hidden_size={HIDDEN}, moe 后缀={MOE!r}）")
