# ---
# title: AgentRLDataset 过 DataLoader：默认 collate vs 自定义 collate
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 batch_size 从 2 改成 1：默认 collate 这次成功了吗？为什么（提示：一条样本内部不需要跟别人对齐）"
#   - "在 agent_collate_fn 的 return 之前加一行 torch.tensor([b['gt'] for b in batch])：报什么错？连 gt 这么简单的字段为什么也拼不成 tensor？"
# ---
import tempfile
from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from learnkit import *
from dataset.lm_dataset import AgentRLDataset

tok = get_tokenizer()
dst = Path(tempfile.mkdtemp()) / "toy_agent.jsonl"  # agent_rl_math.jsonl 有 18MB，只取前 4 行
with open("dataset/agent_rl_math.jsonl", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in islice(fin, 4): fout.write(line)

ds = AgentRLDataset(str(dst), tok)
print("单条样本 keys:", list(ds[0].keys()), " messages 的 role:", [m["role"] for m in ds[0]["messages"]],
      " gt:", ds[0]["gt"], " tools 个数:", len(ds[0]["tools"]))

# train_agent.py:462 里的那一行：三个字段各自收集成 list，不做任何张量拼接
def agent_collate_fn(batch):
    return {k: [b[k] for b in batch] for k in ("messages", "tools", "gt")}

BATCH = 2  # 👉 改这里
rows = []
for name, fn in [("自定义 agent_collate_fn", agent_collate_fn), ("DataLoader 默认 collate", None)]:
    try:
        b = next(iter(DataLoader(ds, batch_size=BATCH, collate_fn=fn)))
        rows.append([name, "成功", f"gt = {b['gt']}"])
    except Exception as e:
        rows.append([name, f"{type(e).__name__}", str(e)[:80]])
table(rows, headers=["collate_fn", "结果", "细节"], title=f"batch_size={BATCH} 时取一个 batch")
