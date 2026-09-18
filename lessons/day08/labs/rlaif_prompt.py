# ---
# title: RLAIFDataset 渲染出来的 prompt 停在哪里
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 thinking_ratio 的两个取值改成 0.5 和 0.5，多跑几次：两行结尾还固定吗？"
#   - "加一行打印 ds[0]['prompt'].count('<|im_start|>') 和 len(ds.samples[0]['conversations'])：两个数一样吗？（渲染时少了一条消息、又多了一个生成提示）"
# ---
import tempfile
from itertools import islice
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import RLAIFDataset

tok = get_tokenizer()
dst = Path(tempfile.mkdtemp()) / "toy_rlaif.jsonl"  # rlaif.jsonl 有 23MB，只取前 3 行
with open("dataset/rlaif.jsonl", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in islice(fin, 3): fout.write(line)

for ratio in (1.0, 0.0):  # 👉 改这里
    ds = RLAIFDataset(str(dst), tok, thinking_ratio=ratio)
    item = ds[0]
    print(f"thinking_ratio={ratio}  返回的 keys={list(item.keys())}  answer={item['answer']!r}")
    print(f"   prompt 结尾 50 个字符: {item['prompt'][-50:]!r}")
    print(f"   prompt 是 {type(item['prompt']).__name__}，不是 tensor；长度 {len(tok(item['prompt']).input_ids)} 个 token\n")

last = ds.samples[0]["conversations"][-1]
print("数据里最后一条消息（被 conversations[:-1] 扔掉的那条）:", str(last)[:90])
