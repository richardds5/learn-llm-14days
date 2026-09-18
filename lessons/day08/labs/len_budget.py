# ---
# title: 定长方案的账：不同 max_length 的截断率与 padding 占比
# timeout: 120
# tasks:
#   - "把 CANDIDATES 里加上 340（train_pretrain.py 的默认值）和 1024（train_dpo.py 的默认值）：padding 占比分别到多少？"
#   - "把 add_system_ratio 改回默认的 0.2 再跑一次：长度统计会不会变得不稳定？"
# ---
import statistics
from learnkit import *
from dataset.lm_dataset import SFTDataset, pre_processing_chat

tok = get_tokenizer()
ds = SFTDataset("dataset/lora_identity.jsonl", tok, max_length=4096)  # 先用一个大到不会截断的值，只量真实长度
lens = []
for i in range(len(ds)):
    conv = pre_processing_chat(list(ds.samples[i]["conversations"]), add_system_ratio=0.0)  # 关掉随机 system
    lens.append(len(tok(ds.create_chat_prompt(conv)).input_ids))

CANDIDATES = [64, 96, 128, 256, 768]  # 👉 改这里；768 是 train_full_sft.py 的默认值
rows = []
for ML in CANDIDATES:
    trunc = sum(1 for l in lens if l > ML)
    pad = sum(max(0, ML - l) for l in lens) / (ML * len(lens))
    rows.append([ML, trunc, f"{trunc / len(lens):.1%}", f"{pad:.1%}"])
table(rows, headers=["max_length", "被截断的样本数", "截断占比", "平均 padding 占比"],
      title=f"lora_identity.jsonl 共 {len(lens)} 条，长度 min={min(lens)} / 中位数={int(statistics.median(lens))} / max={max(lens)}")
