# ---
# title: create_chat_prompt 渲染出来的 ChatML 文本
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "在 conv 最前面插一条 {'role': 'system', 'content': '你是 minimind'}：渲染结果最前面多了哪一段？token 数涨了多少？"
#   - "把最后一行改成 tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) 再打印 repr：结尾多出来的那一段是什么？"
# ---
import json, tempfile
from pathlib import Path
from learnkit import *
from dataset.lm_dataset import SFTDataset

tok = get_tokenizer()
conv = [  # 👉 改这里：加一轮 / 删一轮，看渲染结果多出或少掉哪一段
    {"role": "user", "content": "1+1="},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": "再加 3 呢"},
    {"role": "assistant", "content": "5"},
]
path = Path(tempfile.mkdtemp()) / "toy_sft.jsonl"
path.write_text(json.dumps({"conversations": conv}, ensure_ascii=False), encoding="utf-8")

ds = SFTDataset(str(path), tok, max_length=256)
prompt = ds.create_chat_prompt(conv)

print("=== 原样打印（人眼看结构）===")
print(prompt)
print("=== repr（看得见每一个 \\n）===")
print(repr(prompt))
ids = tok(prompt).input_ids
print(f"\ntokenize 之后 {len(ids)} 个 token；第 0 个 = {ids[0]} = {tok.decode([ids[0]])!r}"
      f"（add_bos_token=False，所以没有被额外插入特殊 token）")
