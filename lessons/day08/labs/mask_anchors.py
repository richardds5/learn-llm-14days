# ---
# title: bos_id / eos_id 这两段锚点到底是什么
# timeout: 60
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 PROBE 里的 f'{tok.bos_token}assistant\\n' 改成 f'{tok.bos_token}assistant'（去掉换行）：长度变成几？它还等于 ds.bos_id 吗？"
#   - "在 PROBE 里加一行 '<|im_start|>assistant\\n'（直接写字面量，不用 tok.bos_token）：编码结果和 ds.bos_id 一样吗？"
# ---
from learnkit import *
from dataset.lm_dataset import SFTDataset

tok = get_tokenizer()
ds = SFTDataset("dataset/lora_identity.jsonl", tok, max_length=128)

PROBE = [f"{tok.bos_token}assistant\n", f"{tok.eos_token}\n",   # 👉 改这里
         f"{tok.bos_token}user\n", f"{tok.bos_token}system\n", f"{tok.bos_token}tool\n"]
rows = []
for s in PROBE:
    ids = tok(s, add_special_tokens=False).input_ids
    rows.append([repr(s), ids, len(ids), [tok.decode([i]) for i in ids],
                 "✅" if ids == ds.bos_id else ("eos_id" if ids == ds.eos_id else "❌")])
table(rows, headers=["文本", "token ids", "长度", "逐 token 解码", "== ds.bos_id ?"],
      title="滑动匹配要比对的是一整段 token，不是单个 id")
print("ds.bos_id =", ds.bos_id, "  ds.eos_id =", ds.eos_id)
print("四种角色的第 0 个 token 都是", ds.bos_id[0], "=", repr(tok.decode([ds.bos_id[0]])),
      "—— 只看第 0 个 id 分不出是谁在说话")
