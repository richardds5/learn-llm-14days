# ---
# title: 真实权重上开/关 inference_rope_scaling 的 loss 对比
# timeout: 180
# tasks:
#   - "把 LONG 从 3072 改成 1024（仍超过训练用的 768，但离 orig_max=2048 更近）：两条配置的差距变大还是变小？"
#   - "在 build 里 cfg 那一行后面加一句 `if yarn: cfg.rope_scaling['original_max_position_embeddings'] = 768`（改成这份权重真实的训练长度）：loss 是被救回来了，还是更差？"
# ---
import itertools, json, torch
from learnkit import *
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

SHORT, LONG = 512, 3072          # 👉 SFT 训练时 max_seq_len 只有 768，LONG 远超训练见过的位置
tok, texts = get_tokenizer(), []
with open("dataset/dpo.jsonl", encoding="utf-8") as f:                  # 仓库里没有 pretrain 语料，借 dpo 的回答拼长文本
    for line in itertools.islice(f, 60):
        texts.extend(m["content"] for m in json.loads(line)["chosen"])
ids_all = tok(" ".join(texts), return_tensors="pt").input_ids

def build(yarn):
    cfg = MiniMindConfig(inference_rope_scaling=yarn)                   # 👉 想换 YaRN 超参就在这后面改 cfg.rope_scaling
    m = MiniMindForCausalLM(cfg)
    m.load_state_dict(torch.load(weight_path("full_sft", 768), map_location="cpu"), strict=True)
    return m.eval()

models = {"关闭 YaRN": build(False), "开启 YaRN（factor=16）": build(True)}
rows = []
for L in (SHORT, LONG):
    ids = ids_all[:, :L]
    for name, m in models.items():
        with torch.no_grad():
            rows.append([f"T={ids.shape[1]}", name, round(m(ids, labels=ids).loss.item(), 4)])

table(rows, headers=["输入长度", "配置", "整段平均 loss"],
      title="full_sft_768：训练时只见过 768 以内的位置，YaRN 却假设见过 2048")
