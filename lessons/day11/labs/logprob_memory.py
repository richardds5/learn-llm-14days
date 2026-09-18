# ---
# title: log_softmax 那一份 [2B, T, V] 有多大
# timeout: 60
# tasks:
#   - "把 V 改成 32000（Qwen 量级的词表）：同样的 batch_size 下峰值涨到多少？"
#   - "把 BYTES 改成 2（bf16 混合精度下 logits 是 2 字节）：batch_size=4 的那一格变成多少 MB？"
# ---
from learnkit import *

T, V, BYTES = 1024, 6400, 4          # 👉 train_dpo.py 的默认 max_seq_len / 词表 / fp32
SIZES = [1, 2, 4, 8, 16]             # 👉 --batch_size

mb = lambda n: n * T * V * BYTES / 1e6
bars([str(s) for s in SIZES], [mb(2 * s) for s in SIZES], highlight=[2],
     title=f"一份 [2·batch_size, {T}, {V}] fp32 张量的大小 (MB)，高亮的是默认 batch_size=4",
     xlabel="--batch_size（DPO 里实际进模型的行数是它的 2 倍）")

table([[s, 2 * s, f"{mb(2 * s):.0f}", f"{mb(2 * s) * 3:.0f}", f"{mb(s):.0f}"] for s in SIZES],
      headers=["batch_size", "实际行数 2B", "一份 (MB)", "DPO 至少 3 份 (MB)", "SFT 的 logits [B,T,V] (MB)"],
      title="三份 = policy 的 logits + policy 的 log_softmax（要留到 backward）+ ref 的一份（算完即弃）")
print(f"默认 batch_size=4：DPO 光这三份就要 {mb(8) * 3:.0f} MB，而同样 batch_size 的 SFT，logits 只有 {mb(4):.0f} MB。")
