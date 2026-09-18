# ---
# title: 左 padding 的 batch：漏传 attention_mask 会改掉谁的预测
# timeout: 60
# tasks:
#   - "把 texts 里的短句 '你好' 换成一句和长句差不多长的话：padding 变少之后，误差和 argmax 还会变吗？"
#   - "把 tok.padding_side 改成 'right'：短句那一行的 logits[:, -1, :] 取的是哪个位置的输出？（右 padding 时最后一列本身就是 pad）"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft", flash_attn=False), get_tokenizer()
tok.padding_side = "left"                         # 👉 推理常用左 padding：真实的最后一个 token 对齐在同一列
texts = ["请介绍一下你自己，越详细越好", "你好"]
enc = tok(texts, return_tensors="pt", padding=True)
ids, am = enc["input_ids"], enc["attention_mask"]

logits_ok = model(ids, attention_mask=am).logits[:, -1, :]      # 正确写法
logits_bug = model(ids, attention_mask=None).logits[:, -1, :]   # 👉 常见 bug：忘了传 attention_mask
top = lambda t: repr(tok.decode([t.argmax(-1).item()]))

table([[repr(t), int((m == 0).sum()), f"{(a - b).abs().max().item():.3f}", top(a), top(b)]
       for t, m, a, b in zip(texts, am, logits_ok, logits_bug)],
      headers=["输入", "pad 了几个", "最后一位 logits 的最大绝对误差", "传了 mask 的 next token", "漏传 mask 的 next token"],
      title=f"同一个 batch（pad_token_id={tok.pad_token_id}，即 {tok.pad_token}）：没有 padding 的那一行完全不受影响")
