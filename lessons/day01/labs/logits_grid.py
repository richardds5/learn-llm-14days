# ---
# title: 逐位置对照：模型猜的下一个 token vs 真实的下一个 token
# timeout: 90
# tasks:
#   - "把 QUESTION 换成一句你自己的问题：user 那几行仍然大面积猜错、`<|im_start|>assistant` 之后仍然几乎全对吗？"
#   - "把 logits[0].argmax(-1) 改成 logits[0].topk(2, -1).indices[:, 1]（取第二名）：猜对的行数掉到多少？"
# ---
import torch
from learnkit import *

tok = get_tokenizer()
model = load_model("full_sft")                       # out/full_sft_768.pth，真实训练过的权重

QUESTION = "天空是什么颜色的？"                        # 👉 换成你自己的问题
prompt = tok.apply_chat_template([{"role": "user", "content": QUESTION}],
                                 tokenize=False, add_generation_prompt=True)
input_ids = tok(prompt, return_tensors="pt")["input_ids"]      # [B=1, T]
T = input_ids.shape[1]

with torch.no_grad():
    logits = model(input_ids).logits                           # [B=1, T, V=6400]
pred = logits[0].argmax(-1)                                    # 每个位置各自的 top-1

rows, hit = [], 0
for t in range(T):
    guess = repr(tok.decode([pred[t].item()]))
    truth = repr(tok.decode([input_ids[0, t + 1].item()])) if t + 1 < T else "（还不存在 → 要生成的就是它）"
    hit += guess == truth
    rows.append([t, repr(tok.decode([input_ids[0, t].item()])), guess, truth, "✓" if guess == truth else ""])

table(rows, headers=["位置 t", "输入 token[t]", "argmax logits[0, t]（猜的 token[t+1]）", "真实 token[t+1]", "对"],
      title=f"logits 是 {list(logits.shape)}：T={T} 个位置各猜一次，前 {T - 1} 个有标准答案，猜对 {hit} 个")
