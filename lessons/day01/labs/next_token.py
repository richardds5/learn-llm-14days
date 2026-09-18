# ---
# title: logits[0, -1] 的 top-5，和贪心续写出来的前 24 个 token
# timeout: 90
# tasks:
#   - "把 QUESTION 换成「你叫什么名字？」：top-1 还是不是问题的第一个字？续写的第一个 token 和 top-1 对得上吗？"
#   - "把 argmax 换成 torch.multinomial(torch.softmax(nxt_logits / 0.85, -1), 1)（温度采样）：同一个问题连跑两次，续写还一样吗？"
# ---
import torch
from learnkit import *

tok = get_tokenizer()
model = load_model("full_sft")

QUESTION = "天空是什么颜色的？"                        # 👉 换个问题再跑
prompt = tok.apply_chat_template([{"role": "user", "content": QUESTION}],
                                 tokenize=False, add_generation_prompt=True)
input_ids = tok(prompt, return_tensors="pt")["input_ids"]
T = input_ids.shape[1]

with torch.no_grad():
    probs = torch.softmax(model(input_ids).logits[0, -1].float(), dim=-1)   # 只取最后一个位置
top5 = torch.topk(probs, 5)
bars([repr(tok.decode([i])) for i in top5.indices.tolist()], top5.values,
     title=f"logits[0, -1] → 第 {T} 个位置预测的「下一个 token」top-5", highlight=[0])

# 贪心续写：把 argmax 拼回 input_ids 再 forward 一次（没用 KV cache，Day 7 才省）
cur = input_ids
with torch.no_grad():
    for _ in range(24):
        nxt_logits = model(cur).logits[0, -1]
        cur = torch.cat([cur, nxt_logits.argmax().view(1, 1)], dim=1)
        if cur[0, -1].item() == tok.eos_token_id: break
print("top-1 =", repr(tok.decode([top5.indices[0].item()])), f"  p={top5.values[0].item():.3f}")
print("贪心续写 =", repr(tok.decode(cur[0, T:].tolist())))
