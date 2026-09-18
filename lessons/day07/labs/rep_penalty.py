# ---
# title: 已出现过的 token 被 repetition_penalty 改成什么样
# timeout: 60
# tasks:
#   - "把 RP 从 1.5 改成 1.0：两列惩罚后的 logit 会回到原值吗？（源码里 `if repetition_penalty != 1.0` 会整段跳过）"
#   - "把 RP 改成 0.5（小于 1）：源码写法下正 logit 和负 logit 各往哪个方向走？这时「惩罚」变成了什么？"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
text = tok.apply_chat_template([{"role": "user", "content": "用一句话介绍北京"}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt")["input_ids"]
ids = model.generate(ids, max_new_tokens=32, do_sample=False, eos_token_id=None)   # 贪心生成到开始复读
with torch.no_grad():
    logits = model(ids, use_cache=False).logits[0, -1, :].clone()

RP = 1.5                                                    # 👉 repetition_penalty
seen = torch.unique(ids[0])                                 # generate L269-L270：这一行里出现过的所有 token
score = logits[seen]
fixed = torch.where(score > 0, score / RP, score * RP)      # 源码写法：正数除、负数乘
naive = score / RP                                          # 常见的想当然写法：一律相除
after = lambda new: torch.softmax(logits.clone().index_put_((seen,), new), dim=-1)
p0, p_fixed, p_naive = torch.softmax(logits, -1), after(fixed), after(naive)

pick = torch.argsort(score)[[0, 2, -2, -1]]                 # 两个负 logit + 两个正 logit
table([[repr(tok.decode([seen[j]]))[1:-1], f"{score[j]:+.2f}", f"{fixed[j]:+.2f}", f"{naive[j]:+.2f}",
        f"×{p_fixed[seen[j]] / p0[seen[j]]:.2f}", f"×{p_naive[seen[j]] / p0[seen[j]]:.2f}"] for j in pick.tolist()],
      headers=["已出现过的 token", "原始 logit", f"源码 where 之后", "一律相除之后", "概率变化（源码）", "概率变化（一律相除）"],
      title=f"repetition_penalty={RP}：{len(seen)} 个已出现的 token 里挑 4 个看")
