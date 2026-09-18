# ---
# title: 同一个问题，pretrain 和 full_sft 拿到的输入文本不一样
# timeout: 90
# tasks:
#   - "把 pretrain 那一行也换成 apply_chat_template 的文本：它会因为没见过 <|im_start|> 而输出更奇怪的东西吗？"
#   - "把 PROMPT 换成一个陈述句（比如 '中国的首都是北京'）：pretrain 的续写是不是反而更顺？"
# ---
import torch
from learnkit import *

tok, PROMPT, MAX_NEW = get_tokenizer(), "请用一句话介绍北京", 40

def run(weight, text):
    model = load_model(weight)
    ids = tok(text, return_tensors="pt")["input_ids"]
    out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=tok.eos_token_id)
    return [weight, repr(text)[1:-1][-46:], tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)]

# eval_llm.py L73-L76：权重名里带 'pretrain' 就只拼 bos_token，否则套 chat 模板
rows = [run("pretrain", tok.bos_token + PROMPT),
        run("full_sft", tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                                tokenize=False, add_generation_prompt=True))]
table(rows, headers=["权重", "喂进 tokenizer 的文本（尾部 46 字）", f"贪心生成的前 {MAX_NEW} 个 token"],
      title=f"同一个问题「{PROMPT}」：一个在续写，一个在回答")
