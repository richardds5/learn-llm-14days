# ---
# title: batch 里两行在不同的步数结束，各自的结尾是什么
# timeout: 90
# tasks:
#   - "把 torch.manual_seed(7) 换成 3：两行结束的步数变了吗？「最后 5 个 token id」那一列呢？"
#   - "把 eos_token_id=EOS 改成 eos_token_id=None：L279 的强制覆盖和 L285 的提前 break 都失效，循环会跑满 80 步——第 1 行在第 16 步之后还会不会接着说出新的句子？"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
EOS = tok.eos_token_id
text = tok.apply_chat_template([{"role": "user", "content": "用一句话介绍北京"}],
                               tokenize=False, add_generation_prompt=True)
prompt = tok(text, return_tensors="pt")["input_ids"].repeat(2, 1)   # 同一个 prompt 两行，随机采样各走各的

torch.manual_seed(7)                            # 👉 换个种子，两行结束的步数就会变
out = model.generate(prompt, max_new_tokens=80, do_sample=True, temperature=0.85, top_p=0.85, eos_token_id=EOS)
new = out[:, prompt.shape[1]:]                  # 只看新生成的部分 [2, n]

rows = [[f"行 {i}",
         (new[i] == EOS).nonzero().flatten()[0].item(),
         (new[i] == EOS).sum().item(),
         str(new[i, -5:].tolist()),
         tok.decode(new[i], skip_special_tokens=True)] for i in range(2)]
table(rows, headers=["batch 行", "第一次采到 eos 是第几步", f"这一行里 id={EOS} 共出现几次", "最后 5 个 token id", "解码文本"],
      title=f"两行一起生成，循环跑了 {new.shape[1]} 步才停（id={EOS} 就是 eos）")
