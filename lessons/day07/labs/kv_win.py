# ---
# title: 关掉 KV cache：同样生成 64 个 token，算多少、慢多少
# timeout: 90
# tasks:
#   - "把 N_NEW 从 64 改成 16：两个比值（token·层 的比、实测耗时比）各自怎么变？"
#   - "把 do_sample 改成 True（两次调用前都加一句 torch.manual_seed(0)）：两条路径生成的 token 还完全相同吗？"
# ---
import time
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
text = tok.apply_chat_template([{"role": "user", "content": "用一句话介绍北京"}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt")["input_ids"]
P, N_NEW = ids.shape[1], 64                      # 👉 N_NEW 越大，两条路径的差距越明显

def run(use_cache):
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, eos_token_id=None, use_cache=use_cache)
    return out, time.time() - t0

out_c, t_c = run(True)
out_n, t_n = run(False)

# 有 cache：第 0 步喂 P 个 token，之后每步 1 个；无 cache：第 k 步要把 P+k 个 token 全部重算一遍
tok_cache = P + (N_NEW - 1)
tok_nocache = sum(P + k for k in range(N_NEW))
table([["送进 embed 的 token 总数", f"{tok_cache:,}", f"{tok_nocache:,}", f"{tok_nocache / tok_cache:.1f}x"],
       ["实测耗时", f"{t_c:.2f}s", f"{t_n:.2f}s", f"{t_n / t_c:.1f}x"],
       ["生成出来的 token 序列", "—", "和左边逐位相同" if torch.equal(out_c, out_n) else "和左边不同", "—"]],
      headers=["对比项", "use_cache=True", "use_cache=False", "无 cache / 有 cache"],
      title=f"prompt {P} 个 token，贪心生成 {N_NEW} 个（本机 CPU，数字每次会有波动）")
