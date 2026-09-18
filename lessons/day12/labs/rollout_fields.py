# ---
# title: RolloutResult 的六个字段
# timeout: 180
# sources:
#   - trainer/rollout_engine.py
# tasks:
#   - "把 max_new_tokens 从 24 改成 96：R 变成多少？（提示：只有所有序列都写出 EOS 时 generate 才会提前 break）"
#   - "把 num_generations 从 1 改成 3：哪几个字段的第 0 维变了？"
# ---
import torch
from learnkit import *
from trainer.rollout_engine import TorchRolloutEngine

tokenizer = get_tokenizer()
actor_model = load_model("full_sft")
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True)
           for q in ["请解释一下什么是光合作用？", "你好", "1+1 等于几？"]]   # 👉 三条长度不同的 prompt
enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=256, padding_side="left")                       # 👈 prompt 必须左 padding
torch.manual_seed(1)
engine = TorchRolloutEngine(actor_model, tokenizer, device="cpu")
r = engine.rollout(prompt_ids=enc.input_ids, attention_mask=enc.attention_mask,
                   num_generations=1, max_new_tokens=24, temperature=0.8)

B, P, R = enc.input_ids.size(0), enc.input_ids.size(1), r.completion_ids.size(1)
table([["output_ids", "[B, P+R]", str(list(r.output_ids.shape)), "prompt 和 response 拼好的完整序列"],
       ["completion_ids", "[B, R]", str(list(r.completion_ids.shape)), "只有 response 段"],
       ["per_token_logps", "[B, R]", str(list(r.per_token_logps.shape)), "old_logp，在 no_grad 里算的"],
       ["completions", "list[str]", f"{len(r.completions)} 条", "skip_special_tokens 解码，喂给 reward"],
       ["prompt_lens", "[B]", str(r.prompt_lens.tolist()), "torch 引擎恒等于 P"],
       ["completion_mask", "[B, R]", f"每行 1 的个数 {r.completion_mask.sum(1).tolist()}", "torch 引擎恒为全 1"]],
      headers=["字段", "shape", "实测", "说明"], title=f"TorchRolloutEngine.rollout 的六个返回值（B={B}, P={P}, R={R}）")
eos_hit = r.completion_ids.eq(tokenizer.eos_token_id)
print(f"prompt_ids[1] 开头 = {enc.input_ids[1, :6].tolist()}  ← 左 padding 填的是 pad_token_id={tokenizer.pad_token_id}")
print(f"每条 response 第一个 EOS(={tokenizer.eos_token_id}) 的下标 = "
      f"{torch.where(eos_hit.any(1), eos_hit.int().argmax(1), torch.full((B,), -1)).tolist()}  （-1 = 24 步内没写出 EOS）")
