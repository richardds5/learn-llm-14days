# ---
# title: 逐行看 TorchRolloutEngine.rollout 怎么拼出 RolloutResult
# timeout: 120
# sources:
#   - trainer/rollout_engine.py
# tasks:
#   - "把 max_new_tokens 从 16 改成 4：trace 里 completion_ids、per_token_logps 两行的 shape 各变成什么？"
#   - "把 num_generations 改成 1：repeat_interleave 那一行的输出第 0 维变成多少？（PPO 就是这么调这个引擎的）"
# ---
import torch
from learnkit import *
from trainer.rollout_engine import TorchRolloutEngine, create_rollout_engine

tok, model = get_tokenizer(), load_model("full_sft")
B, G, R = 3, 4, 16

prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in ["你是谁？", "太阳为什么是热的？", "推荐一本科普书。"]]
enc = tok(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
          padding_side="left", add_special_tokens=False)
P = enc["input_ids"].size(1)

engine = create_rollout_engine("torch", policy_model=model, tokenizer=tok, device="cpu")
torch.manual_seed(0)
with trace(fns=[TorchRolloutEngine.rollout], dims=dict(B=B, G=G, BG=B * G, P=P, R=R), tree=False,
           title="TorchRolloutEngine.rollout 逐行（rollout_engine.py:71-92）", max_calls=1):
    rr = engine.rollout(prompt_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        num_generations=G, max_new_tokens=R, temperature=0.9)

note(f"`RolloutResult` 的六个字段，torch 引擎给的值：\n\n"
     f"- `output_ids` {tuple(rr.output_ids.shape)} = [B*G, P+R]，prompt 和 completion 拼在一起\n"
     f"- `completion_ids` {tuple(rr.completion_ids.shape)} = `output_ids[:, P:]`\n"
     f"- `per_token_logps` {tuple(rr.per_token_logps.shape)}，走的是 `compute_per_token_logps`（`logits_to_keep=R+1`）\n"
     f"- `completions` list[str]，len={len(rr.completions)}\n"
     f"- `prompt_lens` {tuple(rr.prompt_lens.shape)}，每行都是常量 {rr.prompt_lens[0].item()}（左 padding 后大家一样长）\n"
     f"- `completion_mask` {tuple(rr.completion_mask.shape)}，全 1 = {bool(rr.completion_mask.all())}（没有右 padding）\n\n"
     f"最后两个字段在 torch 引擎里是常量/全 1，看着多余 —— 它们是给 sglang 引擎准备的。")
