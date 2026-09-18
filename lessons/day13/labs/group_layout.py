# ---
# title: "[B*G] 的行序：同一个 prompt 的 G 个回答连续占 G 行"
# timeout: 90
# sources:
#   - trainer/rollout_engine.py
# tasks:
#   - "把 G 从 4 改成 1：表格剩几行？k、i、j 三列变成什么关系？（这就是 PPO 复用同一个引擎时的样子）"
#   - "在源码标签页把 rollout 里两处 repeat_interleave(num_generations, dim=0) 都改成 repeat(num_generations, 1)：k=1 这一行匹配到的 prompt 变成了谁？"
# ---
import torch
from learnkit import *
from trainer.rollout_engine import create_rollout_engine

tok, model = get_tokenizer(), load_model("full_sft")
B, G, R = 3, 4, 16  # 3 个 prompt × 每个采 4 条 = 12 条轨迹，每条最多 16 个 token

prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in ["你是谁？", "太阳为什么是热的？", "推荐一本科普书。"]]
# 和 train_grpo.py:74 一样：左 padding、不加特殊 token（chat template 里已经有了）
enc = tok(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
          padding_side="left", add_special_tokens=False)
P = enc["input_ids"].size(1)

engine = create_rollout_engine("torch", policy_model=model, tokenizer=tok, device="cpu")
torch.manual_seed(0)
rr = engine.rollout(prompt_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    num_generations=G, max_new_tokens=R, temperature=0.9)  # 👉 内部是 repeat_interleave(G, dim=0)

# 拿 outputs[k, :P] 去和 B 个原始 prompt 逐行比对，反查第 k 行到底在回答谁
hit = [(rr.output_ids[k, :P] == enc["input_ids"]).all(dim=1).nonzero()[0, 0].item() for k in range(B * G)]
table([[k, k // G, k % G, hit[k], repr(rr.completions[k][:20])] for k in range(B * G)],
      headers=["下标 k", "源码算的 i=k//G", "j=k%G", "实测匹配到的 prompt 行", "completion 前 20 字"],
      title=f"B={B}, G={G} → {B * G} 行；第 4、5 列自己对照着看")
note(f"`output_ids` 是 {tuple(rr.output_ids.shape)} = [B*G, P+R]，P={P}。\n\n"
     f"第 3 列（源码 `calculate_rewards` 用的 `i = k // G`）和第 4 列（实测这一行带的 prompt）完全一致，"
     f"这就是 `repeat_interleave` 的排布；`repeat` 会给出 `[p0,p1,p2,p0,p1,p2,…]`，两列立刻对不上。")
