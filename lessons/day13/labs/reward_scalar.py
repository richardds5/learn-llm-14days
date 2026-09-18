# ---
# title: 一条轨迹一个标量：跑一遍仓库里真实的 calculate_rewards
# timeout: 60
# sources:
#   - trainer/train_grpo.py
# tasks:
#   - "把 k=4 那条回答换成不重复的一句（比如「光合作用发生在叶绿体里，需要光、二氧化碳和水共同参与。」）：它的 reward 涨了多少？涨的是哪一项？"
#   - "把 k=2 那条回答里的 </think> 删掉：它的 reward 掉了多少？（提示：源码里 </think> 分支一次能加 1.25 分）"
# ---
import torch
from types import SimpleNamespace
from learnkit import *
import trainer.train_grpo as grpo

B, G = 2, 3
grpo.args = SimpleNamespace(num_generations=G, device="cpu")  # 源码里 calculate_rewards 读的是模块级 args


class ZeroRewardModel:  # 本地没有 internlm2-1_8b-reward，用恒 0 的替身，只看规则分量
    def get_score(self, messages, answer): return 0.0


tok = get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in ["你是谁？", "介绍一下光合作用。"]]  # 👉 只有 B=2 个 prompt
responses = [  # 👉 但有 B*G=6 条回答，下标 k = i*G + j
    "我是 MiniMind，一个参数量很小的中文语言模型。",
    "我。",
    "<think>\n我需要先想清楚我到底是谁，再组织一下语言回答用户。\n</think>\n我是 MiniMind。",
    "光合作用是绿色植物把光能转化成化学能储存起来的过程。",
    "光合作用很重要，光合作用很重要，光合作用很重要，光合作用很重要。",
    "植物靠光合作用把二氧化碳和水变成葡萄糖，并放出氧气。",
]
rewards = grpo.calculate_rewards(prompts, responses, ZeroRewardModel())  # [B*G]

show(rewards=rewards)
table([[k, k // G, k % G, len(responses[k]), "有" if "</think>" in responses[k] else "—",
        round(grpo.rep_penalty(responses[k]), 3), round(rewards[k].item(), 3)] for k in range(B * G)],
      headers=["下标 k", "prompt i", "gen j", "字符数", "</think>", "rep_penalty", "reward"],
      title="reward 是序列级的：每条轨迹一个标量，没有 token 维")
