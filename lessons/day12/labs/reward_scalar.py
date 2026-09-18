# ---
# title: calculate_rewards 的四个分量
# timeout: 90
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把第 4 条 response 的 </think> 删掉：总分变了多少？消失的是哪两项？"
#   - "把最后一行的 FakeRewardModel(3.0) 改成 FakeRewardModel(10.0)：总分跟着涨到 10 以上了吗？看 trainer_utils.py:177"
# ---
from types import SimpleNamespace
from learnkit import *
import trainer.train_ppo as tp

tp.args = SimpleNamespace(device="cpu")     # calculate_rewards 只用到全局 args.device


class FakeRewardModel:      # 替身：本地没有 InternLM2-1.8B-Reward；截断规则与 trainer_utils.py:177 一致
    def __init__(self, s): self.s = s
    def get_score(self, messages, response): return max(min(self.s, 3.0), -3.0)


tokenizer = get_tokenizer()
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "什么是光合作用？"}],
                                       tokenize=False, open_thinking=True, add_generation_prompt=True)
responses = ["太短",
             "光合作用是绿色植物利用光能，把二氧化碳和水转化成有机物并释放氧气的过程。",
             "abc def ghi abc def ghi abc def ghi abc def ghi",           # 3-gram 大量重复
             "先想一想：植物靠叶绿体吸收光能，把水和二氧化碳变成糖。</think>光合作用是植物把光能转成化学能的过程。",
             "嗯</think>好的</think>再说一遍"]                              # 两个 </think> + thinking 过短
rows = []
for r in responses:                         # 逐项复刻 train_ppo.py:61-67，总分仍用源码函数算
    answer = r.split('</think>', 1)[1].strip() if '</think>' in r else r
    rows.append([r[:20] + ("…" if len(r) > 20 else ""), len(r.strip()),
                 0.5 if 20 <= len(r.strip()) <= 800 else -0.5,
                 (1.0 if 20 <= len(r.split('</think>')[0].strip()) <= 300 else -0.5) if '</think>' in r else "—",
                 (0.25 if r.count('</think>') == 1 else -0.25) if '</think>' in r else "—",
                 -round(tp.rep_penalty(answer), 4),
                 round(tp.calculate_rewards([prompt], [r], FakeRewardModel(0.0))[0].item(), 4)])
table(rows, headers=["response", "字符数", "长度 ±0.5", "think 长度 +1/-0.5", "</think> 计数 ±0.25",
                     "-rep_penalty", "总 reward（RM=0）"], title="calculate_rewards 逐项拆解 (train_ppo.py:61-67)")
print(f"rewards 的 shape = {tuple(tp.calculate_rewards([prompt] * 3, responses[:3], FakeRewardModel(0.0)).shape)}"
      f"  ← 一条序列一个标量；把 RM 换成 +3 时第 4 条的总分 = "
      f"{tp.calculate_rewards([prompt], [responses[3]], FakeRewardModel(3.0))[0].item():.4f}")
