# ---
# title: 同一条样本取 200 次：system 注入与空 think 块的实际频率
# timeout: 90
# sources:
#   - dataset/lm_dataset.py
# tasks:
#   - "把 pre_processing_chat(conv) 改成 pre_processing_chat(conv, add_system_ratio=0.9)：表格第一行的比例变成多少？"
#   - "把 post_processing_chat(prompt) 改成 post_processing_chat(prompt, empty_think_ratio=1.0)：空 think 块还会被删掉吗？"
# ---
import random
from learnkit import *
from dataset.lm_dataset import SFTDataset, pre_processing_chat, post_processing_chat

tok = get_tokenizer()
ds = SFTDataset("dataset/lora_identity.jsonl", tok, max_length=128)
conv = ds.samples[3]["conversations"]  # 一条普通的 user/assistant 两轮对话，没有 tools
random.seed(0)  # 固定随机种子，跑出来的数才可复现

N = 200  # 👉 改这里：样本数越大，比例越接近源码里的 0.2
n_sys = n_think = 0
for _ in range(N):
    processed = pre_processing_chat(conv)                     # 概率性在最前面插一条 system
    n_sys += processed[0]["role"] == "system"
    prompt = post_processing_chat(ds.create_chat_prompt(processed))  # 概率性删掉空 <think> 块
    n_think += "<think>\n\n</think>\n\n" in prompt

table([["最前面被插入 system 消息", n_sys, f"{n_sys / N:.0%}", "add_system_ratio=0.2"],
       ["保留了空 <think>\\n\\n</think>\\n\\n", n_think, f"{n_think / N:.0%}", "empty_think_ratio=0.2"]],
      headers=["事件", f"{N} 次里出现", "实测比例", "源码里的默认参数"],
      title="__getitem__ 每次调用都重新掷一次骰子：同一个 index 拿到的文本并不固定")
