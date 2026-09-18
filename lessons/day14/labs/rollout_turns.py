# ---
# title: 真实多轮 rollout：一条轨迹被切成几轮
# timeout: 120
# sources:
#   - trainer/train_agent.py
# tasks:
#   - "把 MAX_TURNS 改成 1：unfinished 变成什么？轨迹还会不会有第 2 轮？（看 train_agent.py:133 那一行写在 break 之后）"
#   - "把 thinking_ratio 改成 1.0：generation prompt 末尾从空 think 壳变成开着的 '<think>\\n'，模型这一轮的输出多了什么"
# ---
import json, random, torch
from learnkit import *
from trainer.train_agent import rollout_single
from trainer.rollout_engine import TorchRolloutEngine

MAX_TURNS, MAX_NEW = 3, 96                      # 👉 这两个值决定这条轨迹能滚多久
tok, dev = get_tokenizer(), best_device()
engine = TorchRolloutEngine(load_model("full_sft", device=dev), tok, device=dev)

with open("dataset/agent_rl_math.jsonl", encoding="utf-8") as f:
    sample = json.loads(next(iter(f)))           # AgentRLDataset.__getitem__ 读的就是这一行
messages = [dict(m) for m in sample["conversations"]][:-1]   # parse_conversations 丢掉最后一条 assistant
tools = next((json.loads(m["tools"]) for m in messages if m.get("tools")), None)
question = messages[-1]["content"]                           # 先存下来：rollout_single 会往 messages 里 append

torch.manual_seed(0); random.seed(0)
final, ctx, p_ids, r_ids, r_mask, r_lp, turns, unfinished = rollout_single(
    engine, tok, messages, tools, max_turns=MAX_TURNS, max_new_tokens=MAX_NEW,
    thinking_ratio=0.0, device=dev)

table([[f"turn {i}", repr(t)[:78], len(tok(t, add_special_tokens=False)["input_ids"])]
       for i, t in enumerate(turns)],
      headers=["轮次", "这一轮模型自己写的 new_text", "文本 token 数"],
      title=f"user={question!r} gt={sample['gt']} → 共 {len(turns)} 轮, unfinished={unfinished}")

print(f"prompt_ids  = {len(p_ids):4d} tok  ← 只在第 1 轮 tokenize 一次，之后原样复用")
print(f"response_ids= {len(r_ids):4d} tok  = 模型生成的 + 工具结果 + 模板补的下一轮 generation prompt")
print(f"其中 response_mask=1（参与 loss）的有 {sum(r_mask)} 个，其余 {len(r_mask) - sum(r_mask)} 个是环境喂进来的")
print(f"要是还有下一轮，送进 generate 的就是 prompt_ids + response_ids = {len(p_ids) + len(r_ids)} 个 token 的全量重放")
