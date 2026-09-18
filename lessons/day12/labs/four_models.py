# ---
# title: 四个模型的来源与梯度归属
# timeout: 120
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 strict=False 改成 strict=True：在哪一行报什么错、点名了哪两个 key？"
#   - "去掉 ref_model 后面的 .requires_grad_(False)：表里「其中要梯度」哪一行变了、变成多少？"
# ---
import torch
from learnkit import *
from model.model_minimind import MiniMindConfig
import trainer.train_ppo as tp

lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
state_dict = torch.load(weight_path("full_sft"), map_location="cpu")

actor_model = load_model("full_sft")                                # policy，要梯度
ref_model = load_model("full_sft").requires_grad_(False)            # 👉 冻结的参考策略
critic_model = tp.CriticModel(lm_config)                            # 仓库源码里的 Critic
info = critic_model.load_state_dict(state_dict, strict=False)       # 👈 strict=False

n_all = lambda m: sum(p.numel() for p in m.parameters())
n_grad = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
table([["actor", "init_model(lm_config, 'full_sft')", n_all(actor_model), n_grad(actor_model), "AdamW lr=3e-7"],
       ["ref", "同一份 full_sft + .eval().requires_grad_(False)", n_all(ref_model), n_grad(ref_model), "无"],
       ["critic", "CriticModel(lm_config) + load_state_dict(strict=False)", n_all(critic_model),
        n_grad(critic_model), "AdamW lr=5e-7"],
       ["reward", "LMForRewardModel(InternLM2-1.8B-Reward, fp16)", "≈1.8B（本地没有）", 0, "无"]],
      headers=["模型", "从哪来", "参数量", "其中要梯度", "优化器"], title="PPO 的四个模型 (train_ppo.py:383-392)")
print("critic.load_state_dict(state_dict, strict=False) 的 missing_keys =", info.missing_keys)
print("unexpected_keys =", info.unexpected_keys)
print(f"value_head 共 {sum(p.numel() for p in critic_model.value_head.parameters())} 个参数，权重是随机初始化的："
      f"std={critic_model.value_head.weight.std().item():.4f}, bias={critic_model.value_head.bias.item():.4f}")
