# ---
# title: CriticModel.forward 逐行：从 [B,T,C] 到 [B,T]
# timeout: 120
# sources:
#   - trainer/train_ppo.py
# tasks:
#   - "把 hidden_size 从 128 改成 256：trace 里 hidden_states 的最后一维变成什么？values 的 shape 变了吗？"
#   - "把源码第 40 行的 nn.Linear(params.hidden_size, 1) 改成 nn.Linear(params.hidden_size, 2)：第 47 行那两步的 shape 各变成什么？squeeze(-1) 还起作用吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import MiniMindConfig
import trainer.train_ppo as tp

cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)   # 👉 小 config，只为了看 shape
torch.manual_seed(42)
critic = tp.CriticModel(cfg).eval()

B, T = 3, 7
input_ids = torch.randint(0, cfg.vocab_size, (B, T))
attention_mask = torch.ones(B, T, dtype=torch.long)
with trace(fns=[tp.CriticModel.forward], dims=dict(B=B, T=T),
           title="CriticModel.forward：hidden_states [B,T,C] → value_head → [B,T,1] → squeeze(-1) → [B,T]",
           focus=(44, 48), expand="L47", tree=False):
    values = critic(input_ids=input_ids, attention_mask=attention_mask)

# 第 45 行的 self.model.norm 是「第二次」归一化（MiniMindModel.forward 末尾已经做过一次）。
# 随机初始化时 norm.weight 全 1 看不出差别，换成训练好的权重就能看出它不是恒等变换。
real = tp.CriticModel(MiniMindConfig())
real.load_state_dict(torch.load(weight_path("full_sft"), map_location="cpu"), strict=False)
with torch.no_grad():
    h = real.model(input_ids=input_ids, attention_mask=attention_mask)[0]    # 已经 norm 过一次
    v_once, v_twice = real.value_head(h).squeeze(-1), real.value_head(real.model.norm(h)).squeeze(-1)
print(f"full_sft 的 model.norm.weight.mean() = {real.model.norm.weight.mean().item():.4f}；"
      f"第二次 norm 让 values 变了 {((v_once - v_twice).abs().max() / v_twice.abs().max() * 100).item():.2f}%")
