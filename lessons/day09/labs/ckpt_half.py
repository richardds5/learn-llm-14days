# ---
# title: lm_checkpoint 存下来的两个文件，和读回来的权重
# timeout: 90
# sources:
#   - trainer/trainer_utils.py
# tasks:
#   - "把 lm_checkpoint(...) 里的 scaler=scaler 改成 scaler=None：resume 文件里还有 'scaler' 这个 key 吗？（最后一行会因此 KeyError —— **kwargs 只收非 None 的值）"
#   - "把 trainer_utils.py:73 的 v.half().cpu() 改成 v.cpu()：表格里的 dtype 和两行误差各变成什么？"
# ---
import os, tempfile
import torch
from learnkit import *
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import lm_checkpoint

torch.manual_seed(0)
lm_config = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
model = MiniMindForCausalLM(lm_config)                      # 随机初始化 = 货真价实的 fp32 权重
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
ids = torch.randint(0, 6400, (3, 7))
res = model(ids, labels=ids.clone()); (res.loss + res.aux_loss).backward(); optimizer.step()
save_dir = tempfile.mkdtemp()

scaler = torch.cuda.amp.GradScaler(enabled=False)
lm_checkpoint(lm_config, weight="demo", model=model, optimizer=optimizer, scaler=scaler,
              epoch=0, step=5, save_dir=save_dir)           # 给了 model= → 保存模式
ckp = lm_checkpoint(lm_config, weight="demo", save_dir=save_dir)   # 不给 model= → 加载模式

before = model.state_dict()
rel = lambda k: ((ckp["model"][k].float() - before[k]).norm() / before[k].norm()).item()
worst = max(before, key=rel)
opt_state = next(iter(ckp["optimizer"]["state"].values()))
table([["save_dir 里的文件", str(sorted(os.listdir(save_dir)))],
       ["resume 文件里的 key", str(list(ckp.keys()))],
       ["ckp['model'] 的 dtype", str(ckp["model"][worst].dtype)],
       ["‖Δw‖/‖w‖ 最大的张量", f"{worst}  →  {rel(worst):.2e}"],
       ["逐元素相对误差（|w|>0.01 的位置）",
        f"{max((((ckp['model'][k].float() - before[k]).abs() / before[k].abs())[before[k].abs() > 0.01]).max().item() for k in before):.2e}"
        f"   ← fp16 尾数 10 位，上界 2⁻¹¹ = {2 ** -11:.2e}"],
       ["optimizer 的动量 dtype", f"exp_avg={opt_state['exp_avg'].dtype}, exp_avg_sq={opt_state['exp_avg_sq'].dtype}"],
       ["ckp['scaler'] / ['world_size']", f"{ckp['scaler']} / {ckp['world_size']}"]],
      headers=["检查项", "结果"],
      title=f"续训文件里的模型权重存成了 {ckp['model'][worst].dtype}，而优化器状态是完整 fp32")
