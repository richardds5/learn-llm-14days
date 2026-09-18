# ---
# title: 四种配置下 gate.weight.grad 的量级
# timeout: 90
# tasks:
#   - "把 torch.manual_seed(0) 改成别的种子多跑几次：第 1 行的量级稳定停在 1e-9 附近吗？"
#   - "加一组 dict(num_experts_per_tok=2, norm_topk_prob=False, router_aux_loss_coef=0.0)：k≥2 时不归一化，梯度还在吗？"
# ---
import torch
from learnkit import *
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

CASES = [                                                     # 👉 加一组自己的组合
    dict(num_experts_per_tok=1, norm_topk_prob=True, router_aux_loss_coef=0.0),
    dict(num_experts_per_tok=1, norm_topk_prob=True, router_aux_loss_coef=5e-4),
    dict(num_experts_per_tok=1, norm_topk_prob=False, router_aux_loss_coef=0.0),
    dict(num_experts_per_tok=2, norm_topk_prob=True, router_aux_loss_coef=0.0),
]
rows = []
for case in CASES:
    torch.manual_seed(0)                                      # 四组共用同一套初始权重，只有配置不同
    cfg = MiniMindConfig(num_hidden_layers=2, use_moe=True, moe_intermediate_size=256, **case)
    model = MiniMindForCausalLM(cfg)
    model.train()                                             # aux_loss 只在 train() 下才算
    ids = torch.randint(0, cfg.vocab_size, (3, 7))
    out = model(ids, labels=ids)
    (out.loss + out.aux_loss).backward()                      # trainer/train_pretrain.py:37 的写法
    g = model.model.layers[0].mlp.gate.weight.grad
    rows.append([case["num_experts_per_tok"], case["norm_topk_prob"], case["router_aux_loss_coef"],
                 f"{out.aux_loss.item():.2e}", f"{g.abs().max().item():.2e}"])

table(rows, headers=["k", "norm_topk_prob", "coef", "aux_loss", "gate.weight.grad 绝对值最大"],
      title="同一套初始权重、同一批数据，反传到 router 的梯度量级（gate.weight 是 [E, C]）")
