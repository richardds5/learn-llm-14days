# ---
# title: ref_model 的三道冻结分别关掉了什么
# timeout: 120
# tasks:
#   - "把 ref.requires_grad_(False) 这一行注释掉（eval() 保留），再跑：哪几行结果翻转了？"
#   - "把 policy 那一行末尾的 .train() 去掉（于是 policy 也是 eval）：policy 的 requires_grad / grad_fn 变了吗？说明 .eval() 管的不是建图。"
# ---
import torch

from learnkit import *

torch.manual_seed(0)
ids = torch.randint(0, 6400, (3, 7))
policy = build_model(num_hidden_layers=2).train()        # 可训练的 policy
ref = build_model(num_hidden_layers=2)                   # 👉 和 train_dpo.py 一样：同一份权重加载两遍
ref.eval(); ref.requires_grad_(False)                    # 👉 两道冻结

ref_logits = ref(ids).logits                             # 注意：故意不套 torch.no_grad()
pol_logits = policy(ids).logits
opt = torch.optim.AdamW(policy.parameters(), lr=4e-8)    # 优化器只拿到 policy 的参数
opt_ids = {id(p) for g in opt.param_groups for p in g['params']}

moe = build_model(num_hidden_layers=2, use_moe=True)     # MoE 下 aux_loss 才是真的
aux_eval = moe.eval()(ids).aux_loss.item()
aux_train = moe.train()(ids).aux_loss.item()

table([["ref 的参数 requires_grad", str({p.requires_grad for p in ref.parameters()})],
       ["ref 在 no_grad 之外前向：logits.requires_grad", str(ref_logits.requires_grad)],
       ["ref 在 no_grad 之外前向：logits.grad_fn", str(ref_logits.grad_fn)],
       ["policy 前向：logits.requires_grad / grad_fn", f"{pol_logits.requires_grad} / {type(pol_logits.grad_fn).__name__}"],
       ["optimizer 里有 ref 的参数吗", str(any(id(p) in opt_ids for p in ref.parameters()))],
       ["MoE 模型 .eval() 时的 aux_loss", f"{aux_eval:.6f}"],
       ["MoE 模型 .train() 时的 aux_loss", f"{aux_train:.6f}"]],
      headers=["检查项", "结果"], title="三道冻结里，只有 .eval() 是不可替代的")
