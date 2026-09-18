# ---
# title: 没分到 token 的 expert，backward 之后还有梯度吗
# timeout: 90
# tasks:
#   - "把 model.train() 改成 model.eval()：还会有『没分到 token 却拿到 .grad』的 expert 吗？（elif self.training 的条件）"
#   - "把 num_experts 从 32 改成 4：21 个 token 够不够喂满 4 个 expert？『闲置 expert 数』变成多少？"
# ---
import torch
import torch.nn.functional as F
from learnkit import *

# moe_intermediate_size 调小只是为了 32 个 expert 也能秒建，不影响这里要看的现象
model = build_model(num_hidden_layers=1, use_moe=True, num_experts=32, moe_intermediate_size=128)
model.train()                                      # 👉 y[0,0] += 0*sum(...) 只在 training 下执行

moe = model.model.layers[0].mlp
box = {}
h = moe.gate.register_forward_hook(lambda m, i, o: box.__setitem__("g", o.detach()))
ids = torch.randint(0, model.config.vocab_size, (3, 7))            # 21 个 token < 32 个 expert
out = model(ids, labels=ids)
h.remove()
(out.loss + out.aux_loss).backward()

top1 = F.softmax(box["g"], -1).topk(1, dim=-1).indices.flatten()
load = torch.bincount(top1, minlength=model.config.num_experts)
idle_g = [moe.experts[i].gate_proj.weight.grad for i in (load == 0).nonzero().flatten().tolist()]
mx = max((g.abs().max().item() for g in idle_g if g is not None), default=None)

table([["expert 总数", model.config.num_experts],
       ["这一步分到了 token 的 expert 数", int((load > 0).sum())],
       ["闲置 expert 数（mask.any() 为 False）", len(idle_g)],
       ["backward 后 .grad 不是 None 的 expert 数", sum(e.gate_proj.weight.grad is not None for e in moe.experts)],
       ["其中闲置 expert 有 .grad 的个数", sum(g is not None for g in idle_g)],
       ["闲置 expert 的梯度绝对值最大", f"{mx:.1e}" if mx is not None else "—"]],
      headers=["检查项", "结果"], title="21 个 token 喂不满 32 个 expert：谁进了计算图，谁没进")
