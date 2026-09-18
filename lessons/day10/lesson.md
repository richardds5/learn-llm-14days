---
day: 10
format: points
title: "LoRA 与知识蒸馏：手写实现，不套 peft"
subtitle: "给方阵 Linear 挂一条 768→16→768 的旁路；再用 teacher 的软分布当标签"
minutes: 90
mainline: "跟着 q_proj 旁边那条旁路走：A/B 怎么让 ΔW 从 0 开始、apply_lora 怎么把它挂上去、只训练它、存成 195 KB 再合并回 W；最后换一条路，用 teacher 的 [M, V] 软分布喂 student"
files:
  - model/model_lora.py#L1-L65
  - trainer/train_lora.py#L128-L152
  - trainer/train_distillation.py#L25-L93
goals:
  - 能报出 LoRA 旁路上每一步的 shape，并用 r·(in+out) 算出任意 rank 下的参数占比
  - 能说清 apply_lora 的筛选条件为什么只命中 q_proj/o_proj，以及把这个条件去掉会撞上什么
  - 能解释 forward_with_lora 的两个默认参数在防什么，以及去掉之后为什么不报错却算错
  - 能说出 save_lora / load_lora / merge_lora 各自存了什么、靠什么 key 对上、误差从哪来
  - 能写出 distillation_loss 的公式，说清 kl_div 的参数顺序、batchmean 的分母、T² 在补偿什么
---

Day 9 讲的是全参 SFT：64M 个参数全都进 optimizer，存一次盘几百 MB。今天两个脚本问的是同一件事的两个方向——**能不能少动一点、少存一点**。

**LoRA**（`model/model_lora.py` + `trainer/train_lora.py`）：基座一个参数都不动，只在少数几个 Linear 旁边挂一条低秩旁路，训练时只更新它——实测 0.611% 的参数、存盘 779 KB，还能像插卡一样换着用。**知识蒸馏**（`trainer/train_distillation.py`）：换一个更小的 student，拿训练好的 teacher 在每个位置上的完整概率分布当标签，而不只是那一个正确 token。两者都复用 Day 8 的 `SFTDataset` 和 Day 9 的训练循环骨架，也都是手写实现、不套 peft。

四章的分工：第一章拆开 `LoRA` 这个只有十来行的模块；第二章看 `apply_lora` 怎么把它挂上去，今天三个坑全在这一章；第三章跟着 `train_lora.py` 走一遍冻结、存盘、合并、热插拔；第四章换到蒸馏，盯住 `distillation_loss` 那四行和它的输入 `[M, V]`。

{{flow: hidden_states [B,T,768] | q_proj 原始 Linear | *lora.A → [B,T,16]* | *lora.B → [B,T,768]* | 两路相加 | q_proj 输出 [B,T,768]}}

# LoRA 模块：一条 768 → 16 → 768 的旁路

`LoRA` 类只有两个 `nn.Linear` 和两行初始化，但这两件事分别回答了「为什么省参数」和「为什么挂上去不会把模型搅坏」。这一章把它们各看一遍，后面两章讲的都是「怎么把这个模块塞进已有模型」。

## A 与 B 的两条边：rank 卡在中间 {#lora-shapes}

`LoRA` 就是两个不带 bias 的 `nn.Linear` 串起来：`A` 把 `in_features` 压到 `rank`，`B` 再把 `rank` 抬回 `out_features`。数学上它等价于给原权重矩阵 $W$ 加一个增量 $\Delta W = B A$，其中 $B \in \mathbb{R}^{out \times r}$、$A \in \mathbb{R}^{r \times in}$，乘出来正好 `[out, in]`——和 `module.weight` 同形。

{{source:model/model_lora.py#L6-L18}}

省的是参数量。`q_proj.weight` 是 `768 × 768 = 589,824` 个数，而一条 rank=16 的旁路是 `r·(in + out) = 16 × (768 + 768) = 24,576`，只有 1/24。注意这个公式对 `in ≠ out` 也成立，`LoRA.__init__` 根本没要求两者相等——后面会看到，是 `apply_lora` 自己加的条件。

秩这件事在数值上也是真的：`B @ A` 的秩最多是 `r`，因为中间被 16 维卡了一道。前向时这道卡口直接看得见——`x` 进来是 `[B, T, 768]`，过完 `A` 变成 `[B, T, 16]`，过完 `B` 回到 `[B, T, 768]`。整条旁路的输入输出和原 Linear 完全一样，所以才能直接相加。

一个和 peft 的差异值得先记下：这里的 `forward` 就是 `self.B(self.A(x))`，**没有** `scaling = alpha / rank` 这个缩放系数。想把 peft 训出来的 LoRA 搬过来（或反过来），ΔW 会差一个因子。

下面的实验用 `expand=True` 把 `self.B(self.A(x))` 这一行流拆开，中间那个 `[B, T, r=16]` 就是低秩瓶颈本身。

{{lab:lora_shapes}}

{{quiz:q1}}

> [!KEY]
> LoRA = `A: in→r` + `B: r→out`，参数量 `r·(in+out)`；rank=16 时一条旁路只有 24,576 个数，是 `q_proj.weight` 的 1/24。

> [!MORE] rank 开到多大就不划算了
> 令 `r·(in+out) = in·out`，在 `in = out = 768` 时解出 `r = 384`。也就是说 rank 超过 384，旁路比原矩阵还大。实测全模型（8 层）的 LoRA 占比：rank=4 → 0.154%，rank=16 → 0.611%，rank=64 → 2.402%，严格线性。

## B 的全零初始化：挂上去的那一刻等于没挂 {#lora-zero-init}

`A` 用 `normal_(mean=0.0, std=0.02)` 高斯初始化，`B` 用 `zero_()` 全零。矩阵乘法里只要有一个因子全零，结果就是全零，所以刚 `apply_lora` 完的那一刻，不管 `A` 抽到什么随机数，`B(A(x))` 恒等于 0。

{{source:model/model_lora.py#L12-L15}}

这不是「差别很小」，是**逐比特相同**：实验里 `torch.equal(out_before, out_after)` 直接返回 `True`，差值张量的 min/max/mean/std 全是精确的 0。对一个已经训练好的基座来说这很重要——挂上 LoRA 结构本身不应该动到任何能力，能力的变化只能来自训练。随着梯度更新，`B` 才慢慢离开 0。实验最后手动把一个 `B` 改成非零，logits 立刻出现 0.2 量级的差异。

那为什么是「A 随机 + B 全零」，不是反过来？因为要保证 ΔW 起步为 0，`A`、`B` 至少有一个是 0；而如果**两个都是 0**，这一对就死在原地了：`dL/dB` 正比于 `A(x)`，`dL/dA` 里带着 `B`，两边同时为零意味着两边的梯度也同时为零（实测两个 `.grad` 的绝对值最大都是 0.0）。所以必须有一个随机、一个全零。至于哪个随机，`B = 0` 更顺手——它保证输出为 0，而 `A = 0` 只是让中间量为 0，效果相同但第一步的梯度路径不一样。

{{lab:lora_zero_init}}

{{quiz:q2,q3}}

> [!KEY]
> `A` 高斯、`B` 全零 ⇒ ΔW ≡ 0 ⇒ `apply_lora` 前后 logits 逐比特相同；两个都置零则梯度也全是 0，这一对永远学不动。

# apply_lora：一道筛子、一次 monkey-patch、一对默认参数

`apply_lora` 只有 12 行，却同时做了三件互不相干的事：按形状挑模块、把 `lora` 挂成子模块、把 `module.forward` 整个换掉。今天最容易踩的三个坑刚好一件一个，这一章一节一个。

## in_features == out_features 这道筛子 {#lora-square}

{{source:model/model_lora.py#L21-L24}}

判断条件是 `isinstance(module, nn.Linear) and module.in_features == module.out_features`——**只有方阵 Linear 会被注入**。算一下 MiniMind 默认配置（`hidden_size=768`、`num_attention_heads=8` → `head_dim=96`、`num_key_value_heads=4`）下每个 Linear 的形状：`q_proj` 768→768 ✅，`o_proj` 768→768 ✅，`k_proj`/`v_proj` 768→`KV·D`=384 ❌（GQA 把 KV 头砍掉一半），`gate_proj`/`up_proj` 768→2432 ❌，`down_proj` 2432→768 ❌，`lm_head` 768→6400 ❌。

所以每层只有 `q_proj` 和 `o_proj` 中签，8 层共 16 个模块、393,216 个参数、占全模型 0.611%。请注意这**不是**「只想微调 attention」这种设计意图——判断条件纯粹是形状匹配，凑巧在这份配置里只有 q/o_proj 满足。换个配置结论就变：把 `num_attention_heads` 改成 4，`head_dim` 变成 192，于是 `k_proj`/`v_proj` 也成了 768→768，一样会被注入。

那把这个条件删掉、让所有 Linear 都挂上 LoRA，会怎样？先下注再跑。

{{predict:p1}}

{{lab:lora_square}}

实验最后那段复现给出了答案：`RecursionError`。`named_modules()` 是个**懒生成器**，一边遍历一边产出；你刚 `setattr` 上去的 `lora.A` 本身就是一个 `nn.Linear`，会被同一轮循环遍历到、再被注入一个 LoRA，无限套娃。换成 `nn.Identity()` 就能跑完——问题不在条件本身，而在「挂上去的东西又满足条件」。

{{quiz:q4}}

> [!KEY]
> 筛子是纯形状条件，默认配置下只命中每层的 `q_proj`/`o_proj`（8 层共 16 个、0.611%）；去掉条件不是「全都注入」，而是被懒生成器拖进 `RecursionError`。

## 被换掉的 forward 与被自动收编的子模块 {#lora-patch}

`apply_lora` 没有新建一个 `LoRALinear(nn.Module)` 把原 Linear 包进去，而是直接改写实例上的 `forward`。这个选择换来两件好事，也留下两处代价。

{{source:model/model_lora.py#L24-L32}}

好处一：原模型的类结构和 `state_dict` 的 key 一个字都没变，`q_proj.weight` 还是 `q_proj.weight`，和没加 LoRA 之前训出的 `full_sft` 权重完全兼容。好处二：`setattr(module, "lora", lora)` 里 `lora` 是个 `nn.Module`，而 `nn.Module.__setattr__` 对这种赋值有特殊处理——会自动注册进 `_modules`。于是不写任何额外代码，`named_modules()` / `named_parameters()` / `state_dict()` 就能发现它们：实验里 2 层模型的 state_dict key 从 25 变成 33，多出的正是 8 个 `.lora.` key（2 层 × 2 模块 × A/B）。`save_lora`/`load_lora`/`merge_lora` 全靠这一点。

代价一：`module.forward` 现在是**实例属性**而不是类方法，`torch.compile` 的图捕获接不住，所以 `train_lora.py:164-166` 干脆在 `use_compile=1` 时打印一句提醒、直接把它关掉。代价二更隐蔽：`forward_with_lora` 里 `layer1(x)` 是**直接函数调用**，绕过了 `nn.Module.__call__`，不触发 forward hook；而 `layer2(x)` 走的是 `lora.__call__`，会触发。这门课的 `trace` 正是靠 forward hook 画模块树的——所以实验的模块树里，`q_proj` 节点下面只挂着一个 `lora` 子调用，那次真正的基座矩阵乘法**根本没有出现**。任何基于 hook 的工具（profiler、可解释性工具）都有这个盲点。

{{lab:lora_patch}}

{{quiz:q5,q6}}

> [!KEY]
> monkey-patch 保住了 state_dict 的 key 兼容性，`setattr` 让 `lora` 自动成为子模块被收进 state_dict；代价是和 `torch.compile` 不兼容，且 forward hook 看不见基座那一路。

## layer1= 和 layer2= 这两个默认参数 {#lora-closure}

{{source:model/model_lora.py#L26-L32}}

看 `def forward_with_lora(x, layer1=original_forward, layer2=lora):` 这一行——两个外层变量被写成了**默认参数**，而不是在函数体里直接引用。源码里那句注释「显式绑定」就是在说这件事，它不是风格偏好，是必须的。

原因是 Python 闭包捕获的是**变量本身**（外层栈帧里的 cell），不是定义那一刻的值。`for` 循环每进一次 `if` 分支都会重新绑定 `original_forward` 和 `lora`，循环跑完之后这两个 cell 里留下的是**最后一次进入 if 分支时**的值。注意是「最后一次进入 if 分支」，不是「最后一个模块」：`lm_head` 是 768→6400，从来不满足条件，压根没机会碰到这两个变量；`module` 这个变量确实会走到 `lm_head`，但函数体里根本没引用 `module`。所以闭包最终指向的是 `layers.L-1.self_attn.o_proj`。

用默认参数就不一样了：默认值在 **`def` 执行的那一刻**求值并复制一份，每轮循环各自绑定各自的那一对，互不干扰。

那么，把默认参数去掉、改成直接引用外层变量，`apply_lora` 之后立刻跑一次前向，会发生什么？

{{predict:p2}}

{{lab:lora_closure}}

实验的答案是：**什么异常都不抛**。因为 MiniMind 里所有被 patch 的方阵 Linear 形状完全相同（768→768），换上「错的」`original_forward`/`lora` 也不会触发任何 shape 检查，只是权重张冠李戴——所有层的 `q_proj`/`o_proj` 都偷偷用上了 `layers.1.self_attn.o_proj` 的基座和旁路。`torch.allclose` 是 `False`，max diff 约 4（2 层）/ 3.6（8 层）。实验最后两行直接把闭包 cell 掏出来确认了它指向谁。

> [!KEY]
> 去掉默认参数**不会报错**，而是让所有被 patch 的层共用最后一个方阵 Linear 的权重和 LoRA，logits 静默算错（max diff ≈ 4）——这是 Python 闭包晚绑定，不是 `AttributeError`。

# 训练、存盘、合并与热插拔

模型改造完了，接下来是 `train_lora.py` 的四件事：冻结掉谁、optimizer 收谁、存盘存什么、拿到权重之后怎么用。这一章每一节都有一个可以直接看数字的证据。

## 冻结之后真正在动的那 0.6% {#lora-freeze}

和 Day 9 的 `train_full_sft.py` 相比，`train_lora.py` 在「谁能被训练」上多了两步：先 `apply_lora(model)`，再按参数名里有没有 `'lora'` 子串决定 `requires_grad`——有就设 `True` 并收进 `lora_params`，没有就设 `False`。

{{source:trainer/train_lora.py#L139-L152}}

这个 `lora_params` 列表之后被用在两处：`optim.AdamW(lora_params, lr=...)` 和 `clip_grad_norm_(lora_params, args.grad_clip)`。注意这两处收窄**不是**为了省显存（原因见折叠块），而是让「谁在训练」只有一处定义、并且万一哪天有人漏设 `requires_grad=False`，optimizer 也伤不到基座。对比 `trainer/train_distillation.py:99` 的 `clip_grad_norm_(model.parameters(), ...)`：那边整个 student 都在训练，没有这层筛选。

数据用 `lora_identity.jsonl`（91 条自我认知对话）或 `lora_medical.jsonl`，格式和 Day 8 的 SFT 多轮对话完全一样，只是量小得多——LoRA 本来就是为「少量数据快速适配一个垂域」设计的。实验跑 25 步，最后直接量一量两个张量：基座 `q_proj.weight` 的变化量是**精确的 0.0**，旁路 `lora.B` 是 2.5e-3。把冻结那一支改成全参微调，同样 25 步 loss 会从 2.2 掉到 0.4（LoRA 这 25 步还在 1.5~1.7 附近晃），代价是 `q_proj.weight` 也跟着动了 1.9e-3。

{{lab:lora_freeze}}

{{quiz:q7}}

> [!KEY]
> 按名字里的 `'lora'` 子串分流：可训练参数 393,216 / 64,305,408 = 0.611%，optimizer 和 clip 都只收这一小撮；跑完 25 步基座权重的变化量是精确的 0。

> [!MORE] 传整个 model.parameters() 真的会多占显存吗
> 不会。AdamW 的 `_init_group` 里写着 `if p.grad is None: continue`——只有第一次真的拿到梯度的参数才会被分配 `exp_avg` / `exp_avg_sq`。实测把整个 `model.parameters()` 交给 AdamW 再跑一步，2 层模型的 32 个参数张量里只有 8 个进了 `optimizer.state`，状态元素总数 196,616 ≈ 2 × 98,304，和只传 `lora_params` 完全一样。所以收窄省的不是显存，是 `clip_grad_norm_` 每步少遍历几千万个参数，以及代码意图的清晰。

## save_lora 存下来的 195 KB {#lora-save}

`save_lora` 不是从完整 `state_dict()` 里过滤，而是遍历 `named_modules()` 找 `hasattr(module, 'lora')` 的模块，自己拼出 `{clean_name}.lora.{k}` 这样的 key，并且 `.cpu().half()`。

{{source:model/model_lora.py#L45-L53}}

三个细节。**一、体积**：2 层模型存出来 8 个 key、195 KB；官方的 `lora_identity_768.pth` 是 8 层、32 个 key、393,216 个数、779 KB。对比一个完整 checkpoint 的几百 MB，这就是 LoRA 能当「插卡」分发的原因。改成不 `.half()`、保留 fp32，体积正好翻倍到 387 KB。

**二、`module.` 前缀**：`save_lora` 里的 `name[7:] if name.startswith("module.")` 和 `load_lora` 里的 `k[7:] if k.startswith('module.')` 都在处理 `DistributedDataParallel`——DDP 会把模型包成 `model.module`，所有 key 前面多出 `"module."` 这七个字符。训练脚本存、单卡脚本加载，不统一去掉前缀 key 就对不上。同理 `getattr(model, '_orig_mod', model)` 是在剥 `torch.compile` 的包装。

**三、`load_lora` 不重建结构**：它按 `f'{name}.lora.' in k` 把属于当前模块的 key 挑出来，喂给已经存在的 `module.lora`。所以必须先 `apply_lora` 把空壳搭好——`eval_llm.py` 的 `--lora_weight` 就是 `apply_lora(model)` 再 `load_lora(...)` 这个顺序。反过来，在一个没 `apply_lora` 的模型上调用 `load_lora`，`hasattr` 一个都不命中，**不报错也不做任何事**。

{{lab:lora_save}}

{{quiz:q8}}

> [!KEY]
> `save_lora` 只存 `hasattr(module,'lora')` 那几个模块的 A/B，转 fp16，8 层 779 KB；`load_lora` 只填不建，必须先 `apply_lora`，否则静默无操作。

## merge_lora：把 B @ A 加回 W {#lora-merge}

推理时每个被注入的 Linear 要多跑两次矩阵乘法。既然 ΔW 是个和 `W` 同形的普通矩阵，那就可以在部署前一次性加进去，让模型退回成完全普通的结构。

{{source:model/model_lora.py#L56-L65}}

形状是对得上的：`module.lora.B.weight` 是 `[out, r] = [768, 16]`，`module.lora.A.weight` 是 `[r, in] = [16, 768]`，`B.weight @ A.weight` 得到 `[768, 768]`，正好是 `module.weight` 的 shape，可以直接 `+=`。函数还顺手把所有 `.lora.` 开头的 key 过滤掉，存出来的是一份干净权重——实验里用 `strict=True` 加载进一个**没有任何 lora 子模块**的模型，能一次通过。

但等价性只是「容差内」的，不是逐比特。实验在 logits 标准差约 0.55 的量级下测到最大绝对误差 2.2e-3：`atol=1e-2` 过得了，`atol=1e-3` 就过不了。误差来自两处 `.half()`——`save_lora` 存 A/B 时转了一次 fp16，`merge_lora` 存合并权重时又转了一次。把这两处都改回 fp32，同一个实验的误差掉到 5e-6，`atol=1e-5` 都能过。所以这是存盘精度的锅，不是算法有问题。

{{lab:lora_merge}}

{{quiz:q9}}

> [!KEY]
> `B.weight @ A.weight` 是 `[out, r] @ [r, in] = [out, in]`，正好加回 `module.weight`；合并前后只在 `atol=1e-2` 量级 allclose，误差全部来自两处 `.half()`。

## 同一个基座热插拔两份 LoRA {#lora-swap}

前面都是形状和数字，这一节看行为。仓库带了两份官方 LoRA 权重：`lora_identity_768.pth`（自我认知，让模型稳定回答「我是 MiniMind」）和 `lora_medical_768.pth`（医学问答），两份都配 `full_sft` 基座。

挂载顺序就是 `eval_llm.py` 里那三步：`load_model("full_sft")` → `apply_lora(model)` → `load_lora(model, path)`。因为 `load_lora` 只覆盖已有 `module.lora` 的权重、不碰结构，换第二份 LoRA 时**不需要**重新 `apply_lora`，直接再 `load_lora` 一次就行——这就是「插卡」。

实测很有意思。基座被问「你叫什么」会答「我叫通义千问，是阿里巴巴集团旗下的通义实验室研发的超大规模语言模型」——这是 `full_sft` 语料里混入的黑盒蒸馏数据留下的痕迹。挂上 `lora_identity` 之后同一个问题变成「您好，我叫 MiniMind，是由 Jingyao Gong 开发的人工智能助手」。换成 `lora_medical`，医学问题的回答明显转向病因罗列的专业腔；而这时再问「你叫什么」，模型会把它当成一个医学术语问题来答——说明这条低秩增量改的不只是知识，还有整体的应答风格。把所有 `B` 清零，回答立刻一字不差地退回基座版本。

{{lab:lora_swap}}

{{quiz:q10}}

> [!KEY]
> 一个基座 + `apply_lora` 一次 + 反复 `load_lora`，就能在几百 KB 的粒度上切换行为；`B` 清零即刻退回基座。

# 知识蒸馏：把 teacher 的软分布当标签

README 把蒸馏分成两类：**黑盒蒸馏**只看得到 teacher 说了什么（`full_sft` 语料里混的大模型回答就是这一类，代码上和普通 SFT 没区别）；**白盒蒸馏**还要拟合 teacher 在每个位置上对全部 6400 个候选 token 的概率分布。`train_distillation.py` 实现的是后者，这一章精读它。

## distillation_loss 的四行与 kl_div 的参数顺序 {#distill-kldiv}

整个白盒蒸馏的核心只有这四行：teacher logits 除以温度做 softmax（包在 `no_grad` 里，teacher 不需要梯度），student logits 除以同一个温度做 **log**_softmax，两者丢进 `F.kl_div`，最后乘 `T²`。

{{source:trainer/train_distillation.py#L25-L36}}

$$\mathcal{L}_{KD} = T^2 \cdot \mathrm{KL}\big(\mathrm{softmax}(z_t / T) \,\big\|\, \mathrm{softmax}(z_s / T)\big)$$

最容易写反的是参数顺序。`F.kl_div(input, target)` 里 **`input` 必须已经是 log 概率、`target` 是普通概率**，算的是 $\mathrm{KL}(\text{target} \,\|\, \exp(\text{input}))$。所以源码里第一个位置放的是 `student_log_probs`、第二个是 `teacher_probs`，而不是反过来。两个参数对调不会报错，只会安静地返回 `nan`（因为 `log_softmax` 的输出是负数，被当成概率做了 `log`）——这是一个很难查的 bug。

第二个要盯的是 `reduction='batchmean'`：它按输入张量的 **第 0 维长度** 求平均。`train_epoch` 传进来的已经是展平并过滤后的二维张量 `[M, V]`，所以分母是 `M` = 这个 batch 里参与蒸馏的有效 token 数，不是 batch_size。换成 `'sum'` 数值会乘上 `M` 倍，换成 `'mean'` 则是除以 `M·V`（相当于再除 6400），量级完全对不上。

实验用手造的 `[M=21, V=6400]` 逐行追踪这个函数，你会看到 `kl` 是个标量，返回值是它的 `T²` 倍。

{{lab:distill_kldiv}}

{{quiz:q11,q12}}

> [!KEY]
> `F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')`：第一个参数必须是 log 概率，对调只会得到 `nan`；`batchmean` 的分母是有效 token 数 `M`。

## 温度 T 与那个 T² {#distill-temperature}

温度做的事很直白：`softmax(z / T)` 里 T 越大，logits 之间的差距被压得越小，分布越平、熵越大。蒸馏要的就是这个——teacher 说「这里 80% 是 A、15% 是 B」比「这里就是 A」携带更多信息，而升温把那些被 top-1 压住的次优 token 的相对关系放大出来，这就是所谓的「暗知识」。

{{source:trainer/train_distillation.py#L36}}

代价是 KL 本身会塌下去。同一对 logits，温度从 1 升到 4，两个分布都被拉向均匀分布，它们之间的 KL 自然越来越小。那么真正返回的 `T² · KL` 会跟着塌吗？先下注。

{{predict:p3}}

{{lab:distill_temperature}}

实验表格给出了答案：裸 KL 是 8.63 → 2.27 → 0.57（掉了 15 倍），而 `T² · KL` 是 8.63 → 9.08 → 9.05，**基本不动**。这就是 Hinton 那篇蒸馏论文里乘 `T²` 的用意：升温会把损失以及它对 logits 的梯度一起压小，乘回 `T²` 把损失量级拉回同一个尺度，于是换温度时不必跟着重调学习率、也不必重新平衡 `alpha`。顺带看熵：teacher 第 0 个 token 的熵从 4.83 涨到 8.50，而 `ln(6400) = 8.76` 就是均匀分布的上限——T=4 时已经快贴着天花板了，再升温就只剩噪声。`train_distillation.py` 的默认值是 1.5，帮助里写的推荐区间是 1.0~2.0。

> [!KEY]
> T 升高把 teacher 分布拉平（熵 4.83 → 8.50，上限 `ln V = 8.76`），裸 KL 因此掉 15 倍；乘回 `T²` 让 loss 量级稳在 9 附近，换温度不用重调 lr。

## 从 [B,T] 的 labels 到 [M,V] 的蒸馏输入 {#distill-mask}

`distillation_loss` 要的是 `[M, V]`，而 dataloader 给的是 `[B, T]`。中间这几行变形和 Day 8/9 的 SFT loss 是同一套，只是最后多了一次布尔索引。

{{source:trainer/train_distillation.py#L59-L66}}

{{source:trainer/train_distillation.py#L84-L88}}

三步走。**一、shift**：`logits[..., :-1, :]` 丢掉最后一个位置、`labels[..., 1:]` 丢掉第一个，对齐「预测下一个 token」；实验里 `[3, 128]` 的 input 出来是 `[3, 127, 6400]`。**二、mask**：`loss_mask = (labels[..., 1:] != -100).float()` 得到 `[3, 127]`，`-100` 标的是 prompt 和 padding；展平成 `[381]` 之后有效位只有 94 个（24.7%），实验的 token 彩带里能直接看出 `<|im_start|>assistant` 之后才开始亮。**三、布尔索引**：`student_logits.view(-1, V)[loss_mask_flat == 1]` 筛出 `[94, 6400]`——这就是 `M`。CE 和蒸馏用的是同一个 mask，prompt 和 padding 两边都不参与。

还有一行防御性代码值得一提：`vocab_size_student = student_logits.size(-1); teacher_logits = teacher_logits[..., :vocab_size_student]`，用来兼容「teacher 词表比 student 大」。当前 teacher 和 student 共用同一个 tokenizer、`vocab_size` 都是 6400，这一行是彻底的 no-op。

{{lab:distill_mask}}

{{quiz:q13}}

> [!KEY]
> `[B,T]` → shift → `[B,T-1,V]` → 按 `labels != -100` 展平筛选 → `[M, V]`；实测 381 个位置里只有 94 个参与，CE 和 KD 共用这一个 mask。

## alpha 混合，以及整个 student 都在训练 {#distill-train}

最后一步是把两种监督拼起来：`loss = alpha * ce_loss + (1 - alpha) * distill_loss`，`alpha` 默认 0.5。CE 用的是硬标签（那一个正确 token），KD 用的是 teacher 的整条软分布；`alpha=1` 退化成普通 SFT，`alpha=0` 则完全不看 ground truth。

{{source:trainer/train_distillation.py#L92-L93}}

和 LoRA 最大的结构性差别在这里：`optim.AdamW(model.parameters(), ...)` 和 `clip_grad_norm_(model.parameters(), ...)` 收的是**整个 student**——蒸馏训练的是一个完整（通常更小）的模型，不存在「只更新一小撮」这回事。teacher 那边则是 `eval()` + `requires_grad_(False)`，前向还额外包在 `torch.no_grad()` 里，连计算图都不建。顺带一提，`MiniMindConfig` 默认 `dropout=0.0`，所以 `eval()` 在当前配置下不改变任何数值，纯属防御性写法。

teacher 和 student 的结构可以完全不同：`--student_hidden_size`/`--student_num_layers` 和 `--teacher_*` 是分开的两组参数，各自构造一份 `MiniMindConfig`。实验拿官方 `full_sft`（8 层 63.9M）当 teacher、一个随机初始化的 2 层模型（19.7M）当 student，跑 20 步，三条曲线一起看：ce 从 9 附近降到 6 上下，distill 从 13.4 附近降到 11 上下，total 夹在两者中间。把 `alpha` 推到 1.0（纯 CE），distill 这一项 20 步只挪了 1.2；推到 0.0（纯 KD），两项都降但 ce 明显慢下来（降 2.4 而不是 3.3）——两种监督确实在往不同方向拽。

{{lab:distill_train}}

{{quiz:q14}}

> [!KEY]
> `alpha` 在硬标签 CE 和软分布 KD 之间插值（默认 0.5）；蒸馏的 optimizer/clip 收的是整个 `model.parameters()`，这正是它和 LoRA 在「更新谁」上的分水岭。
