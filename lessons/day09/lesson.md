---
day: 9
format: points
title: "Pretrain 与 SFT 的训练循环"
subtitle: "一个 batch 在 train_epoch 里走完的七拍：lr、autocast、累积、裁剪、日志、续训"
minutes: 90
mainline: "跟着一个 (input_ids, labels) 走完 train_epoch 的一次迭代：算 lr → autocast 前向出 loss → 除以 N → backward 攒梯度 → 每 N 步 unscale/clip/step/zero → 日志与 checkpoint"
files:
  - trainer/train_pretrain.py#L24-L81
  - trainer/trainer_utils.py#L40-L157
  - trainer/train_full_sft.py#L84-L110
goals:
  - 能按顺序说出 train_epoch 一次迭代的七拍，以及每一拍读写了哪个对象
  - 能写出 get_lr 的 cosine 公式、算出起点和终点，并说清楚喂进去的是全局 step 而不是 epoch 内的步号
  - 能解释 loss 为什么要除以 accumulation_steps、日志里为什么又乘回去，以及「累积 N 步 = 大 batch」在什么条件下不成立
  - 能说出 unscale_ → clip → step → zero_grad 这个顺序里每一步的必要性，以及 GradScaler 在 bfloat16 / 无 CUDA 时是个空壳
  - 能对着 lm_checkpoint 和 SkipBatchSampler 说出续训存了什么、恢复时靠什么做到不多训也不漏训
  - 能列出 train_pretrain.py 与 train_full_sft.py 的全部差异，并解释学习率差两个数量级意味着什么
---

Day 5 之后模型已经能吃 `input_ids` 吐 `loss`，Day 8 之后数据集已经能把 jsonl 变成 `(input_ids, labels)`。今天看的是把这两头接起来、让权重真正动起来的那几十行：`trainer/train_pretrain.py` 里的 `train_epoch`，加上 `trainer_utils.py` 里的 `get_lr`、`lm_checkpoint`、`SkipBatchSampler`。跑完它们就得到了 Day 7 用来聊天的那两个权重文件。

今天只跟一样东西走：**一个 batch**。它从 `loader` 里出来，被搬上设备，走完固定的七拍，最后被 `del` 掉；几万次之后权重就训好了。

五章分工：七拍骨架与学习率、前向那三行、真正改权重的五行、存盘与续训、pretrain 与 SFT 的差异及 DDP。`train_full_sft.py` 的 `train_epoch` 和 `train_pretrain.py` **逐字节相同**（只有 `lm_checkpoint` 调用里 `scaler=` 的书写位置不同），所以精读一份就够。

{{flow: (input_ids, labels) [B,T] | *get_lr → param_groups* | autocast 前向 | (loss + aux_loss) / N | *backward → .grad* | 每 N 步 unscale→clip→step→zero | 日志 / checkpoint}}

# 一次迭代的七拍

这一章先把镜头拉远：`train_epoch` 的函数签名长什么样、一次迭代按什么顺序做哪七件事、它凭什么能用到那些没在参数表里出现的名字。然后钻进第一拍——学习率，看清 `get_lr` 的曲线形状，以及喂给它的那个 step 到底是谁。

## train_epoch 的七拍与它依赖的全局变量 {#loop-skeleton}

`train_epoch` 只有五个参数：`epoch`（第几轮）、`loader`、`iters`（这一轮总共多少个 batch）、`start_step`（续训时从第几步接着数）、`wandb`。真正干活要用到的 `args`、`model`、`optimizer`、`scaler`、`autocast_ctx`、`lm_config` 一个都不在参数表里——它们全是 `if __name__ == "__main__":` 块里定义的**模块级变量**，函数体靠 Python 的全局查找直接抓。代价是 `from trainer.train_pretrain import train_epoch` 之后没法直接调用它：那几个名字在别的模块里根本不存在，一执行到 `args.device` 就是 `NameError`。

{{source:trainer/train_pretrain.py#L24-L30}}

`enumerate(loader, start=start_step + 1)` 定了两件事：step 从 1 开始数而不是 0，续训时接着上次的编号往下走。所以「第 step 步」在整个训练里是唯一的，lr、日志、`save_interval` 全靠它对齐。

循环体固定七拍：① 把 batch 搬到 `args.device`；② 按全局 step 算 lr、写回 `optimizer.param_groups`；③ 在 `autocast_ctx` 里前向；④ `res.loss + res.aux_loss` 再除以 `accumulation_steps`；⑤ `scaler.scale(loss).backward()` 把梯度攒进 `.grad`；⑥ 每 `accumulation_steps` 步做一次 unscale → clip → step → zero_grad；⑦ 按间隔打日志、存权重。最后一行 `del input_ids, labels, res, loss` 把这一步的引用全部放掉——显存吃紧时这一行是有意义的。

下面的实验把这七拍原样抄成一份局部变量的版本（变量名和源码一致），在玩具语料上跑 120 步，看 loss 真的往下掉。

{{lab:loop_skeleton}}

{{quiz:q1}}

> [!KEY]
> 一次迭代固定七拍：取 batch → 设 lr → 前向 → 除以 N → backward → 每 N 步更新 → 日志/保存；`args`/`model`/`optimizer`/`scaler` 都是模块级全局变量，所以 `train_epoch` 不能 import 进来直接调用。

> [!MORE] iters 为什么要当参数传进来
> `iters` 不是 `len(loader)` 现算的，而是调用方给的。正常训练时 `train_epoch(epoch, loader, len(loader), 0, wandb)`；续训时是 `train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)`——因为 `loader` 已经被 `SkipBatchSampler` 砍掉了前 `skip` 个 batch，`len(loader)` 会偏小，lr 的分母和日志里的 `(step/iters)` 就都对不上了。把 `skip` 加回去，续训的 lr 曲线才和不中断时完全重合（[见跳过已训 batch 一节](#skip-batch)）。

## get_lr：一条没有 warmup 的余弦 {#lr-cosine}

整个学习率调度只有一行代码，没有 `torch.optim.lr_scheduler`，也没有任何状态。

{{source:trainer/trainer_utils.py#L40-L41}}

把两端代进去：`current_step = 0` 时 `cos(0) = 1`，系数是 `0.1 + 0.45×2 = 1.0`，也就是**起点就是最大学习率**；`current_step = total_steps` 时 `cos(π) = -1`，系数是 `0.1 + 0.45×0 = 0.1`，**终点是 `0.1·lr`**；半程 `cos(π/2) = 0`，系数 `0.55`。中间严格单调下降。

所以这条曲线里**没有 warmup**：主流预训练脚本一般会先花几百到几千步把 lr 从 0 线性拉到峰值，避免随机初始化的模型在第一步就被一个大梯度带偏。MiniMind 直接从峰值开工，靠的是另外两道保险：`grad_clip=1.0` 把第一步的梯度范数摁住（[见梯度裁剪一节](#clip-grad)），以及 64M 这个体量本身对大 lr 比较宽容。

还有一点值得注意：终点是 `0.1·lr` 而不是 0。很多 cosine 调度会衰减到 0，让最后几步几乎不更新参数；这里保留了 10% 的步长，意味着训练结束时模型仍在小幅移动——如果你在最后一个 checkpoint 上看到 loss 还有抖动，这是原因之一。公式里那两个常数（`0.1` 和 `0.45`）就是在控制这两端，改它们等于换一条调度。

{{lab:lr_cosine}}

{{quiz:q2}}

> [!KEY]
> `get_lr` 是纯余弦衰减：起点 = `lr`（没有 warmup），半程 = `0.55·lr`，终点 = `0.1·lr`（不衰减到 0）。

## 喂给 get_lr 的是全局 step {#global-step}

`get_lr` 本身不知道「现在是第几步」，这两个数是调用处算出来的，而算法很关键。

{{source:trainer/train_pretrain.py#L31-L33}}

`epoch * iters + step` 是**跨 epoch 累计**的全局步号，`args.epochs * iters` 是整个训练的总步数。于是 2 个 epoch 不是两条各自从峰值开始的余弦，而是**同一条余弦的前半段和后半段**：第 1 个 epoch 结束时 lr 掉到 `0.55·lr`，第 2 个 epoch 从那里接着往下走，直到 `0.1·lr`。如果误写成 `get_lr(step, iters, ...)`，每个 epoch 都会把 lr 重新拉回峰值，变成一个谁也没打算要的 cosine restart。

第二件事是**每一步手动写回** `optimizer.param_groups`，而不是建一个 scheduler 对象再 `scheduler.step()`。好处是没有额外状态要存进 checkpoint：续训时只要 `epoch` 和 `step` 对，lr 自然就对（[见 lm_checkpoint 一节](#ckpt-half)）；坏处是 `param_groups` 里的 `lr` 全部被同一个值覆盖，想给不同参数组不同学习率（比如 embedding 用更小的 lr）就得自己改这两行。

顺带记一下日志那边：`current_lr = optimizer.param_groups[-1]['lr']` 读的是**最后一组**——这里所有组的值都一样，所以随便读哪一组都行。

{{lab:global_step}}

{{quiz:q3}}

> [!KEY]
> 传给 `get_lr` 的是 `epoch * iters + step` 这个全局步号、分母是 `args.epochs * iters`，所以多个 epoch 共享一条连续的余弦；lr 每步手写进 `param_groups`，没有 scheduler 状态要存。

# 前向那三行

`with autocast_ctx:` 包住的只有三行，却决定了这一步用什么精度算、loss 由哪两项组成、以及真正拿去 backward 的数是多少。这一章逐行拆开，最后用一个实验检验「梯度累积等于大 batch」这句话到底在什么条件下成立。

## autocast_ctx 的那行 device 判断 {#autocast-ctx}

混合精度的开关在 `main` 里，三行就配好了，`train_epoch` 只负责 `with` 它一下。

{{source:trainer/train_pretrain.py#L121-L123}}

逐行看：第一行用**字符串包含**判断设备类别——`args.device` 里含不含 `"cuda"` 这三个字母，含就是 `"cuda"`，不含就是 `"cpu"`，只有这两种结果。第二行把 `--dtype` 这个字符串翻译成真正的 `torch.dtype`，`"bfloat16"` 之外的任何取值都会落到 `float16`。第三行据 `device_type` 二选一：`"cpu"` 用 `nullcontext()`（进出都什么都不做的空上下文），否则用 `torch.cuda.amp.autocast(dtype=dtype)`。

`autocast` 的作用是在上下文内部把 `nn.Linear`、`matmul` 这类算子自动降到低精度执行（softmax、归一化等数值敏感的算子仍留在 fp32）；`nullcontext()` 则是纯粹的占位符。写成这样是为了让 `train_epoch` 里那句 `with autocast_ctx:` 在任何设备上都成立，不用写 `if`。

Mac 上 `best_device()` 给出的是 `"mps"`。那么在 MPS 上跑这份脚本（`--dtype bfloat16`），`autocast_ctx` 最终是什么、前向用的是什么精度？先下注，再看实验最后一行的实测。

{{predict:p1}}

{{lab:autocast_ctx}}

实验表格里 `mps` 那一行和 `cpu` 完全一样：`device_type = cpu`、`autocast_ctx = nullcontext`，实测 `logits.dtype = torch.float32`。这行判断把「不是 CUDA」和「是 CPU」当成了同一件事，于是 **MPS 上混合精度静默失效，全程 fp32**。这解释了为什么本课的训练实验都不必操心 dtype。

{{quiz:q4}}

> [!KEY]
> `"cuda" in args.device` 只认 CUDA，MPS / XPU 都会掉进 `"cpu"` 分支让 `autocast_ctx` 变成 `nullcontext()`，混合精度静默失效。

> [!MORE] torch.cuda.amp 这个写法
> `torch.cuda.amp.autocast` / `torch.cuda.amp.GradScaler` 在新版 PyTorch 里已经是 `torch.amp.autocast("cuda", ...)` / `torch.amp.GradScaler("cuda", ...)` 的别名，调用时会发 `FutureWarning`——脚本开头那句 `warnings.filterwarnings('ignore')` 把这类提示全吞掉了。设备无关的新接口第一个参数就是 device_type，所以只要把判断改对，同一行代码在 cuda / cpu / mps 上都能用。

## res.loss + res.aux_loss 与那次除法 {#loss-terms}

前向只有一次调用：`res = model(input_ids, labels=labels)`。`labels` 一旦给了，`MiniMindForCausalLM.forward` 内部就把 shift、`ignore_index=-100`、交叉熵全做完了（Day 5 讲过），所以这里拿到的 `res.loss` 已经是一个标量。

{{source:trainer/train_pretrain.py#L35-L38}}

第二项 `res.aux_loss` 是 MoE 的路由均衡损失（Day 6）。稠密模型里它不是 `None`、也不是 `0.0`，而是 `hidden_states.new_zeros(1).squeeze()` 对空列表求和的结果——一个值为 0、`requires_grad=False` 的 0 维 tensor。正因为它永远存在且形状统一，这行加法对稠密和 MoE 两种模型通用，不需要任何分支判断；稠密模型加上它既不改数值也不引入梯度。

第三行的除法是为梯度累积做的：`accumulation_steps` 次 `backward()` 的梯度会累加在 `.grad` 里，每次都先除以 N，加起来才相当于对 N 个 micro-batch 求**平均**而不是求和。注意除的是 loss 不是梯度——因为求导是线性的，这两种做法等价，但除 loss 只需要一次标量除法。

被除过的 `loss` 会一路用到日志：`current_loss = loss.item() * args.accumulation_steps` 把它乘回去，只是为了让打印出来的数和「不做累积」时量纲一致，不影响任何梯度。日志里还把总 loss 拆成 `logits_loss` 和 `aux_loss` 两项分别记录，方便训 MoE 时盯住路由是否失衡。

{{lab:loss_terms}}

{{quiz:q5,q6}}

> [!KEY]
> `loss = (res.loss + res.aux_loss) / accumulation_steps`：`aux_loss` 在稠密模型里是个值为 0 的 0 维 tensor（所以不用分支），那次除法是为了让 N 次累积等于求平均，日志里再乘回去。

## 累积 N 步等于大 batch 的条件 {#grad-accum}

「`batch_size=32` + `accumulation_steps=8` 等价于 `batch_size=256`」是梯度累积的标准说法，MiniMind 的预训练默认配置正是这么设的。它成立的依据很简单：`backward()` 把新梯度**加到** `.grad` 上而不是覆盖，而求导是线性的，所以 N 次 `(loss_i / N).backward()` 得到的就是 `mean(loss_i)` 的梯度。

{{source:trainer/train_pretrain.py#L38-L40}}

不过要注意 `loss` 是怎么来的。`MiniMindForCausalLM` 内部调的是 `F.cross_entropy(..., ignore_index=-100)`，默认 `reduction='mean'`——分母是**这一次调用里的有效 token 数**，而不是全局 token 数。所以 `loss_i` 不是「这 micro-batch 的总损失」，而是「这 micro-batch 的每 token 平均损失」。

另一边，`PretrainDataset` 会把 pad 位置的 label 设成 `-100`（Day 8），`SFTDataset` 更是只保留 assistant 回答那一段。同一批样本长短不一，切成 micro-batch 之后每份的有效 token 数自然也不一样。

那么：把 8 条长短不齐的样本拆成 4 个 micro-batch 累积，和一次性算这 8 条，得到的梯度差多少？先下注再跑。

{{predict:p2}}

{{lab:grad_accum}}

实验的对比很干脆：每条样本都是 11 个有效 token 时，两种算法的梯度最大差 `2e-08`（纯浮点误差）；换成 `[3, 5, 9, 11, 4, 10, 6, 7]` 这种长度之后，差距变成 `1.4e-01`——和梯度本身一个量级。原因就在上面那个分母：「先按各自的 token 数取平均、再对 4 份取平均」和「对全部 token 取一次平均」只有在各份 token 数相等时才相同，否则短样本会被赋予更高的权重。这不是 bug，而是「等价」这个词的适用范围。

{{quiz:q7}}

> [!KEY]
> 梯度累积等于大 batch，前提是每个 micro-batch 的**有效 token 数相同**；因为 `ignore_index` 让每次 `cross_entropy` 各自按自己的有效 token 数取平均，长短不齐时两者会真的不一样。

# 真正改权重的五行

前面所有步骤都只是在往 `.grad` 里攒数。真正改动权重的是 `if step % accumulation_steps == 0:` 里的五行，顺序一行都不能换。这一章从最容易被误解的 `scaler` 开始，再看裁剪，最后把顺序和 epoch 末尾那段「尾巴」讲清楚。

## GradScaler：只有 fp16 才真的在缩放 {#scaler}

`scaler` 的构造只有一行，但 `enabled` 的取值决定了后面四处调用（`scale` / `unscale_` / `step` / `update`）是真干活还是走过场。

{{source:trainer/train_pretrain.py#L138}}

`enabled=(args.dtype == 'float16')`：脚本默认 `--dtype bfloat16`，所以默认情况下 `enabled=False`。这时 `scaler.scale(loss)` **原样返回同一个对象**（`scale(loss) is loss` 为 `True`），`unscale_` 什么都不做，`scaler.step(optimizer)` 直接转调 `optimizer.step()`，`update()` 是空操作——`GradScaler` 全程只是一层透明包装，让代码不用为两种 dtype 写两套分支。

为什么 bfloat16 不需要它？因为 loss scaling 是给 fp16 擦屁股的：fp16 能表示的最小正规数约 `6e-5`，反向传播里小梯度很容易直接下溢成 0；把 loss 乘上一个大因子（默认起始 65536）再 backward，梯度整体抬高，`unscale_` 时再除回去。bfloat16 的指数位和 fp32 一样宽，根本不会下溢，自然不用缩放。

还有一个容易踩空的地方：`torch.cuda.amp.GradScaler(enabled=True)` 在**没有 CUDA** 的机器上会自己把 `enabled` 改回 `False`（只发一条警告）。也就是说在 Mac 上哪怕显式传 `--dtype float16`，scaler 依然是空壳。实验里用 `torch.amp.GradScaler("cpu", enabled=True)` 做对照，能看到真正启用时是什么样。

{{lab:scaler_noop}}

{{quiz:q8}}

> [!KEY]
> `GradScaler` 只有在 `--dtype float16` **且** 机器真有 CUDA 时才启用；否则 `scale(loss)` 原样返回 loss，`unscale_`/`step`/`update` 全部退化成透传。

## clip_grad_norm_ 与它的返回值 {#clip-grad}

裁剪只有一行，但它和上一行的顺序是有讲究的。

{{source:trainer/train_pretrain.py#L43-L44}}

`clip_grad_norm_(params, max_norm)` 做的是**全局**裁剪：先把所有参数的 `.grad` 当成一个长向量算出总 L2 范数 `total_norm`，如果 `total_norm > max_norm`，就把每个 `.grad` 原地乘上 `max_norm / total_norm`。注意是「所有参数一起缩放同一个系数」，不是逐参数各裁各的——所以梯度的**方向**完全不变，只是长度被截短，这正是它比逐元素 `clip_grad_value_` 温和的原因。

它的返回值是**裁剪前**的 `total_norm`，这是个很好用的监控量：训练炸掉之前，grad norm 往往先出现尖峰。MiniMind 没有把它记进日志，但你自己加一行 `wandb.log({"grad_norm": ...})` 是最省事的排障手段。

为什么必须排在 `scaler.unscale_(optimizer)` 后面？因为 fp16 训练时 `.grad` 里装的是被放大过 65536 倍的值，直接拿去和 `max_norm=1.0` 比较，会导致几乎每一步都被判定为「超标」并被压回 1.0/65536 的真实尺度——等于把学习率砍掉四个数量级。`unscale_` 先把梯度换算回真实尺度，裁剪阈值才有意义。

{{lab:grad_clip}}

{{quiz:q9}}

> [!KEY]
> `clip_grad_norm_` 按全局范数等比缩放所有 `.grad`（方向不变），返回值是**裁剪前**的范数；它必须排在 `unscale_` 之后，否则阈值比较的是被放大过的假梯度。

## 更新的顺序，与 epoch 末尾的尾巴 {#update-tail}

五行的顺序是固定的：先 `unscale_` 把梯度换算回真实尺度，再 `clip_grad_norm_` 截短，然后 `scaler.step(optimizer)` 更新参数（启用时它还会检查有没有 inf/nan，有就跳过这一步），`scaler.update()` 调整下一轮的缩放因子，最后 `optimizer.zero_grad(set_to_none=True)` 清空梯度。

{{source:trainer/train_pretrain.py#L42-L49}}

`set_to_none=True` 让 `.grad` 直接变成 `None` 而不是一堆 0：省一次内存写，下一次 `backward()` 会直接赋值而不是累加。副作用是清零之后 `p.grad` 是 `None`，写监控代码时得判空。

问题出在循环的边界。`if step % accumulation_steps == 0` 只在整除时触发，如果这个 epoch 的 batch 总数不是 `accumulation_steps` 的整数倍（比如 10 个 batch、每 3 步更新一次），最后几步的梯度会一直躺在 `.grad` 里等一个永远不会到来的整除时刻，循环结束就被下一个 epoch 的 `zero_grad` 冲掉了。

{{source:trainer/train_pretrain.py#L75-L80}}

于是 epoch 末尾补了一段完全一样的五行，条件是 `last_step > start_step`（这个 epoch 确实训过东西）**且** `last_step % accumulation_steps != 0`（确实有没消费掉的梯度）。整除时这段不会执行，也就不会出现「同一批梯度被 step 两次」。

{{lab:update_tail}}

{{quiz:q10,q11}}

> [!KEY]
> 顺序是 unscale_ → clip → step → update → zero_grad，缺一不可；epoch 末尾用 `last_step % accumulation_steps != 0` 判断有没有攒了一半的梯度，有就补一次更新。

# 存盘与续训

训练几个小时的脚本必须能中断重来。MiniMind 把这件事拆成两半：`lm_checkpoint` 负责「存了什么、读回什么」，`SkipBatchSampler` 负责「读回来之后从哪个 batch 接着训」。这一章把两半拼起来，顺便看一个关于精度的细节。

## lm_checkpoint 存的两个文件 {#ckpt-half}

`lm_checkpoint` 一个函数身兼保存和加载，靠 `model is not None` 区分：传了 `model=` 就是保存，不传就是去 `checkpoints/` 里探测有没有可续训的存档，有就返回一个 dict，没有就返回 `None`。

保存模式会往 `save_dir`（脚本传的是 `../checkpoints`）写两个文件。第一个是 `{weight}_{hidden}.pth`，内容和 `train_epoch` 刚写进 `out/` 的那份权重一模一样：

{{source:trainer/trainer_utils.py#L69-L76}}

三个细节：`raw_model` 那两行在**剥壳**（见下面的折叠块）；`{k: v.half().cpu()}` 把参数转成 fp16 再挪回 CPU，文件体积减半；最后是先写 `.tmp` 再 `os.replace` 的**原子写**。

第二个文件 `{weight}_{hidden}_resume.pth` 装续训状态：`model`、`optimizer.state_dict()`、`epoch`、`step`、`world_size`、`wandb_id`，外加 `**kwargs` 里任何带 `.state_dict()` 的对象（脚本传的是 `scaler=scaler`）。值得盯一眼的是 `resume_data['model']`：它填的**不是**重新取的一份 `raw_model.state_dict()`，而是上面那段代码里的 `state_dict` 变量本身。

那么，存一次再把 resume 文件读回来，权重和保存前相比会变吗？变多少？

{{predict:p3}}

{{lab:ckpt_half}}

实测：`ckp['model']` 的 dtype 是 `float16`，`‖Δw‖/‖w‖` 最大 `2.1e-04`，逐元素相对误差最大 `4.88e-04`——正好是 fp16 尾数 10 位的上界 `2⁻¹¹`。而 Adam 的 `exp_avg` / `exp_avg_sq` 是完整 fp32：每存一次盘权重就被舍入一次，动量却分毫不差。

{{quiz:q12}}

> [!KEY]
> `lm_checkpoint` 原子写两个文件，续训用的 `'model'` 复用的是**已经 half() 过**的同一份 state_dict（每次存盘舍入一次，相对误差 ~5e-4），而优化器动量是完整 fp32。

> [!MORE] 剥壳、原子写、空 scaler：三个边角
> 1. **剥壳**：`model` 可能被 `DistributedDataParallel` 和 `torch.compile` 各包了一层，所以先 `.module` 再 `._orig_mod`（`getattr(x, '_orig_mod', x)` 没包过就原样返回）。不剥的话 state_dict 的 key 会多出 `module.` / `_orig_mod.` 前缀，换个环境加载就全对不上。
> 2. **不对称的原子写**：`train_epoch` 里写 `out/` 下那份「最终交付权重」用的是裸 `torch.save(..., ckp)`，**没有** `.tmp` + `os.replace`；而 `lm_checkpoint` 内部两次保存都是原子写。同一次保存里续训存档比对外权重更安全，这是个不太对称的细节。
> 3. **空 scaler**：`enabled=False` 的 `GradScaler`，它的 `state_dict()` 是空字典 `{}`（实验表格最后一行）。拿一份在 CPU/bfloat16 下存的 checkpoint 到 fp16 的 CUDA 机器上续训，`scaler.load_state_dict({})` 会直接抛 `RuntimeError: The source state dict is empty`。

## SkipBatchSampler：跳过已经训过的 batch {#skip-batch}

读回 `step` 之后还有一半工作：这个 epoch 的前 `step` 个 batch 已经训过了，不能再训一遍。`SkipBatchSampler` 就干这件事。

{{source:trainer/trainer_utils.py#L140-L157}}

它是个 **batch sampler**（给 `DataLoader(batch_sampler=...)` 用的，一次 yield 一个下标列表）。`__iter__` 按 `batch_size` 攒下标，攒够一批就看 `skipped < skip_batches`：是就丢掉这一批、计数加一、继续；否则正常 `yield`。所以**被跳过的和保留下来的是同一条序列上的前后两段**，顺序完全不受影响——这正是续训能对齐的关键。`__len__` 则是「总批数减去跳过的」，向上取整那一下 `(len + bs - 1) // bs` 说明最后一个不满的批也算数。

它包住的 `sampler` 可以是一个普通 list（单卡时的 `indices`），也可以是任何有 `__iter__` / `__len__` 的 `Sampler`（DDP 时的 `DistributedSampler`），接口完全一致。

最后一块拼图是 `world_size`：`step` 记的是「每张卡各自的 DataLoader 走到第几个 batch」。8 卡训的存档换到 4 卡上续训，每张卡要多分担一倍的数据，所以加载模式里会做 `step = step * saved_ws // current_ws` 的换算，并打一条日志。

{{lab:skip_batch}}

{{quiz:q13}}

> [!KEY]
> `SkipBatchSampler` 只丢掉前 `skip_batches` 批、不改变剩下的顺序，所以续训既不重训也不漏训；GPU 数变化时 `step` 会按 `saved_ws / current_ws` 的比例换算。

# Pretrain、SFT 与多卡

最后一章回到开头那句话：两个脚本的 `train_epoch` 逐字节相同，差异全在 `main` 里的默认参数和数据集类。我们先把差异列全、再用真实权重验证其中最要命的一条（学习率），最后看 DDP 到底动了哪几行。

## 两个脚本的差异清单 {#pretrain-vs-sft}

`train_full_sft.py` 是从 `train_pretrain.py` 复制出来改的，差异只有下面这些：

{{source:trainer/train_full_sft.py#L88-L94}}

| 项 | train_pretrain.py | train_full_sft.py | 为什么 |
|---|---|---|---|
| 数据集类 | `PretrainDataset` | `SFTDataset` | 纯文本续写 vs 只对 assistant 回答算 loss（Day 8） |
| `learning_rate` | `5e-4` | `1e-5` | 从随机初始化建立能力 vs 在已经会说话的模型上微调 |
| `accumulation_steps` | `8` | `1` | 预训练要大等效 batch（32×8=256）稳住梯度 |
| `batch_size` / `max_seq_len` | `32` / `340` | `16` / `768` | 对话样本更长，同样显存下只能缩小 batch |
| `from_weight` | `'none'`（从 0 训） | `'pretrain'` | 这就是两个阶段的衔接点 |
| `save_weight` / `data_path` | `pretrain` / `pretrain_t2t_mini.jsonl` | `full_sft` / `sft_t2t_mini.jsonl` | 各自的输入输出文件名 |

`from_weight` 那一行由 `init_model` 消费：`'none'` 就直接用随机初始化的 `MiniMindForCausalLM(lm_config)`，否则去 `{save_dir}/{from_weight}_{hidden}.pth` 加载，并且用的是 `strict=False`——多余或缺失的 key 都不报错，给「结构改了一点还想热启动」留了余地（Day 10 的 LoRA 会用上）。

学习率差两个数量级是这张表里最值得亲手验证的一条。下面的实验在真实的 `out/full_sft_768.pth` 上，用完全相同的 12 个 batch，分别按 `1e-5` 和 `5e-4` 训一遍。

{{lab:sft_lr}}

两条曲线从同一个起点出发（`2.4` 上下，每次抽到的样本略有不同）：`1e-5` 一路降到 `0.7` 左右；`5e-4` 第 3 步就冲到 `15` 上下，12 步之后才勉强爬回起点附近——**训了等于没训，甚至更差**。预训练的学习率直接拿来做 SFT，等于把辛苦训出来的权重先砸碎一遍。

{{quiz:q14}}

> [!KEY]
> 两个脚本的 `train_epoch` 完全相同，差异只在默认参数和数据集类；其中 `5e-4` vs `1e-5` 是量变引起质变——预训练的 lr 用在 SFT 上会直接把已有能力打散。

> [!MORE] README 里的训练开销
> 作者在 README 「Ⅰ 训练开销」一节给了单卡 3090 的经验估算（`7￥ ≈ 1 美元`，3090 约 `1.3￥/h`）：`minimind-3`（64M）跑 `pretrain_t2t_mini` 约 1.21h / 1.57￥，跑 `sft_t2t_mini` 约 1.10h / 1.43￥；MoE 版本（198M-A64M）分别是 1.69h 和 1.54h。两个阶段各 1 epoch 合计约 2.31 小时、约 3.0 元——就是首页那句「3 块钱、2 小时训出一个 LLM」的来源。

## DDP 动过的那几处 {#ddp-touches}

单卡和多卡跑的是同一份脚本，DDP 相关的改动一共就散落在六处：`init_distributed_mode()` 看 `RANK` 环境变量决定要不要 `init_process_group`（`torchrun` 会设置它，不设就直接返回 0）；种子加上 rank，让各进程的随机状态错开；`DistributedSampler` 切分数据；`DistributedDataParallel` 包模型（必须在 `torch.compile` 之后，包在最外层）；`is_main_process()` 保证只有 rank 0 写文件和打日志；结尾 `dist.barrier()` + `destroy_process_group()`。

{{source:trainer/train_pretrain.py#L158-L163}}

训练循环这几行里藏着一个短路表达式：`SkipBatchSampler(train_sampler or indices, ...)`。单卡时 `train_sampler is None`，取右边的 `indices`——也就是上一行用 `setup_seed(args.seed + epoch)` 重新种子之后 `randperm` 出来的顺序，保证每个 epoch 换一种打乱方式、而且可复现。DDP 时 `train_sampler` 是个非空对象，表达式直接短路取左边，刚算出来的 `indices` **一行都用不上**，真正决定顺序的是 `train_sampler.set_epoch(epoch)`。

`set_epoch` 这一步不能省：`DistributedSampler` 内部用 epoch 当随机种子，不调用它的话每个 epoch 每张卡都会拿到一模一样的切分顺序。写法 `train_sampler and train_sampler.set_epoch(epoch)` 也是同一个短路技巧——`None` 时整个表达式直接求值成 `None`，省掉一个 `if`。

{{lab:ddp_sampler}}

{{quiz:q15}}

> [!KEY]
> DDP 只改了六处（进程组、seed+rank、Sampler、DDP 包装、is_main_process、barrier）；`train_sampler or indices` 靠短路让单卡走 `randperm`、多卡走 `DistributedSampler`。
