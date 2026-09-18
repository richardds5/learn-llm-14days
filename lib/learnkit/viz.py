"""learnkit.viz —— 在网页输出区画东西的小工具；终端里运行则退化成文本。"""
import torch
import torch.nn.functional as F

from .core import emit, describe, fmt_desc, build_dim_table


def _t(x):
    if isinstance(x, torch.Tensor): return x.detach().float().cpu()
    return torch.as_tensor(x, dtype=torch.float32)


def _round(x, nd=4):
    return [[_r(v, nd) for v in row] for row in x] if x and isinstance(x[0], list) else [_r(v, nd) for v in x]


def _r(v, nd=4):
    if isinstance(v, float) and v == v and abs(v) != float("inf"):
        if v.is_integer() and abs(v) < 1e15: return v  # 65536.0 这类整数值不能被舍成 65540
        return float(f"{v:.{nd}g}")
    return v


def _preview(t: torch.Tensor, limit=8):
    """取一个能看的小角落：前面的维度取 index 0，最后两维各取前 limit 个。"""
    v = t.detach()
    note = ""
    while v.dim() > 2:
        v = v[0]
        note = "前面的维度取 [0]"
    if v.dim() == 2: v = v[:limit, :limit]
    elif v.dim() == 1: v = v[:limit * 2]
    v = v.float().cpu() if v.is_floating_point() or v.dtype == torch.bool else v.cpu()
    return v.tolist(), note


def show(*args, dims=None, title=None, **tensors):
    """查看 tensor：shape / dtype / 统计量 / 数值预览。 用法: show(xq=xq, xk=xk) 或 show(x, title='x')"""
    for i, a in enumerate(args): tensors[f"arg{i}" if len(args) > 1 or not title else title] = a
    table = build_dim_table(dims, None)
    items = []
    for name, v in tensors.items():
        d = describe(v, table)
        item = {"name": name, "desc": d}
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            f = v.detach().float()
            finite = f[torch.isfinite(f)]
            if finite.numel():
                item["stats"] = {"min": finite.min().item(), "max": finite.max().item(), "mean": finite.mean().item(),
                                 "std": finite.std().item() if finite.numel() > 1 else 0.0}
            item["numel"] = v.numel()
            item["bytes"] = v.numel() * v.element_size()
            item["preview"], item["preview_note"] = _preview(v)
            item["preview"] = _round(item["preview"]) if v.dim() else item["preview"]
        items.append(item)
    if not emit({"type": "tensors", "title": title, "items": items}):
        for it in items:
            s = it.get("stats")
            st = f"  min={s['min']:.4g} max={s['max']:.4g} mean={s['mean']:.4g} std={s['std']:.4g}" if s else ""
            print(f"{it['name']}: {fmt_desc(it['desc'])}{st}")


def heatmap(t, title=None, xlabels=None, ylabels=None, labels=None, facet_titles=None, max_size=96, cmap="viridis", vmin=None, vmax=None):
    """热力图。t 可以是 2D [R, C]，也可以是 3D [N, R, C]（画成 N 个小图，比如每个 attention head 一张）。
    labels: 行列共用的标签（比如 token 列表）。-inf 会被画成特殊的灰色格子。"""
    t = _t(t)
    if t.dim() == 1: t = t[None]
    if t.dim() == 2: t = t[None]
    if t.dim() != 3: raise ValueError(f"heatmap 需要 2D 或 3D tensor，拿到的是 {list(t.shape)}")
    orig = list(t.shape)
    n, r, c = t.shape
    if r > max_size or c > max_size:
        t = F.adaptive_avg_pool2d(t[None], (min(r, max_size), min(c, max_size)))[0]
        xlabels = ylabels = labels = None
    if labels is not None: xlabels, ylabels = xlabels or labels, ylabels or labels
    ev = {"type": "heatmap", "title": title, "shape": orig, "cmap": cmap, "vmin": vmin, "vmax": vmax,
          "facets": [_round(m.tolist()) for m in t[:32]], "facet_titles": facet_titles,
          "xlabels": [str(x) for x in xlabels] if xlabels is not None else None,
          "ylabels": [str(y) for y in ylabels] if ylabels is not None else None}
    if not emit(ev): print(f"[heatmap] {title or ''} shape={orig}  (在网页里运行可以看到图)")


def plot(series, title=None, xlabel=None, ylabel=None, x=None, logy=False):
    """折线图。series: list/tensor，或 {名字: list/tensor}。"""
    if not isinstance(series, dict): series = {ylabel or "y": series}
    data = {k: _round(_t(v).flatten().tolist()) for k, v in series.items()}
    xticks = None
    if x is not None and not isinstance(x, torch.Tensor) and any(isinstance(v, str) for v in x):
        xticks, x = [str(v) for v in x], None  # 字符串 x：当成类别标签，横坐标用 0..n-1
    ev = {"type": "plot", "title": title, "xlabel": xlabel, "ylabel": ylabel, "logy": logy, "series": data,
          "x": _round(_t(x).flatten().tolist()) if x is not None else None, "xticks": xticks}
    if not emit(ev):
        for k, v in data.items(): print(f"[plot] {title or ''} {k}: n={len(v)} first={v[:3]} last={v[-3:]}")


def live(chart, series, x, y):
    """训练循环里实时追加一个点: live('loss 曲线', 'train_loss', step, loss.item())"""
    y = float(y)
    if not emit({"type": "live", "chart": chart, "series": series, "x": x, "y": y}):
        print(f"[{chart}] {series} x={x} y={y:.4f}")


def bars(labels, values, title=None, highlight=None, xlabel=None):
    """条形图，常用来看 next-token 概率分布。highlight: 要高亮的下标列表。"""
    vals = _round(_t(values).flatten().tolist())
    ev = {"type": "bars", "title": title, "labels": [str(l) for l in labels], "values": vals,
          "highlight": list(highlight) if highlight is not None else None, "xlabel": xlabel}
    if not emit(ev):
        print(f"[bars] {title or ''}")
        for l, v in zip(labels, vals): print(f"  {str(l)!r:>12}  {v}")


def table(rows, headers=None, title=None):
    rows = [[c if isinstance(c, (int, float, str, bool)) or c is None else str(c) for c in r] for r in rows]
    if not emit({"type": "table", "title": title, "headers": headers, "rows": [_round(r) for r in rows]}):
        if title: print(f"[table] {title}")
        if headers: print(" | ".join(str(h) for h in headers))
        for r in rows: print(" | ".join(str(c) for c in _round(r)))  # 和网页端一样做舍入，终端里不再出现 0.10000000149011612


def token_strip(tokens, values=None, title=None, legend=None):
    """把 token 序列画成一条彩带。values 与 tokens 等长：0/1 的 mask，或者任意浮点数 (如每个 token 的 loss)。"""
    vals = _round(_t(values).flatten().tolist()) if values is not None else None
    ev = {"type": "tokens", "title": title, "tokens": [str(t) for t in tokens], "values": vals, "legend": legend}
    if not emit(ev):
        print(f"[tokens] {title or ''}")
        print(" ".join(f"{t!r}" + (f"({v:g})" if vals else "") for t, v in zip(tokens, vals or [0] * len(tokens))))


def note(md: str):
    """在输出区插一段 markdown 说明。"""
    if not emit({"type": "note", "md": md}): print(md)
