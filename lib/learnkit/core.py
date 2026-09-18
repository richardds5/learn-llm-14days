"""learnkit.core —— 事件通道 + tensor 描述 + 维度符号标注。

在网页 (server.py) 里运行时，环境变量 LEARNKIT_EVENTS 指向一个 jsonl 文件，
所有可视化事件都追加写到这个文件里，前端轮询后渲染。
直接在终端里 `python runner.py xxx.py` 运行时没有这个变量，learnkit 会退化成纯文本打印。
"""
import os
import re
import json
import math
import threading
from pathlib import Path

import torch

from minimind_root import find_minimind_root  # lib/ 在 sys.path 里（learnkit 能被 import 就说明在）

REPO_ROOT = find_minimind_root()  # minimind 源码目录：labs 的 cwd，out/ 权重和 model/ tokenizer 都在它下面
_EVENTS_PATH = os.environ.get("LEARNKIT_EVENTS")
IS_WEB = bool(_EVENTS_PATH)
_lock = threading.Lock()
_fh = None


def _sanitize(o):
    """JSON 不支持 NaN/Infinity，统一转成字符串，前端再还原。"""
    if isinstance(o, float):
        if math.isnan(o): return "nan"
        if math.isinf(o): return "inf" if o > 0 else "-inf"
        return o
    if isinstance(o, dict): return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_sanitize(v) for v in o]
    if isinstance(o, (str, int, bool)) or o is None: return o
    if isinstance(o, torch.Tensor): return _sanitize(o.detach().float().cpu().tolist())
    if isinstance(o, Path): return str(o)
    try:
        import numpy as np
        if isinstance(o, np.generic): return _sanitize(o.item())
        if isinstance(o, np.ndarray): return _sanitize(o.tolist())
    except Exception:
        pass
    return str(o)


def emit(event: dict) -> bool:
    """写一条可视化事件；非网页模式返回 False（调用方自己决定怎么打印）。"""
    global _fh
    if not IS_WEB: return False
    with _lock:
        if _fh is None: _fh = open(_EVENTS_PATH, "a", encoding="utf-8")
        _fh.write(json.dumps(_sanitize(event), ensure_ascii=False) + "\n")
        _fh.flush()
    return True


# ----------------------------------------------------------------------------
# 维度符号：把 [3, 7, 8, 96] 标注成 [B, T, H, D]
# ----------------------------------------------------------------------------
CONFIG_DIMS = [  # (符号, config 属性, 含义)
    ("H", "num_attention_heads", "query 头数"),
    ("KV", "num_key_value_heads", "key/value 头数"),
    ("D", "head_dim", "每个头的维度"),
    ("C", "hidden_size", "隐藏层维度 (残差流宽度)"),
    ("I", "intermediate_size", "FFN 中间层维度"),
    ("V", "vocab_size", "词表大小"),
    ("E", "num_experts", "MoE 专家数"),
]
DIM_MEANINGS = {"B": "batch size", "T": "序列长度 (token 数)", **{s: m for s, _, m in CONFIG_DIMS}}


def dims_from_config(config) -> dict:
    out = {}
    for sym, attr, _ in CONFIG_DIMS:
        v = getattr(config, attr, None)
        if sym == "E" and not getattr(config, "use_moe", False): continue
        if isinstance(v, int) and v > 1: out[sym] = v
    return out


def build_dim_table(user_dims: dict = None, config=None) -> dict:
    """用户给的符号优先级最高，其次是 config 推出来的，最后是常见乘积 (B*T, H*D, KV*D)。"""
    table = {}
    for k, v in (user_dims or {}).items():
        if isinstance(v, int): table[k] = v
    if config is not None:
        for k, v in dims_from_config(config).items(): table.setdefault(k, v)
    used = set(table.values())
    for a, b in [("B", "T"), ("KV", "D"), ("H", "D")]:
        if a in table and b in table:
            p = table[a] * table[b]
            if p not in used:
                table[f"{a}*{b}"] = p
                used.add(p)
    return table


# 两个符号取值相同 (默认配置下 KV=4、E=4) 时，按「当前在哪个函数/模块里」挑更合理的那个
AFFINITY = {
    "E": r"moe|expert|router|mlp\.gate(\s|$)",
    "KV": r"attention|attn|rotary|repeat_kv",
    "H": r"attention|attn|rotary|repeat_kv",
    "D": r"attention|attn|rotary|repeat_kv|freqs",
    "I": r"feedforward|mlp",
    "V": r"lm_head|embed|causallm|log_probs|loss",
}


def label_dims(shape, table: dict, hint: str = ""):
    if not table: return [None] * len(shape)
    rev = {}
    for k, v in table.items(): rev.setdefault(v, []).append(k)  # 先出现的优先
    out = []
    for s in shape:
        cands = rev.get(int(s))
        if not cands: out.append(None); continue
        pick = cands[0]
        if len(cands) > 1 and hint:
            pick = next((c for c in cands if c in AFFINITY and re.search(AFFINITY[c], hint, re.I)), pick)
        out.append(pick)
    return out


# ----------------------------------------------------------------------------
# 把任意 python 值描述成可 JSON 化的 dict（只关心 tensor 的形状等元信息）
# ----------------------------------------------------------------------------
def _has_tensor(v, depth=0):
    if isinstance(v, torch.Tensor): return True
    if depth < 3 and isinstance(v, (tuple, list)): return any(_has_tensor(x, depth + 1) for x in v[:16])
    if depth < 3 and isinstance(v, dict): return any(_has_tensor(x, depth + 1) for x in list(v.values())[:16])
    return False


def describe(v, table: dict = None, depth=0, hint: str = ""):
    if isinstance(v, torch.Tensor):
        d = {"k": "tensor", "shape": [int(s) for s in v.shape], "dtype": str(v.dtype).replace("torch.", ""),
             "labels": label_dims(v.shape, table, hint)}
        if v.device.type != "cpu": d["device"] = str(v.device)
        if v.requires_grad: d["grad"] = True
        if v.numel() == 1:
            try: d["value"] = v.detach().float().item() if v.is_floating_point() else v.item()
            except Exception: pass
        return d
    if isinstance(v, bool): return {"k": "scalar", "value": v, "py": "bool"}
    if isinstance(v, int): return {"k": "scalar", "value": v, "py": "int"}
    if isinstance(v, float): return {"k": "scalar", "value": v, "py": "float"}
    if v is None: return {"k": "none"}
    if isinstance(v, str): return {"k": "str", "value": v if len(v) <= 60 else v[:57] + "..."}
    if isinstance(v, (tuple, list)) and depth < 3:
        if not _has_tensor(v): return None
        return {"k": "seq", "py": type(v).__name__, "len": len(v),
                "items": [describe(x, table, depth + 1, hint) for x in v[:4]]}
    if hasattr(v, "items") and depth < 3:  # dict / transformers ModelOutput
        try: items = list(v.items())
        except Exception: return None
        if not any(_has_tensor(x) for _, x in items): return None
        return {"k": "dict", "py": type(v).__name__,
                "items": {str(k): describe(x, table, depth + 1, hint) for k, x in items[:12] if x is not None}}
    return None


def fmt_desc(d) -> str:
    """纯文本模式下的一行描述。"""
    if d is None: return "?"
    k = d.get("k")
    if k == "tensor":
        dims = ", ".join(f"{l}={s}" if l else str(s) for s, l in zip(d["shape"], d["labels"]))
        val = f" = {d['value']:.4g}" if "value" in d and isinstance(d["value"], float) else (f" = {d['value']}" if "value" in d else "")
        return f"[{dims}] {d['dtype']}{val}"
    if k == "scalar": return repr(d["value"])
    if k == "str": return repr(d["value"])
    if k == "none": return "None"
    if k == "seq":
        names = d.get("names") or []
        inner = ", ".join((f"{names[i]}=" if i < len(names) and names[i] else "") + fmt_desc(x) for i, x in enumerate(d["items"]))
        more = f", …共{d['len']}项" if d["len"] > len(d["items"]) else ""
        return f"{d['py']}({inner}{more})"
    if k == "dict": return f"{d['py']}{{" + ", ".join(f"{n}: {fmt_desc(x)}" for n, x in d["items"].items()) + "}"
    return "?"
