"""learnkit.tracer —— 两种粒度的 tensor 维度追踪。

1. 模块级：给 model 的每个子模块挂 forward hook，得到一棵「调用树」，每个节点有输入/输出 shape。
2. 行级：用 sys.settrace 盯住指定函数（比如 Attention.forward），每执行完一行就 diff 一次局部变量，
   记录「这一行读了哪些 tensor、写出了哪些 tensor、shape 是什么」。

用法：
    with trace(model, fns=[Attention.forward], dims=dict(B=3, T=7)) as tr:
        model(input_ids)
    tr.captured  # 如果传了 capture=['scores']，这里能拿到对应局部变量的真实 tensor
"""
import os
import re
import ast
import sys
import copy
import json
import inspect
import textwrap
import importlib
from pathlib import Path

import torch
from torch import nn

from .core import emit, describe, fmt_desc, build_dim_table, REPO_ROOT, DIM_MEANINGS

_SKIP_NAMES = {"self", "cls", "__class__"}
_OVERRIDES = {str(Path(v).resolve()): k for k, v in json.loads(os.environ.get("LEARNKIT_OVERRIDES") or "{}").items()}


def _resolve_fn(f):
    """接受函数 / 方法 / 类(取 forward) / 'pkg.mod:Class.method' 字符串。"""
    if isinstance(f, str):
        mod_name, _, qual = f.partition(":")
        obj = importlib.import_module(mod_name)
        for part in qual.split("."): obj = getattr(obj, part)
        f = obj
    if inspect.isclass(f): f = f.forward
    f = inspect.unwrap(getattr(f, "__func__", f))
    if not hasattr(f, "__code__"): raise TypeError(f"trace(fns=...) 里的 {f!r} 不是 python 函数")
    return f


_FWD_NAMES = {}


def _forward_names(module):
    """从 forward 的签名和 AST 里挖出名字，让模块调用树不再是一排匿名色块：
    params  —— 位置参数名 (跳过 self)；varargs —— *args 的名字
    unpack  —— {参数名: [元素名]}，来自 `cos, sin = position_embeddings` 这种解包语句
    returns —— 每条 return 语句的名字：`return output, past_kv` → ['output', 'past_kv']；`return x` → 'x'
    """
    fn = getattr(type(module), "forward", None)
    fn = inspect.unwrap(getattr(fn, "__func__", fn)) if fn is not None else None
    if fn in _FWD_NAMES: return _FWD_NAMES[fn]
    info = {"params": [], "varargs": None, "unpack": {}, "returns": []}
    try:
        for p in list(inspect.signature(fn).parameters.values())[1:]:
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD): info["params"].append(p.name)
            elif p.kind == p.VAR_POSITIONAL: info["varargs"] = p.name
        fdef = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        name_of = lambda e: e.id if isinstance(e, ast.Name) else (s if len(s := ast.unparse(e)) <= 24 else None)
        todo = list(fdef.body)
        while todo:
            n = todo.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)): continue
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple) \
                    and all(isinstance(e, ast.Name) for e in n.targets[0].elts):
                info["unpack"].setdefault(n.value.id, [e.id for e in n.targets[0].elts])
            if isinstance(n, ast.Return) and n.value is not None:
                info["returns"].append([name_of(e) for e in n.value.elts] if isinstance(n.value, ast.Tuple) else (n.value.id if isinstance(n.value, ast.Name) else None))
            todo.extend(ast.iter_child_nodes(n))
    except Exception:
        pass  # 内置模块 (nn.Linear 等) 的 forward 也能拿到签名；拿不到就退回匿名
    _FWD_NAMES[fn] = info
    return info


def _sig(v, depth=0):
    """变量的「指纹」：用来判断一行代码执行后这个变量有没有变。"""
    if isinstance(v, torch.Tensor):
        try: ver = v._version
        except Exception: ver = 0  # inference tensor 没有 version counter
        return ("T", id(v), tuple(v.shape), v.dtype, ver)
    if isinstance(v, (bool, int, float, str, type(None))): return ("V", v)
    if isinstance(v, (tuple, list)) and depth < 2: return ("S", len(v), tuple(_sig(x, depth + 1) for x in v[:8]))
    return ("O", id(v))


class _FnInfo:
    def __init__(self, fn):
        self.fn = fn
        self.code = fn.__code__
        self.name = fn.__qualname__
        lines, first = inspect.getsourcelines(fn)
        self.first_line = first
        self.source = [l.rstrip("\n") for l in lines]
        f = Path(inspect.getsourcefile(fn) or "?").resolve()
        self.modified = str(f) in _OVERRIDES  # 网页里改过的源码：显示成仓库里的原路径
        try: self.file = _OVERRIDES.get(str(f)) or str(f.relative_to(REPO_ROOT))
        except ValueError: self.file = str(f)
        self.stmts, self.src_text = {}, ""  # expand 用：{首行号: 语句的 ast 节点}
        self.reads, self.stmt_start = self._analyze()
        self.calls = []
        self.total_calls = 0

    def _analyze(self):
        """用 ast 找出每条语句读了哪些变量名；多行语句统一归到首行。"""
        reads, stmt_start = {}, {}
        self.src_text = textwrap.dedent("\n".join(self.source))
        try: tree = ast.parse(self.src_text)
        except SyntaxError: return reads, stmt_start
        off = self.first_line - 1
        compound = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt): continue
            if isinstance(node, compound):
                heads = [getattr(node, a) for a in ("test", "iter") if hasattr(node, a)]
                heads += [i.context_expr for i in getattr(node, "items", [])]
            else:
                heads = [node]
                self.stmts[node.lineno + off] = node
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1): stmt_start[ln + off] = node.lineno + off
            names = []
            for h in heads:
                for n in ast.walk(h):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in names: names.append(n.id)
            if names: reads.setdefault(node.lineno + off, names)
        return reads, stmt_start


_EXPAND_SKIP = (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)
_EXPAND_LEAF = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.IfExp, ast.BoolOp, ast.JoinedStr)
_EXPAND_EVAL = (ast.Call, ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Attribute, ast.Compare)


class Trace:
    def __init__(self, model=None, fns=(), dims=None, title=None, max_calls=4, max_events=400, capture=(), max_roots=3, focus=None, tree=True, expand=None):
        self.model, self.title, self.focus, self.tree, self.expand = model, title, focus, tree, expand
        self._expanding, self._focus_cache = False, {}
        self.max_calls, self.max_events, self.max_roots = max_calls, max_events, max_roots
        self.capture = set(capture or ())
        self.captured = {}  # {fn_name: [ {var: tensor}, ... 每次调用一个 dict ]}
        config = getattr(model, "config", None)
        self.table = build_dim_table(dims, config)
        self.infos = {}
        for f in (fns or ()):
            info = _FnInfo(_resolve_fn(f))
            self.infos[info.code] = info
        self.module_names = {id(m): n for n, m in model.named_modules()} if isinstance(model, nn.Module) else {}
        self._frames, self._handles, self._stack = {}, [], []
        self._seq_descs, self._seq_named = {}, {}  # 见 _name_items
        self.roots, self.total_roots = [], 0
        self._prev_trace = None

    # ------------------------------------------------------------------ 行级
    def _snapshot(self, frame):
        vals = {}
        for k, v in frame.f_locals.items():
            if k in _SKIP_NAMES or k.startswith("."): continue
            if isinstance(v, nn.Module) or callable(v) and not isinstance(v, torch.Tensor): continue
            vals[k] = v
        return vals

    def _global_tracer(self, frame, event, arg):
        info = self.infos.get(frame.f_code)
        if info is None or event != "call": return None
        info.total_calls += 1
        if len(info.calls) >= self.max_calls: return None
        vals = self._snapshot(frame)
        slf = frame.f_locals.get("self")
        rec = {"index": info.total_calls - 1, "self": self.module_names.get(id(slf)), "events": [], "ret": None, "truncated": False, "hits": {}}
        args = [{"name": k, "desc": d, "change": "arg"} for k, v in vals.items() if (d := describe(v, self.table, hint=info.name)) is not None]
        if args: rec["events"].append({"line": info.first_line, "reads": [], "writes": args, "is_args": True})
        info.calls.append(rec)
        cap = {}
        if self.capture: self.captured.setdefault(info.name, []).append(cap)
        st = {"info": info, "rec": rec, "vals": vals, "sigs": {k: _sig(v) for k, v in vals.items()}, "line": None, "pending_reads": [], "cap": cap}
        self._frames[id(frame)] = st
        return self._local_tracer

    def _flush(self, st, frame):
        """把「上一行」造成的变量变化记下来。"""
        if st["line"] is None: return
        info, rec = st["info"], st["rec"]
        vals = self._snapshot(frame)
        sigs = {k: _sig(v) for k, v in vals.items()}
        writes = []
        for k, s in sigs.items():
            old = st["sigs"].get(k)
            if old == s: continue
            change = "new" if old is None else ("inplace" if old[0] == "T" and s[0] == "T" and old[1] == s[1] else "assign")
            d = describe(vals[k], self.table, hint=info.name)
            if d is None: continue
            writes.append({"name": k, "desc": d, "change": change})
            if k in self.capture and isinstance(vals[k], torch.Tensor): st["cap"][k] = vals[k].detach().clone()
        if writes or st["pending_reads"] or st.get("pending_expr"):
            if len(rec["events"]) < self.max_events:
                ev = {"line": st["line"], "reads": st["pending_reads"], "writes": writes}
                if st.get("pending_expr"): ev["expr"] = st["pending_expr"]
                rec["events"].append(ev)
            else: rec["truncated"] = True
        st["vals"], st["sigs"] = vals, sigs

    def _local_tracer(self, frame, event, arg):
        st = self._frames.get(id(frame))
        if st is None: return None
        info = st["info"]
        if event == "line":
            ln = info.stmt_start.get(frame.f_lineno, frame.f_lineno)
            raw, st["raw"] = st.get("raw"), frame.f_lineno
            if ln == st["line"] and raw != frame.f_lineno: return self._local_tracer  # 多行语句内部的跳转，忽略
            self._flush(st, frame)
            hits = st["rec"]["hits"]
            hits[ln] = hits.get(ln, 0) + 1
            st["line"] = ln
            st["pending_reads"] = [{"name": n, "desc": d} for n in info.reads.get(ln, [])
                                   if n in st["vals"] and (d := describe(st["vals"][n], self.table, hint=info.name)) is not None and d["k"] in ("tensor", "seq", "dict")]
            st["pending_expr"] = self._expand_line(info, frame, ln) if self._want_expand(info, ln) else []
        elif event == "return":
            self._flush(st, frame)
            st["rec"]["ret"] = describe(arg, self.table, hint=info.name)
            st["rec"]["ret_line"] = st["line"]
            self._frames.pop(id(frame), None)
        return self._local_tracer

    # ---------------------------------------------------------------- 把一行拆开：子表达式的 shape
    def _want_expand(self, info, ln):
        e = self.expand
        if not e: return False
        if e is True:  # 跟着 focus 走；没有 focus 就全部展开
            if info.code not in self._focus_cache: self._focus_cache[info.code] = self._focus_for(info)
            f = self._focus_cache[info.code]
            if not f: return True
            if "lines" in f: return f["lines"][0] <= ln <= f["lines"][1]
            stmt = info.stmts.get(ln)
            targets = {n.id for t in getattr(stmt, "targets", []) for n in ast.walk(t) if isinstance(n, ast.Name)}
            return bool(targets & set(f["names"])) or (isinstance(stmt, ast.Return) and "return" in f["names"])
        if isinstance(e, str):
            m = re.match(r"L?(\d+)(?:\s*-\s*L?(\d+))?$", e.strip())
            return bool(m) and int(m.group(1)) <= ln <= int(m.group(2) or m.group(1))
        return ln in ([e] if isinstance(e, int) else list(e))

    def _expand_line(self, info, frame, ln):
        """行级追踪只能看到「一行执行完写出了什么」。像 `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` 这种一行流，
        中间的 [B,T,1] 是看不到的。这里在该行执行之前，把右侧表达式按求值顺序逐个子表达式算一遍 (no_grad、RNG 隔离、不触发 hook)，
        记下每一步的 shape。算过的子表达式会被替换成临时变量，所以总开销约等于把这一行多跑一遍。只对作者点名的行做 (有副作用的行不要 expand)。"""
        stmt = info.stmts.get(ln)
        root = copy.deepcopy(getattr(stmt, "value", None))
        if root is None or any(isinstance(n, _EXPAND_SKIP) for n in ast.walk(root)): return []
        env, out, tr = dict(frame.f_locals), [], self

        class Sub(ast.NodeTransformer):
            def visit(self, node):
                if isinstance(node, _EXPAND_LEAF): return node
                node = self.generic_visit(node)
                if node is root or not isinstance(node, _EXPAND_EVAL) or not isinstance(getattr(node, "ctx", None) or ast.Load(), ast.Load): return node
                try: val = eval(compile(ast.fix_missing_locations(ast.Expression(body=node)), "<learnkit-expand>", "eval"), frame.f_globals, env)
                except Exception: return node
                name = f"__lk{len(env)}"
                env[name] = val
                d = describe(val, tr.table, hint=info.name)
                seg = ast.get_source_segment(info.src_text, node) or ""
                if d is not None and d["k"] in ("tensor", "seq") and len(out) < 14 and not (out and out[-1]["src"] == seg):
                    out.append({"src": re.sub(r"\s+", " ", seg), "desc": d})
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

        self._expanding = True
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=[]): Sub().visit(root)
        except Exception: pass
        finally: self._expanding = False
        return out

    # ---------------------------------------------------------------- 模块级
    def _pre_hook(self, module, args, kwargs):
        if self._expanding: return
        hint = f"{self.module_names.get(id(module), '')} {type(module).__name__}"
        names = _forward_names(module)
        inputs, input_names = [], []
        for i, a in enumerate(args):
            if (d := describe(a, self.table, hint=hint)) is None: continue
            pname = names["params"][i] if i < len(names["params"]) else (f"*{names['varargs']}[{i - len(names['params'])}]" if names["varargs"] else None)
            self._name_items(a, d, names["unpack"].get(pname))
            inputs.append(d); input_names.append(pname)
        kw = {}
        for k, v in kwargs.items():
            if (d := describe(v, self.table, hint=hint)) is None or d["k"] == "none": continue
            self._name_items(v, d, names["unpack"].get(k)); kw[k] = d
        node = {"name": self.module_names.get(id(module), "?"), "cls": type(module).__name__,
                "params": sum(p.numel() for p in module.parameters()),
                "inputs": inputs, "input_names": input_names, "kwargs": kw,
                "output": None, "output_name": None, "children": []}
        if self._stack: self._stack[-1]["children"].append(node)
        else:
            self.total_roots += 1
            if len(self.roots) < self.max_roots: self.roots.append(node)
        self._stack.append(node)

    def _name_items(self, obj, desc, elem_names):
        """给 tuple/list 参数里的每个元素起名。名字来自「真正解包它的那个 forward」(如 Attention 里的 cos, sin = position_embeddings)，
        所以外层模块 (MiniMindBlock 只是把它原样往下传) 要等内层解包时再回填 —— 靠对象 id 认出是同一个 tuple。"""
        if desc.get("k") != "seq": return
        if elem_names and len(elem_names) == desc["len"]:
            for d in self._seq_descs.pop(id(obj), []) + [desc]: d["names"] = elem_names[:len(d["items"])]
            self._seq_named[id(obj)] = elem_names
        elif id(obj) in self._seq_named and len(self._seq_named[id(obj)]) == desc["len"]:
            desc["names"] = self._seq_named[id(obj)][:len(desc["items"])]
        else:
            self._seq_descs.setdefault(id(obj), []).append(desc)

    def _post_hook(self, module, args, kwargs, output):
        if self._expanding: return
        if self._stack:
            node = self._stack.pop()
            node["output"] = d = describe(output, self.table, hint=f"{node['name']} {node['cls']}")
            for r in _forward_names(module)["returns"]:
                if d and d["k"] == "seq" and isinstance(r, list) and len(r) == d["len"]: d["names"] = r[:len(d["items"])]; break
                if d and d["k"] == "tensor" and isinstance(r, str): node["output_name"] = r; break
            if not self._stack: self._seq_descs.clear(); self._seq_named.clear()  # 一次完整 forward 结束，id 可能被复用

    # ------------------------------------------------------------------ 生命周期
    def __enter__(self):
        if isinstance(self.model, nn.Module):
            for m in self.model.modules():
                self._handles.append(m.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
                self._handles.append(m.register_forward_hook(self._post_hook, with_kwargs=True))
        if self.infos:
            self._prev_trace = sys.gettrace()
            sys.settrace(self._global_tracer)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.infos: sys.settrace(self._prev_trace)
        for h in self._handles: h.remove()
        self._handles.clear()
        self._stack.clear()
        self.result = self._build()
        if not emit(self.result): self._print_text()
        return False

    def _focus_for(self, info):
        """focus：这个知识点只关心函数里的哪几行。(a, b) / 'L112-L116' = 行号范围；['xq', 'xk'] = 只看写出这些变量的行；dict = 按函数名分别给。"""
        f = self.focus
        if isinstance(f, dict): f = f.get(info.name) or f.get(info.name.split(".")[0])
        if not f: return None
        if isinstance(f, str):
            m = re.match(r"L?(\d+)\s*-\s*L?(\d+)$", f.strip())
            f = (int(m.group(1)), int(m.group(2))) if m else [f]
        f = list(f)
        if len(f) == 2 and all(isinstance(x, int) for x in f):
            last = info.first_line + len(info.source) - 1
            if f[1] < info.first_line or f[0] > last: return None
            return {"lines": [max(f[0], info.first_line), min(f[1], last)]}
        return {"names": [str(x) for x in f]}

    def _build(self):
        fns = [{"name": i.name, "file": i.file, "modified": i.modified, "first_line": i.first_line, "source": i.source,
                "calls": i.calls, "total_calls": i.total_calls, "focus": self._focus_for(i)} for i in self.infos.values()]
        return {"type": "trace", "title": self.title, "dims": self.table,
                "dim_meanings": {k: DIM_MEANINGS.get(k, "") for k in self.table},
                "modules": (self.roots or None) if self.tree else None, "total_roots": self.total_roots, "fns": fns}

    # ------------------------------------------------------------------ 纯文本回退
    def _print_text(self):
        print(f"\n━━ trace: {self.title or ''}  dims={self.table}")
        for root in self.roots[:1] if self.tree else []:
            print("── 模块调用树 (输入 → 输出)")
            self._print_node(root, 0)
        for i in self.infos.values():
            print(f"\n── {i.name}  ({i.file}:{i.first_line})  共调用 {i.total_calls} 次，下面是第 1 次")
            if not i.calls: continue
            by_line = {}
            for ev in i.calls[0]["events"]: by_line.setdefault(ev["line"], ev)
            for off, src in enumerate(i.source):
                ln = i.first_line + off
                print(f"{ln:4d} | {src}")
                ev = by_line.get(ln)
                if ev:
                    for x in ev.get("expr") or []: print(f"     |     ┆ {x['src']}  →  {fmt_desc(x['desc'])}")
                    for w in ev["writes"]: print(f"     |   {'⇒' if not ev.get('is_args') else '·'} {w['name']}: {fmt_desc(w['desc'])}" + ("   (in-place：同一个 tensor 被原地修改)" if w.get("change") == "inplace" else ""))
            if i.calls[0]["ret"]: print(f"     |   return {fmt_desc(i.calls[0]['ret'])}")

    def _print_node(self, node, depth):
        ins = ", ".join((f"{n}=" if n else "") + fmt_desc(d) for n, d in zip(node["input_names"], node["inputs"])) or "-"
        out = (f"{node['output_name']}=" if node.get("output_name") else "") + fmt_desc(node["output"])
        print(f"{'  ' * depth}{node['name'] or '(root)'} <{node['cls']}>  {ins}  →  {out}")
        kids = node["children"]
        for idx, c in enumerate(kids):
            if idx >= 1 and re.search(r"\.\d+$", c["name"]) and c["cls"] == kids[idx - 1]["cls"]:
                if idx == 1 or kids[idx - 2]["cls"] != c["cls"]:  # 这一段重复序列里第一个被省略的
                    n_same = 0
                    for k in kids[idx:]:
                        if k["cls"] != c["cls"]: break
                        n_same += 1
                    print(f"{'  ' * (depth + 1)}… 后面 {n_same} 个 {c['cls']} 结构相同，省略")
                continue
            self._print_node(c, depth + 1)


def trace(model=None, fns=(), dims=None, title=None, **kw) -> Trace:
    """追踪 tensor 维度。

    model : nn.Module，可选。给了就记录模块调用树，并自动从 model.config 推出 H/KV/D/C/I/V 等符号。
    fns   : 要逐行追踪的函数列表，如 [Attention.forward, apply_rotary_pos_emb]，也可以传类(取其 forward)。
    dims  : 你自己知道的符号，如 dict(B=3, T=7)。建议让各符号取值互不相同，标注才不会混淆。
    capture : 想拿到真实数值的局部变量名，如 ['scores']，结束后在 tr.captured 里取。
    focus : 这个知识点只关心哪几行：(112, 116) 行号范围 / ['xq', 'xk'] 变量名 / {函数名: 前两者之一}。网页上默认只展示这几行，其余折叠。
    tree  : False = 不要模块调用树 (只想看逐行追踪时用)。
    expand : 把「一行流」拆开，显示每个子表达式的 shape。True = focus 的那几行 (没给 focus 就是全部)；也可以 "L57" / "L112-L116" / [57, 60]。
             实现方式是在该行执行前把子表达式各求值一遍，所以不要用在有副作用的行上 (原地修改、KV cache 拼接、随机采样)。
    max_calls : 每个函数最多详细记录前几次调用 (默认 4)。
    """
    if model is not None and not isinstance(model, nn.Module) and not fns:
        fns, model = model, None
    if callable(fns) or isinstance(fns, str): fns = [fns]
    return Trace(model=model, fns=fns, dims=dims, title=title, **kw)
