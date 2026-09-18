"""课程内容体检：格式、引用、行号、quiz 结构；加 --run 会把所有 lab 真跑一遍。

    venv/bin/python tools/check_lessons.py            # 只做静态检查
    venv/bin/python tools/check_lessons.py --run      # 静态检查 + 运行全部实验 (并行 4 个)
    venv/bin/python tools/check_lessons.py --run --days 4 9
"""
import re
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))
from server import parse_frontmatter, parse_lab, split_points, PY, ROOT  # noqa: E402  ROOT = minimind 源码目录

DIRECTIVE = re.compile(r"^[ \t]*\{\{(source|lab|quiz|predict|flow)(?::([^}]*))?\}\}[ \t]*$", re.M)
LOOSE = re.compile(r"\{\{(source|lab|quiz|predict|flow)[^}]*\}\}")


def check_day(d: Path):
    errs, warns = [], []
    text = (d / "lesson.md").read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    if "_error" in meta: errs.append(f"frontmatter YAML 错误: {meta['_error']}")
    for k in ("day", "title", "goals"):
        if not meta.get(k): errs.append(f"frontmatter 缺少 {k}")
    labs = {p.stem: parse_lab(p) for p in sorted((d / "labs").glob("*.py"))}
    try: quiz = json.loads((d / "quiz.json").read_text(encoding="utf-8"))
    except Exception as e: quiz = []; errs.append(f"quiz.json 解析失败: {e}")
    qids = [q.get("id") for q in quiz]
    if len(set(qids)) != len(qids): errs.append("quiz id 重复")
    for q in quiz:
        n = len(q.get("options") or [])
        ans = q.get("answer")
        if not q.get("question") or n < 2: errs.append(f"{q.get('id')}: 缺题干或选项")
        if not isinstance(ans, list) or not ans or any((not isinstance(a, int)) or a < 0 or a >= n for a in ans): errs.append(f"{q.get('id')}: answer 非法 {ans}")
        elif q.get("type", "single") == "single" and len(ans) != 1: errs.append(f"{q.get('id')}: single 题应只有一个答案")
        if not q.get("explain"): warns.append(f"{q.get('id')}: 没有解析")
    pos = [a for q in quiz if q.get("type", "single") == "single" for a in (q.get("answer") or [])[:1]]
    if len(pos) >= 5 and max(pos.count(i) for i in set(pos)) > 0.6 * len(pos): warns.append(f"正确答案位置过于集中: {pos}")

    used_labs, strict = set(), set()
    for m in DIRECTIVE.finditer(body):
        strict.add(m.group(0).strip())
        kind, arg = m.group(1), (m.group(2) or "").strip()
        if kind == "lab":
            used_labs.add(arg)
            if arg not in labs: errs.append(f"{{{{lab:{arg}}}}} 找不到 labs/{arg}.py")
        elif kind in ("quiz", "predict") and arg:
            for qid in [s.strip() for s in arg.split(",")]:
                if qid not in qids: errs.append(f"{{{{{kind}:{qid}}}}} 不存在")
        elif kind == "predict": errs.append("{{predict}} 必须写明题目 id")
        elif kind == "source":
            sm = re.match(r"^(.+?)(?:#L(\d+)(?:-L?(\d+))?)?$", arg)
            f = ROOT / sm.group(1)
            if not f.is_file(): errs.append(f"source 文件不存在: {arg}"); continue
            total = len(f.read_text(encoding="utf-8", errors="replace").split("\n"))
            a, b = int(sm.group(2) or 1), int(sm.group(3) or sm.group(2) or total)
            if a > b or b > total: errs.append(f"source 行号越界: {arg} (文件共 {total} 行)")
    for m in LOOSE.finditer(body):  # 写在句子中间的提及：前端会渲染成链接，并在该段落后自动嵌入
        if m.group(0) in strict: continue
        im = re.match(r"\{\{(lab|quiz):([^}]+)\}\}", m.group(0))
        if not im: errs.append(f"{m.group(0)} 必须独占一行"); continue
        for x in [s.strip() for s in im.group(2).split(",")]:
            if im.group(1) == "lab":
                used_labs.add(x)
                if x not in labs: errs.append(f"行内提及的 lab 不存在: {x}")
            elif x not in qids: errs.append(f"行内提及的 quiz 不存在: {x}")
    for lid in labs:
        if lid not in used_labs: warns.append(f"lab {lid} 没有在正文中引用 (会被放到文末「更多实验」)")
    for lid, lab in labs.items():
        if lab.get("error"): errs.append(f"lab {lid} 头部 YAML 解析失败 (title/timeout/tasks 全部丢失；title 以 [ 或含 ': ' 时要加引号): {str(lab['error'])[:120]}")
        elif not lab["tasks"]: warns.append(f"lab {lid} 没有 tasks")
    if meta.get("format") == "points": check_points(meta, body, labs, quiz, errs, warns)
    return meta, labs, quiz, errs, warns


def _prose_len(text):
    """知识点正文字数：去掉 MORE 折叠块、KEY、代码块、表格、指令之后剩下的字符数。"""
    text = re.sub(r"^>[ \t]*\[!(MORE|KEY)\][^\n]*\n(?:>[^\n]*(?:\n|$))*", "", text, flags=re.M)
    text = re.sub(r"(^|\n)(```|~~~)[^\n]*\n[\s\S]*?\n\2", "", text)
    text = re.sub(r"^\|.*$", "", text, flags=re.M)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # 链接只算文字部分
    return len(re.sub(r"\s+", "", text))


def check_points(meta, body, labs, quiz, errs, warns):
    """「总览 → 章 → 小节」格式的规则，见 AUTHORING.md 第 0、2 节。"""
    if not meta.get("mainline"): errs.append("format: points 需要 frontmatter 里的 mainline (一句话主线)")
    intro, chapters, points = split_points(body)
    n_intro = _prose_len(intro)
    if n_intro < 120: warns.append(f"总览只有 {n_intro} 字：要讲清今天在全局的位置、分哪几章 (200~400 字)")
    if n_intro > 520: warns.append(f"总览 {n_intro} 字，压到 400 字以内")
    if "{{flow:" not in intro: warns.append("总览里没有 {{flow: ...}} 数据流")
    if not chapters: warns.append("没有分章 (`# 章标题`)：小节太散，需要提炼成 3~5 个主题")
    elif not 3 <= len(chapters) <= 6: warns.append(f"{len(chapters)} 章，建议 3~5 章")
    for i, c in enumerate(chapters, 1):
        if not c["points"]: errs.append(f"第 {i} 章「{c['title']}」下面没有小节")
        elif len(c["points"]) > 5: warns.append(f"第 {i} 章「{c['title']}」有 {len(c['points'])} 个小节，建议 2~4 个")
        if not c["intro"].strip(): warns.append(f"第 {i} 章「{c['title']}」没有导语")
        if c["title"].rstrip().endswith(("？", "?")): warns.append(f"章标题「{c['title']}」不要用问句")
    if chapters and any(p["ch"] is None for p in points): errs.append("有小节写在第一个 `# 章` 之前")
    if not 8 <= len(points) <= 14: warns.append(f"小节 {len(points)} 个，建议 9~13 个")
    ids = [p["id"] for p in points]
    if len(set(ids)) != len(ids): errs.append(f"小节 id 重复: {ids}")
    seen_q, pred_q, core, core_with_lab = set(), set(), 0, 0
    for p in points:
        tag, text = f"小节 {p['num']} #{p['id']}", p["body"]
        if re.match(r"^point-\d+$", p["id"]): errs.append(f"「{p['title']}」没有写 {{#id}}")
        if p["title"].rstrip().endswith(("？", "?")): warns.append(f"{tag}: 标题「{p['title']}」是问句——改成陈述式名词短语，问题放到小节里面")
        n_key = len(re.findall(r"^>[ \t]*\[!KEY\]", text, flags=re.M))
        if n_key != 1: errs.append(f"{tag}: 需要恰好 1 个 [!KEY]，现在有 {n_key} 个")
        elif len(re.sub(r"\s+", "", p["key"])) > 140: warns.append(f"{tag}: KEY 太长 ({len(p['key'])} 字)，压到 2 句以内")
        srcs = re.findall(r"\{\{source:([^}]+)\}\}", text)
        if len(srcs) > 2: warns.append(f"{tag}: {len(srcs)} 个 source，建议 ≤ 1")
        for a in srcs:
            m = re.search(r"#L(\d+)(?:-L?(\d+))?", a)
            if m and int(m.group(2) or m.group(1)) - int(m.group(1)) + 1 > 22: warns.append(f"{tag}: source {a} 超过 20 行，框窄一点")
        if len(p["labs"]) > 2: warns.append(f"{tag}: {len(p['labs'])} 个 lab，建议 ≤ 1")
        if p["predict"] and not p["labs"]: errs.append(f"{tag}: 有 predict 题但没有实验——predict 的答案要靠同一小节里的实验来揭晓")
        if p["predict"] and p["labs"] and text.find("{{predict:") > text.find("{{lab:"): warns.append(f"{tag}: predict 应该放在 lab 前面 (先猜再跑)")
        nq = len(p["predict"]) + len(p["quiz"])
        if not p["side"]:
            core += 1; core_with_lab += bool(p["labs"])
            if nq == 0 and not p["labs"]: warns.append(f"{tag}: 主线小节没有任何互动 (lab / quiz)")
        if nq > 3: warns.append(f"{tag}: {nq} 道题，建议 1~2 道")
        for q in p["predict"] + p["quiz"]:
            if q in seen_q: errs.append(f"{tag}: 题目 {q} 被多个小节引用")
            seen_q.add(q)
        pred_q.update(p["predict"])
        n = _prose_len(text)
        if n < 220: warns.append(f"{tag}: 讲解只有 {n} 字——只有问题没有讲解。先把这个点讲清楚 (300~700 字)")
        if n > 850: warns.append(f"{tag}: 讲解 {n} 字 (上限 700)，多半是两个点，拆开或挪进 [!MORE]")
        for lid in p["labs"]:
            lab = labs.get(lid)
            if not lab: continue
            n_code = len([l for l in lab["code"].split("\n") if l.strip() and not l.strip().startswith("#")])
            if n_code > 42: warns.append(f"{tag}: 微实验 {lid} 有 {n_code} 行代码 (上限 30；训练/rollout 类可到 40)")
            if len(lab["tasks"]) > 2: warns.append(f"{tag}: 微实验 {lid} 有 {len(lab['tasks'])} 条 tasks (上限 2)")
    if core and core_with_lab / core < 0.7: warns.append(f"主线小节里只有 {core_with_lab}/{core} 个带实验 (要求 ≥ 70%)")
    if len(pred_q) > max(4, len(points) // 3): warns.append(f"predict 题 {len(pred_q)} 道偏多：只在结果反直觉时用，其余用普通 quiz")
    rest = [q.get("id") for q in quiz if q.get("id") not in seen_q]
    if not 2 <= len(rest) <= 12: warns.append(f"综合测验 (未被小节引用的题) 有 {len(rest)} 道，建议 3~5 道 (Day 14 的毕业测验可以到 10 道)")
    if re.search(r"^[ \t]*\{\{quiz\}\}[ \t]*$", body, flags=re.M): warns.append("章节格式不需要 {{quiz}}：未被引用的题会自动进入最后的综合测验")


def run_lab(day, lab_id, path, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([PY, str(LEARN / "runner.py"), str(path)], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
        code, tail = p.returncode, (p.stdout + p.stderr)[-1200:]
    except subprocess.TimeoutExpired:
        code, tail = "TIMEOUT", ""
    return day, lab_id, code, time.time() - t0, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--days", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs, n_err = [], 0
    for d in sorted((LEARN / "lessons").glob("day[0-9][0-9]")):
        day = int(d.name[3:])
        if args.days and day not in args.days: continue
        if not (d / "lesson.md").exists(): print(f"❌ {d.name}: 没有 lesson.md"); n_err += 1; continue
        meta, labs, quiz, errs, warns = check_day(d)
        n_err += len(errs)
        lines = len((d / "lesson.md").read_text(encoding="utf-8").split("\n"))
        if meta.get("format") == "points": _, chs, pts = split_points(parse_frontmatter((d / 'lesson.md').read_text(encoding='utf-8'))[1]); fmt = f"{len(chs)} 章 {len(pts)} 小节 · "
        else: fmt = "长文 · "
        print(f"{'❌' if errs else '✅'} {d.name}  {meta.get('title', '?')}   [{fmt}{lines} 行 · {len(labs)} labs · {len(quiz)} quiz]")
        for e in errs: print(f"     ✗ {e}")
        for w in warns: print(f"     · {w}")
        for lid, lab in labs.items(): jobs.append((day, lid, d / "labs" / f"{lid}.py", max(lab["timeout"], 60) + 30))
    if args.run:
        print(f"\n运行 {len(jobs)} 个实验 (并行 {args.workers}) …")
        with ThreadPoolExecutor(args.workers) as ex:
            for day, lid, code, dt, tail in ex.map(lambda j: run_lab(*j), jobs):
                ok = code == 0
                n_err += 0 if ok else 1
                print(f"  {'✅' if ok else '❌'} day{day:02d}/{lid:<32} exit={code}  {dt:5.1f}s")
                if not ok: print("      " + tail.replace("\n", "\n      "))
    print(f"\n{'全部通过 🎉' if n_err == 0 else f'共 {n_err} 个错误'}")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
