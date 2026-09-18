"""MiniMind 学习站本地服务。

    venv/bin/python server.py            # http://127.0.0.1:8877
    venv/bin/python server.py --port 9000

只监听 127.0.0.1。它会在子进程里执行网页发来的 python 代码（这就是它存在的意义），所以不要把它暴露到公网。

环境变量：MINIMIND_ROOT（minimind 源码目录，默认 ./minimind）、LEARN_PYTHON（跑实验用的解释器，默认 ./venv/bin/python）、
LEARN_STATE_DIR / LEARN_LESSONS_DIR（进度和课程目录，测试用）。
"""
import os
import re
import sys
import json
import time
import uuid
import shutil
import signal
import argparse
import threading
import subprocess
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory, abort

LEARN = Path(__file__).resolve().parent
sys.path.insert(0, str(LEARN / "lib"))
from minimind_root import find_minimind_root  # noqa: E402

ROOT = find_minimind_root()  # minimind 源码目录：/api/source 只能读它下面的文件，实验以它为 cwd
STATE = Path(os.environ.get("LEARN_STATE_DIR") or LEARN)  # 进度/工作区放哪；测试时指到临时目录，避免弄脏真实进度
WEB, LESSONS, RUNS, WORKSPACE = LEARN / "web", Path(os.environ.get("LEARN_LESSONS_DIR") or LEARN / "lessons"), LEARN / ".runs", STATE / "workspace"
PROGRESS = STATE / "progress.json"
API_VERSION = 3  # 前端据此判断「server.py 改过但还没重启」：web/js/app.js 里的 NEED_API 要同步
PY = os.environ.get("LEARN_PYTHON") or next((str(p) for p in (LEARN / "venv/bin/python", ROOT / "venv/bin/python") if p.exists()), sys.executable)
EDITABLE = re.compile(r"^(model|dataset|trainer|scripts)/[\w\-]+\.py$|^eval_llm\.py$")
READABLE_SUFFIX = {".py", ".json", ".md", ".txt", ".jinja", ".yaml", ".yml"}

app = Flask(__name__, static_folder=None)
app.json.ensure_ascii = False
app.json.sort_keys = False
jobs, jobs_lock, progress_lock = {}, threading.Lock(), threading.Lock()


# --------------------------------------------------------------------------- 安全
@app.before_request
def guard():
    host = (request.host or "").split(":")[0]
    if host not in ("127.0.0.1", "localhost"): abort(403)
    if request.method != "GET":
        origin = request.headers.get("Origin")
        if origin and not re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin): abort(403)
        if not request.is_json: abort(415)


def safe_repo_file(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if ROOT not in p.parents or not p.is_file() or p.suffix not in READABLE_SUFFIX: abort(404)
    if any(part in ("venv", "out", ".git", ".runs") for part in p.relative_to(ROOT).parts): abort(404)
    return p


# --------------------------------------------------------------------------- 课程解析
def parse_frontmatter(text: str):
    m = re.match(r"^---\n(.*?)\n---\n?", text, re.S)
    if not m: return {}, text
    try: meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e: meta = {"_error": str(e)}
    return meta, text[m.end():]


def parse_lab(path: Path):
    text = path.read_text(encoding="utf-8")
    lines, meta, body_start = text.split("\n"), {}, 0
    if lines and lines[0].strip() == "# ---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "# ---":
                raw = "\n".join(re.sub(r"^# ?", "", l) for l in lines[1:i])
                try: meta = yaml.safe_load(raw) or {}
                except yaml.YAMLError as e: meta = {"_error": str(e)}
                if not isinstance(meta, dict): meta = {"_error": "头部 YAML 不是 key: value 形式"}
                body_start = i + 1
                break
    code = "\n".join(lines[body_start:]).lstrip("\n")
    return {"id": path.stem, "title": meta.get("title") or path.stem, "timeout": int(meta.get("timeout") or 60),
            "sources": [s for s in (meta.get("sources") or []) if EDITABLE.match(s)], "tasks": meta.get("tasks") or [], "code": code,
            "error": meta.get("_error") if isinstance(meta, dict) else "头部 YAML 不是一个 mapping"}


POINT_H = re.compile(r"^##[ \t]+(.+?)[ \t]*(?:\{#([\w\-]+)((?:[ \t]+\.[\w\-]+)*)\})?[ \t]*$")
KEY_RE = re.compile(r"^>[ \t]*\[!KEY\][^\n]*\n((?:>[^\n]*(?:\n|$))*)", re.M)


def _ids(kind, text):
    return [x.strip() for arg in re.findall(r"\{\{%s:([^}]+)\}\}" % kind, text) for x in arg.split(",") if x.strip()]


CHAPTER_H = re.compile(r"^#[ \t]+(.+?)[ \t]*$")


def split_points(body: str):
    """章节格式 (format: points)：`# 章标题` 开始一章 (后面跟章导语)，`## 小节标题 {#id .side}` 开始一个小节；第一个 # 之前是总览。
    返回 (总览 markdown, chapters, points)。没有写 `# 章` 的课 chapters 为空，所有小节平铺。"""
    intro, chapters, points, cur, ch, fence = [], [], [], None, None, None
    for line in body.split("\n"):
        f = re.match(r"^\s*(```|~~~)", line)
        if f: fence = None if fence == f.group(1) else (fence or f.group(1))
        h = None if fence else POINT_H.match(line)
        c = None if fence or h else CHAPTER_H.match(line)
        if c:
            ch, cur = {"title": c.group(1).strip(), "_lines": [], "points": []}, None
            chapters.append(ch)
        elif h:
            cur = {"id": h.group(2) or f"point-{len(points) + 1}", "title": h.group(1).strip(), "side": ".side" in (h.group(3) or ""),
                   "ch": len(chapters) - 1 if ch else None, "_lines": []}
            if ch: ch["points"].append(cur["id"])
            cur["num"] = f"{len(chapters)}.{len(ch['points'])}" if ch else str(len(points) + 1)
            points.append(cur)
        else:
            (cur["_lines"] if cur else ch["_lines"] if ch else intro).append(line)
    for c in chapters: c["intro"] = "\n".join(c.pop("_lines")).strip("\n")
    for pt in points:
        text = "\n".join(pt.pop("_lines")).strip("\n")
        k = KEY_RE.search(text)
        pt.update(body=text, labs=_ids("lab", text), predict=_ids("predict", text), quiz=_ids("quiz", text),
                  key=re.sub(r"^>[ \t]?", "", k.group(1), flags=re.M).strip() if k else "")
    return "\n".join(intro).strip(), chapters, points


def day_dir(day: int) -> Path:
    return LESSONS / f"day{int(day):02d}"


def load_lesson(day: int, full=True):
    d = day_dir(day)
    f = d / "lesson.md"
    if not f.exists(): return None
    meta, body = parse_frontmatter(f.read_text(encoding="utf-8"))
    labs = [parse_lab(p) for p in sorted((d / "labs").glob("*.py"))] if (d / "labs").exists() else []
    try: quiz = json.loads((d / "quiz.json").read_text(encoding="utf-8")) if (d / "quiz.json").exists() else []
    except json.JSONDecodeError as e: quiz, meta["_quiz_error"] = [], str(e)
    out = {"day": int(day), "meta": meta, "n_labs": len(labs), "n_quiz": len(quiz),
           "lab_ids": [l["id"] for l in labs], "quiz_ids": [q.get("id") for q in quiz]}
    if meta.get("format") == "points":
        intro, chapters, points = split_points(body)
        out["points"] = points if full else [{k: v for k, v in pt.items() if k not in ("body", "key")} for pt in points]
        out["chapters"] = chapters
        if full: out["intro"] = intro
    if full:
        for lab in labs:
            ws = WORKSPACE / f"day{int(day):02d}" / f"{lab['id']}.json"
            lab["saved"] = json.loads(ws.read_text(encoding="utf-8")) if ws.exists() else None
        out.update(body=body, labs=labs, quiz=quiz)
    return out


@app.get("/api/plan")
def api_plan():
    days = [l for n in range(1, 32) if (l := load_lesson(n, full=False))]
    return jsonify(api=API_VERSION, days=days, progress=read_progress(), repo=str(ROOT), weights=sorted(p.name for p in (ROOT / "out").glob("*.pth")))


@app.get("/api/lesson/<int:day>")
def api_lesson(day):
    lesson = load_lesson(day)
    if lesson is None: abort(404)
    return jsonify(lesson)


@app.get("/api/source")
def api_source():
    rel = request.args.get("path", "")
    p = safe_repo_file(rel)
    if p.stat().st_size > 3_000_000: abort(413)
    lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
    start, end = int(request.args.get("start") or 1), int(request.args.get("end") or len(lines))
    start, end = max(1, start), min(len(lines), end)
    return jsonify(path=rel, abs=str(p), start=start, end=end, total=len(lines), lines=lines[start - 1:end], editable=bool(EDITABLE.match(rel)))


# --------------------------------------------------------------------------- 进度 / 工作区
def read_progress():
    with progress_lock:
        if PROGRESS.exists():
            try: return json.loads(PROGRESS.read_text(encoding="utf-8"))
            except json.JSONDecodeError: pass
        return {"days": {}, "started": None}


@app.get("/api/progress")
def api_progress_get():
    return jsonify(read_progress())


@app.post("/api/progress")
def api_progress_set():
    """body: {path: ["days","4","quiz","q1"], value: ...}  value 为 null 表示删除"""
    body = request.get_json()
    path, value = body.get("path") or [], body.get("value")
    data = read_progress()
    with progress_lock:
        node = data
        for key in path[:-1]: node = node.setdefault(str(key), {})
        if path:
            if value is None: node.pop(str(path[-1]), None)
            else: node[str(path[-1])] = value
        data["started"] = data.get("started") or time.strftime("%Y-%m-%d")
        data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = PROGRESS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, PROGRESS)
    return jsonify(data)


@app.get("/api/workspace")
def api_workspace_get():
    day, lab = int(request.args.get("day") or 0), re.sub(r"[^\w\-]", "", request.args.get("lab") or "")
    f = WORKSPACE / f"day{day:02d}" / f"{lab}.json"
    return jsonify(saved=json.loads(f.read_text(encoding="utf-8")) if f.exists() else None)


@app.post("/api/workspace")
def api_workspace():
    """保存读者对某个实验的修改: {day, lab, code, overrides: {relpath: content}}；code 和 overrides 都为空则删除。"""
    body = request.get_json()
    day, lab = int(body["day"]), re.sub(r"[^\w\-]", "", body["lab"])
    f = WORKSPACE / f"day{day:02d}" / f"{lab}.json"
    code, overrides = body.get("code"), {k: v for k, v in (body.get("overrides") or {}).items() if EDITABLE.match(k)}
    if code is None and not overrides:
        f.unlink(missing_ok=True)
        return jsonify(ok=True, deleted=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"code": code, "overrides": overrides, "ts": time.time()}, ensure_ascii=False), encoding="utf-8")
    return jsonify(ok=True)


# --------------------------------------------------------------------------- 执行代码
def kill_job(job):
    proc = job["proc"]
    if proc.poll() is None:
        try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError): proc.kill()


def pump(job):
    proc = job["proc"]
    for chunk in iter(lambda: proc.stdout.read1(65536), b""):
        job["log"] += chunk.decode("utf-8", errors="replace")
    proc.wait()
    job["exit"], job["ended"] = proc.returncode, time.time()


def watchdog(job):
    while job["exit"] is None:
        if time.time() - job["started"] > job["timeout"]:
            job["timed_out"] = True
            kill_job(job)
            return
        time.sleep(0.5)


def cleanup_runs(keep=30):
    if not RUNS.exists(): return
    dirs = sorted((d for d in RUNS.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    active = {j["dir"] for j in jobs.values() if j["exit"] is None}
    for d in dirs[:-keep]:
        if d not in active: shutil.rmtree(d, ignore_errors=True)


@app.post("/api/run")
def api_run():
    body = request.get_json()
    code = body.get("code") or ""
    timeout = max(5, min(int(body.get("timeout") or 60), 1800))
    job_id = time.strftime("%H%M%S-") + uuid.uuid4().hex[:6]
    d = RUNS / job_id
    (d / "overrides").mkdir(parents=True)
    (d / "lab.py").write_text(code, encoding="utf-8")
    events = d / "events.jsonl"
    events.touch()
    mapping = {}
    for rel, content in (body.get("overrides") or {}).items():
        if not EDITABLE.match(rel): continue
        target = d / "overrides" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        mapping[rel] = str(target)
    (d / "overrides.json").write_text(json.dumps(mapping), encoding="utf-8")
    env = dict(os.environ, LEARNKIT_EVENTS=str(events), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
    proc = subprocess.Popen([PY, "-u", str(LEARN / "runner.py"), str(d / "lab.py"), "--overrides", str(d / "overrides.json")],
                            cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    job = {"id": job_id, "proc": proc, "dir": d, "events": events, "log": "", "exit": None, "started": time.time(),
           "ended": None, "timeout": timeout, "timed_out": False}
    with jobs_lock:
        jobs[job_id] = job
        for old in [k for k, j in jobs.items() if j["exit"] is not None and time.time() - (j["ended"] or 0) > 600]: jobs.pop(old)
    threading.Thread(target=pump, args=(job,), daemon=True).start()
    threading.Thread(target=watchdog, args=(job,), daemon=True).start()
    cleanup_runs()
    return jsonify(job=job_id)


@app.get("/api/job/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if job is None: abort(404)
    ev_off, log_off = int(request.args.get("ev") or 0), int(request.args.get("log") or 0)
    done = job["exit"] is not None  # 先读状态再读文件，保证 done=True 时事件已经写完
    with open(job["events"], "rb") as f:
        f.seek(ev_off)
        raw = f.read()
    cut = raw.rfind(b"\n") + 1  # 只取完整的行
    events = []
    for line in raw[:cut].split(b"\n"):
        if not line.strip(): continue
        try: events.append(json.loads(line))
        except json.JSONDecodeError: events.append({"type": "stdout", "text": line.decode("utf-8", "replace") + "\n"})
    log = job["log"]
    return jsonify(events=events, ev=ev_off + cut, log=log[log_off:], log_off=len(log), done=done, exit=job["exit"],
                   timed_out=job["timed_out"], timeout=job["timeout"], elapsed=round((job["ended"] or time.time()) - job["started"], 2))


@app.post("/api/job/<job_id>/stop")
def api_job_stop(job_id):
    job = jobs.get(job_id)
    if job is None: abort(404)
    kill_job(job)
    return jsonify(ok=True)


# --------------------------------------------------------------------------- 静态文件
@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(WEB, path)


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8877)
    args = ap.parse_args()
    print(f"\n  MiniMind 学习站:  http://127.0.0.1:{args.port}\n  python: {PY}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
