"""从各天 lesson.md 的 frontmatter 生成 PLAN.md（14 天计划总览）。  venv/bin/python tools/make_plan.py"""
import sys
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEARN))
from server import load_lesson  # noqa: E402

PHASES = [("第一阶段 · 模型与推理（Day 1-7）", "把 model/model_minimind.py 这 287 行逐行读透", range(1, 8)),
          ("第二阶段 · 数据与训练（Day 8-10）", "从 jsonl 到 loss 下降：Pretrain / SFT / LoRA / 蒸馏", range(8, 11)),
          ("第三阶段 · 对齐与 Agent（Day 11-14）", "DPO / PPO / GRPO / Agent RL 与部署", range(11, 15))]

out = ["# MiniMind 14 天学习计划", "",
       "> 读者画像：Python 精通、LLM 原理了解、没手写过 LLM。方式：读源码 → 跑实验看每个 tensor 的维度 → 改代码看 diff → 选择题。每天 1~1.5 小时。",
       "> 打开方式：`./start.sh` → http://127.0.0.1:8877 （进度自动保存在 `progress.json`）", ""]
total_min = total_labs = total_quiz = 0
for name, note, days in PHASES:
    out += [f"## {name}", "", f"*{note}*", ""]
    for n in days:
        L = load_lesson(n)
        if not L: out += [f"### Day {n}（缺失）", ""]; continue
        m = L["meta"]
        total_min += int(m.get("minutes") or 75); total_labs += L["n_labs"]; total_quiz += L["n_quiz"]
        out += [f"### Day {n} · {m.get('title', '')}", ""]
        if m.get("subtitle"): out += [f"{m['subtitle']}", ""]
        out += [f"- ⏱ {m.get('minutes', 75)} 分钟 · 🧪 {L['n_labs']} 个实验 · ✍️ {L['n_quiz']} 道选择题"]
        if m.get("files"): out += ["- 📄 精读：" + "、".join(f"`{f}`" for f in m["files"])]
        out += ["- 🎯 学完能做到："] + [f"  - {g}" for g in (m.get("goals") or [])]
        if L.get("points"):  # 知识点格式：列出主线和每个知识点 (问题)
            if m.get("mainline"): out += [f"- 🧵 主线：{m['mainline']}"]
            out += [f"- 📍 {len(L.get('chapters') or [])} 章 · {len(L['points'])} 个小节（一次只看一个）："]
            last = object()
            for pt in L["points"]:
                if pt.get("ch") != last and pt.get("ch") is not None: last = pt["ch"]; out += [f"  - **{L['chapters'][last]['title']}**"]
                out += [f"    - {pt['num']} {pt['title']}" + ("（选学）" if pt["side"] else "")]
            out += [""]
        else:
            out += ["- 🧪 实验：" + "；".join(lab["title"] for lab in L["labs"]), ""]
out[5:5] = [f"**总量**：14 天 · 约 {total_min / 60:.0f} 小时 · {total_labs} 个可运行实验 · {total_quiz} 道选择题", ""]
(LEARN / "PLAN.md").write_text("\n".join(out), encoding="utf-8")
print(f"PLAN.md: {total_min} min, {total_labs} labs, {total_quiz} quiz")
