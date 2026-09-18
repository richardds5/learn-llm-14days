// app.js —— 路由、首页(学习计划)、课程页、选择题
import { el, api, renderMarkdown, mdToHtml, codeLines, openSource } from './util.js';
import { Lab } from './lab.js';
import { renderMap } from './map.js';

const PHASES = [
  { name: '第一阶段 · 模型与推理', note: '把 model_minimind.py 这 287 行逐行读透', days: [1, 2, 3, 4, 5, 6, 7] },
  { name: '第二阶段 · 数据与训练', note: '从 jsonl 到 loss 下降：Pretrain / SFT / LoRA / 蒸馏', days: [8, 9, 10] },
  { name: '第三阶段 · 对齐与 Agent', note: 'DPO / PPO / GRPO / Agent RL 与部署', days: [11, 12, 13, 14] },
];
const state = { plan: [], progress: { days: {} }, labs: [], weights: [] };
const main = document.getElementById('main');
const dayProg = (d) => ((state.progress.days ||= {})[String(d)] ||= {});

async function setProgress(day, path, value) {
  let node = dayProg(day); path.slice(0, -1).forEach((k) => (node = node[k] ||= {})); node[path[path.length - 1]] = value;
  renderSidebar();
  try { await api('/api/progress', { path: ['days', String(day), ...path], value }); } catch (e) { console.warn(e); }
}

function dayStats(d) {
  const p = dayProg(d.day), labsRun = d.lab_ids.filter((id) => p.labs?.[id]?.runs).length, answered = d.quiz_ids.filter((id) => p.quiz?.[id]), right = answered.filter((id) => p.quiz[id].correct).length;
  const total = d.n_labs + d.n_quiz, got = labsRun + answered.length;
  return { labsRun, answered: answered.length, right, pct: p.done ? 100 : total ? Math.round((got / total) * 100) : 0, done: !!p.done, started: got > 0 };
}

/** 一个知识点的完成情况：它带的实验都跑过、题都答过 = 完成；没有任何互动的知识点，看过就算完成 */
function pointStatus(day, pt) {
  const p = dayProg(day), qs = [...(pt.predict || []), ...(pt.quiz || [])];
  const got = (pt.labs || []).filter((id) => p.labs?.[id]?.runs).length + qs.filter((id) => p.quiz?.[id]).length, total = (pt.labs || []).length + qs.length, seen = !!p.points?.[pt.id];
  return { done: total ? got === total : seen, started: got > 0 || seen, got, total };
}
const pointHref = (day, id) => `#/day/${day}/p/${id}`;
const CN_NUM = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'];
const NEED_API = 3;  // 和 server.py 的 API_VERSION 同步

// ---------------------------------------------------------------- sidebar
function renderSidebar() {
  const nav = document.getElementById('daynav'); nav.innerHTML = '';
  const hm = location.hash.match(/^#\/day\/(\d+)(?:\/p\/([\w\-]+))?/), cur = hm?.[1], curPt = hm?.[2] || state.lessonView?.cur;
  PHASES.forEach((ph) => {
    nav.append(el('div', { class: 'phase' }, ph.name.split(' · ')[1]));
    ph.days.forEach((n) => { const d = state.plan.find((x) => x.day === n); if (!d) { nav.append(el('div', { class: 'daylink', style: 'opacity:.4' }, [el('span', { class: 'daynum' }, String(n)), el('span', {}, '（编写中）'), el('span')])); return; }
      const s = dayStats(d); nav.append(el('a', { class: 'daylink' + (String(n) === cur ? ' active' : '') + (s.done ? ' done' : s.started ? ' started' : ''), href: `#/day/${n}` }, [el('span', { class: 'daynum' }, s.done ? '✓' : String(n)), el('span', {}, d.meta.title || `Day ${n}`), el('span', { class: 'pct' }, s.done || !s.started ? '' : s.pct + '%')]));
      if (String(n) === cur && d.points?.length) { // 当天的「板书大纲」：一行一个知识点
        const box = el('div', { class: 'ptnav' }); let lastCh = null;
        d.points.forEach((pt) => {
          if (pt.ch != null && pt.ch !== lastCh) { lastCh = pt.ch; box.append(el('div', { class: 'ptch' }, `${CN_NUM[pt.ch] || pt.ch + 1}、${d.chapters?.[pt.ch]?.title || ''}`)); }
          const st = pointStatus(n, pt); box.append(el('a', { class: 'ptlink' + (pt.id === curPt ? ' cur' : '') + (st.done ? ' done' : st.started ? ' started' : '') + (pt.side ? ' side' : ''), href: pointHref(n, pt.id), title: pt.title }, [el('i', {}, st.done ? '✓' : pt.num), el('span', {}, pt.title)]));
        });
        nav.append(box);
      } });
  });
  const done = state.plan.filter((d) => dayProg(d.day).done).length;
  document.getElementById('overall-text').textContent = `${done} / ${state.plan.length || 14} 天`;
  document.getElementById('overall-bar').style.width = (state.plan.length ? (done / state.plan.length) * 100 : 0) + '%';
  document.querySelectorAll('.side-link[data-route]').forEach((a) => a.classList.toggle('active', location.hash === `#/${a.dataset.route}`));
}

// ---------------------------------------------------------------- 计划日期
const DAY_MS = 86400000, WEEK = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
const parseDate = (s) => { const [y, m, d] = s.split('-').map(Number); return new Date(y, m - 1, d); };
const fmtDate = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
const today0 = () => { const n = new Date(); return new Date(n.getFullYear(), n.getMonth(), n.getDate()); };
function schedule() { // → {start, dateOf(dayN), todayN}  todayN: 按计划今天该学第几天 (可能 <1 或 >14)
  const s = state.progress.plan_start; if (!s) return null;
  const start = parseDate(s);
  return { start, dateOf: (n) => new Date(start.getTime() + (n - 1) * DAY_MS), todayN: Math.round((today0() - start) / DAY_MS) + 1 };
}
async function setPlanStart(v) { state.progress.plan_start = v; try { await api('/api/progress', { path: ['plan_start'], value: v }); } catch (e) { console.warn(e); } route(); }

function renderPlanBar(doneDays) {
  const sch = schedule(), total = state.plan.length || 14, box = el('div', { class: 'planbar' });
  const input = el('input', { type: 'date', value: state.progress.plan_start || fmtDate(today0()) });
  if (!sch) {
    const btn = el('button', { class: 'btn primary', type: 'button' }, '📅 从这一天开始我的 14 天计划'); btn.addEventListener('click', () => setPlanStart(input.value));
    box.append(el('span', {}, '还没有设定开始日期：'), input, btn); return box;
  }
  const end = sch.dateOf(total), n = sch.todayN, expected = Math.max(0, Math.min(total, n - 1)), lag = expected - doneDays;
  let msg, cls = 'ok';
  if (doneDays >= total) msg = '🎓 14 天全部完成！';
  else if (n < 1) msg = `计划 ${-n + 1} 天后开始`;
  else if (n > total) { msg = `计划已于 ${fmtDate(end)} 到期，还剩 ${total - doneDays} 天的内容`; cls = 'bad'; }
  else if (lag > 0) { msg = `今天是计划的第 ${n} 天 · 按计划应已完成 ${expected} 天，你完成了 ${doneDays} 天 · 落后 ${lag} 天`; cls = lag >= 2 ? 'bad' : 'warn'; }
  else msg = `今天是计划的第 ${n} 天 · 已完成 ${doneDays} 天 · ${lag < 0 ? `超前 ${-lag} 天 🚀` : '进度正常 ✓'}`;
  input.addEventListener('change', () => input.value && setPlanStart(input.value));
  const clear = el('button', { class: 'btn ghost', type: 'button', title: '清除计划日期' }, '✕'); clear.addEventListener('click', () => setPlanStart(null));
  box.append(el('span', { class: 'plan-msg ' + cls }, msg), el('span', { style: 'flex:1' }), el('span', { class: 'muted' }, `${fmtDate(sch.start)} → ${fmtDate(end)} · 开始日期`), input, clear);
  return box;
}

// ---------------------------------------------------------------- dashboard
function renderHome() {
  const page = el('div', { class: 'page' });
  const stats = state.plan.map(dayStats), doneDays = stats.filter((s) => s.done).length;
  const labsRun = stats.reduce((a, s) => a + s.labsRun, 0), labsAll = state.plan.reduce((a, d) => a + d.n_labs, 0);
  const ans = stats.reduce((a, s) => a + s.answered, 0), right = stats.reduce((a, s) => a + s.right, 0), quizAll = state.plan.reduce((a, d) => a + d.n_quiz, 0);
  const next = state.plan.find((d) => !dayProg(d.day).done) || state.plan[0];
  page.append(el('div', { class: 'hero' }, [
    el('div', {}, [el('h1', {}, '14 天，读透 MiniMind 的每一行'), el('p', {}, '不讲你已经懂的原理，只讲实现：每天精读一段源码，在网页里直接运行仓库的真实代码，逐行看到每个 tensor 的维度，改一改、看 diff、再跑一遍，最后用选择题检验。每天 1~1.5 小时。')]),
    next ? el('a', { class: 'btn primary', href: `#/day/${next.day}`, style: 'padding:10px 20px;font-size:15px' }, doneDays || stats.some((s) => s.started) ? `继续学习 · Day ${next.day} →` : '从 Day 1 开始 →') : null,
  ]));
  page.append(el('div', { class: 'stats' }, [
    el('div', { class: 'stat' }, [el('b', {}, `${doneDays}/${state.plan.length}`), el('span', {}, '已完成天数')]),
    el('div', { class: 'stat' }, [el('b', {}, `${labsRun}/${labsAll}`), el('span', {}, '跑过的实验')]),
    el('div', { class: 'stat' }, [el('b', {}, `${ans}/${quizAll}`), el('span', {}, '做过的选择题')]),
    el('div', { class: 'stat' }, [el('b', {}, ans ? Math.round((right / ans) * 100) + '%' : '—'), el('span', {}, '答题正确率')]),
  ]));
  page.append(renderPlanBar(doneDays));
  const sch = schedule();
  PHASES.forEach((ph) => {
    page.append(el('div', { class: 'phase-title' }, [el('h2', {}, ph.name), el('span', {}, ph.note)]));
    const cards = el('div', { class: 'cards' });
    ph.days.forEach((n) => {
      const d = state.plan.find((x) => x.day === n);
      if (!d) { cards.append(el('div', { class: 'card missing' }, [el('div', { class: 'card-top' }, `DAY ${String(n).padStart(2, '0')}`), el('h3', {}, '内容编写中…')])); return; }
      const s = dayStats(d), m = d.meta;
      cards.append(el('a', { class: 'card', href: `#/day/${n}` }, [
        el('div', { class: 'card-top' }, [el('span', {}, `DAY ${String(n).padStart(2, '0')} · ${m.minutes || 75} 分钟` + (sch ? ` · ${sch.dateOf(n).getMonth() + 1}/${sch.dateOf(n).getDate()} ${WEEK[sch.dateOf(n).getDay()]}` : '')),
          s.done ? el('span', { class: 'badge ok' }, '✓ 已完成') : sch && sch.todayN === n ? el('span', { class: 'badge today' }, '📍 今天') : sch && sch.todayN > n ? el('span', { class: 'badge late' }, `逾期 ${sch.todayN - n} 天`) : s.started ? el('span', { class: 'badge acc' }, '进行中') : null]),
        el('h3', {}, m.title || ''), m.subtitle ? el('p', {}, m.subtitle) : null,
        el('ul', {}, (m.goals || []).slice(0, 3).map((g) => el('li', { html: mdToHtml(String(g)).replace(/^<p>|<\/p>\s*$/g, '') }))),
        el('div', { class: 'card-foot' }, [d.points ? el('span', { title: '知识点' }, `📍 ${d.points.filter((pt) => pointStatus(n, pt).done).length}/${d.points.length}`) : null, el('span', {}, `🧪 ${s.labsRun}/${d.n_labs}`), el('span', {}, `✍️ ${s.answered}/${d.n_quiz}`), el('div', { class: 'bar' }, [el('i', { style: `width:${s.pct}%` })])]),
      ]));
    });
    page.append(cards);
  });
  page.append(el('p', { class: 'muted', style: 'margin-top:36px' }, `仓库：${state.repo || ''} · 已下载权重：${state.weights.join('、') || '无'} · 进度保存在 progress.json，你改过的代码保存在 workspace/`));
  return page;
}

// ---------------------------------------------------------------- quiz
/** opts.predict = {hasRun(): 同一知识点里的实验是否已经跑成功过}。predict 题：选完只记下猜测，实验跑完 (box._reveal) 才揭晓。 */
function renderQuiz(day, q, index, opts = {}) {
  const saved = dayProg(day).quiz?.[q.id], multi = q.type === 'multi', answer = new Set(q.answer), predict = opts.predict;
  const box = el('div', { class: 'quiz' + (predict ? ' predict' : ''), id: `quiz-${q.id}` });
  const picked = new Set(saved?.picked || []); let graded = !!saved, revealed = !predict || !!saved?.revealed || (graded && predict.hasRun());
  const letters = () => [...picked].sort().map((i) => 'ABCDEFGH'[i]).join('、');
  const optsEl = el('div'), after = el('div');
  const paint = () => {
    optsEl.innerHTML = ''; after.innerHTML = '';
    q.options.forEach((o, i) => {
      let cls = 'opt'; if (graded && revealed) { if (answer.has(i)) cls += ' right'; else if (picked.has(i)) cls += ' wrong'; } else if (picked.has(i)) cls += ' sel';
      const b = el('button', { class: cls, type: 'button', disabled: graded && revealed }, [el('span', { class: 'k' }, 'ABCDEFGH'[i]), el('span', { html: mdToHtml(String(o)) })]);
      b.addEventListener('click', () => { if (multi) { picked.has(i) ? picked.delete(i) : picked.add(i); graded = false; paint(); } else { picked.clear(); picked.add(i); grade(); } });
      optsEl.append(b);
    });
    if (!graded && multi) { const sub = el('button', { class: 'btn primary', type: 'button', disabled: !picked.size }, '提交'); sub.addEventListener('click', grade); after.append(el('div', { class: 'quiz-actions' }, [sub, el('span', { class: 'muted' }, '多选题')])); }
    if (graded && !revealed) { // 已下注，等实验揭晓
      const peek = el('button', { class: 'btn ghost', type: 'button' }, '不跑了，直接看答案'); peek.addEventListener('click', () => box._reveal());
      after.append(el('div', { class: 'bet' }, [el('span', {}, `🎲 已记下你的猜测：${letters()}（还可以改）。现在运行下面的实验 —— 跑完自动揭晓。`), peek]));
    }
    if (graded && revealed) {
      const ok = picked.size === answer.size && [...picked].every((i) => answer.has(i));
      const ex = renderMarkdown(q.explain || ''); ex.className = 'explain';
      ex.prepend(el('div', { class: 'verdict ' + (ok ? 'ok' : 'bad') }, ok ? '✓ 答对了' : `✗ 正确答案：${[...answer].sort().map((i) => 'ABCDEFGH'[i]).join('、')}`));
      if (q.ref) { const m = String(q.ref).match(/^(.+?)#L(\d+)(?:-L?(\d+))?$/); if (m) { const a = el('a', { class: 'srcref', href: '#' }, `→ ${m[1]}:${m[2]}${m[3] ? '-' + m[3] : ''}`); a.addEventListener('click', (e) => { e.preventDefault(); openSource(m[1], +m[2], +(m[3] || m[2])); }); ex.append(el('p', {}, [a])); } }
      const redo = el('button', { class: 'btn ghost', type: 'button' }, '↺ 重做'); redo.addEventListener('click', () => { graded = false; picked.clear(); setProgress(day, ['quiz', q.id], null); delete dayProg(day).quiz[q.id]; paint(); opts.onChange?.(); });
      after.append(ex, el('div', { class: 'quiz-actions' }, [redo]));
    }
  };
  const save = () => setProgress(day, ['quiz', q.id], { picked: [...picked], correct: picked.size === answer.size && [...picked].every((i) => answer.has(i)), revealed: predict ? revealed : undefined, ts: Date.now() });
  const grade = () => { graded = true; revealed = !predict || predict.hasRun(); save(); paint(); opts.onChange?.(); };
  box._reveal = () => { if (!graded || revealed) return; revealed = true; save(); paint(); box.classList.add('flash'); opts.onChange?.(); };
  const qEl = renderMarkdown(q.question || ''); qEl.className = 'q';
  box.append(el('div', { class: 'quiz-top' }, [el('span', {}, (predict ? '🤔 先猜，再跑' : `选择题 ${index}`) + (multi ? ' · 多选' : '')), el('span', {}, q.id)]), qEl, optsEl, after); paint();
  return box;
}

// ---------------------------------------------------------------- lesson
async function renderSourceBlock(holder, arg) {
  const m = arg.match(/^(.+?)(?:#L(\d+)(?:-L?(\d+))?)?$/), path = m[1], a = m[2] ? +m[2] : null, b = m[3] ? +m[3] : a;
  try {
    const s = await api(`/api/source?path=${encodeURIComponent(path)}` + (a ? `&start=${a}&end=${b}` : ''));
    const open = el('a', { href: '#' }, '查看完整文件'); open.addEventListener('click', (e) => { e.preventDefault(); openSource(path, s.start, s.end); });
    holder.replaceWith(el('div', { class: 'srcblock' }, [el('div', { class: 'srcblock-head' }, [el('span', {}, `📄 ${path}` + (a ? `  ·  L${s.start}–L${s.end}` : '')), el('span', {}, [open, '  ·  ', el('a', { href: `vscode://file/${s.abs}:${s.start}` }, 'VS Code ↗')])]), codeLines(s.lines, s.start, null, path)]));
  } catch (e) { holder.replaceWith(el('div', { class: 'callout warning' }, `源码片段加载失败：${arg}`)); }
}

/** 把一段 markdown (整篇长文，或一个知识点) 渲染出来，并把里面的 {{source}} {{lab}} {{quiz}} {{predict}} 指令换成真正的组件。
 *  ctx: {used:Set 已经出过的题, qn 题号计数, micro 微实验样式, labs:[] 本段创建的 Lab, quizEls:[] 本段的题, labIds 本段的实验 id, onChange() 进度变了} */
function renderBody(L, day, md, ctx) {
  md = md.replace(/^(#{1,4}[ \t].*?)[ \t]*\{#[\w\-]+(?:[ \t]+\.[\w\-]+)*\}[ \t]*$/gm, '$1');  // 小节 id 只给程序用，永远不显示
  let body = md.replace(/^[ \t]*\{\{(source|lab|quiz|predict|flow)(?::([^}]*))?\}\}[ \t]*$/gm, (_, kind, arg) => `\n\n<div data-directive="${kind}" data-arg="${(arg || '').trim().replace(/"/g, '&quot;')}"></div>\n\n`);
  // 写在句子中间的 {{lab:x}} / {{quiz:q1}}：渲染成跳转链接，并在首次提及的段落后面自动嵌入
  body = body.replace(/`?\{\{(lab|quiz):([^}]+)\}\}`?/g, (_, kind, arg) => `<a class="mention" data-kind="${kind}" data-arg="${arg.trim()}"></a>`);
  const prose = renderMarkdown(body); prose.className = 'prose wide';
  const blockHas = (kind, id) => [...prose.querySelectorAll(`[data-directive="${kind}"]`)].some((h) => h.dataset.arg.split(',').map((s) => s.trim()).includes(id));
  prose.querySelectorAll('a.mention').forEach((m) => {
    const kind = m.dataset.kind, ids = m.dataset.arg.split(',').map((s) => s.trim());
    const lab = kind === 'lab' ? L.labs.find((x) => x.id === ids[0]) : null;
    m.textContent = kind === 'lab' ? `🧪 实验「${lab?.title || ids[0]}」` : `✍️ 下面的选择题`;
    m.href = '#'; m.addEventListener('click', (e) => { e.preventDefault(); document.getElementById(`${kind}-${ids[0]}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); });
    const missing = ids.filter((id) => !blockHas(kind, id)); if (!missing.length) return;
    let blk = m; while (blk.parentElement && blk.parentElement !== prose) blk = blk.parentElement;
    let after = blk; while (after.nextElementSibling?.matches('[data-directive][data-auto]')) after = after.nextElementSibling;
    after.after(el('div', { 'data-directive': kind, 'data-arg': missing.join(','), 'data-auto': '1' }));
  });
  [...prose.children].forEach((c) => { if (!c.matches('[data-directive]')) c.style.maxWidth = '860px'; });
  prose.querySelectorAll('a[href^="#"]').forEach((a) => { const id = a.getAttribute('href').slice(1); if ((L.points || []).some((pt) => pt.id === id)) a.setAttribute('href', pointHref(day, id)); });  // [见 params](#params) → 跳到那个知识点
  const p = dayProg(day), onProgress = (path, value) => { setProgress(day, path, value); ctx.onChange?.(); };
  const hasRun = () => (ctx.labIds || []).some((id) => dayProg(day).labs?.[id]?.ok);
  prose.querySelectorAll('[data-directive]').forEach((h) => {
    const kind = h.dataset.directive, arg = h.dataset.arg;
    if (kind === 'source') renderSourceBlock(h, arg);
    else if (kind === 'flow') h.replaceWith(renderFlow(arg));
    else if (kind === 'lab') { const lab = L.labs.find((x) => x.id === arg); if (!lab) return h.replaceWith(el('div', { class: 'callout warning' }, `找不到实验 ${arg}`)); const w = new Lab({ day, lab, progress: p, onProgress, micro: !!ctx.micro, onRun: (ok) => { if (ok) ctx.quizEls.forEach((q) => q._reveal?.()); ctx.onChange?.(); } }); state.labs.push(w); ctx.labs.push(w); h.replaceWith(w.root); }
    else if (kind === 'quiz' || kind === 'predict') {
      const later = new Set([...prose.querySelectorAll('[data-directive="quiz"],[data-directive="predict"]')].filter((x) => x !== h).flatMap((x) => x.dataset.arg.split(',').map((s) => s.trim())).filter(Boolean));
      const ids = arg ? arg.split(',').map((s) => s.trim()) : L.quiz.filter((q) => !ctx.used.has(q.id) && !later.has(q.id)).map((q) => q.id); const frag = document.createDocumentFragment();
      ids.forEach((id) => { const q = L.quiz.find((x) => x.id === id); if (q && !ctx.used.has(id)) { ctx.used.add(id); const box = renderQuiz(day, q, ++ctx.qn, { predict: kind === 'predict' && (ctx.labIds || []).length ? { hasRun } : null, onChange: ctx.onChange }); ctx.quizEls.push(box); frag.append(box); } });
      h.replaceWith(frag);
    }
  });
  return prose;
}
/** {{flow: input_ids [B,T] | *embed_tokens* | hidden_states [B,T,C]}} → 一条横向数据流；*星号* = 今天的主角 */
function renderFlow(arg) {
  const box = el('div', { class: 'flow' });
  arg.split('|').map((x) => x.trim()).filter(Boolean).forEach((part, i) => {
    if (i) box.append(el('span', { class: 'flow-arrow' }, '→'));
    const hot = /^\*.*\*$/.test(part), txt = hot ? part.slice(1, -1) : part, m = txt.match(/^(.*?)\s*(\[[^\]]+\])\s*$/);
    box.append(el('span', { class: 'flow-node' + (hot ? ' hot' : '') + (m && !hot ? ' data' : '') }, m ? [el('b', {}, m[1]), el('code', {}, m[2])] : [el('b', {}, txt)]));
  });
  return box;
}
const newCtx = (extra) => ({ used: new Set(), qn: 0, labs: [], quizEls: [], ...extra });

function lessonFoot(day) {
  const p = dayProg(day);
  const notes = el('textarea', { placeholder: '我的笔记（自动保存）…' }); notes.value = p.notes || ''; let nt; notes.addEventListener('input', () => { clearTimeout(nt); nt = setTimeout(() => setProgress(day, ['notes'], notes.value), 700); });
  const doneBtn = el('button', { class: 'btn ' + (p.done ? 'on' : 'primary'), type: 'button', style: 'padding:8px 18px;font-size:14px' }, p.done ? '✓ 今天已完成（点击取消）' : '✓ 标记今天完成');
  doneBtn.addEventListener('click', async () => { await setProgress(day, ['done'], !dayProg(day).done || null); const on = !!dayProg(day).done; doneBtn.className = 'btn ' + (on ? 'on' : 'primary'); doneBtn.textContent = on ? '✓ 今天已完成（点击取消）' : '✓ 标记今天完成'; });
  const prev = state.plan.find((d) => d.day === day - 1), next = state.plan.find((d) => d.day === day + 1);
  return el('div', { class: 'lesson-foot' }, [el('div', {}, [el('b', {}, '📝 我的笔记'), notes]), el('div', { class: 'nav-row' }, [prev ? el('a', { class: 'btn', href: `#/day/${day - 1}` }, `← Day ${day - 1} ${prev.meta.title || ''}`) : el('span'), doneBtn, next ? el('a', { class: 'btn', href: `#/day/${day + 1}` }, `Day ${day + 1} ${next.meta.title || ''} →`) : el('a', { class: 'btn', href: '#/' }, '回到首页')])]);
}

async function renderLesson(day, pid) {
  const L = await api(`/api/lesson/${day}`), m = L.meta, page = el('div', { class: 'page' });
  state.labs.forEach((l) => l.dispose()); state.labs = [];
  const planDay = state.plan.find((d) => d.day === day); if (planDay && L.points) planDay.points = L.points;  // 保持 sidebar 大纲最新
  const head = el('div', { class: 'lesson-head' }, [el('div', { class: 'kicker' }, `DAY ${String(day).padStart(2, '0')} / 14`), el('h1', {}, m.title || `Day ${day}`), m.subtitle ? el('p', { class: 'sub' }, m.subtitle) : null]);
  const metaRow = el('div', { class: 'meta-row' }, [el('span', { class: 'badge' }, `⏱ ${m.minutes || 75} 分钟`), L.points ? el('span', { class: 'badge' }, `📍 ${(L.chapters || []).length ? L.chapters.length + ' 章 · ' : ''}${L.points.length} 个小节`) : null, el('span', { class: 'badge' }, `🧪 ${L.labs.length} 个实验`), el('span', { class: 'badge' }, `✍️ ${L.quiz.length} 道题`)]);
  (m.files || []).forEach((f) => { const mm = String(f).match(/^(.+?)(?:#L(\d+)(?:-L?(\d+))?)?$/); const a = el('a', { class: 'srcref', href: '#' }, String(f).replace('#L', ':').replace('-L', '-')); a.addEventListener('click', (e) => { e.preventDefault(); openSource(mm[1], mm[2] ? +mm[2] : null, mm[3] ? +mm[3] : null); }); metaRow.append(a); });
  head.append(metaRow); page.append(head);
  if (L.points) return renderPointsLesson(L, day, pid, page, head);

  // ---- 第一版：整篇长文
  if (m.goals?.length) page.append(el('div', { class: 'goals' }, [el('b', {}, '今天学完，你应该能…'), el('ul', {}, m.goals.map((g) => el('li', { html: mdToHtml(String(g)).replace(/^<p>|<\/p>\s*$/g, '') })))]));
  const ctx = newCtx(), prose = renderBody(L, day, L.body, ctx), p = dayProg(day), onProgress = (path, value) => setProgress(day, path, value); page.append(prose);
  const leftover = L.labs.filter((lab) => !ctx.labs.some((w) => w.lab.id === lab.id));
  if (leftover.length) { prose.append(el('h2', {}, '更多实验')); leftover.forEach((lab) => { const w = new Lab({ day, lab, progress: p, onProgress }); state.labs.push(w); prose.append(w.root); }); }
  const restQ = L.quiz.filter((q) => !ctx.used.has(q.id)); if (restQ.length) { prose.append(el('h2', {}, '今日测验')); restQ.forEach((q) => prose.append(renderQuiz(day, q, ++ctx.qn))); }
  page.append(lessonFoot(day));
  return page;
}

// ---------------------------------------------------------------- 章节步进模式 (format: points)
/** 总览 → 章 → 小节，一次只显示一个小节。卡片序列：总览 intro → 各小节 → 小结 recap。每张卡片第一次显示时才渲染，之后缓存 (实验输出不会丢)。 */
function renderPointsLesson(L, day, pid, page, head) {
  const m = L.meta, pts = L.points, chs = L.chapters || [], cards = ['intro', ...pts.map((p) => p.id), 'recap'], cache = new Map();
  if (m.mainline) head.append(el('div', { class: 'mainline' }, [el('b', {}, '🧵 今日主线'), el('span', { html: mdToHtml(String(m.mainline)).replace(/^<p>|<\/p>\s*$/g, '') })]));
  const rail = el('div', { class: 'rail' }), stage = el('div', { class: 'stage' }), foot = el('div', { class: 'stepfoot' });
  page.append(rail, stage, foot);
  const view = { day, page, cur: null }; state.lessonView = view;
  const byId = (id) => pts.find((p) => p.id === id), chName = (i) => `${CN_NUM[i] || i + 1}、${chs[i]?.title || ''}`;
  const groups = chs.length ? chs.map((c, i) => ({ ch: i, pts: pts.filter((p) => p.ch === i) })) : [{ ch: null, pts }];
  const firstTodo = () => (pts.find((p) => !p.side && !pointStatus(day, p).done) || pts[0]).id;
  const stCls = (p) => { const st = pointStatus(day, p); return (st.done ? ' done' : st.started ? ' started' : '') + (p.side ? ' side' : ''); };

  const paintRail = () => {
    rail.innerHTML = ''; const segs = el('div', { class: 'segs' });
    segs.append(el('a', { class: 'seg cap' + (view.cur === 'intro' ? ' cur' : ''), href: pointHref(day, 'intro'), title: '总览：今天讲什么、分哪几章' }, '总览'));
    groups.forEach((g) => {
      const grp = el('span', { class: 'seg-group' + (g.pts.some((p) => p.id === view.cur) ? ' cur' : '') });
      if (g.ch != null) grp.append(el('span', { class: 'seg-ch', title: chs[g.ch].title }, CN_NUM[g.ch] || g.ch + 1));
      g.pts.forEach((p) => grp.append(el('a', { class: 'seg' + (p.id === view.cur ? ' cur' : '') + stCls(p), href: pointHref(day, p.id), title: `${p.num} ${p.title}${p.side ? '（选学）' : ''}` }, pointStatus(day, p).done ? '✓' : p.num.split('.').pop())));
      segs.append(grp);
    });
    segs.append(el('a', { class: 'seg cap' + (view.cur === 'recap' ? ' cur' : ''), href: pointHref(day, 'recap'), title: '今日小结 + 综合测验' }, '小结'));
    const core = pts.filter((p) => !p.side), done = core.filter((p) => pointStatus(day, p).done).length;
    rail.append(segs, el('span', { class: 'rail-text' }, `主线 ${done}/${core.length}` + (pts.length > core.length ? ` · 选学 ${pts.filter((p) => p.side && pointStatus(day, p).done).length}/${pts.length - core.length}` : '')));
  };

  const paintFoot = () => {
    foot.innerHTML = ''; const i = cards.indexOf(view.cur), prev = cards[i - 1], next = cards[i + 1], pt = byId(view.cur), st = pt && pointStatus(day, pt);
    const label = (id) => (id === 'intro' ? '总览' : id === 'recap' ? '今日小结' : `${byId(id).num} ${byId(id).title}`);
    foot.append(prev ? el('a', { class: 'btn', href: pointHref(day, prev) }, `← ${label(prev)}`) : el('span'),
      el('span', { class: 'muted' }, (pt && st.total ? (st.done ? '✓ 这一节的实验和小测都做完了　' : `这一节还有 ${st.total - st.got} 个互动没做　`) : '') + '键盘 ← → 翻页'),
      next ? el('a', { class: 'btn primary', href: pointHref(day, next) }, `${label(next)} →`) : el('span'));
  };

  const chapterCards = () => { // 总览里的「目录」：章 → 小节，带完成情况
    const box = el('div', { class: 'chapters' });
    groups.forEach((g) => {
      const c = g.ch != null ? chs[g.ch] : null, card = el('div', { class: 'chcard' });
      if (c) { card.append(el('h3', {}, chName(g.ch))); if (c.intro) { const pr = renderMarkdown(c.intro); pr.className = 'chintro'; card.append(pr); } }
      g.pts.forEach((p) => { const st = pointStatus(day, p); card.append(el('a', { class: 'ol-row' + stCls(p), href: pointHref(day, p.id) }, [el('i', {}, st.done ? '✓' : p.num), el('span', {}, p.title), p.side ? el('em', {}, '选学') : null, el('small', {}, st.total ? `${st.got}/${st.total}` : '')])); });
      box.append(card);
    });
    return box;
  };

  const build = (id) => {
    const card = el('div', { class: 'pcard' });
    if (id === 'intro') {
      card.append(el('h2', { class: 'ptitle', style: 'margin-bottom:10px' }, '总览：今天讲什么'));
      if (L.intro) card.append(renderBody(L, day, L.intro, newCtx()));
      if (m.goals?.length) card.append(el('div', { class: 'goals' }, [el('b', {}, '今天学完，你应该能…'), el('ul', {}, m.goals.map((g) => el('li', { html: mdToHtml(String(g)).replace(/^<p>|<\/p>\s*$/g, '') })))]));
      card.append(el('h3', { class: 'ol-title' }, `今天的结构：${chs.length ? chs.length + ' 章 · ' : ''}${pts.length} 个小节，一次只讲一个点`), el('div', { 'data-outline': '1' }));
      card.append(el('div', { style: 'margin-top:18px' }, [el('a', { class: 'btn primary', style: 'padding:9px 20px;font-size:15px', 'data-start': '1', href: '#' }, '开始 →')]));
    } else if (id === 'recap') {
      card.append(el('h2', { class: 'ptitle' }, '今日小结：每个小节一句话'));
      const list = el('div', { class: 'recap' });
      groups.forEach((g) => {
        if (g.ch != null) list.append(el('h3', { class: 'recap-ch' }, chName(g.ch)));
        g.pts.forEach((p) => { const st = pointStatus(day, p), k = renderMarkdown(p.key || '（这个小节没有写结论）'); k.className = 'recap-key'; list.append(el('div', { class: 'recap-row' + (p.side ? ' side' : '') + (st.done ? ' done' : '') }, [el('a', { class: 'recap-q', href: pointHref(day, p.id) }, [el('i', {}, st.done ? '✓' : p.num), p.title]), k])); });
      });
      card.append(list);
      const usedInPoints = new Set(pts.flatMap((p) => [...p.predict, ...p.quiz])), rest = L.quiz.filter((q) => !usedInPoints.has(q.id));
      if (rest.length) { card.append(el('h2', { class: 'ptitle', style: 'margin-top:34px' }, '综合测验'), el('p', { class: 'muted' }, '这几道题跨多个小节，答错了就点上面的结论回去看。')); const box = el('div', { class: 'prose wide' }); rest.forEach((q, i) => box.append(renderQuiz(day, q, i + 1, { onChange: refresh }))); card.append(box); }
      card.append(lessonFoot(day));
    } else {
      const pt = byId(id), ctx = newCtx({ micro: true, labIds: pt.labs, onChange: () => refresh() }), c = pt.ch != null ? chs[pt.ch] : null;
      if (c) { // 你在哪：第几章 › 第几节；每章第一节顺带给出章导语
        card.append(el('div', { class: 'crumb' }, [el('a', { href: pointHref(day, 'intro') }, '总览'), ' › ', el('b', {}, chName(pt.ch)), ` › 第 ${pt.num.split('.').pop()} / ${c.points.length} 节`]));
        if (c.points[0] === id && c.intro) { const pr = renderMarkdown(c.intro); pr.className = 'chlead'; pr.prepend(el('b', { class: 'ct' }, '本章导语')); card.append(pr); }
      }
      card.append(el('div', { class: 'phead' }, [el('span', { class: 'pnum' + (pt.side ? ' side' : '') }, pt.num), el('h2', { class: 'ptitle' }, pt.title), pt.side ? el('span', { class: 'badge' }, '选学') : null]));
      const prose = renderBody(L, day, pt.body, ctx); card.append(prose);
      const key = prose.querySelector('.callout.key b.ct'); if (key) key.textContent = '🔑 这一节的结论';
    }
    return card;
  };

  function refresh() {
    paintRail(); paintFoot(); renderSidebar();
    const intro = cache.get('intro'); if (!intro) return;
    const o = intro.querySelector('[data-outline]'); o.innerHTML = ''; o.append(chapterCards());
    const st = intro.querySelector('[data-start]'), t = byId(firstTodo()); st.href = pointHref(day, t.id); st.textContent = pts.some((p) => pointStatus(day, p).started) ? `继续：${t.num} ${t.title} →` : `从 ${t.num} ${t.title} 开始 →`;
  }

  view.show = (id) => {
    if (!id || !cards.includes(id)) id = dayProg(day).last && cards.includes(dayProg(day).last) ? dayProg(day).last : 'intro';
    view.cur = id; document.getElementById('drawer').hidden = true; document.getElementById('inspector').hidden = true;
    if (!cache.has(id)) { const c = build(id); cache.set(id, c); stage.append(c); }
    if (id === 'recap') { const fresh = build('recap'); cache.get('recap').replaceWith(fresh); cache.set('recap', fresh); }  // 小结里的 ✓ 要反映最新进度
    cache.forEach((c, k) => (c.hidden = k !== id));
    page.classList.toggle('in-point', id !== 'intro');  // 看小节时把页头收起来，一屏留给这一个点
    if (byId(id) && !dayProg(day).points?.[id]) setProgress(day, ['points', id], Date.now());
    if (dayProg(day).last !== id) setProgress(day, ['last'], id);
    refresh(); scrollTo(0, 0);
  };
  view.step = (d) => { const i = cards.indexOf(view.cur) + d; if (i >= 0 && i < cards.length) location.hash = pointHref(day, cards[i]); };
  view.show(pid);
  return page;
}

// ---------------------------------------------------------------- playground
function renderPlayground() {
  const page = el('div', { class: 'page' });
  page.append(el('div', { class: 'lesson-head' }, [el('div', { class: 'kicker' }, 'PLAYGROUND'), el('h1', {}, '自由实验场'), el('p', { class: 'sub' }, '随便写、随便跑。三个源码标签页可以直接改（只对这里的运行生效，不写回仓库），用 Diff 看你改了什么。')]));
  const code = `import torch\nfrom learnkit import *\nfrom model.model_minimind import *\n\nmodel = build_model(num_hidden_layers=2, flash_attn=False)      # 换成 load_model("full_sft") 就是真实权重\nx = torch.randint(0, 6400, (3, 7))\n\nwith trace(model, fns=[MiniMindBlock.forward, Attention.forward, FeedForward.forward], dims=dict(B=3, T=7)):\n    out = model(x)\n\nshow(logits=out.logits)\n`;
  const lab = new Lab({ day: 0, lab: { id: 'playground', title: 'Playground', timeout: 300, sources: ['model/model_minimind.py', 'dataset/lm_dataset.py', 'model/model_lora.py'], tasks: [], code, saved: state.playgroundSaved }, progress: dayProg(0), onProgress: (path, v) => setProgress(0, path, v) });
  state.labs.push(lab); page.append(lab.root);
  const api_md = '### learnkit 速查\n```python\nmodel = build_model(num_hidden_layers=2, **MiniMindConfig参数)   # 随机初始化\nmodel = load_model("full_sft")  # 或 "pretrain"；load_model("full_sft", use_moe=True)\ntok = get_tokenizer()\n\nwith trace(model, fns=[Attention.forward], dims=dict(B=3, T=7), capture=["scores"]) as tr:\n    model(x)\ntr.captured["Attention.forward"][0]["scores"]\n\nshow(x=x)  heatmap(t, labels=tokens)  bars(labels, values)  plot({"loss": ys})\nlive("loss", "train", step, v)  table(rows, headers)  token_strip(tokens, values)  note("md")\n```';
  const help = renderMarkdown(api_md); help.className = 'prose'; page.append(help);
  return page;
}

// ---------------------------------------------------------------- router
async function route() {
  const h = location.hash || '#/';
  const dm = h.match(/^#\/day\/(\d+)(?:\/p\/([\w\-]+))?/);
  if (dm && state.lessonView?.day === +dm[1] && document.body.contains(state.lessonView.page)) { state.lessonView.show(dm[2]); return; }  // 同一天内翻知识点
  state.lessonView = null;
  state.labs.forEach((l) => l.dispose()); state.labs = [];
  document.getElementById('drawer').hidden = true; document.getElementById('inspector').hidden = true;
  renderSidebar(); main.innerHTML = '<div class="loading">加载中…</div>';
  try {
    let page; const m = h.match(/^#\/day\/(\d+)/);
    if (m) page = await renderLesson(+m[1], dm?.[2]); else if (h === '#/map') page = renderMap(); else if (h === '#/playground') { state.playgroundSaved = (await api('/api/workspace?day=0&lab=playground').catch(() => null))?.saved; page = renderPlayground(); } else page = renderHome();
    main.innerHTML = ''; main.append(page); const anchor = h.split('@')[1]; if (anchor) document.getElementById(anchor)?.scrollIntoView(); else scrollTo(0, 0);
  } catch (e) { console.error(e); main.innerHTML = ''; main.append(el('div', { class: 'loading' }, `加载失败：${e.message}`)); }
}

async function boot() {
  document.getElementById('theme-toggle').addEventListener('click', () => { const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = t; try { localStorage.setItem('mm-theme', t); } catch {} document.dispatchEvent(new Event('mm-theme')); });
  const plan = await api('/api/plan'); state.plan = plan.days;
  if ((plan.api || 1) < NEED_API) document.body.prepend(el('div', { class: 'stale-banner' }, '⚠️ server.py 已经更新，但正在运行的还是旧服务：页面会显示不完整（看不到总览/章节、标题里露出 {#id}）。请到启动它的终端按 Ctrl+C，再运行 ./start.sh，然后刷新本页。')); state.progress = plan.progress || { days: {} }; state.progress.days ||= {}; state.weights = plan.weights; state.repo = plan.repo;
  document.addEventListener('keydown', (e) => { // ← → 翻知识点 (在编辑器/输入框里不生效)
    if (!state.lessonView || e.metaKey || e.ctrlKey || e.altKey || e.shiftKey || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
    if (e.target.closest?.('input, textarea, select, [contenteditable], .monaco-editor')) return;
    e.preventDefault(); state.lessonView.step(e.key === 'ArrowRight' ? 1 : -1);
  });
  addEventListener('hashchange', route); route();
}
boot();
