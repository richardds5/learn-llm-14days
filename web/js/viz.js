// viz.js —— 把 learnkit 发来的事件渲染成 DOM。核心是「维度色块」：同一个符号 (B/T/H/D…) 全站同一种颜色，
// 这样 view / transpose / reshape 前后维度怎么搬家一眼就能看出来。
import { el, md, fmtNum, fmtBytes, openSource, highlightLine } from './util.js';

export const DIM_COLORS = { B: '#64748b', T: '#0ea5e9', H: '#f59e0b', KV: '#ef4444', D: '#a855f7', C: '#10b981', I: '#ec4899', V: '#f97316', E: '#14b8a6' };
const EXTRA = ['#6366f1', '#84cc16', '#06b6d4', '#d946ef', '#ca8a04', '#f43f5e', '#22c55e', '#8b5cf6', '#0d9488', '#e11d48'];
const DTYPE = { float32: 'f32', float16: 'f16', bfloat16: 'bf16', float64: 'f64', int64: 'i64', int32: 'i32', bool: 'bool', uint8: 'u8', int8: 'i8' };
const DTYPE_BYTES = { float32: 4, float16: 2, bfloat16: 2, float64: 8, int64: 8, int32: 4, bool: 1, uint8: 1, int8: 1 };

function hash(s) { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) >>> 0; return h; }
export function dimColor(label) {
  if (!label) return null;
  if (label.includes('*')) { const [a, b] = label.split('*'); return `linear-gradient(100deg, ${dimColor(a)} 45%, ${dimColor(b)} 55%)`; }
  return DIM_COLORS[label] || EXTRA[hash(label) % EXTRA.length];
}
export function dimEl(size, label) {
  const d = el('span', { class: 'dim' + (label ? '' : ' plain') });
  if (label) { d.style.background = dimColor(label); d.append(label, el('i', {}, String(size))); } else d.textContent = String(size);
  return d;
}
const shapeStr = (d) => d && d.k === 'tensor' ? `[${d.shape.join(', ')}]` : '';

// ---------------------------------------------------------------- chip
export function chip(name, desc, opts = {}) {
  const c = el('span', { class: 'tchip' + (opts.small ? ' small' : '') });
  if (name != null) c.append(el('span', { class: 'tn' }, name));
  if (!desc) { c.classList.add('none'); c.append('?'); return c; }
  if (desc.k === 'tensor') {
    if (!desc.shape.length) c.append(el('span', { class: 'br' }, 'scalar'));
    desc.shape.forEach((s, i) => c.append(dimEl(s, desc.labels?.[i])));
    c.append(el('span', { class: 'dt' }, DTYPE[desc.dtype] || desc.dtype));
    if (desc.value !== undefined) c.append(el('span', { class: 'dt' }, '= ' + fmtNum(desc.value)));
    if (desc.grad) c.append(el('span', { class: 'flag', title: 'requires_grad=True' }, '∇'));
    if (desc.device) c.append(el('span', { class: 'flag' }, desc.device));
    if (opts.change === 'inplace') c.append(el('span', { class: 'flag', title: '原地修改 (同一个 tensor 对象，_version 变了)' }, 'in-place'));
    if (opts.was) { c.classList.add('changed'); c.append(el('span', { class: 'was', title: '上一次运行时的 shape' }, '上次 ' + opts.was)); }
    c.addEventListener('click', (e) => { e.stopPropagation(); openInspector(c, name, desc, opts.context); });
  } else if (desc.k === 'scalar' || desc.k === 'str') { c.classList.add('scalar'); c.append('= ' + (desc.k === 'str' ? JSON.stringify(desc.value) : fmtNum(desc.value))); }
  else if (desc.k === 'none') { c.classList.add('none'); c.append('= None'); }
  else if (desc.k === 'seq') {
    c.classList.add('scalar'); c.append(el('span', { class: 'br' }, desc.py + '('));
    desc.items.forEach((it, i) => c.append(chip(desc.names?.[i] ?? null, it, { small: true, context: opts.context })));
    if (desc.len > desc.items.length) c.append(el('span', { class: 'br' }, `… 共 ${desc.len} 项`));
    c.append(el('span', { class: 'br' }, ')'));
  } else if (desc.k === 'dict') {
    c.classList.add('scalar'); c.append(el('span', { class: 'br' }, desc.py + '{'));
    Object.entries(desc.items).forEach(([k, v]) => c.append(chip(k, v, { small: true, context: opts.context })));
    c.append(el('span', { class: 'br' }, '}'));
  }
  return c;
}

// ---------------------------------------------------------------- tensor 示意图
const len = (n) => 14 + 13 * Math.log2(Math.max(1, n));
export function tensorSvg(shape, labels = []) {
  const n = shape.length, NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  const mk = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; svg.append(e); return e; };
  const col = (i) => (labels[i] ? (labels[i].includes('*') ? dimColor(labels[i].split('*')[0]) : dimColor(labels[i])) : '#8d877a');
  const lab = (i) => (labels[i] ? `${labels[i]}=${shape[i]}` : String(shape[i]));
  if (n === 0) { svg.setAttribute('viewBox', '0 0 120 40'); svg.setAttribute('width', 120); mk('circle', { cx: 60, cy: 20, r: 7, fill: '#8d877a' }); return svg; }
  const cols = shape[n - 1], rows = n >= 2 ? shape[n - 2] : 1, depth = n >= 3 ? shape[n - 3] : 1;
  const w = len(cols), h = n >= 2 ? len(rows) : 16, layers = Math.min(depth, 6), off = 7;
  const padL = 64, padT = 30 + (layers - 1) * off, W = padL + w + (layers - 1) * off + 70, H = padT + h + 26;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', Math.min(W, 340));
  for (let k = layers - 1; k >= 0; k--) {
    const x = padL + k * off, y = padT - k * off;
    mk('rect', { x, y, width: w, height: h, rx: 3, fill: k ? 'var(--panel-2)' : 'var(--panel)', stroke: 'var(--line-2)', 'stroke-width': 1.2 });
  }
  if (cols <= 16) for (let i = 1; i < cols; i++) mk('line', { x1: padL + (w * i) / cols, y1: padT, x2: padL + (w * i) / cols, y2: padT + h, stroke: 'var(--line)', 'stroke-width': 1 });
  if (n >= 2 && rows <= 16) for (let i = 1; i < rows; i++) mk('line', { x1: padL, y1: padT + (h * i) / rows, x2: padL + w, y2: padT + (h * i) / rows, stroke: 'var(--line)', 'stroke-width': 1 });
  mk('line', { x1: padL, y1: padT + h + 6, x2: padL + w, y2: padT + h + 6, stroke: col(n - 1), 'stroke-width': 3.5, 'stroke-linecap': 'round' });
  mk('text', { x: padL + w / 2, y: padT + h + 21, 'text-anchor': 'middle', 'font-size': 11, 'font-family': 'var(--mono)', fill: col(n - 1), 'font-weight': 600 }, lab(n - 1));
  if (n >= 2) {
    mk('line', { x1: padL - 6, y1: padT, x2: padL - 6, y2: padT + h, stroke: col(n - 2), 'stroke-width': 3.5, 'stroke-linecap': 'round' });
    mk('text', { x: padL - 12, y: padT + h / 2 + 4, 'text-anchor': 'end', 'font-size': 11, 'font-family': 'var(--mono)', fill: col(n - 2), 'font-weight': 600 }, lab(n - 2));
  }
  if (n >= 3) {
    const x0 = padL + w + 5, y0 = padT, x1 = x0 + (layers - 1) * off + 4, y1 = y0 - (layers - 1) * off - 4;
    mk('line', { x1: x0, y1: y0, x2: x1, y2: y1, stroke: col(n - 3), 'stroke-width': 3.5, 'stroke-linecap': 'round' });
    mk('text', { x: x1 + 5, y: y1 + 4, 'font-size': 11, 'font-family': 'var(--mono)', fill: col(n - 3), 'font-weight': 600 }, lab(n - 3) + (depth > layers ? ' 层' : ''));
  }
  if (n >= 4) mk('text', { x: 4, y: 14, 'font-size': 11, 'font-family': 'var(--mono)', fill: 'var(--ink-2)' }, '外层还有 ' + shape.slice(0, n - 3).map((_, i) => lab(i)).join(' × ') + ' 份');
  return svg;
}

// ---------------------------------------------------------------- inspector
const inspector = () => document.getElementById('inspector');
export function closeInspector() { inspector().hidden = true; }
export function openInspector(anchor, name, desc, context) {
  const box = inspector();
  box.innerHTML = '';
  const numel = desc.shape.reduce((a, b) => a * b, 1), bytes = numel * (DTYPE_BYTES[desc.dtype] || 4);
  box.append(el('h5', {}, [el('span', {}, name || 'tensor'), el('button', { onclick: closeInspector }, '✕')]));
  const dims = el('div', { class: 'legend' }); desc.shape.forEach((s, i) => dims.append(dimEl(s, desc.labels?.[i]))); box.append(dims);
  box.append(tensorSvg(desc.shape, desc.labels || []));
  const meanings = context?.meanings || {};
  const rows = [['shape', `[${desc.shape.join(', ')}]`], ['dtype', desc.dtype], ['元素个数', numel.toLocaleString()], ['占用', fmtBytes(bytes)]];
  desc.shape.forEach((s, i) => { const l = desc.labels?.[i]; if (l) rows.push([`dim ${i}`, `${l} = ${s}` + (meanings[l] ? `  · ${meanings[l]}` : (l.includes('*') ? '  · 两个维度合并' : ''))]); else rows.push([`dim ${i}`, String(s)]); });
  const t = el('table'); rows.forEach(([a, b]) => t.append(el('tr', {}, [el('td', {}, a), el('td', {}, b)]))); box.append(t);
  if (context?.event) {
    const f = el('div', { class: 'flow' });
    if (context.src) f.append(el('div', { class: 'muted mono' }, `第 ${context.event.line} 行： ${context.src.trim().slice(0, 120)}`));
    if (context.event.reads?.length) { const r = el('div', { class: 'io' }, [el('span', { class: 'muted' }, '这一行读入 ')]); context.event.reads.forEach((x) => r.append(chip(x.name, x.desc, { small: true }), ' ')); f.append(r); }
    if (context.event.writes?.length) { const r = el('div', { class: 'io' }, [el('span', { class: 'muted' }, '这一行写出 ')]); context.event.writes.forEach((x) => r.append(chip(x.name, x.desc, { small: true }), ' ')); f.append(r); }
    box.append(f);
  }
  box.hidden = false;
  const r = anchor.getBoundingClientRect(), bw = box.offsetWidth, bh = box.offsetHeight;
  let x = Math.min(r.left, innerWidth - bw - 12), y = r.bottom + 8;
  if (y + bh > innerHeight - 8) y = Math.max(8, r.top - bh - 8);
  box.style.left = Math.max(8, x) + 'px'; box.style.top = y + 'px';
}
document.addEventListener('click', (e) => { const b = inspector(); if (b && !b.hidden && !b.contains(e.target)) closeInspector(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeInspector(); });

// ---------------------------------------------------------------- tooltip
const tip = () => document.getElementById('tooltip');
function showTip(e, text) { const t = tip(); t.textContent = text; t.hidden = false; t.style.left = Math.min(e.clientX + 14, innerWidth - t.offsetWidth - 8) + 'px'; t.style.top = e.clientY + 16 + 'px'; }
function hideTip() { tip().hidden = true; }

function card(type, title, extraHead) {
  const body = el('div', { class: 'vcard-body' });
  const head = el('div', { class: 'vcard-head' }, [el('span', { class: 'vtype' }, type), el('b', {}, title || '')]);
  if (extraHead) head.append(extraHead);
  return { root: el('div', { class: 'vcard' }, [head, body]), head, body };
}

// ---------------------------------------------------------------- trace
function legend(ev) {
  const lg = el('div', { class: 'legend' });
  Object.entries(ev.dims || {}).forEach(([k, v]) => { const d = dimEl(v, k); d.title = ev.dim_meanings?.[k] || (k.includes('*') ? '两个维度的乘积' : '自定义符号'); lg.append(d); });
  return lg;
}

function collectShapes(ev) { // 给「和上一次运行对比」用
  const m = new Map();
  (ev.fns || []).forEach((fn) => fn.calls.forEach((c, ci) => { const seen = {}; c.events.forEach((e) => e.writes.forEach((w) => { const k0 = `${fn.name}|${ci}|${e.line}|${w.name}`; seen[k0] = (seen[k0] || 0) + 1; if (seen[k0] === 1) m.set(k0, shapeStr(w.desc)); })); }));
  return m;
}

function renderFn(fn, ev, prev) {
  const wrap = el('div');
  const info = el('div', { class: 'trace-info' });
  const link = el('a', { class: 'srcref', href: '#' }, `${fn.file}:${fn.first_line}`);
  link.addEventListener('click', async (e) => { e.preventDefault(); const lab = link.closest('.lab')?._lab; if (!(lab && (await lab.revealSource(fn.file, fn.first_line)))) openSource(fn.file, fn.first_line, fn.first_line + fn.source.length - 1); });
  link.title = '点击定位到源码（如果本实验允许改这个文件，会直接跳到编辑器里）';
  info.append(link);
  if (fn.modified) info.append(el('span', { class: 'badge acc' }, '你改过的版本'));
  const body = el('div');
  if (!fn.calls.length) { wrap.append(info, el('div', { class: 'trace-info' }, '⚠️ 这个函数在追踪期间一次也没被调用。')); return wrap; }
  info.append(el('span', {}, `共调用 ${fn.total_calls} 次` + (fn.total_calls > fn.calls.length ? `，详细记录了前 ${fn.calls.length} 次` : '')));
  const callsEl = el('span', { class: 'calls' }); info.append(callsEl);
  const readsBtn = el('button', { class: 'btn', type: 'button' }, '显示每行读入的 tensor');
  const trajBtn = el('button', { class: 'btn', type: 'button', title: '同一个变量的 shape 在函数里是怎么一步步变的' }, '🧬 变量轨迹');
  info.append(el('span', { style: 'flex:1' }), trajBtn, readsBtn);
  const traj = el('div', { class: 'traj', hidden: true });
  let cur = 0;
  const drawTraj = () => {
    traj.innerHTML = ''; const call = fn.calls[cur], vars = new Map();
    call.events.forEach((e) => e.writes.forEach((w) => { if (w.desc?.k !== 'tensor') return; const arr = vars.get(w.name) || []; const last = arr[arr.length - 1]; if (last && JSON.stringify(last.w.desc.shape) === JSON.stringify(w.desc.shape) && last.w.desc.dtype === w.desc.dtype) { last.lines.push(e.line); return; } arr.push({ w, e, lines: [e.line] }); vars.set(w.name, arr); }));
    if (!vars.size) { traj.append(el('div', { class: 'muted' }, '这次调用里没有 tensor 变量。')); return; }
    [...vars.entries()].sort((a, b) => b[1].length - a[1].length).forEach(([name, steps]) => {
      const row = el('div', { class: 'traj-row' }, [el('span', { class: 'traj-name' }, name)]);
      steps.forEach((s, i) => { if (i) row.append(el('span', { class: 'arrow' }, '→')); const src = fn.source[s.e.line - fn.first_line] || ''; row.append(el('span', { class: 'traj-step' }, [el('small', {}, s.e.is_args ? '入参' : 'L' + s.lines[0] + (s.lines.length > 1 ? `…L${s.lines[s.lines.length - 1]}` : '')), chip(null, s.w.desc, { small: true, context: { event: s.e, src, meanings: ev.dim_meanings } })])); });
      traj.append(row);
    });
  };
  trajBtn.addEventListener('click', () => { trajBtn.classList.toggle('on'); traj.hidden = !trajBtn.classList.contains('on'); if (!traj.hidden) drawTraj(); });
  const draw = (ci) => {
    cur = ci; if (!traj.hidden) drawTraj();
    [...callsEl.children].forEach((b, i) => b.classList.toggle('on', i === ci));
    body.innerHTML = ''; body.append(renderCall(fn, fn.calls[ci], ci, ev, prev));
    body.firstChild.classList.toggle('show-reads', readsBtn.classList.contains('on'));
  };
  fn.calls.forEach((c, i) => { const b = el('button', { class: 'btn', type: 'button', title: c.self || '' }, `#${c.index + 1}` + (c.self ? ` ${c.self.replace(/^model\./, '')}` : '')); b.addEventListener('click', () => draw(i)); callsEl.append(b); });
  if (fn.calls.length < 2) callsEl.hidden = true;
  readsBtn.addEventListener('click', () => { readsBtn.classList.toggle('on'); body.firstChild.classList.toggle('show-reads'); });
  wrap.append(info, traj, body); draw(0);
  return wrap;
}

function renderCall(fn, call, ci, ev, prev) {
  const tl = el('div', { class: 'tl' });
  const byLine = {}; call.events.forEach((e) => (byLine[e.line] ||= []).push(e));
  const fc = fn.focus, names = new Set(fc?.names || []);
  const inFocus = (ln, evs, isRet) => !fc || (fc.lines ? ln >= fc.lines[0] && ln <= fc.lines[1] : evs.some((e) => e.writes.some((w) => names.has(w.name))) || (isRet && names.has('return')));
  let nFocus = 0;
  const seenKey = {};
  fn.source.forEach((src, i) => {
    const ln = fn.first_line + i, evs = byLine[ln] || [], hits = call.hits?.[ln] || 0;
    const executed = hits > 0 || evs.length > 0, isRet = call.ret_line === ln && call.ret;
    const focused = inFocus(ln, evs, isRet); nFocus += focused ? 1 : 0;
    const row = el('div', { class: 'tl-row ' + (executed ? 'exec' : 'skipped') + (evs.some((e) => e.writes.length || e.expr?.length) || isRet ? ' has-io' : '') + (fc ? (focused ? ' focus' : ' offfocus') : '') });
    const main = el('div', {}, [el('pre', { html: highlightLine(src) })]);
    const withWrites = evs.filter((e) => e.writes.length || e.expr?.length), shown = withWrites.length ? withWrites : evs.slice(0, 1);
    if (shown.length || isRet) {
      const io = el('div', { class: 'tl-io' });
      let idx = 0;
      const paint = () => {
        io.innerHTML = '';
        const e = shown[idx];
        if (e) {
          const ctx = { event: e, src, meanings: ev.dim_meanings };
          if (e.reads?.length) { const r = el('div', { class: 'io reads' }, [el('span', { class: 'lbl' }, '读 ←')]); e.reads.forEach((x) => r.append(chip(x.name, x.desc, { small: true, context: ctx }))); io.append(r); }
          if (e.expr?.length) { // 把这一行拆开：每个子表达式的 shape (按求值顺序)
            const x = el('div', { class: 'io expr' }, [el('span', { class: 'lbl' }, '拆开')]), list = el('div', { class: 'expr-list' });
            e.expr.forEach((it, k) => list.append(el('div', { class: 'expr-row' }, [el('span', { class: 'expr-n' }, String(k + 1)), el('code', { html: highlightLine(it.src.length > 90 ? it.src.slice(0, 88) + ' …' : it.src), title: it.src }), el('span', { class: 'arrow' }, '→'), chip(null, it.desc, { small: true, context: { ...ctx, src: it.src } })])));
            x.append(list); io.append(x);
          }
          if (e.writes.length) {
            const w = el('div', { class: 'io' }, [el('span', { class: 'lbl' }, e.is_args ? '入参' : '写 ⇒')]);
            e.writes.forEach((x) => { const key = `${fn.name}|${ci}|${ln}|${x.name}`; let was = null; if (idx === 0 && prev?.has(key) && prev.get(key) !== shapeStr(x.desc) && prev.get(key)) was = prev.get(key); w.append(chip(x.name, x.desc, { change: x.change, was, context: ctx })); });
            if (shown.length > 1) { const pg = el('span', { class: 'pager' }); const b1 = el('button', { type: 'button' }, '◀'), b2 = el('button', { type: 'button' }, '▶'); b1.onclick = (z) => { z.stopPropagation(); idx = (idx + shown.length - 1) % shown.length; paint(); }; b2.onclick = (z) => { z.stopPropagation(); idx = (idx + 1) % shown.length; paint(); }; pg.append(b1, `第 ${idx + 1}/${shown.length} 次执行`, b2); w.append(pg); }
            io.append(w);
          }
        }
        if (isRet) io.append(el('div', { class: 'io' }, [el('span', { class: 'lbl' }, '返回'), chip(null, call.ret, { context: { meanings: ev.dim_meanings } })]));
      };
      paint(); main.append(io);
      row.addEventListener('click', () => row.classList.toggle('open'));
    }
    row.append(el('span', { class: 'ln' }, String(ln)), main, el('span', { class: 'hits', title: '这一行执行了几次' }, hits > 1 ? '×' + hits : ''));
    tl.append(row);
  });
  if (call.truncated) tl.append(el('div', { class: 'trace-info' }, '（事件太多，后面的被截断了）'));
  if (fc && nFocus && nFocus < fn.source.length) { // 这个知识点只关心其中几行：其余默认折叠
    tl.classList.add('focused');
    const bar = el('div', { class: 'trace-info focusbar' }), btn = el('button', { class: 'btn', type: 'button' });
    const paintBtn = () => { const on = tl.classList.contains('focused'); btn.textContent = on ? `展开整个函数（${fn.source.length} 行）` : '只看这个知识点相关的行'; bar.firstChild.textContent = on ? `🎯 只显示和这个知识点相关的 ${nFocus} 行 ` : '🎯 相关的行用色条标出 '; };
    bar.append(el('span', {}, ''), btn); btn.addEventListener('click', () => { tl.classList.toggle('focused'); paintBtn(); }); paintBtn(); tl.prepend(bar);
  }
  return tl;
}

function sameShape(a, b) { return JSON.stringify([a.inputs, a.output, a.cls]) === JSON.stringify([b.inputs, b.output, b.cls]); }
function renderModuleNode(node, depth, ctx) {
  const wrap = el('div', { class: 'mnode' });
  const kids = el('div', { class: 'mkids' });
  const hasKids = node.children.length > 0;
  const caret = el('span', { class: 'caret' }, hasKids ? '▾' : '·');
  const short = node.name ? node.name.split('.').slice(-1)[0] : '(model)';
  const row = el('div', { class: 'mrow' }, [caret, el('span', { class: 'mname', title: node.name }, /^\d+$/.test(short) ? node.name.split('.').slice(-2).join('.') : short), el('span', { class: 'mcls' }, node.cls), el('span', { class: 'mparams' }, node.params ? fmtNum(node.params, true) + ' 参数' : '')]);
  node.inputs.forEach((d, i) => row.append(chip(node.input_names?.[i] ?? null, d, { small: true, context: ctx })));
  Object.entries(node.kwargs || {}).forEach(([k, d]) => { if (d.k === 'tensor' || d.k === 'seq') row.append(chip(k, d, { small: true, context: ctx })); });
  row.append(el('span', { class: 'arrow' }, '→'), chip(node.output_name ?? null, node.output, { context: ctx }));
  let open = depth < 2;
  const setOpen = (v) => { open = v; kids.hidden = !v; if (hasKids) caret.textContent = v ? '▾' : '▸'; };
  caret.addEventListener('click', () => setOpen(!open));
  let prevKid = null;
  node.children.forEach((c) => {
    const k = renderModuleNode(c, depth + 1, ctx);
    if (prevKid && /\.\d+$/.test(c.name) && sameShape(prevKid, c)) { k.querySelector('.mrow').append(el('span', { class: 'same' }, '← 维度同上一层')); k._collapse?.(); }
    prevKid = c; kids.append(k);
  });
  wrap._collapse = () => setOpen(false);
  wrap._setOpenAll = (v) => { setOpen(v); [...kids.children].forEach((k) => k._setOpenAll?.(v)); };
  setOpen(open); wrap.append(row, kids);
  return wrap;
}

export function renderTrace(ev, prevShapes) {
  const { root, body, head } = card('TRACE', ev.title || 'tensor 维度追踪');
  head.append(el('span', { style: 'flex:1' }), legend(ev));
  body.style.padding = '0';
  const tabs = el('div', { class: 'trace-tabs' }), pane = el('div');
  const views = [];
  (ev.fns || []).forEach((fn) => views.push({ label: fn.name, make: () => renderFn(fn, ev, prevShapes) }));
  if (ev.modules) views.push({ label: '🌲 模块调用树', make: () => {
    const box = el('div'), info = el('div', { class: 'trace-info' }), tree = el('div', { class: 'mtree' });
    const ctx = { meanings: ev.dim_meanings };
    const nodes = ev.modules.map((m) => renderModuleNode(m, 0, ctx));
    const b1 = el('button', { class: 'btn', type: 'button' }, '全部展开'), b2 = el('button', { class: 'btn', type: 'button' }, '全部折叠');
    b1.onclick = () => nodes.forEach((n) => n._setOpenAll(true)); b2.onclick = () => nodes.forEach((n) => { n._setOpenAll(false); });
    info.append(el('span', {}, '每个子模块一行： forward 的各个实参 → 返回值，名字取自源码里的形参 / return 语句。tuple( … ) 表示这个参数或返回值本身是一个 python tuple，括号里是它的各个元素。点 ▸ 展开，点色块看 tensor 详情。' + (ev.total_roots > ev.modules.length ? ` （模型共被调用 ${ev.total_roots} 次，这里显示前 ${ev.modules.length} 次）` : '')), el('span', { style: 'flex:1' }), b1, b2);
    nodes.forEach((n) => tree.append(n)); box.append(info, tree); return box;
  } });
  const first = Math.max(0, (ev.fns || []).findIndex((fn) => fn.focus && fn.calls.length));  // 有 focus 的函数优先展示：那才是这个知识点要看的
  views.forEach((v, i) => { const t = el('button', { class: 'tab', type: 'button' }, v.label); t.addEventListener('click', () => { [...tabs.children].forEach((x) => x.classList.remove('active')); t.classList.add('active'); pane.innerHTML = ''; pane.append(v.make()); }); tabs.append(t); if (i === first) queueMicrotask(() => t.click()); });
  body.append(tabs, pane);
  root._shapes = collectShapes(ev);
  return root;
}

// ---------------------------------------------------------------- show()
function renderTensors(ev) {
  const { root, body } = card('SHOW', ev.title || 'tensor');
  const list = el('div', { class: 'tcards' });
  ev.items.forEach((it) => {
    const box = el('div', {}, [chip(it.name, it.desc)]);
    if (it.stats) { const s = it.stats; box.append(el('div', { class: 'tcard-stats' }, [`min ${fmtNum(s.min)}`, `max ${fmtNum(s.max)}`, `mean ${fmtNum(s.mean)}`, `std ${fmtNum(s.std)}`, `${(it.numel || 0).toLocaleString()} 个元素`, fmtBytes(it.bytes || 0)].map((x) => el('span', {}, x)))); }
    if (it.preview != null && it.desc?.shape?.length) {
      const rows = Array.isArray(it.preview[0]) ? it.preview : [it.preview];
      const flat = rows.flat().filter((v) => typeof v === 'number'), lo = Math.min(...flat), hi = Math.max(...flat);
      const t = el('table', { class: 'pv' });
      rows.forEach((r) => { const tr = el('tr'); r.forEach((v) => { const td = el('td', {}, typeof v === 'number' ? fmtNum(v) : String(v)); if (typeof v === 'number' && hi > lo) td.style.background = `color-mix(in srgb, var(--accent) ${Math.round(((v - lo) / (hi - lo)) * 38)}%, transparent)`; tr.append(td); }); t.append(tr); });
      box.append(t, el('div', { class: 'muted' }, '数值预览（左上角）' + (it.preview_note ? '，' + it.preview_note : '')));
    }
    list.append(box);
  });
  body.append(list); return root;
}

// ---------------------------------------------------------------- heatmap
const VIRIDIS = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];
const RDBU = [[33, 102, 172], [103, 169, 207], [209, 229, 240], [247, 247, 247], [253, 219, 199], [239, 138, 98], [178, 24, 43]];
function cmap(stops, t) { t = Math.max(0, Math.min(1, t)) * (stops.length - 1); const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i; return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f)); }
const num = (v) => (typeof v === 'number' ? v : v === 'inf' ? Infinity : v === '-inf' ? -Infinity : NaN);

function renderHeatmap(ev) {
  const { root, body } = card('HEATMAP', ev.title || '', el('span', { class: 'muted mono' }, `shape [${ev.shape.join(', ')}]`));
  const all = ev.facets.flat(2).map(num).filter(Number.isFinite);
  let lo = ev.vmin ?? Math.min(...all), hi = ev.vmax ?? Math.max(...all); if (!(hi > lo)) hi = lo + 1;
  const diverging = ev.cmap === 'rdbu' || ev.cmap === 'diverging'; if (diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
  const stops = diverging ? RDBU : VIRIDIS;
  const facets = el('div', { class: 'facets' });
  ev.facets.forEach((mat, fi) => {
    const R = mat.length, C = mat[0].length, target = ev.facets.length > 1 ? 200 : 420;
    const cell = Math.max(2, Math.min(34, Math.floor(target / Math.max(R, C))));
    const showLabels = cell >= 13 && (ev.xlabels || ev.ylabels) && ev.facets.length <= 4;
    const padL = showLabels && ev.ylabels ? 74 : 0, padT = showLabels && ev.xlabels ? 58 : 0;
    const cv = el('canvas'), dpr = devicePixelRatio || 1, W = padL + C * cell, H = padT + R * cell;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const g = cv.getContext('2d'); g.scale(dpr, dpr);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const v = num(mat[r][c]);
      if (Number.isFinite(v)) { const [a, b, d] = cmap(stops, (v - lo) / (hi - lo)); g.fillStyle = `rgb(${a},${b},${d})`; } else g.fillStyle = v === -Infinity ? '#3f3f46' : '#a1a1aa';
      g.fillRect(padL + c * cell, padT + r * cell, cell, cell);
      if (!Number.isFinite(v) && cell >= 8) { g.strokeStyle = 'rgba(255,255,255,.25)'; g.beginPath(); g.moveTo(padL + c * cell, padT + (r + 1) * cell); g.lineTo(padL + (c + 1) * cell, padT + r * cell); g.stroke(); }
    }
    if (showLabels) {
      g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--ink-2'); g.font = '11px ui-monospace, Menlo, monospace';
      const clip = (s) => (s.length > 8 ? s.slice(0, 7) + '…' : s).replace(/\n/g, '↵');
      if (ev.ylabels) { g.textAlign = 'right'; g.textBaseline = 'middle'; ev.ylabels.slice(0, R).forEach((s, r) => g.fillText(clip(s), padL - 5, padT + r * cell + cell / 2)); }
      if (ev.xlabels) { g.textAlign = 'left'; ev.xlabels.slice(0, C).forEach((s, c) => { g.save(); g.translate(padL + c * cell + cell / 2, padT - 5); g.rotate(-Math.PI / 3); g.fillText(clip(s), 0, 0); g.restore(); }); }
    }
    cv.addEventListener('mousemove', (e) => {
      const b = cv.getBoundingClientRect(), c = Math.floor((e.clientX - b.left - padL) / cell), r = Math.floor((e.clientY - b.top - padT) / cell);
      if (r < 0 || c < 0 || r >= R || c >= C) return hideTip();
      const yl = ev.ylabels?.[r], xl = ev.xlabels?.[c];
      showTip(e, `行 ${r}${yl != null ? ` (${yl})` : ''} · 列 ${c}${xl != null ? ` (${xl})` : ''}\n值 = ${typeof mat[r][c] === 'number' ? fmtNum(mat[r][c]) : mat[r][c]}`);
    });
    cv.addEventListener('mouseleave', hideTip);
    facets.append(el('div', { class: 'facet' }, [ev.facets.length > 1 || ev.facet_titles ? el('small', {}, ev.facet_titles?.[fi] ?? `#${fi}`) : null, cv]));
  });
  const grad = stops.map((s, i) => `rgb(${s.join(',')}) ${(i / (stops.length - 1)) * 100}%`).join(',');
  const cb = el('div', { class: 'cbar' }, [fmtNum(lo), el('i', { style: `background:linear-gradient(90deg,${grad})` }), fmtNum(hi)]);
  if (ev.facets.flat(2).some((v) => v === '-inf')) cb.append(el('span', { style: 'margin-left:10px' }, '▨ 深灰 = -inf (被 mask 掉)'));
  body.append(facets, cb); return root;
}

// ---------------------------------------------------------------- plot / live
const SERIES = ['#5b4bdb', '#f97316', '#10b981', '#ef4444', '#0ea5e9', '#d946ef', '#ca8a04', '#64748b'];
function niceTicks(lo, hi, n = 5) { if (!(hi > lo)) { hi = lo + 1; lo = lo - 1; } const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0); const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(12)); return out; }
function drawPlot(holder, { series, xlabel, ylabel, logy, xticks }) {
  holder.innerHTML = '';
  const names = Object.keys(series), W = 720, H = 270, m = { l: 58, r: 14, t: 10, b: 34 };
  const lg = el('div', { class: 'plot-legend' }); names.forEach((n, i) => lg.append(el('span', {}, [el('i', { style: `background:${SERIES[i % SERIES.length]}` }), n])));
  const pts = names.map((n) => series[n].x.map((x, i) => [x, num(series[n].y[i])]).filter((p) => Number.isFinite(p[1]) && (!logy || p[1] > 0)));
  const xs = pts.flat().map((p) => p[0]), ys = pts.flat().map((p) => (logy ? Math.log10(p[1]) : p[1]));
  if (!xs.length) { holder.append(el('div', { class: 'muted' }, '（还没有数据点）')); return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys); if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 0.5; y0 -= 0.5; } const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
  const sx = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r), sy = (y) => H - m.b - (((logy ? Math.log10(y) : y) - y0) / (y1 - y0)) * (H - m.t - m.b);
  const NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const mk = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; svg.append(e); return e; };
  niceTicks(y0, y1).forEach((t) => { const y = H - m.b - ((t - y0) / (y1 - y0)) * (H - m.t - m.b); mk('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y, y2: y }); mk('text', { x: m.l - 6, y: y + 3.5, 'text-anchor': 'end' }, logy ? '1e' + fmtNum(t) : fmtNum(t)); });
  if (xticks?.length) { const step = Math.ceil(xticks.length / 12); xticks.forEach((s, i) => { if (i % step === 0) mk('text', { x: sx(i), y: H - m.b + 15, 'text-anchor': 'middle' }, s.length > 10 ? s.slice(0, 9) + '…' : s); }); }
  else niceTicks(x0, x1, 7).forEach((t) => { mk('text', { x: sx(t), y: H - m.b + 15, 'text-anchor': 'middle' }, fmtNum(t)); });
  mk('line', { class: 'axis', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }); mk('line', { class: 'axis', x1: m.l, x2: m.l, y1: m.t, y2: H - m.b });
  if (xlabel) mk('text', { x: (W + m.l - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, xlabel);
  if (ylabel) mk('text', { x: 12, y: (H - m.b + m.t) / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${(H - m.b + m.t) / 2})` }, ylabel);
  pts.forEach((p, i) => { if (!p.length) return; const c = SERIES[i % SERIES.length]; mk('path', { d: p.map((q, j) => (j ? 'L' : 'M') + sx(q[0]).toFixed(1) + ' ' + sy(q[1]).toFixed(1)).join(''), fill: 'none', stroke: c, 'stroke-width': 1.8, 'stroke-linejoin': 'round' }); if (p.length <= 40) p.forEach((q) => mk('circle', { cx: sx(q[0]), cy: sy(q[1]), r: 2.4, fill: c })); });
  const cross = mk('line', { class: 'axis', y1: m.t, y2: H - m.b, x1: -10, x2: -10, 'stroke-dasharray': '3 3' });
  svg.addEventListener('mousemove', (e) => { const b = svg.getBoundingClientRect(), xv = x0 + (((e.clientX - b.left) / b.width) * W - m.l) / (W - m.l - m.r) * (x1 - x0); const lines = []; let snap = null; pts.forEach((p, i) => { if (!p.length) return; const q = p.reduce((a, c) => (Math.abs(c[0] - xv) < Math.abs(a[0] - xv) ? c : a)); snap = q[0]; lines.push(`${names[i]}: ${fmtNum(q[1])}`); }); if (snap == null) return; cross.setAttribute('x1', sx(snap)); cross.setAttribute('x2', sx(snap)); showTip(e, `x = ${xticks?.[snap] ?? fmtNum(snap)}\n` + lines.join('\n')); });
  svg.addEventListener('mouseleave', () => { hideTip(); cross.setAttribute('x1', -10); cross.setAttribute('x2', -10); });
  holder.append(lg, svg);
}
function renderPlot(ev) {
  const { root, body } = card('PLOT', ev.title || ''); const holder = el('div', { class: 'plot' });
  const series = {}; Object.entries(ev.series).forEach(([n, y]) => (series[n] = { x: ev.x && ev.x.length === y.length ? ev.x : y.map((_, i) => i), y }));
  drawPlot(holder, { series, xlabel: ev.xlabel, ylabel: ev.ylabel, logy: ev.logy, xticks: ev.xticks }); body.append(holder); return root;
}

// ---------------------------------------------------------------- bars / table / tokens
function renderBars(ev) {
  const { root, body } = card('BARS', ev.title || ''); const g = el('div', { class: 'barsv' });
  const vals = ev.values.map(num), max = Math.max(...vals.filter(Number.isFinite).map(Math.abs), 1e-12), hl = new Set(ev.highlight || []);
  ev.labels.forEach((l, i) => g.append(el('span', { class: 'bl', title: l }, l.replace(/\n/g, '↵')), el('span', { class: 'bt' + (hl.has(i) ? ' hl' : '') }, [el('i', { style: `width:${Number.isFinite(vals[i]) ? (Math.abs(vals[i]) / max) * 100 : 0}%` })]), el('span', { class: 'bv' }, fmtNum(ev.values[i]))));
  body.append(g); if (ev.xlabel) body.append(el('div', { class: 'muted' }, ev.xlabel)); return root;
}
function renderTable(ev) {
  const { root, body } = card('TABLE', ev.title || ''); const t = el('table', { class: 'vt' });
  if (ev.headers) t.append(el('tr', {}, ev.headers.map((h) => el('th', {}, String(h)))));
  ev.rows.slice(0, 300).forEach((r) => t.append(el('tr', {}, r.map((c) => el('td', { class: typeof c === 'number' ? 'num' : '' }, c == null ? '' : typeof c === 'number' ? fmtNum(c) : String(c))))));
  body.append(el('div', { class: 'vt-wrap' }, [t])); if (ev.rows.length > 300) body.append(el('div', { class: 'muted' }, `… 共 ${ev.rows.length} 行，只显示前 300 行`)); return root;
}
const CAT = ['#c7d2fe', '#fed7aa', '#bbf7d0', '#fecaca', '#bae6fd', '#f5d0fe', '#fde68a', '#99f6e4', '#e2e8f0', '#fbcfe8', '#d9f99d', '#ddd6fe'];
function renderTokens(ev) {
  const { root, body } = card('TOKENS', ev.title || '', el('span', { class: 'muted mono' }, `${ev.tokens.length} 个 token`));
  const vals = ev.values ? ev.values.map(num) : null, wrap = el('div', { class: 'tokens' });
  let mode = 'none', lo = 0, hi = 1, cats = [];
  if (vals) { const fin = vals.filter(Number.isFinite), uniq = [...new Set(fin)]; if (uniq.every((v) => v === 0 || v === 1)) mode = 'binary'; else if (uniq.every(Number.isInteger) && uniq.length <= 12) { mode = 'cat'; cats = uniq.sort((a, b) => a - b); } else { mode = 'seq'; lo = Math.min(...fin); hi = Math.max(...fin); if (!(hi > lo)) hi = lo + 1; } }
  const colorOf = (v) => { if (mode === 'binary') return v === 1 ? 'color-mix(in srgb, var(--accent) 38%, transparent)' : 'transparent'; if (mode === 'cat') return CAT[cats.indexOf(v) % CAT.length]; if (mode === 'seq' && Number.isFinite(v)) return `color-mix(in srgb, #ef4444 ${Math.round(((v - lo) / (hi - lo)) * 75)}%, transparent)`; return 'transparent'; };
  const lgd = el('div', { class: 'tok-legend' }), L = ev.legend;
  if (mode === 'binary') [[1, L?.['1'] ?? '1'], [0, L?.['0'] ?? '0']].forEach(([v, t]) => lgd.append(el('span', {}, [el('i', { style: `background:${colorOf(v)}` }), String(t)])));
  else if (mode === 'cat') cats.forEach((v) => lgd.append(el('span', {}, [el('i', { style: `background:${colorOf(v)}` }), String(L?.[String(v)] ?? v)])));
  else if (mode === 'seq') lgd.append(el('span', {}, `颜色越红数值越大： ${fmtNum(lo)} → ${fmtNum(hi)}` + (typeof L === 'string' ? ` （${L}）` : '')));
  if (typeof L === 'string' && mode !== 'seq') lgd.append(el('span', {}, L));
  ev.tokens.forEach((tk, i) => { const s = el('span', { class: 'tok' }, tk.replace(/\n/g, '↵').replace(/ /g, '·') || '∅'); if (vals) { s.style.background = colorOf(vals[i]); if (mode === 'cat') s.style.color = '#1d1b16'; if (mode === 'binary' && vals[i] !== 1) s.style.opacity = 0.55; } s.addEventListener('mousemove', (e) => showTip(e, `#${i}  ${JSON.stringify(tk)}` + (vals ? `\n值 = ${ev.values[i]}` : ''))); s.addEventListener('mouseleave', hideTip); wrap.append(s); });
  if (lgd.children.length) body.append(lgd); body.append(wrap); return root;
}

// ---------------------------------------------------------------- 输出流
export class OutputStream {
  constructor(container) { this.container = container; this.prevShapes = []; this.reset(); }
  reset() { this.container.innerHTML = ''; this.lastPre = null; this.live = new Map(); this.traceIdx = 0; this.newShapes = []; }
  finish() { this.prevShapes = this.newShapes; }
  append(node) { this.lastPre = null; this.container.append(node); }
  text(text, cls) { if (!this.lastPre || this.lastPre.dataset.cls !== (cls || '')) { const p = el('pre', { class: 'stdout ' + (cls || '') }); p.dataset.cls = cls || ''; this.container.append(p); this.lastPre = p; } this.lastPre.append(text); this.lastPre.scrollTop = this.lastPre.scrollHeight; }
  push(ev) {
    switch (ev.type) {
      case 'stdout': return this.text(ev.text);
      case 'error': return this.text(ev.traceback, 'err');
      case 'note': return this.append(el('div', { class: 'out-note', html: md(ev.md) }));
      case 'trace': { const node = renderTrace(ev, this.prevShapes[this.traceIdx]); this.newShapes[this.traceIdx++] = node._shapes; return this.append(node); }
      case 'tensors': return this.append(renderTensors(ev));
      case 'heatmap': return this.append(renderHeatmap(ev));
      case 'plot': return this.append(renderPlot(ev));
      case 'bars': return this.append(renderBars(ev));
      case 'table': return this.append(renderTable(ev));
      case 'tokens': return this.append(renderTokens(ev));
      case 'live': {
        let st = this.live.get(ev.chart);
        if (!st) { const { root, body } = card('LIVE', ev.chart); const holder = el('div', { class: 'plot' }); body.append(holder); st = { holder, series: {}, raf: 0 }; this.live.set(ev.chart, st); this.append(root); }
        const s = (st.series[ev.series] ||= { x: [], y: [] }); s.x.push(ev.x); s.y.push(ev.y);
        if (!st.raf) st.raf = requestAnimationFrame(() => { st.raf = 0; drawPlot(st.holder, { series: st.series, xlabel: 'step' }); });
        return;
      }
      default: return this.text(JSON.stringify(ev) + '\n', 'sys');
    }
  }
}
