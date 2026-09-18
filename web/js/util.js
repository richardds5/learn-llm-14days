// util.js —— DOM 小工具、markdown(+公式)渲染、源码抽屉、API 封装
export function el(tag, attrs = {}, children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'style') e.style.cssText = v; else e.setAttribute(k, v === true ? '' : v);
  }
  const add = (c) => { if (c == null || c === false) return; if (Array.isArray(c)) c.forEach(add); else e.append(c); };
  add(children);
  return e;
}

export function fmtNum(v, compact = false) {
  if (typeof v !== 'number') return String(v);
  if (!Number.isFinite(v)) return String(v);
  if (compact) { const a = Math.abs(v); if (a >= 1e9) return (v / 1e9).toFixed(2) + 'B'; if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M'; if (a >= 1e3) return (v / 1e3).toFixed(1) + 'K'; return String(v); }
  if (Number.isInteger(v)) return Math.abs(v) >= 1e15 ? v.toExponential(3) : String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return String(+v.toPrecision(4));
}
export function fmtBytes(b) { if (b >= 1 << 30) return (b / (1 << 30)).toFixed(2) + ' GB'; if (b >= 1 << 20) return (b / (1 << 20)).toFixed(2) + ' MB'; if (b >= 1024) return (b / 1024).toFixed(1) + ' KB'; return b + ' B'; }
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------- 代码高亮
export function highlightLine(src, lang = 'python') {
  if (!window.hljs || !src.trim()) return esc(src) || ' ';
  try { return window.hljs.highlight(src, { language: lang, ignoreIllegals: true }).value; } catch { return esc(src); }
}
const langOf = (path) => (/\.json$/.test(path || '') ? 'json' : /\.(md|txt)$/.test(path || '') ? 'plaintext' : /\.ya?ml$/.test(path || '') ? 'yaml' : 'python');
export function codeLines(lines, start, hot, path) {
  const box = el('div', { class: 'codelines' }), lang = langOf(path);
  lines.forEach((l, i) => { const ln = start + i; const row = el('div', { class: 'cl' + (hot && ln >= hot[0] && ln <= hot[1] ? ' hot' : '') }, [el('span', { class: 'ln' }, String(ln)), el('pre', { html: l.length > 600 ? esc(l.slice(0, 600)) + ' …' : highlightLine(l, lang) })]); row.dataset.ln = ln; box.append(row); });
  return box;
}

// ---------------------------------------------------------------- 源码抽屉
export async function openSource(path, from, to) {
  const drawer = document.getElementById('drawer'), body = document.getElementById('drawer-body');
  document.getElementById('drawer-title').textContent = path + (from ? `:${from}` + (to && to !== from ? `-${to}` : '') : '');
  body.innerHTML = '<div class="loading">读取中…</div>'; drawer.hidden = false;
  try {
    let s = await api(`/api/source?path=${encodeURIComponent(path)}&start=1&end=1`);
    const big = s.total > 2500, a = big ? Math.max(1, (from || 1) - 300) : 1, b = big ? Math.min(s.total, (to || from || 1) + 900) : s.total;
    s = await api(`/api/source?path=${encodeURIComponent(path)}&start=${a}&end=${b}`);
    document.getElementById('drawer-vscode').href = `vscode://file/${s.abs}:${from || 1}`;
    body.innerHTML = ''; if (big) body.append(el('div', { class: 'muted', style: 'padding:8px 16px' }, `文件共 ${s.total} 行，这里只显示 L${a}–L${b}；完整内容请用 VS Code 打开。`));
    body.append(codeLines(s.lines, a, from ? [from, to || from] : null, path));
    const target = body.querySelector(`[data-ln="${Math.max(1, (from || 1) - 4)}"]`); if (target) target.scrollIntoView({ block: 'start' });
  } catch (e) { body.innerHTML = `<div class="loading">读取失败：${esc(e.message)}</div>`; }
}
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('drawer-close')?.addEventListener('click', () => (document.getElementById('drawer').hidden = true));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') document.getElementById('drawer').hidden = true; });
});

// ---------------------------------------------------------------- markdown
const SRC_RE = /\b((?:model|dataset|trainer|scripts)\/[\w\-]+\.(?:py|json)|eval_llm\.py)(?::|#L)(\d+)(?:-L?(\d+))?/g;
function linkSrcRefs(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) => (n.parentElement.closest('pre, a, .katex, textarea') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
  const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((n) => {
    SRC_RE.lastIndex = 0; if (!SRC_RE.test(n.nodeValue)) return; SRC_RE.lastIndex = 0;
    const frag = document.createDocumentFragment(); let last = 0, m;
    while ((m = SRC_RE.exec(n.nodeValue))) {
      frag.append(n.nodeValue.slice(last, m.index));
      const [full, path, a, b] = m, link = el('a', { class: 'srcref', href: '#', title: '点击查看源码' }, full.replace('#L', ':').replace('-L', '-'));
      link.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); openSource(path, +a, b ? +b : +a); });
      frag.append(link); last = m.index + full.length;
    }
    frag.append(n.nodeValue.slice(last)); n.parentNode.replaceChild(frag, n);
  });
}

const CALLOUTS = { TIP: ['tip', '💡 提示'], WARNING: ['warning', '⚠️ 注意'], KEY: ['key', '🔑 关键结论'], QUESTION: ['question', '🤔 先想一想'], NOTE: ['tip', '📝 备注'], IMPORTANT: ['key', '🔑 重点'] };
function upgradeCallouts(root) {
  root.querySelectorAll('blockquote').forEach((bq) => {
    const p = bq.querySelector('p'); if (!p) return;
    const more = p.innerHTML.match(/^\s*\[!MORE\][ \t]*([^\n<]*)(?:<br>|\n)?\s*/i);
    if (more) { // 折叠的延伸内容：不在主线上的细节
      p.innerHTML = p.innerHTML.slice(more[0].length); if (!p.innerHTML.trim()) p.remove();
      const d = el('details', { class: 'more' }, [el('summary', { html: '延伸 · ' + (more[1].trim() || '想深入再点开') })]); const inner = el('div', { class: 'more-body' }); inner.append(...bq.childNodes); d.append(inner); bq.replaceWith(d); return;
    }
    const m = p.innerHTML.match(/^\s*\[!(\w+)\]\s*(?:<br>)?\s*/); if (!m || !CALLOUTS[m[1].toUpperCase()]) return;
    const [cls, title] = CALLOUTS[m[1].toUpperCase()]; p.innerHTML = p.innerHTML.slice(m[0].length); if (!p.innerHTML.trim()) p.remove();
    const box = el('div', { class: 'callout ' + cls }, [el('b', { class: 'ct' }, title)]); box.append(...bq.childNodes); bq.replaceWith(box);
  });
}

/** markdown → HTML 字符串。先把代码和公式抠出来保护好，再交给 marked，最后用 KaTeX 渲染公式。 */
export function mdToHtml(src) {
  const stash = [];
  const keep = (s) => { stash.push(s); return `§§C${stash.length - 1}§§C`; };
  let text = String(src ?? '');
  text = text.replace(/(^|\n)(```|~~~)[^\n]*\n[\s\S]*?\n\2[ \t]*(?=\n|$)/g, (m) => keep(m));
  text = text.replace(/`[^`\n]+`/g, (m) => keep(m));
  const math = [];
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => { math.push([t, true]); return `§§M${math.length - 1}§§M`; });
  text = text.replace(/(^|[^\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])/g, (_, pre, t) => { math.push([t, false]); return `${pre}§§M${math.length - 1}§§M`; });
  text = text.replace(/§§C(\d+)§§C/g, (_, i) => stash[+i]);
  let html = window.marked ? window.marked.parse(text, { gfm: true, breaks: false }) : `<pre>${esc(text)}</pre>`;
  html = html.replace(/§§M(\d+)§§M/g, (_, i) => { const [t, display] = math[+i]; try { return window.katex.renderToString(t, { displayMode: display, throwOnError: false }); } catch { return esc(t); } });
  return html;
}
export function md(src) { return mdToHtml(src); }

export function renderMarkdown(src) {
  const box = el('div', { html: mdToHtml(src) });
  box.querySelectorAll('pre code').forEach((c) => { const lang = (c.className.match(/language-(\w+)/) || [])[1]; if (window.hljs && lang && window.hljs.getLanguage(lang)) { try { c.innerHTML = window.hljs.highlight(c.textContent, { language: lang }).value; } catch {} } });
  upgradeCallouts(box); linkSrcRefs(box);
  return box;
}
