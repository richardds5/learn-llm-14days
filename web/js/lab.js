// lab.js —— 可运行实验：Monaco 编辑器 (带 diff) + 运行 + 输出流
import { el, api, mdToHtml } from './util.js';
import { OutputStream } from './viz.js';

const CDN = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min';
let monacoPromise = null;
export function loadMonaco() {
  if (!monacoPromise) monacoPromise = new Promise((resolve, reject) => {
    window.MonacoEnvironment = { getWorkerUrl: () => `data:text/javascript;charset=utf-8,${encodeURIComponent(`self.MonacoEnvironment={baseUrl:'${CDN}/'};importScripts('${CDN}/vs/base/worker/workerMain.js');`)}` };
    const s = el('script', { src: `${CDN}/vs/loader.js` });
    s.onload = () => { window.require.config({ paths: { vs: `${CDN}/vs` } }); window.require(['vs/editor/editor.main'], () => resolve(window.monaco), reject); };
    s.onerror = () => reject(new Error('Monaco 加载失败'));
    document.head.append(s);
    setTimeout(() => reject(new Error('Monaco 加载超时')), 15000);
  }).catch((e) => { console.warn(e); return null; });
  return monacoPromise;
}
const monacoTheme = () => (document.documentElement.dataset.theme === 'dark' ? 'vs-dark' : 'vs');
document.addEventListener('mm-theme', () => window.monaco?.editor.setTheme(monacoTheme()));

/** 极简行级 diff（Monaco 加载失败时的兜底） */
function simpleDiff(a, b) {
  const A = a.split('\n'), B = b.split('\n'), n = A.length, m = B.length, dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = []; let i = 0, j = 0;
  while (i < n && j < m) { if (A[i] === B[j]) { i++; j++; } else if (dp[i + 1][j] >= dp[i][j + 1]) out.push('- ' + A[i++]); else out.push('+ ' + B[j++]); }
  while (i < n) out.push('- ' + A[i++]); while (j < m) out.push('+ ' + B[j++]);
  return out.join('\n') || '（没有改动）';
}

export class Lab {
  /** opts: {day, lab:{id,title,timeout,sources,tasks,code,saved}, progress, onProgress(path, value), micro?, onRun?(ok)} */
  constructor(opts) {
    Object.assign(this, opts);
    const saved = this.lab.saved || {};
    this.files = [{ key: '@lab', label: '实验脚本', original: this.lab.code, current: saved.code ?? this.lab.code, loaded: true }];
    (this.lab.sources || []).forEach((p) => this.files.push({ key: p, label: p, original: null, current: saved.overrides?.[p] ?? null, loaded: false }));
    this.active = 0; this.diff = false; this.job = null; this.models = new Map();
    this.build();
  }

  build() {
    const L = this.lab, prog = this.progress?.labs?.[L.id];
    this.statusEl = el('span', { class: 'run-status' }, prog ? `已运行 ${prog.runs} 次` : '');
    this.runBtn = el('button', { class: 'btn primary', type: 'button', title: '⌘/Ctrl + Enter' }, '▶ 运行');
    this.stopBtn = el('button', { class: 'btn danger', type: 'button', hidden: true }, '■ 停止');
    this.diffBtn = el('button', { class: 'btn', type: 'button', title: '对比原始代码和你的修改' }, '⇄ Diff');
    this.resetBtn = el('button', { class: 'btn', type: 'button', title: '把当前标签页恢复成原始代码' }, '↺ 还原');
    this.fullBtn = el('button', { class: 'btn ghost', type: 'button', title: '全屏' }, '⛶');
    this.tabsEl = el('div', { class: 'tabs' });
    this.editorEl = el('div', { class: 'editor' });
    this.outEl = el('div', { class: 'output' });
    this.out = new OutputStream(this.outEl);
    const head = el('div', { class: 'lab-head' }, [el('span', { class: 'tag' }, 'LAB'), el('h4', {}, L.title), this.statusEl]);
    const bar = el('div', { class: 'lab-bar' }, [this.tabsEl, el('span', { style: 'flex:1' }), this.diffBtn, this.resetBtn, this.fullBtn, this.stopBtn, this.runBtn]);
    this.root = el('div', { class: 'lab', id: `lab-${L.id}` }, [head]); this.root._lab = this;
    let tasksBox = null;
    if (L.tasks?.length) {
      const done = this.progress?.tasks?.[L.id] || [];
      const box = tasksBox = el('div', { class: 'tasks' + (this.micro ? ' below' : '') }, [el('b', {}, this.micro ? '🔧 改一处，再跑一次' : '动手改一改')]);
      L.tasks.forEach((t, i) => { const cb = el('input', { type: 'checkbox' }); cb.checked = !!done[i]; const row = el('label', { class: 'task' + (done[i] ? ' done' : '') }, [cb, el('span', { html: mdToHtml(String(t)) })]); cb.addEventListener('change', () => { done[i] = cb.checked; row.classList.toggle('done', cb.checked); this.onProgress(['tasks', L.id], done.slice()); }); box.append(row); });
      if (!this.micro) this.root.append(box);
    }
    this.root.append(bar, el('div', { class: 'editor-wrap' }, [this.editorEl]), this.outEl);
    if (this.micro && tasksBox) this.root.append(tasksBox);  // 微实验：先跑、看结果，再改
    this.runBtn.addEventListener('click', () => this.run());
    this.stopBtn.addEventListener('click', () => this.job && api(`/api/job/${this.job}/stop`, {}));
    this.diffBtn.addEventListener('click', () => this.toggleDiff());
    this.resetBtn.addEventListener('click', () => this.resetActive());
    this.fullBtn.addEventListener('click', () => { this.root.classList.toggle('full'); this.layout(); });
    this.renderTabs();
    new IntersectionObserver((ents, obs) => { if (ents.some((e) => e.isIntersecting)) { obs.disconnect(); this.mountEditor(); } }, { rootMargin: '600px' }).observe(this.root);
  }

  renderTabs() {
    this.tabsEl.innerHTML = '';
    this.files.forEach((f, i) => { const dirty = f.loaded && f.current !== f.original; const t = el('button', { class: 'tab' + (i === this.active ? ' active' : '') + (dirty ? ' dirty' : ''), type: 'button', title: i ? '仓库源码：可以直接改，只对本实验的运行生效，不会写回仓库' : '' }, [i ? '📄 ' : '🧪 ', f.label, el('span', { class: 'dot', title: '已修改' })]); t.addEventListener('click', () => this.switchTo(i)); this.tabsEl.append(t); });
    this.tabsEl.hidden = this.files.length < 2;
  }

  async ensureLoaded(f) {
    if (f.loaded) return;
    const s = await api(`/api/source?path=${encodeURIComponent(f.key)}`);
    f.original = s.lines.join('\n'); if (f.current == null) f.current = f.original; f.loaded = true;
  }

  async mountEditor() {
    const monaco = await loadMonaco();
    if (!monaco) { // 兜底：textarea
      this.textarea = el('textarea', { class: 'fallback', spellcheck: 'false' }); this.textarea.value = this.files[0].current;
      this.textarea.addEventListener('input', () => { this.files[this.active].current = this.textarea.value; this.changed(); });
      this.textarea.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); this.run(); } });
      this.editorEl.replaceWith(this.textarea); return;
    }
    this.monaco = monaco;
    const common = { theme: monacoTheme(), fontSize: 13, lineHeight: 20, minimap: { enabled: false }, scrollBeyondLastLine: false, automaticLayout: true, tabSize: 4, renderLineHighlight: 'line', scrollbar: { alwaysConsumeMouseWheel: false, verticalScrollbarSize: 10 }, padding: { top: 8, bottom: 8 }, fontFamily: 'JetBrains Mono, SF Mono, Menlo, monospace', wordWrap: 'off', unicodeHighlight: { ambiguousCharacters: false, invisibleCharacters: false } };
    this.editor = monaco.editor.create(this.editorEl, { ...common, model: this.modelFor(this.files[0]) });
    this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => this.run());
    this.editor.onDidContentSizeChange(() => this.layout());
    this.common = common; this.layout();
  }

  modelFor(f) {
    let m = this.models.get(f.key);
    if (!m) { m = this.monaco.editor.createModel(f.current ?? '', 'python'); m.onDidChangeContent(() => { f.current = m.getValue(); this.changed(); }); this.models.set(f.key, m); }
    return m;
  }

  layout() {
    const ed = this.diff ? this.diffEditor?.getModifiedEditor() : this.editor; if (!ed) return;
    const full = this.root.classList.contains('full'), h = Math.max(140, Math.min(ed.getContentHeight() + 4, full ? innerHeight * 0.62 : 560));
    this.editorEl.style.height = h + 'px'; (this.diff ? this.diffEditor : this.editor).layout();
  }

  async switchTo(i) {
    const f = this.files[i]; await this.ensureLoaded(f); this.active = i; this.renderTabs();
    if (this.textarea) { this.textarea.value = f.current; return; }
    if (!this.monaco) return;
    if (this.diff) this.showDiff(); else { this.editor.setModel(this.modelFor(f)); this.layout(); }
  }

  toggleDiff() {
    this.diff = !this.diff; this.diffBtn.classList.toggle('on', this.diff);
    const f = this.files[this.active];
    if (this.textarea) { if (this.diff) { this.out.reset(); this.out.text(simpleDiff(f.original, f.current) + '\n'); } this.diff = false; this.diffBtn.classList.remove('on'); return; }
    if (!this.monaco) return;
    if (this.diff) { this.editor.dispose(); this.editor = null; this.editorEl.innerHTML = ''; this.showDiff(); }
    else { this.diffEditor.dispose(); this.diffEditor = null; this.origModel?.dispose(); this.origModel = null; this.editorEl.innerHTML = ''; this.editor = this.monaco.editor.create(this.editorEl, { ...this.common, theme: monacoTheme(), model: this.modelFor(f) }); this.editor.addCommand(this.monaco.KeyMod.CtrlCmd | this.monaco.KeyCode.Enter, () => this.run()); this.editor.onDidContentSizeChange(() => this.layout()); this.layout(); }
  }

  showDiff() {
    const f = this.files[this.active], monaco = this.monaco;
    if (!this.diffEditor) { this.diffEditor = monaco.editor.createDiffEditor(this.editorEl, { ...this.common, theme: monacoTheme(), renderSideBySide: this.editorEl.clientWidth > 900, originalEditable: false, ignoreTrimWhitespace: false, hideUnchangedRegions: { enabled: true, contextLineCount: 3, minimumLineCount: 5, revealLineCount: 10 } }); this.diffEditor.getModifiedEditor().addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => this.run()); this.diffEditor.getModifiedEditor().onDidContentSizeChange(() => this.layout()); }
    this.origModel?.dispose(); this.origModel = monaco.editor.createModel(f.original, 'python');
    this.diffEditor.setModel({ original: this.origModel, modified: this.modelFor(f) }); this.layout();
  }

  resetActive() {
    const f = this.files[this.active]; if (!f.loaded || f.current === f.original) return;
    if (!confirm(`把「${f.label}」恢复成原始内容？你的修改会丢失。`)) return;
    f.current = f.original; if (this.textarea) this.textarea.value = f.original; else this.models.get(f.key)?.setValue(f.original);
    this.changed();
  }

  changed() {
    this.renderTabs(); clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => {
      const code = this.files[0].current !== this.files[0].original ? this.files[0].current : null, overrides = this.overrides();
      api('/api/workspace', { day: this.day, lab: this.lab.id, code, overrides }).catch(() => {});
    }, 900);
  }

  /** trace 里点 file:line 时调用：如果这个文件是本实验可编辑的源码，就切到那个标签页并定位到该行 */
  async revealSource(path, line) {
    const i = this.files.findIndex((f) => f.key === path); if (i < 0 || !this.monaco) return false;
    if (this.diff) this.toggleDiff();
    await this.switchTo(i); this.editor.revealLineInCenter(line); this.editor.setPosition({ lineNumber: line, column: 1 }); this.editor.focus();
    this.root.querySelector('.editor-wrap').scrollIntoView({ block: 'center', behavior: 'smooth' }); return true;
  }

  overrides() { const o = {}; this.files.slice(1).forEach((f) => { if (f.loaded && f.current !== f.original) o[f.key] = f.current; }); return o; }

  setStatus(text, cls, spin) { this.statusEl.className = 'run-status ' + (cls || ''); this.statusEl.innerHTML = ''; if (spin) this.statusEl.append(el('span', { class: 'spinner' })); this.statusEl.append(text); }

  async run() {
    if (this.job) return;
    this.out.reset(); this.runBtn.disabled = true; this.stopBtn.hidden = false; this.setStatus('启动中…', '', true);
    const t0 = performance.now();
    try {
      const { job } = await api('/api/run', { code: this.files[0].current, overrides: this.overrides(), timeout: this.lab.timeout || 60 });
      this.job = job; let ev = 0, log = 0, sysLog = '';
      for (;;) {
        const r = await api(`/api/job/${job}?ev=${ev}&log=${log}`);
        ev = r.ev; log = r.log_off; sysLog += r.log; r.events.forEach((e) => this.out.push(e));
        this.setStatus(`运行中 ${((performance.now() - t0) / 1000).toFixed(1)}s`, '', true);
        if (r.done) {
          const clean = sysLog.replace(/\[runner\][^\n]*\n/g, '').trim();
          if (clean) this.out.text((r.exit === 0 ? '── stderr / 系统日志 ──\n' : '') + clean + '\n', r.exit === 0 ? 'sys' : 'err');
          if (r.timed_out) this.out.text(`⏱ 超过 ${r.timeout}s 被终止。\n`, 'err');
          const ok = r.exit === 0; this.setStatus(ok ? `✓ 完成 · ${r.elapsed}s` : `✗ 退出码 ${r.exit} · ${r.elapsed}s`, ok ? 'ok' : 'bad');
          const prev = this.progress?.labs?.[this.lab.id] || { runs: 0 };
          this.onProgress(['labs', this.lab.id], { runs: prev.runs + 1, ok: ok || !!prev.ok, ts: Date.now() });
          this.onRun?.(ok);
          break;
        }
        await new Promise((res) => setTimeout(res, 250));
      }
    } catch (e) { this.out.text(`运行失败：${e.message}\n（server.py 还在运行吗？）\n`, 'err'); this.setStatus('✗ 出错', 'bad'); }
    this.out.finish(); this.job = null; this.runBtn.disabled = false; this.stopBtn.hidden = true;
  }

  dispose() { this.editor?.dispose(); this.diffEditor?.dispose(); this.origModel?.dispose(); this.models.forEach((m) => m.dispose()); if (this.job) api(`/api/job/${this.job}/stop`, {}).catch(() => {}); }
}
