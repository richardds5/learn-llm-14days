// map.js —— 模型全景图：调参数 → 真实跑一次 forward → 把每条边上的 tensor 维度画出来
import { el, api } from './util.js';
import { chip, renderTrace, dimEl } from './viz.js';

const FIELDS = [
  ['B', 'batch', 3], ['T', '序列长度', 7], ['hidden_size', 'C · hidden_size', 768], ['num_hidden_layers', '层数', 2],
  ['num_attention_heads', 'H · query 头数', 8], ['num_key_value_heads', 'KV · kv 头数', 4], ['vocab_size', 'V · 词表', 6400],
];

function genCode(v) {
  const mlp = v.use_moe ? 'MOEFeedForward' : 'FeedForward';
  return `import torch
from learnkit import *
from model.model_minimind import *

model = build_model(hidden_size=${v.hidden_size}, num_hidden_layers=${v.num_hidden_layers}, num_attention_heads=${v.num_attention_heads},
                    num_key_value_heads=${v.num_key_value_heads}, vocab_size=${v.vocab_size}, use_moe=${v.use_moe ? 'True' : 'False'}, num_experts_per_tok=${v.topk}, flash_attn=False)
input_ids = torch.randint(0, ${v.vocab_size}, (${v.B}, ${v.T}))
fns = [MiniMindForCausalLM.forward, MiniMindModel.forward, MiniMindBlock.forward, Attention.forward, ${mlp}.forward,
       RMSNorm.forward, apply_rotary_pos_emb, repeat_kv]
with trace(model, fns=fns, dims=dict(B=${v.B}, T=${v.T}), title="MiniMind 全景", max_calls=1):
    out = model(input_ids, labels=input_ids)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M   loss(随机初始化≈ln V={torch.log(torch.tensor(${v.vocab_size}.0)):.2f}): {out.loss.item():.3f}")
`;
}

const find = (node, name) => { if (node.name === name) return node; for (const c of node.children) { const r = find(c, name); if (r) return r; } return null; };
const edge = (desc, name) => el('div', { class: 'edge' + (desc ? '' : ' noshape') }, desc ? [chip(name ?? null, desc, { small: true })] : []);

export function renderMap() {
  const page = el('div', { class: 'page' });
  page.append(el('div', { class: 'lesson-head' }, [el('div', { class: 'kicker' }, 'MODEL MAP'), el('h1', {}, '模型全景图'), el('p', { class: 'sub' }, '改参数 → 真实跑一次仓库里的 forward → 每条边上都是实测的 tensor 维度。点任意模块，右边显示它 forward 里逐行的数据流。')]));
  const inputs = {}, ctrls = el('div', { class: 'ctrls' });
  FIELDS.forEach(([k, label, dv]) => { const i = el('input', { type: 'number', value: dv, min: 1 }); inputs[k] = i; ctrls.append(el('label', { class: 'ctrl' }, [label, i])); });
  const moe = el('select', {}, [el('option', { value: '0' }, 'Dense FFN'), el('option', { value: '1' }, 'MoE (4 experts)')]); const topk = el('input', { type: 'number', value: 1, min: 1, max: 4 });
  ctrls.append(el('label', { class: 'ctrl' }, ['FFN 类型', moe]), el('label', { class: 'ctrl' }, ['MoE top-k', topk]));
  const runBtn = el('button', { class: 'btn primary', type: 'button' }, '▶ 运行并绘制'), status = el('span', { class: 'run-status' });
  ctrls.append(runBtn, status); page.append(ctrls);
  const stage = el('div'), detail = el('div', { id: 'map-detail' }); page.append(stage, detail);

  const draw = (ev) => {
    stage.innerHTML = ''; detail.innerHTML = '';
    const root = ev.modules?.[0]; if (!root) return;
    const N = +inputs.num_hidden_layers.value, fnsByName = Object.fromEntries(ev.fns.map((f) => [f.name, f]));
    const colC = el('div', { class: 'arch' }); let selected = null;
    const showFn = (fnName, title, nodeEl) => {
      if (selected) selected.classList.remove('sel'); selected = nodeEl; nodeEl?.classList.add('sel');
      colC.innerHTML = ''; const fn = fnsByName[fnName];
      colC.append(el('h4', {}, [el('span', {}, `③ ${title} 内部的逐行数据流`), el('small', {}, fn ? `${fn.file}:${fn.first_line}` : '')]));
      if (!fn || !fn.calls.length) { colC.append(el('div', { class: 'muted' }, '这个模块是 PyTorch 内置层 (nn.Linear / nn.Embedding…)，没有可追踪的 python forward；它的输入输出维度见左边连线。')); return; }
      const flow = el('div', { class: 'flowcol' }), call = fn.calls[0];
      call.events.forEach((e) => { if (!e.writes.length) return; const src = fn.source[e.line - fn.first_line]?.trim() || '';
        const tens = e.writes.filter((w) => w.desc.k === 'tensor' || w.desc.k === 'seq'); if (!tens.length) return;
        flow.append(el('div', { class: 'node op', title: src, style: 'min-width:0;max-width:100%;text-align:left;font-weight:400;font-size:11px' }, [el('small', {}, e.is_args ? '入参' : `L${e.line}`), e.is_args ? '' : src.length > 64 ? src.slice(0, 62) + '…' : src]));
        const eg = el('div', { class: 'edge' }); const box = el('div', { style: 'display:flex;flex-wrap:wrap;gap:3px;justify-content:center' }); tens.forEach((w) => box.append(chip(w.name, w.desc, { small: true, change: w.change, context: { event: e, src, meanings: ev.dim_meanings } }))); eg.append(box); flow.append(eg); });
      if (call.ret) flow.append(el('div', { class: 'node op', style: 'min-width:0' }, 'return'), edge(call.ret));
      colC.append(flow);
    };
    const node = (label, sub, fnName, title) => { const n = el('div', { class: 'node' }, [label, sub ? el('small', {}, sub) : null]); n.addEventListener('click', () => showFn(fnName, title || label, n)); return n; };

    // ① 宏观
    const model = find(root, 'model'), emb = find(root, 'model.embed_tokens'), l0 = find(root, 'model.layers.0'), norm = find(root, 'model.norm'), head = find(root, 'lm_head');
    const colA = el('div', { class: 'arch' }, [el('h4', {}, [el('span', {}, '① 整体：MiniMindForCausalLM'), el('small', {}, `${(root.params / 1e6).toFixed(2)}M 参数`)])]);
    const fa = el('div', { class: 'flowcol' });
    fa.append(el('div', { class: 'node op' }, 'input_ids'), edge(emb.inputs[0]), node('embed_tokens', `nn.Embedding(V, C) · ${(emb.params / 1e6).toFixed(2)}M`, 'Embedding.forward'), edge(emb.output));
    const g = el('div', { class: 'group' }, [el('span', { class: 'glabel' }, `× ${N} 层`)]); const blockNode = node('MiniMindBlock', `layers.0 … layers.${N - 1} · 每层 ${(l0.params / 1e6).toFixed(2)}M`, 'MiniMindBlock.forward'); g.append(blockNode); fa.append(g, edge(l0.output?.items?.[0] || l0.output));
    fa.append(node('norm', 'RMSNorm(C)', 'RMSNorm.forward', 'RMSNorm'), edge(norm.output), node('lm_head', `nn.Linear(C, V) · 与 embed_tokens 共享权重`, 'Linear.forward'), edge(head.output, 'logits'));
    fa.append(node('shift + cross_entropy', 'MiniMindForCausalLM.forward', 'MiniMindForCausalLM.forward', 'MiniMindForCausalLM.forward'), edge(root.output?.items?.loss, 'loss'));
    const topNode = node('↑ 看 MiniMindModel.forward 的逐行维度', null, 'MiniMindModel.forward', 'MiniMindModel.forward'); topNode.style.marginTop = '12px'; fa.append(topNode);
    colA.append(fa);

    // ② Block 内部
    const ln1 = find(l0, 'model.layers.0.input_layernorm'), attn = find(l0, 'model.layers.0.self_attn'), ln2 = find(l0, 'model.layers.0.post_attention_layernorm'), mlp = find(l0, 'model.layers.0.mlp');
    const colB = el('div', { class: 'arch' }, [el('h4', {}, [el('span', {}, '② 一个 Block 内部 (pre-norm 残差)'), el('small', {}, 'layers.0')])]);
    const fb = el('div', { class: 'flowcol' }); fb.append(el('div', { class: 'node op' }, 'hidden_states'), edge(l0.inputs[0]));
    const r1 = el('div', { class: 'resid' }); r1.append(node('input_layernorm', 'RMSNorm(C)', 'RMSNorm.forward', 'RMSNorm'), edge(ln1.output), node('self_attn', `Attention · GQA H=${inputs.num_attention_heads.value} / KV=${inputs.num_key_value_heads.value} · ${(attn.params / 1e6).toFixed(2)}M`, 'Attention.forward', 'Attention'), edge(attn.output?.items?.[0]));
    const r2 = el('div', { class: 'resid' }); const mlpFn = mlp.cls + '.forward'; r2.append(node('post_attention_layernorm', 'RMSNorm(C)', 'RMSNorm.forward', 'RMSNorm'), edge(ln2.output), node('mlp', `${mlp.cls} · ${(mlp.params / 1e6).toFixed(2)}M`, mlpFn, mlp.cls), edge(mlp.output));
    fb.append(r1, el('div', { class: 'node op' }, '⊕  hidden_states += residual'), edge(l0.inputs[0]), el('div', { style: 'height:14px' }), r2, el('div', { class: 'node op' }, '⊕  hidden_states + mlp(...)'), edge(l0.output?.items?.[0]));
    const extra = el('div', { class: 'split', style: 'margin-top:12px' }); [['RoPE', 'apply_rotary_pos_emb', 'apply_rotary_pos_emb'], ['repeat_kv', 'repeat_kv', 'repeat_kv']].forEach(([lb, fnn, tt]) => { const n = node(lb, null, fnn, tt); n.style.minWidth = '100px'; extra.append(n); }); fb.append(extra);
    colB.append(fb);

    stage.append(el('div', { class: 'legend', style: 'margin:4px 0 14px' }, Object.entries(ev.dims).map(([k, v]) => { const d = dimEl(v, k); d.title = ev.dim_meanings?.[k] || ''; return d; })));
    stage.append(el('div', { class: 'archs' }, [colA, colB, colC]));
    showFn('Attention.forward', 'Attention', r1.querySelectorAll('.node')[1]);
    detail.append(el('h3', {}, '完整追踪（源码逐行 + 模块树）'), renderTrace(ev));
  };

  const run = async () => {
    const v = Object.fromEntries(Object.entries(inputs).map(([k, i]) => [k, Math.max(1, parseInt(i.value) || 1)])); v.use_moe = moe.value === '1'; v.topk = Math.max(1, parseInt(topk.value) || 1);
    if (v.hidden_size % v.num_attention_heads) return (status.textContent = '✗ hidden_size 必须能被 H 整除', status.className = 'run-status bad');
    if (v.num_attention_heads % v.num_key_value_heads) return (status.textContent = '✗ H 必须能被 KV 整除', status.className = 'run-status bad');
    if ((v.hidden_size / v.num_attention_heads) % 2) return (status.textContent = '✗ head_dim 必须是偶数 (RoPE 两两配对)', status.className = 'run-status bad');
    runBtn.disabled = true; status.className = 'run-status'; status.innerHTML = ''; status.append(el('span', { class: 'spinner' }), ' 运行中…');
    try {
      const { job } = await api('/api/run', { code: genCode(v), timeout: 120 }); let evOff = 0, log = '', text = '', traceEv = null, err = null;
      for (;;) { const r = await api(`/api/job/${job}?ev=${evOff}&log=0`); evOff = r.ev; log = r.log; r.events.forEach((e) => { if (e.type === 'trace') traceEv = e; else if (e.type === 'stdout') text += e.text; else if (e.type === 'error') err = e.traceback; }); if (r.done) break; await new Promise((res) => setTimeout(res, 250)); }
      if (traceEv) { draw(traceEv); status.className = 'run-status ok'; status.textContent = '✓ ' + text.trim(); } else { status.className = 'run-status bad'; status.textContent = '✗ 运行失败'; stage.innerHTML = ''; stage.append(el('pre', { class: 'stdout err' }, err || log || '未知错误')); }
    } catch (e) { status.className = 'run-status bad'; status.textContent = '✗ ' + e.message; }
    runBtn.disabled = false;
  };
  runBtn.addEventListener('click', run); queueMicrotask(run);
  return page;
}
