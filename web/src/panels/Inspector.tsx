import { useEffect, useState } from 'react';
import { analysisKey } from '../canvas/flow';
import { categoryColor } from '../canvas/Nodes';
import { formatCount, graphOf, isNest, nestTarget, parseAttrInput, subgraphUsers, ungroup, uniqueName } from '../lib/docops';
import { evaluate } from '../lib/expr';
import { useHub } from '../lib/hub';
import { keyOfPath, requestRelayout, useStore } from '../lib/store';
import type { AttrSpec, AttrValue, Doc, GNode, GraphKey } from '../lib/types';

export function Inspector() {
  const doc = useStore((s) => s.doc);
  const selection = useStore((s) => s.selection);
  if (!doc) return null;
  const ref = selection.length === 1 ? selection[0] : null;
  const key = ref ? keyOfPath(doc, ref.path) : null;
  const node = ref && key !== null ? graphOf(doc, key).nodes.find((n) => n.id === ref.id) : undefined;
  return (
    <aside className="inspector">
      {node && key !== null ? <NodePanel doc={doc} graphKey={key} path={ref!.path} node={node} /> : <DocPanel doc={doc} count={selection.length} />}
    </aside>
  );
}

// ---- shared bits ----------------------------------------------------------------

function useVars() {
  return useStore((s) => s.analysis?.variables ?? {});
}

function TextField({ value, onCommit, placeholder, mono }: { value: string; onCommit: (v: string) => void; placeholder?: string; mono?: boolean }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return (
    <input
      className={mono ? 'mono' : undefined}
      value={text}
      placeholder={placeholder}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => text !== value && onCommit(text)}
      onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
    />
  );
}

function Hint({ text, vars }: { text: string; vars: Record<string, number> }) {
  if (!/[A-Za-z$]/.test(text) || /^(true|false)$/.test(text)) return null;
  try {
    const v = evaluate(text, vars);
    return <span className="hint ok">= {Number.isInteger(v) ? v : v.toPrecision(6)}</span>;
  } catch (e) {
    return <span className="hint bad">{(e as Error).message}</span>;
  }
}

// ---- node panel ------------------------------------------------------------------

function NodePanel({ doc, graphKey, path, node }: { doc: Doc; graphKey: GraphKey; path: number[]; node: GNode }) {
  const specs = useStore((s) => s.specs);
  const analysis = useStore((s) => s.analysis);
  const { edit, toast, openScripts, select } = useStore.getState();
  const vars = useVars();
  const spec = specs[node.type];
  const ga = analysis?.graphs[analysisKey(graphKey)];
  const info = ga?.nodes[String(node.id)];
  const error = ga?.errors[String(node.id)];

  const update = (fn: (n: GNode) => void) =>
    edit((d) => {
      const n = graphOf(d, graphKey).nodes.find((m) => m.id === node.id);
      if (n) fn(n);
    });

  const rename = (name: string) => {
    const g = graphOf(doc, graphKey);
    if (!/^[A-Za-z_][\w.]*$/.test(name)) return toast('Names are letters, digits, _ and . (they prefix the weight names)', true);
    if (g.nodes.some((n) => n.name === name && n.id !== node.id)) return toast(`'${name}' is already used in this graph`, true);
    update((n) => (n.name = name));
  };

  const scriptFile = spec && spec.source !== 'builtin' ? spec.source.split(/[\\/]/).pop()! : null;

  return (
    <div className="panel-scroll">
      <div className="insp-title">
        <span className="dot big" style={{ background: categoryColor(spec?.category) }} />
        <div>
          <div className="insp-type">{node.type}</div>
          <div className="insp-cat">{spec?.category ?? 'unknown type'}</div>
        </div>
      </div>
      {spec?.description && <p className="desc">{spec.description}</p>}
      {scriptFile && (
        <button className="secondary full" onClick={() => openScripts(scriptFile)}>
          Edit Python source
        </button>
      )}
      {error && <div className="error-box">{error}</div>}

      <h4>Node</h4>
      <label className="field">
        <span>name</span>
        <TextField value={node.name} onCommit={rename} mono />
      </label>
      <p className="note">
        The name prefixes this node's weights, e.g. <code>{isNest(node) ? `${node.name}${node.type === 'Repeat' ? '.0' : ''}.<inner node>.weight` : `${node.name}.weight`}</code>.
      </p>

      {isNest(node) && <NestSection doc={doc} graphKey={graphKey} path={path} node={node} />}

      {spec && spec.attrs.filter((a) => a.name !== 'subgraph' && !(a.name === 'index' && spec.role !== 'op')).length > 0 && (
        <>
          <h4>Attributes</h4>
          {spec.attrs
            .filter((a) => a.name !== 'subgraph' && !(a.name === 'index' && spec.role !== 'op'))
            .map((a) => (
              <AttrField key={a.name} spec={a} value={node.attrs[a.name] ?? a.default} vars={vars} doc={doc} graphKey={graphKey} self={node} onChange={(v) => update((n) => (n.attrs[a.name] = v))} />
            ))}
        </>
      )}

      <h4>Shapes</h4>
      <table className="shapes">
        <tbody>
          {(info?.ports.inputs ?? spec?.inputs ?? []).map((p, i) => (
            <tr key={`i${p.name}`}>
              <td className="dim">in</td>
              <td>{p.name}</td>
              <td className="mono">{info?.inputs?.[i]?.text ?? '—'}</td>
            </tr>
          ))}
          {(info?.ports.outputs ?? spec?.outputs ?? []).map((p, i) => (
            <tr key={`o${p.name}`}>
              <td className="dim">out</td>
              <td>{p.name}</td>
              <td className="mono">{info?.outputs?.[i]?.text ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {info?.params !== undefined && (
        <p className="note">
          Parameters: <b>{formatCount(info.params)}</b>
        </p>
      )}
      <div className="row gap">
        <button className="ghost" onClick={() => select([])}>Deselect</button>
        <button className="ghost danger" onClick={() => window.dispatchEvent(new CustomEvent('mg-command', { detail: 'delete' }))}>
          Delete node
        </button>
      </div>
    </div>
  );
}

function NestSection({ doc, graphKey, path, node }: { doc: Doc; graphKey: GraphKey; path: number[]; node: GNode }) {
  const { edit, toast, select } = useStore.getState();
  const vars = useVars();
  const sub = nestTarget(node);
  const users = sub ? subgraphUsers(doc, sub) : 0;
  const update = (fn: (n: GNode, d: Doc) => void, history = true) =>
    edit(
      (d) => {
        const n = graphOf(d, graphKey).nodes.find((m) => m.id === node.id);
        if (n) fn(n, d);
      },
      { history },
    );

  const renameSub = (name: string) => {
    if (!/^[A-Za-z_]\w*$/.test(name)) return toast('Subgraph names are identifiers', true);
    if (doc.subgraphs[name]) return toast(`'${name}' already exists`, true);
    edit((d) => {
      d.subgraphs[name] = d.subgraphs[sub];
      delete d.subgraphs[sub];
      for (const g of [d.graph, ...Object.values(d.subgraphs)]) for (const n of g.nodes) if (isNest(n) && n.attrs.subgraph === sub) n.attrs.subgraph = name;
    });
  };

  return (
    <>
      <h4>Nest</h4>
      <label className="field">
        <span>contents</span>
        <select value={sub} onChange={(e) => update((n) => (n.attrs.subgraph = e.target.value))}>
          {!sub && <option value="">choose…</option>}
          {Object.keys(doc.subgraphs).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </label>
      {sub && (
        <label className="field">
          <span>rename</span>
          <TextField value={sub} onCommit={renameSub} mono />
        </label>
      )}
      {users > 1 && <p className="note">Shared by {users} nests: editing it changes all of them. Their weights stay separate.</p>}
      {node.type === 'Repeat' && (
        <label className="field">
          <span>count</span>
          <TextField value={String(node.attrs.count ?? '')} onCommit={(v) => update((n) => (n.attrs.count = parseAttrInput(v, 'int')))} mono />
          <Hint text={String(node.attrs.count ?? '')} vars={vars} />
        </label>
      )}
      <div className="row gap wrap">
        <button
          className="secondary"
          onClick={() => {
            update((n) => (n.ui = { ...(n.ui ?? {}), expanded: !node.ui?.expanded }), false);
            requestRelayout(path);
          }}
        >
          {node.ui?.expanded ? 'Collapse' : 'Expand'}
        </button>
        {node.type === 'Group' ? (
          <button className="secondary" onClick={() => update((n) => { n.type = 'Repeat'; n.attrs.count = 2; })}>Repeat ×N</button>
        ) : (
          <button className="secondary" onClick={() => update((n) => { n.type = 'Group'; delete n.attrs.count; })}>Single copy</button>
        )}
        {node.type === 'Group' && (
          <button
            className="secondary"
            onClick={() => {
              edit((d) => ungroup(d, graphKey, node.id));
              select([]);
            }}
          >
            Unwrap
          </button>
        )}
        {users > 1 && (
          <button
            className="secondary"
            title="Give this nest its own copy of the subgraph so you can change it independently"
            onClick={() =>
              update((n, d) => {
                let name = `${sub}_${n.name.replace(/\W/g, '_')}`;
                for (let i = 2; d.subgraphs[name]; i++) name = `${sub}_${i}`;
                d.subgraphs[name] = structuredClone(d.subgraphs[sub]);
                n.attrs.subgraph = name;
              })
            }
          >
            Make unique
          </button>
        )}
      </div>
    </>
  );
}

function AttrField({ spec, value, vars, doc, graphKey, self, onChange }: { spec: AttrSpec; value: AttrValue; vars: Record<string, number>; doc: Doc; graphKey: GraphKey; self: GNode; onChange: (v: AttrValue) => void }) {
  const text = String(value);
  let control: React.ReactNode;
  if (spec.kind === 'bool') {
    control = (
      <label className="switch">
        <input type="checkbox" checked={value === true || value === 'true' || value === 1} onChange={(e) => onChange(e.target.checked)} />
        <span />
      </label>
    );
  } else if (spec.kind === 'enum' || spec.kind === 'noderef') {
    const options = spec.kind === 'enum' ? spec.options : graphOf(doc, graphKey).nodes.filter((n) => n.id !== self.id).map((n) => n.name);
    control = (
      <select value={text} onChange={(e) => onChange(e.target.value)}>
        {!options.includes(text) && <option value={text}>{text}</option>}
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    );
  } else {
    const kind = spec.kind === 'int' ? 'int' : spec.kind === 'float' ? 'float' : 'other';
    control = (
      <>
        <TextField value={text} onCommit={(v) => onChange(parseAttrInput(v, kind))} mono />
        {kind !== 'other' && <Hint text={text} vars={vars} />}
      </>
    );
  }
  return (
    <label className="field" title={spec.help}>
      <span>{spec.name}</span>
      {control}
    </label>
  );
}

// ---- Hugging Face download ----------------------------------------------------

function mb(n = 0) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${Math.round(n / 1e6).toLocaleString()} MB`;
}

function HubDownload() {
  const status = useHub((s) => s.status);
  const { start, cancel, resume } = useHub.getState();
  const toast = useStore((s) => s.toast);
  const [repo, setRepo] = useState('');
  useEffect(() => void resume(), [resume]);

  const busy = !!status?.busy;
  const total = status?.total ?? 0;
  const done = status?.done ?? 0;
  const pct = total ? Math.min(100, (done / total) * 100) : 0;
  const go = () => {
    if (!repo.trim() || busy) return;
    start(repo.trim()).catch((e: Error) => toast(e.message, true));
  };

  return (
    <div className="hub">
      <div className="row gap">
        <input
          className="mono grow"
          placeholder="org/model on Hugging Face"
          value={repo}
          disabled={busy}
          onChange={(e) => setRepo(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && go()}
        />
        {busy ? (
          <button className="secondary" onClick={() => void cancel()}>Cancel</button>
        ) : (
          <button className="primary" disabled={!repo.trim()} onClick={go}>Download</button>
        )}
      </div>
      {busy ? (
        <div className="hub-progress">
          <div className="bar"><span style={{ width: `${pct}%` }} /></div>
          <div className="row small dim">
            <span className="mono grow">{status?.phase === 'downloading' ? status.file : status?.phase}</span>
            <span>{total ? `${mb(done)} / ${mb(total)}` : ''}</span>
          </div>
        </div>
      ) : (
        <p className="note">
          Fetches <code>model.safetensors</code>, the tokenizer and config into <code>models/</code>, then fills in the fields below.
          The graph still has to match the model's architecture.
        </p>
      )}
    </div>
  );
}

// ---- document panel --------------------------------------------------------------

function DocPanel({ doc, count }: { doc: Doc; count: number }) {
  const analysis = useStore((s) => s.analysis);
  const { edit, select } = useStore.getState();
  const vars = useVars();
  const [newVar, setNewVar] = useState('');
  const main = analysis?.graphs.main;
  const problems: { key: GraphKey; id: number; text: string }[] = [];
  if (analysis) {
    for (const [g, a] of Object.entries(analysis.graphs)) {
      for (const [id, text] of Object.entries(a.errors)) problems.push({ key: g === 'main' ? '' : g, id: Number(id), text });
    }
  }
  const m = doc.model;
  const setModel = (fn: (m: Doc['model']) => void) => edit((d) => fn(d.model));

  return (
    <div className="panel-scroll">
      {count > 1 && <p className="note">{count} nodes selected. Press <kbd>Ctrl G</kbd> to wrap them in a nest.</p>}
      <h4>Model</h4>
      <label className="field">
        <span>name</span>
        <TextField value={doc.name} onCommit={(v) => edit((d) => (d.name = v))} />
      </label>
      <div className="stat-row">
        <div className="stat">
          <b>{main ? formatCount(main.totalParams) : '—'}</b>
          <span>parameters</span>
        </div>
        <div className={`stat ${problems.length ? 'bad' : 'good'}`}>
          <b>{problems.length || '✓'}</b>
          <span>{problems.length ? 'problems' : 'valid'}</span>
        </div>
      </div>
      {problems.slice(0, 8).map((p, i) => (
        <button
          key={i}
          className="problem"
          onClick={() => {
            // Select the node if it is visible from the main graph.
            if (p.key === '') select([{ path: [], id: p.id }]);
          }}
        >
          <b>{p.key || 'main'}</b> · {graphOf(doc, p.key).nodes.find((n) => n.id === p.id)?.name}: {p.text}
        </button>
      ))}
      {main?.graphErrors.map((e) => <div key={e} className="error-box">{e}</div>)}

      <h4>Variables</h4>
      <p className="note">Use them in any attribute as <code>$name</code>, e.g. <code>$d_model * 4</code>.</p>
      {Object.entries(doc.variables).map(([k, v]) => (
        <label key={k} className="field var">
          <span className="mono">{k}</span>
          <TextField value={String(v)} onCommit={(t) => edit((d) => (d.variables[k] = parseAttrInput(t, 'float')))} mono />
          <span className="hint ok">{vars[k] !== undefined && String(v) !== String(vars[k]) ? `= ${vars[k]}` : ''}</span>
          <button className="icon" title="Remove" onClick={() => edit((d) => delete d.variables[k])}>×</button>
        </label>
      ))}
      <div className="row gap">
        <input className="mono" placeholder="new_variable" value={newVar} onChange={(e) => setNewVar(e.target.value)} />
        <button
          className="secondary"
          disabled={!/^[A-Za-z_]\w*$/.test(newVar) || newVar in doc.variables}
          onClick={() => {
            edit((d) => (d.variables[newVar] = 0));
            setNewVar('');
          }}
        >
          Add
        </button>
      </div>

      <h4>Weights &amp; tokenizer</h4>
      <HubDownload />
      <p className="note">Paths are relative to the graphs/ folder.</p>
      <label className="field">
        <span>checkpoint</span>
        <TextField value={m.checkpoint} placeholder="model.safetensors" onCommit={(v) => setModel((x) => (x.checkpoint = v))} mono />
      </label>
      <label className="field">
        <span>name mapping</span>
        <select value={m.format} onChange={(e) => setModel((x) => (x.format = e.target.value))}>
          <option value="raw">raw (rename rules only)</option>
          <option value="gpt2">gpt2</option>
          <option value="llama">llama</option>
        </select>
      </label>
      <label className="field">
        <span>tokenizer</span>
        <TextField value={m.tokenizer} placeholder="tokenizer.json" onCommit={(v) => setModel((x) => (x.tokenizer = v))} mono />
      </label>
      <label className="field">
        <span>chat template</span>
        <TextField value={m.chat_template} placeholder="tokenizer_config.json" onCommit={(v) => setModel((x) => (x.chat_template = v))} mono />
      </label>
      <label className="field">
        <span>stop tokens</span>
        <TextField
          value={m.eos.join(', ')}
          onCommit={(v) => setModel((x) => (x.eos = v.split(/[\s,]+/).filter(Boolean).map(Number).filter(Number.isFinite)))}
          mono
        />
      </label>
      <details>
        <summary>Rename rules ({m.rename_rules.length})</summary>
        <p className="note">Checkpoint name → weight name. <code>{'{i}'}</code> matches a layer number.</p>
        {m.rename_rules.map((r, i) => (
          <div key={i} className="row gap">
            <TextField value={r.from} onCommit={(v) => setModel((x) => (x.rename_rules[i].from = v))} mono />
            <TextField value={r.to} onCommit={(v) => setModel((x) => (x.rename_rules[i].to = v))} mono />
            <button className="icon" onClick={() => setModel((x) => x.rename_rules.splice(i, 1))}>×</button>
          </div>
        ))}
        <button className="ghost small" onClick={() => setModel((x) => x.rename_rules.push({ from: 'model.layers.{i}.mlp.up_proj.weight', to: 'transformer.h.{i}.mlp.up.weight' }))}>
          + rule
        </button>
      </details>

      <h4>Subgraphs</h4>
      <p className="note">The contents of nests. Double-click a nest on the canvas to open it.</p>
      {Object.keys(doc.subgraphs).map((k) => {
        const users = subgraphUsers(doc, k);
        return (
          <div key={k} className="sub-row">
            <span className="mono">{k}</span>
            <span className="dim">{users ? `${users} nest${users > 1 ? 's' : ''}` : 'unused'}</span>
            {!users && (
              <button className="icon" title="Delete" onClick={() => edit((d) => delete d.subgraphs[k])}>×</button>
            )}
          </div>
        );
      })}
      <button
        className="ghost small"
        onClick={() =>
          edit((d) => {
            // Drop a new empty nest into the main graph.
            const g = d.graph;
            let name = 'Nest';
            for (let i = 2; d.subgraphs[name]; i++) name = `Nest${i}`;
            d.subgraphs[name] = {
              nodes: [
                { id: 1, type: 'Input', name: 'x', attrs: { index: 0, dtype: 'f32', shape: 'B, S, $d_model' }, pos: [0, 0] },
                { id: 2, type: 'Output', name: 'y', attrs: { index: 0 }, pos: [320, 0] },
              ],
              links: [{ from: [1, 'out'], to: [2, 'value'] }],
            };
            const id = Math.max(0, ...g.nodes.map((n) => n.id)) + 1;
            g.nodes.push({ id, type: 'Group', name: uniqueName(g, 'nest'), attrs: { subgraph: name }, pos: [0, 300], ui: { expanded: true } });
          })
        }
      >
        + new empty nest
      </button>
    </div>
  );
}
