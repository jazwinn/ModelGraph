import { python } from '@codemirror/lang-python';
import CodeMirror from '@uiw/react-codemirror';
import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import { useStore } from '../lib/store';
import type { ScriptStatus } from '../lib/types';

const CHEATSHEET = `name = "MyNode"            # node type (required)
inputs = ["x"]             # or [{"name": "x", "dtype": "f32"}]
outputs = ["y"]
attrs = {"beta": 1.0}      # "$d_model" strings are expressions

def params(a):             # optional learned weights
    return {"w": {"shape": [a.dim], "init": "ones"}}

def shape(a, x):           # optional; x is like ("B", "S", 768)
    return x

def forward(ctx, x):       # ctx.attrs, ctx.params, ctx.start_pos
    return x * ie.sigmoid(x)

# Tensor: x.shape, x.reshape(...), x.transpose(a, b), x.mean(axis),
#   + - * / @, x.numpy()     ie.matmul, ie.linear, ie.softmax,
#   ie.silu/gelu/relu/tanh, ie.sdpa, ie.split_heads, ie.merge_heads,
#   ie.tensor(numpy_array), ie.exp/log/pow (CPU fallback)`;

export function ScriptPanel() {
  const open = useStore((s) => s.scriptsOpen);
  const initial = useStore((s) => s.scriptFile);
  const status = useStore((s) => s.status);
  const { closeScripts, toast } = useStore.getState();

  const [scripts, setScripts] = useState<ScriptStatus[]>([]);
  const [dir, setDir] = useState('');
  const [file, setFile] = useState<string | null>(null);
  const [text, setText] = useState('');
  const [saved, setSaved] = useState('');
  const [newName, setNewName] = useState('MyNode');
  const [help, setHelp] = useState(false);

  const refresh = useCallback(async () => {
    const r = await api.scripts();
    setScripts(r.scripts);
    setDir(r.dir);
  }, []);

  useEffect(() => {
    if (open) void refresh();
  }, [open, refresh, status?.scriptGeneration]);

  const load = useCallback(
    async (f: string) => {
      try {
        const r = await api.readScript(f);
        setFile(f);
        setText(r.text);
        setSaved(r.text);
      } catch (e) {
        toast((e as Error).message, true);
      }
    },
    [toast],
  );

  useEffect(() => {
    if (open && initial) void load(initial);
  }, [open, initial, load]);

  const save = useCallback(async () => {
    if (!file) return;
    try {
      await api.writeScript(file, text);
      setSaved(text);
      toast(`Saved ${file} — reloading`);
      setTimeout(() => void refresh(), 700);
    } catch (e) {
      toast((e as Error).message, true);
    }
  }, [file, text, toast, refresh]);

  if (!open) return null;
  const current = scripts.find((s) => s.file === file);

  return (
    <div className="drawer-backdrop" onMouseDown={(e) => e.target === e.currentTarget && closeScripts()}>
      <div className="drawer">
        <header className="drawer-head">
          <b>Python nodes</b>
          <span className="dim mono small">{dir}</span>
          <span className="grow" />
          <button className="ghost" onClick={closeScripts}>Close</button>
        </header>
        <div className="drawer-body">
          <nav className="script-list">
            {scripts.map((s) => (
              <button key={s.file} className={s.file === file ? 'active' : ''} onClick={() => void load(s.file)} title={s.error || s.file}>
                <span className={`status ${s.ok ? 'ok' : 'bad'}`} />
                <span>{s.type || s.file}</span>
                <span className="dim small">{s.file}</span>
              </button>
            ))}
            <div className="new-script">
              <input className="mono" value={newName} onChange={(e) => setNewName(e.target.value)} />
              <button
                className="secondary"
                onClick={async () => {
                  try {
                    const r = await api.newScript(newName);
                    await refresh();
                    await load(r.file);
                  } catch (e) {
                    toast((e as Error).message, true);
                  }
                }}
              >
                New node
              </button>
            </div>
            <button className="ghost small" onClick={() => setHelp(!help)}>{help ? 'Hide' : 'Show'} cheatsheet</button>
            {help && <pre className="cheatsheet">{CHEATSHEET}</pre>}
          </nav>
          <div className="editor">
            {file ? (
              <>
                <div className="editor-bar">
                  <span className="mono">{file}</span>
                  {text !== saved && <span className="pill warn">unsaved</span>}
                  {current && (current.ok ? <span className="pill ok">loaded · {current.type}</span> : <span className="pill bad">error</span>)}
                  <span className="grow" />
                  <button
                    className="ghost danger small"
                    onClick={async () => {
                      if (!confirm(`Delete ${file}?`)) return;
                      await api.deleteScript(file);
                      setFile(null);
                      void refresh();
                    }}
                  >
                    Delete
                  </button>
                  <button className="primary" disabled={text === saved} onClick={() => void save()}>Save <kbd>Ctrl S</kbd></button>
                </div>
                {current && !current.ok && <pre className="error-box mono">{current.error}</pre>}
                <div
                  className="cm-wrap"
                  onKeyDown={(e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
                      e.preventDefault();
                      e.stopPropagation();
                      void save();
                    }
                  }}
                >
                  <CodeMirror value={text} height="100%" theme="dark" extensions={[python()]} onChange={setText} basicSetup={{ tabSize: 4 }} />
                </div>
              </>
            ) : (
              <div className="empty-editor">
                <p>Pick a script, or create a new node.</p>
                <p className="dim">Each .py file defines one node type. Saving reloads it immediately: built models use the new <code>forward()</code> on the next token.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
