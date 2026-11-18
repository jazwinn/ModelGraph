import { useEffect, useRef, useState } from 'react';
import { tidyAll } from '../canvas/elkLayout';
import { api, baseName, isAbsolute } from '../lib/api';
import { getRecent, removeRecent, type RecentFile } from '../lib/recent';
import { emptyDoc, formatCount } from '../lib/docops';
import { useStore } from '../lib/store';
import type { Direction } from '../lib/types';

function Menu({ label, children }: { label: string; children: (close: () => void) => React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [open]);
  return (
    <div className="menu" ref={ref}>
      <button className={`ghost ${open ? 'on' : ''}`} onClick={() => setOpen(!open)}>{label} ▾</button>
      {open && <div className="menu-pop">{children(() => setOpen(false))}</div>}
    </div>
  );
}

// A menu entry that opens a nested panel to the right on hover (or click).
function SubMenu({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  // Close after a short delay so moving the mouse across the gap (or slightly
  // diagonally) into the panel doesn't dismiss it.
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const show = () => {
    clearTimeout(timer.current);
    setOpen(true);
  };
  const hide = () => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(false), 300);
  };
  useEffect(() => () => clearTimeout(timer.current), []);
  return (
    <div className="submenu" onMouseEnter={show} onMouseLeave={hide}>
      <button className={open ? 'on' : ''} onClick={() => setOpen(true)}>
        {label}
        <span className="chev">▸</span>
      </button>
      {open && <div className="menu-pop sub">{children}</div>}
    </div>
  );
}

export function TopBar({ theme, setTheme }: { theme: string; setTheme: (t: string) => void }) {
  const doc = useStore((s) => s.doc);
  const file = useStore((s) => s.file);
  const dirty = useStore((s) => s.dirty);
  const past = useStore((s) => s.past.length);
  const future = useStore((s) => s.future.length);
  const analysis = useStore((s) => s.analysis);
  const { setDoc, undo, redo, toast, saveAs, edit, openScripts } = useStore.getState();
  const [templates, setTemplates] = useState<{ id: string; title: string }[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [recent, setRecent] = useState<RecentFile[]>([]);

  useEffect(() => {
    void api.templates().then((r) => setTemplates(r.templates));
  }, []);

  const confirmDiscard = () => !useStore.getState().dirty || confirm('Discard unsaved changes?');

  // Save to the current file; "save as" (or an unsaved document) asks via the system dialog.
  const save = async (as = false) => {
    try {
      let name = file;
      if (as || !name) {
        const suggested = file ? baseName(file) : `${(doc?.name ?? 'model').replace(/[^\w.-]+/g, '-').toLowerCase()}.json`;
        const r = await api.saveDialog(suggested);
        if (r.cancelled || !r.file) return;
        name = r.file;
      }
      await saveAs(name);
    } catch (e) {
      toast((e as Error).message, true);
    }
  };

  const openFile = async (f: string) => {
    if (!confirmDiscard()) return;
    try {
      setDoc(await api.loadGraph(f), f);
    } catch (e) {
      removeRecent(f);
      toast(`Could not open ${baseName(f)}: ${(e as Error).message}`, true);
    }
  };

  const browse = async () => {
    if (!confirmDiscard()) return;
    try {
      const r = await api.openDialog();
      if (r.cancelled || !r.doc || !r.file) return;
      setDoc(r.doc, r.file);
    } catch (e) {
      toast((e as Error).message, true);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      const typing = (e.target as HTMLElement).closest('input, textarea, .cm-editor');
      if (e.key.toLowerCase() === 's' && !(e.target as HTMLElement).closest('.cm-editor')) {
        e.preventDefault();
        void save();
      } else if (e.key.toLowerCase() === 'o') {
        e.preventDefault();
        void browse();
      } else if (!typing && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      } else if (!typing && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  // Set the flow direction and re-arrange every graph (nests included) to match.
  // Layout runs in ELK (async) on a copy, then lands as one undoable edit.
  const arrange = async (dir: Direction) => {
    const { specs, doc: current } = useStore.getState();
    if (!current) return;
    const draft = structuredClone(current);
    draft.layout = dir;
    try {
      await tidyAll(draft, specs, analysis);
    } catch (e) {
      toast(`Layout failed: ${(e as Error).message}`, true);
      return;
    }
    edit((d) => {
      d.layout = dir;
      d.graph = draft.graph;
      d.subgraphs = draft.subgraphs;
    });
    setTimeout(() => window.dispatchEvent(new CustomEvent('mg-command', { detail: 'fit' })), 80);
  };

  const problems = analysis ? Object.values(analysis.graphs).reduce((n, g) => n + Object.keys(g.errors).length + g.graphErrors.length, 0) : 0;
  const direction: Direction = doc?.layout === 'TB' ? 'TB' : 'LR';

  return (
    <header className="topbar">
      <div className="brand">
        <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
          <rect width="32" height="32" rx="8" fill="var(--accent)" />
          <circle cx="10" cy="11" r="4" fill="white" />
          <circle cx="22" cy="21" r="4" fill="white" />
          <path d="M13 13l6 6" stroke="white" strokeWidth="2.5" />
        </svg>
        ModelGraph
      </div>

      <div
        onMouseEnter={() => {
          void api.listGraphs().then((r) => setFiles(r.files));
          setRecent(getRecent());
        }}
      >
        <Menu label="File">
          {(close) => (
            <>
              <SubMenu label="New">
                <button onClick={() => { close(); if (confirmDiscard()) setDoc(emptyDoc(), ''); }}>Empty model</button>
                <div className="menu-sep" />
                <div className="menu-label">From a template</div>
                {templates.map((t) => (
                  <button
                    key={t.id}
                    onClick={async () => {
                      close();
                      if (!confirmDiscard()) return;
                      setDoc(await api.template(t.id), '');
                    }}
                  >
                    {t.title}
                  </button>
                ))}
              </SubMenu>
              <SubMenu label="Open">
                <button onClick={() => { close(); void browse(); }}>
                  Browse… <kbd>Ctrl O</kbd>
                </button>
                <div className="menu-sep" />
                <div className="menu-label">Recent</div>
                {recent.map((r) => (
                  <button key={r.file} className="file-item" title={r.file} onClick={() => { close(); void openFile(r.file); }}>
                    <span>{r.name || baseName(r.file)}</span>
                    <span className="dim small">{isAbsolute(r.file) ? r.file : `graphs/${r.file}`}</span>
                  </button>
                ))}
                {!recent.length && <div className="menu-empty">Nothing opened yet</div>}
                <div className="menu-sep" />
                <div className="menu-label">Project graphs</div>
                {files.map((f) => (
                  <button key={f} onClick={() => { close(); void openFile(f); }}>
                    {f}
                  </button>
                ))}
                {!files.length && <div className="menu-empty">No graphs in graphs/</div>}
              </SubMenu>
              <div className="menu-sep" />
              <button onClick={() => { close(); void save(); }}>Save <kbd>Ctrl S</kbd></button>
              <button onClick={() => { close(); void save(true); }}>Save as…</button>
            </>
          )}
        </Menu>
      </div>

      <div className="sep" />
      <button className="ghost icon-btn" disabled={!past} onClick={undo} title="Undo (Ctrl Z)">↶</button>
      <button className="ghost icon-btn" disabled={!future} onClick={redo} title="Redo (Ctrl Y)">↷</button>
      <div className="sep" />
      <button className="ghost" onClick={() => window.dispatchEvent(new CustomEvent('mg-command', { detail: 'group' }))} title="Wrap the selected nodes in a nest (Ctrl G)">
        ⧉ Nest selection
      </button>
      <div className="layout-toggle" title="Which way data flows on the canvas">
        {(['LR', 'TB'] as Direction[]).map((d) => (
          <button key={d} className={direction === d ? 'on' : ''} onClick={() => void arrange(d)}>
            {d === 'LR' ? '⇶ Horizontal' : '⇊ Vertical'}
          </button>
        ))}
      </div>
      <button className="ghost" onClick={() => void arrange(direction)} title="Re-arrange every graph">
        Tidy
      </button>
      <button className="ghost" onClick={() => window.dispatchEvent(new CustomEvent('mg-command', { detail: 'fit' }))} title="Fit view (F)">⤢ Fit</button>
      <button className="ghost" onClick={() => openScripts()} title="Write custom nodes in Python">{'{ }'} Python</button>

      <div className="title">
        <span className="doc-name">{doc?.name}</span>
        <span className="dim" title={file}>{file ? baseName(file) : 'unsaved'}</span>
        {dirty && <span className="dirty" title="Unsaved changes" />}
      </div>

      <div className="status-chips">
        <span className="chip">{analysis ? formatCount(analysis.graphs.main?.totalParams ?? 0) : '—'} params</span>
        <span className={`chip ${problems ? 'bad' : 'good'}`}>{problems ? `${problems} problem${problems > 1 ? 's' : ''}` : 'valid'}</span>
        <button className="ghost icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title="Toggle theme">
          {theme === 'dark' ? '☾' : '☀'}
        </button>
      </div>
    </header>
  );
}
