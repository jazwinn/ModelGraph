import { ReactFlowProvider } from '@xyflow/react';
import { useEffect, useState } from 'react';
import { Canvas } from './canvas/Canvas';
import { api, isAbsolute } from './lib/api';
import { useStore } from './lib/store';
import { Inspector } from './panels/Inspector';
import { Palette } from './panels/Palette';
import { RunPanel } from './panels/RunPanel';
import { ScriptPanel } from './panels/ScriptPanel';
import { TopBar } from './panels/TopBar';

function readTheme() {
  try {
    return localStorage.getItem('mg-theme') ?? 'dark';
  } catch {
    return 'dark';
  }
}

export function App() {
  const doc = useStore((s) => s.doc);
  const toasts = useStore((s) => s.toasts);
  const [theme, setTheme] = useState(readTheme);
  const [error, setError] = useState('');

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('mg-theme', theme);
    } catch {
      /* private mode */
    }
  }, [theme]);

  // Boot: node types, then the last-opened or a starter model.
  useEffect(() => {
    (async () => {
      try {
        const { loadSpecs, setDoc } = useStore.getState();
        await loadSpecs();
        let last = '';
        try {
          last = localStorage.getItem('mg-last-file') ?? '';
        } catch {
          /* ignore */
        }
        const { files } = await api.listGraphs();
        const reopen = last && (files.includes(last) || isAbsolute(last)) ? await api.loadGraph(last).catch(() => null) : null;
        if (reopen) setDoc(reopen, last);
        else if (files.includes('gpt2.json')) setDoc(await api.loadGraph('gpt2.json'), 'gpt2.json');
        else setDoc(await api.template('tiny'), '');
      } catch (e) {
        setError((e as Error).message);
      }
    })();
  }, []);

  // Remember the open file.
  const file = useStore((s) => s.file);
  useEffect(() => {
    try {
      if (file) localStorage.setItem('mg-last-file', file);
    } catch {
      /* ignore */
    }
  }, [file]);

  // Poll the server: generation output, build progress, Python reloads.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let alive = true;
    const tick = async () => {
      try {
        useStore.getState().setStatus(await api.status());
        setError('');
      } catch {
        setError('Lost connection to mg_server. Is it still running?');
      }
      if (alive) timer = setTimeout(tick, useStore.getState().status?.busy ? 200 : 1000);
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  // Warn before closing with unsaved work.
  useEffect(() => {
    const onUnload = (e: BeforeUnloadEvent) => {
      if (useStore.getState().dirty) e.preventDefault();
    };
    window.addEventListener('beforeunload', onUnload);
    return () => window.removeEventListener('beforeunload', onUnload);
  }, []);

  return (
    <ReactFlowProvider>
      <div className="app">
        <TopBar theme={theme} setTheme={setTheme} />
        {error && <div className="banner">{error}</div>}
        <div className="workspace">
          <Palette />
          <main className="center">
            {doc ? <Canvas /> : <div className="loading">Loading…</div>}
            {doc && <RunPanel />}
          </main>
          <Inspector />
        </div>
        <ScriptPanel />
        <div className="toasts">
          {toasts.map((t) => (
            <div key={t.id} className={`toast ${t.error ? 'error' : ''}`}>{t.text}</div>
          ))}
        </div>
      </div>
    </ReactFlowProvider>
  );
}
