import { useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import { formatCount } from '../lib/docops';
import { useStore } from '../lib/store';

export function RunPanel() {
  const doc = useStore((s) => s.doc)!;
  const file = useStore((s) => s.file);
  const status = useStore((s) => s.status);
  const analysis = useStore((s) => s.analysis);
  const revision = useStore((s) => s.revision);
  const builtRevision = useStore((s) => s.builtRevision);
  const { edit, toast, markBuilt } = useStore.getState();

  const [open, setOpen] = useState(true);
  const [prompt, setPrompt] = useState('The meaning of life is');
  const [tokens, setTokens] = useState(48);
  const [temperature, setTemperature] = useState(0);
  const [chat, setChat] = useState(true);
  const [showSystem, setShowSystem] = useState(false);
  // Transcript: every Generate adds an entry; the last one streams live.
  const [turns, setTurns] = useState<{ prompt: string; chat: boolean; reply: string }[]>([]);
  const outRef = useRef<HTMLDivElement>(null);
  const logRef = useRef<HTMLPreElement>(null);

  const valid = !!analysis && Object.values(analysis.graphs).every((g) => !Object.keys(g.errors).length && !g.graphErrors.length);
  const busy = !!status?.busy;
  const stale = !!status?.ready && builtRevision !== revision;

  // Freeze the streamed reply into the transcript when generation finishes.
  const wasBusy = useRef(false);
  useEffect(() => {
    if (wasBusy.current && !busy && turns.length) {
      const reply = status?.output ?? '';
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, reply } : x)));
    }
    wasBusy.current = busy;
  }, [busy]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    outRef.current?.scrollTo({ top: outRef.current.scrollHeight });
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [status?.output, status?.log]);

  const run = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
    } catch (e) {
      toast((e as Error).message, true);
    }
  };
  // Show the job as running right away; the next status poll confirms it.
  const started = (phase: string) => {
    const s = useStore.getState().status;
    if (s) useStore.setState({ status: { ...s, busy: true, phase } });
  };
  const build = (mode: 'fresh' | 'checkpoint' | 'keep') =>
    run(async () => {
      await api.build(doc, file, mode);
      started('starting build');
      markBuilt();
    });
  const generate = () =>
    run(async () => {
      const useChat = chat && !!doc.model.chat_template;
      await api.generate({ prompt, maxTokens: tokens, temperature, chat: useChat, system: doc.model.system_prompt ?? '' });
      setTurns((t) => [...t, { prompt, chat: useChat, reply: '' }]);
      started('generating');
    });

  return (
    <section className={`run ${open ? 'open' : ''}`}>
      <header className="run-head" onClick={() => setOpen(!open)}>
        <span className="chev">{open ? '▾' : '▸'}</span>
        <b>Run</b>
        <span className={`pill ${busy ? 'busy' : status?.ready ? 'ok' : ''}`}>
          {busy ? status?.phase || 'working' : status?.ready ? `model ready · ${formatCount(status.parameters)} params` : 'not built'}
        </span>
        {stale && !busy && <span className="pill warn">graph changed since build</span>}
        {status && status.tokensPerSecond > 0 && <span className="pill">{status.tokensPerSecond.toFixed(1)} tok/s</span>}
      </header>
      {open && (
        <div className="run-body">
          <div className="run-col">
            <div className="row gap wrap">
              <button className="primary" disabled={busy || !valid} onClick={() => build(doc.model.checkpoint ? 'checkpoint' : 'fresh')}>
                {doc.model.checkpoint ? 'Build + load weights' : 'Build'}
              </button>
              <button className="secondary" disabled={busy || !valid} onClick={() => build('fresh')} title="Every weight starts random">
                Random weights
              </button>
              <button className="secondary" disabled={busy || !valid || !status?.ready} onClick={() => build('keep')} title="Recompile; weights whose name and shape still match are kept">
                Rebuild, keep weights
              </button>
              <select value={doc.model.device} onChange={(e) => edit((d) => (d.model.device = e.target.value), { history: false })}>
                <option value="cuda">CUDA</option>
                <option value="cpu">CPU</option>
              </select>
              <label className="inline">
                context
                <input className="num" type="number" min={1} value={doc.model.context_size} onChange={(e) => edit((d) => (d.model.context_size = Math.max(1, Number(e.target.value))), { history: false })} />
              </label>
            </div>
            {!valid && <p className="note bad">Fix the graph problems before building.</p>}
            {doc.model.chat_template && chat && (
              <details className="system-prompt" open={showSystem} onToggle={(e) => setShowSystem((e.target as HTMLDetailsElement).open)}>
                <summary>
                  System prompt
                  <span className="dim"> · {(doc.model.system_prompt ?? '').trim() ? (doc.model.system_prompt ?? '').slice(0, 60) : 'none'}</span>
                </summary>
                <textarea
                  rows={3}
                  value={doc.model.system_prompt ?? ''}
                  placeholder="Instructions sent before every message (leave empty for none)"
                  onChange={(e) => edit((d) => (d.model.system_prompt = e.target.value), { history: false })}
                />
              </details>
            )}
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={doc.model.tokenizer ? (chat && doc.model.chat_template ? 'Message' : 'Prompt') : 'No tokenizer: type token ids, e.g. 1 2 3'} />
            <div className="row gap wrap">
              <label className="inline">
                tokens
                <input className="num" type="number" min={1} max={4096} value={tokens} onChange={(e) => setTokens(Number(e.target.value))} />
              </label>
              <label className="inline">
                temperature
                <input type="range" min={0} max={2} step={0.05} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
                <span className="mono">{temperature === 0 ? 'greedy' : temperature.toFixed(2)}</span>
              </label>
              {doc.model.chat_template && (
                <label className="inline">
                  <input type="checkbox" checked={chat} onChange={(e) => setChat(e.target.checked)} /> chat
                </label>
              )}
              <span className="grow" />
              {busy ? (
                <button className="danger" onClick={() => run(api.stop)}>Stop</button>
              ) : (
                <button className="primary" disabled={!status?.ready} onClick={generate}>Generate</button>
              )}
            </div>
          </div>
          <div className="run-col">
            <div className="output" ref={outRef}>
              {turns.length ? (
                turns.map((t, i) => {
                  const live = i === turns.length - 1 && busy;
                  const reply = live ? status?.output ?? '' : t.reply;
                  return (
                    <div key={i} className="turn">
                      {t.chat ? (
                        <>
                          <div className="turn-user">{t.prompt}</div>
                          <div className="turn-reply">
                            {reply.trimStart()}
                            {live && <span className="caret" />}
                          </div>
                        </>
                      ) : (
                        <div className="turn-reply">
                          <span className="dim">{t.prompt}</span>
                          {reply}
                          {live && <span className="caret" />}
                        </div>
                      )}
                    </div>
                  );
                })
              ) : (
                <span className="dim">Generated text appears here.</span>
              )}
            </div>
            {turns.length > 0 && !busy && (
              <button className="ghost small clear-turns" onClick={() => setTurns([])}>
                Clear
              </button>
            )}
            {status?.prompt && turns.length > 0 && (
              <details className="raw-prompt">
                <summary>Exact prompt sent to the model</summary>
                <pre>{status.prompt}</pre>
              </details>
            )}
            <pre className="log" ref={logRef}>
              {(status?.log ?? '').split('\n').filter(Boolean).map((l, i) => (
                <div key={i} className={l.startsWith('error') ? 'bad' : ''}>{l}</div>
              ))}
            </pre>
          </div>
        </div>
      )}
    </section>
  );
}
