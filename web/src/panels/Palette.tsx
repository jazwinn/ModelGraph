import { useMemo, useState } from 'react';
import { categoryColor } from '../canvas/Nodes';
import { useStore } from '../lib/store';
import type { NodeSpec } from '../lib/types';

const ORDER = ['Inputs & Outputs', 'Structure', 'Layers', 'Transformer', 'Attention parts', 'Math', 'Tensor'];

export function Palette() {
  const specs = useStore((s) => s.specs);
  const [q, setQ] = useState('');
  const [closed, setClosed] = useState<Set<string>>(new Set());

  const groups = useMemo(() => {
    const query = q.toLowerCase();
    const map = new Map<string, NodeSpec[]>();
    for (const s of Object.values(specs)) {
      if (query && !s.type.toLowerCase().includes(query) && !s.category.toLowerCase().includes(query) && !s.description.toLowerCase().includes(query)) continue;
      map.set(s.category, [...(map.get(s.category) ?? []), s]);
    }
    const rank = (c: string) => (ORDER.includes(c) ? ORDER.indexOf(c) : -1);
    return [...map.entries()]
      .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
      .map(([c, list]) => [c, list.sort((a, b) => a.type.localeCompare(b.type))] as const);
  }, [specs, q]);

  return (
    <aside className="palette">
      <div className="panel-head">
        <span>Nodes</span>
        <button className="ghost small" onClick={() => useStore.getState().openScripts()} title="Write your own node in Python">
          + Python node
        </button>
      </div>
      <input className="search" placeholder="Search nodes…" value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="palette-list">
        {groups.map(([category, list]) => {
          const open = !closed.has(category) || !!q;
          return (
            <section key={category}>
              <button
                className="cat"
                onClick={() => {
                  const next = new Set(closed);
                  open ? next.add(category) : next.delete(category);
                  setClosed(next);
                }}
              >
                <span className="chev">{open ? '▾' : '▸'}</span>
                <span className="dot" style={{ background: categoryColor(category) }} />
                {category}
                <span className="count">{list.length}</span>
              </button>
              {open &&
                list.map((s) => (
                  <div
                    key={s.type}
                    className="palette-item"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData('application/x-modelgraph-node', s.type);
                      e.dataTransfer.effectAllowed = 'move';
                    }}
                    onDoubleClick={() => window.dispatchEvent(new CustomEvent('mg-add', { detail: s.type }))}
                    title={`${s.description}${s.source !== 'builtin' ? `\n\n${s.source}` : ''}\n\nDrag onto the canvas (or into an open nest), or double-click.`}
                  >
                    <span className="pi-name">{s.type}</span>
                    {s.source !== 'builtin' && <span className="tag">py</span>}
                  </div>
                ))}
            </section>
          );
        })}
      </div>
      <div className="palette-hint">
        Drag nodes in or double-click the canvas. Wire outputs to inputs. Select several and press <kbd>Ctrl G</kbd> to wrap them in a nest.
      </div>
    </aside>
  );
}
