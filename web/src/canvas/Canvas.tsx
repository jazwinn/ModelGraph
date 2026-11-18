import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useNodesInitialized,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type FinalConnectionState,
  type NodeChange,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  addNode,
  connect,
  emptySubgraph,
  graphOf,
  groupNodes,
  isNest,
  removeNodes,
  sameLink,
  ungroup,
  uniqueName,
  uniqueSubgraphName,
  wouldCycle,
} from '../lib/docops';
import { keyOfPath, requestRelayout, useStore } from '../lib/store';
import type { Doc, GLink, GraphKey, NodeRef, NodeSpec } from '../lib/types';
import { edgeTypes } from './Edges';
import { tidyAll } from './elkLayout';
import { analysisKey, buildFlow, overlappingGraphs, parseRfId, rfId, type FlowNode } from './flow';
import { categoryColor, nodeTypes } from './Nodes';

interface Pending {
  path: number[];
  node: number;
  handle: string; // "o:port" or "i:port"
}

interface Popup {
  x: number; // screen position
  y: number;
  pending?: Pending;
}

interface Menu {
  x: number;
  y: number;
  ref: NodeRef;
}

const samePath = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

export function Canvas() {
  const doc = useStore((s) => s.doc)!;
  const specs = useStore((s) => s.specs);
  const analysis = useStore((s) => s.analysis);
  const selection = useStore((s) => s.selection);
  const loadedAt = useStore((s) => s.loadedAt);
  const { edit, checkpoint, select, toast } = useStore.getState();
  const rf = useReactFlow();

  const [selectedEdges, setSelectedEdges] = useState<Set<string>>(new Set());
  const [popup, setPopup] = useState<Popup | null>(null);
  const [menu, setMenu] = useState<Menu | null>(null);
  const wrapper = useRef<HTMLDivElement>(null);

  const flow = useMemo(() => buildFlow(doc, specs, analysis), [doc, specs, analysis]);
  const byId = useMemo(() => new Map(flow.nodes.map((n) => [n.id, n])), [flow]);

  const selectedIds = useMemo(() => new Set(selection.map((r) => rfId(r.path, r.id))), [selection]);
  const nodes = useMemo(() => flow.nodes.map((n) => (selectedIds.has(n.id) ? { ...n, selected: true } : n)), [flow, selectedIds]);
  const edges = useMemo(
    () => flow.edges.map((e) => (selectedEdges.has(e.id) ? { ...e, selected: true, style: { ...e.style, strokeWidth: 3.5 } } : e)),
    [flow, selectedEdges],
  );

  // On open: re-tidy if nodes overlap (e.g. generated templates), then fit.
  // After a nest is expanded or collapsed: re-tidy the graph holding it and
  // every graph above it, so neighbours close the gap (or make room).
  const tidied = useRef(0);
  const [relayout, setRelayout] = useState<number[] | null>(null);
  useEffect(() => {
    if (!analysis) return;
    const opening = tidied.current !== loadedAt;
    if (!opening && !relayout) return;
    tidied.current = loadedAt;
    setRelayout(null);

    let keys: GraphKey[] | undefined;
    if (opening) {
      if (!overlappingGraphs(flow.nodes).length) {
        setFitPending(true);
        return;
      }
    } else {
      const path = relayout!;
      const chain = new Set<GraphKey>(overlappingGraphs(flow.nodes));
      for (let i = path.length; i >= 0; i--) {
        const k = keyOfPath(doc, path.slice(0, i));
        if (k !== null) chain.add(k);
      }
      keys = [...chain];
    }
    const draft = structuredClone(doc);
    void tidyAll(draft, specs, analysis, keys).then(() => {
      if (useStore.getState().doc !== doc) return; // edited meanwhile; skip
      const wasDirty = useStore.getState().dirty;
      edit(
        (d) => {
          d.graph = draft.graph;
          d.subgraphs = draft.subgraphs;
        },
        { history: false },
      );
      if (opening) {
        useStore.setState({ dirty: wasDirty });
        setFitPending(true);
      }
    });
  }, [analysis, doc, edit, flow, loadedAt, relayout, specs]);
  useEffect(() => {
    // detail: the path of the graph that holds the nest that changed.
    const on = (e: Event) => setRelayout((e as CustomEvent<number[]>).detail ?? []);
    window.addEventListener('mg-relayout', on);
    return () => window.removeEventListener('mg-relayout', on);
  }, []);

  // Fit once React Flow has measured the (possibly re-laid-out) nodes.
  const [fitPending, setFitPending] = useState(false);
  const initialized = useNodesInitialized();
  useEffect(() => {
    if (!fitPending || !initialized) return;
    const t = setTimeout(() => {
      rf.fitView({ padding: 0.12, duration: 250 });
      setFitPending(false);
    }, 120);
    return () => clearTimeout(t);
  }, [fitPending, initialized, rf]);

  // ---- changes from React Flow ------------------------------------------------

  const onNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) => {
      const moves = changes.filter((c) => c.type === 'position' && c.position);
      if (moves.length) {
        edit(
          (d) => {
            for (const c of moves) {
              if (c.type !== 'position' || !c.position) continue;
              const fn = byId.get(c.id);
              if (!fn) continue;
              const n = graphOf(d, fn.data.key).nodes.find((m) => m.id === fn.data.node.id);
              if (n) n.pos = [Math.round(c.position.x + fn.data.origin.x), Math.round(c.position.y + fn.data.origin.y)];
            }
          },
          { history: false },
        );
      }
      const sel = changes.filter((c) => c.type === 'select');
      if (sel.length) {
        const next = new Map(useStore.getState().selection.map((r) => [rfId(r.path, r.id), r]));
        for (const c of sel) {
          if (c.type !== 'select') continue;
          if (c.selected) next.set(c.id, parseRfId(c.id));
          else next.delete(c.id);
        }
        select([...next.values()]);
      }
    },
    [byId, edit, select],
  );

  const onEdgesChange = useCallback((changes: EdgeChange<Edge>[]) => {
    setSelectedEdges((prev) => {
      const next = new Set(prev);
      for (const c of changes) if (c.type === 'select') c.selected ? next.add(c.id) : next.delete(c.id);
      return next;
    });
  }, []);

  // ---- connections ------------------------------------------------------------

  const checkConnection = useCallback(
    (c: Connection | Edge): string | null => {
      if (!c.sourceHandle?.startsWith('o:') || !c.targetHandle?.startsWith('i:')) return 'connect an output to an input';
      const s = parseRfId(c.source);
      const t = parseRfId(c.target);
      if (!samePath(s.path, t.path)) return 'both ends must be in the same nest (wire through its Input/Output nodes)';
      if (s.id === t.id) return 'a node cannot feed itself';
      const key = keyOfPath(doc, s.path);
      if (key === null) return 'stale node';
      if (wouldCycle(graphOf(doc, key), s.id, t.id)) return 'that would create a cycle';
      const src = byId.get(c.source)?.data;
      const dst = byId.get(c.target)?.data;
      const port = c.sourceHandle.slice(2);
      const oi = src?.ports.outputs.findIndex((p) => p.name === port) ?? -1;
      const outType = (oi >= 0 ? src?.info?.outputs?.[oi]?.dtype : undefined) ?? src?.ports.outputs[oi]?.dtype;
      const inType = dst?.ports.inputs.find((p) => p.name === c.targetHandle!.slice(2))?.dtype;
      if (outType && inType && outType !== inType) return `type mismatch: ${outType} → ${inType}`;
      return null;
    },
    [byId, doc],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      const problem = checkConnection(c);
      if (problem) return toast(problem, true);
      const s = parseRfId(c.source);
      const t = parseRfId(c.target);
      edit((d) => connect(graphOf(d, keyOfPath(d, s.path)!), s.id, c.sourceHandle!.slice(2), t.id, c.targetHandle!.slice(2)));
    },
    [checkConnection, edit, toast],
  );

  const onConnectEnd = useCallback((event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
    if (state.isValid || !state.fromNode || !state.fromHandle?.id) return;
    const point = 'changedTouches' in event ? event.changedTouches[0] : event;
    const { path, id } = parseRfId(state.fromNode.id);
    setPopup({ x: point.clientX, y: point.clientY, pending: { path, node: id, handle: state.fromHandle.id } });
  }, []);

  // ---- placing nodes ----------------------------------------------------------

  // Which graph is under a screen point: the deepest expanded nest, else main.
  const locate = useCallback(
    (screen: { x: number; y: number }) => {
      const p = rf.screenToFlowPosition(screen);
      let best: { path: number[]; pos: [number, number] } = { path: [], pos: [p.x, p.y] };
      let depth = -1;
      for (const n of flow.nodes) {
        if (n.type !== 'nest' || !n.data.expanded || !n.data.childOrigin) continue;
        const abs = rf.getInternalNode(n.id)?.internals.positionAbsolute;
        if (!abs || p.x < abs.x || p.y < abs.y || p.x > abs.x + (n.width ?? 0) || p.y > abs.y + (n.height ?? 0)) continue;
        if (n.data.path.length > depth) {
          depth = n.data.path.length;
          best = {
            path: [...n.data.path, n.data.node.id],
            pos: [Math.round(p.x - abs.x + n.data.childOrigin.x), Math.round(p.y - abs.y + n.data.childOrigin.y)],
          };
        }
      }
      return best;
    },
    [flow, rf],
  );

  const place = useCallback(
    (spec: NodeSpec, screen: { x: number; y: number }, pending?: Pending) => {
      let { path, pos } = locate(screen);
      if (pending && !samePath(path, pending.path)) {
        // Keep the new node in the graph of the wire it came from.
        const src = byId.get(rfId(pending.path, pending.node))?.data.node;
        path = pending.path;
        pos = src ? [src.pos[0] + (pending.handle.startsWith('o:') ? 300 : -300), src.pos[1]] : pos;
      }
      let created: NodeRef | null = null;
      edit((d) => {
        const key = keyOfPath(d, path);
        if (key === null) throw new Error('that nest no longer exists');
        const g = graphOf(d, key);
        let extra = {};
        if (spec.type === 'Group' || spec.type === 'Repeat') {
          // A fresh nest gets its own empty subgraph to fill in.
          const name = uniqueSubgraphName(d, spec.type === 'Repeat' ? 'Layer' : 'Nest');
          d.subgraphs[name] = emptySubgraph();
          extra = { ui: { expanded: true } };
          const n = addNode(g, spec, pos[0], pos[1], extra);
          n.attrs.subgraph = name;
          created = { path, id: n.id };
        } else {
          created = { path, id: addNode(g, spec, pos[0], pos[1]).id };
        }
        const n = g.nodes.find((m) => m.id === created!.id)!;
        if (pending) {
          const port = pending.handle.slice(2);
          if (pending.handle.startsWith('o:') && spec.inputs[0] && !wouldCycle(g, pending.node, n.id)) connect(g, pending.node, port, n.id, spec.inputs[0].name);
          if (pending.handle.startsWith('i:') && spec.outputs[0]) connect(g, n.id, spec.outputs[0].name, pending.node, port);
        }
      });
      if (created) select([created]);
    },
    [byId, edit, locate, select],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const type = e.dataTransfer.getData('application/x-modelgraph-node');
      const spec = specs[type];
      if (spec) place(spec, { x: e.clientX, y: e.clientY });
    },
    [place, specs],
  );

  // ---- commands ---------------------------------------------------------------

  const deleteSelected = useCallback(() => {
    const sel = useStore.getState().selection;
    const links = flow.edges.filter((e) => selectedEdges.has(e.id) && e.data?.link);
    if (!sel.length && !links.length) return;
    edit((d) => {
      for (const e of links) {
        const g = graphOf(d, keyOfPath(d, e.data!.path as number[])!);
        g.links = g.links.filter((l) => !sameLink(l, e.data!.link as GLink));
      }
      const byPath = new Map<string, { path: number[]; ids: Set<number> }>();
      for (const r of sel) {
        const k = r.path.join('/');
        if (!byPath.has(k)) byPath.set(k, { path: r.path, ids: new Set() });
        byPath.get(k)!.ids.add(r.id);
      }
      for (const { path, ids } of byPath.values()) {
        const key = keyOfPath(d, path);
        if (key !== null) removeNodes(graphOf(d, key), ids);
      }
    });
    select([]);
    setSelectedEdges(new Set());
  }, [edit, flow, select, selectedEdges]);

  const groupSelected = useCallback(() => {
    const sel = useStore.getState().selection;
    if (!sel.length) return toast('Select the nodes to wrap in a nest (shift-drag or ctrl-click)', true);
    if (!sel.every((r) => samePath(r.path, sel[0].path))) return toast('Nest nodes from one graph at a time', true);
    let made: NodeRef | null = null;
    edit((d) => {
      const key = keyOfPath(d, sel[0].path)!;
      const id = groupNodes(d, key, new Set(sel.map((r) => r.id)), useStore.getState().analysis?.graphs[analysisKey(key)]);
      made = { path: sel[0].path, id };
    });
    if (made) select([made]);
  }, [edit, select, toast]);

  // Shortcuts that act on the canvas.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest('input, textarea, select, .cm-editor')) return;
      const mod = e.ctrlKey || e.metaKey;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelected();
      } else if (mod && e.key.toLowerCase() === 'g') {
        e.preventDefault();
        groupSelected();
      } else if (e.key === 'f' && !mod) {
        rf.fitView({ padding: 0.15, duration: 300 });
      } else if (e.key === '/' && !mod) {
        e.preventDefault();
        const r = wrapper.current!.getBoundingClientRect();
        setPopup({ x: r.left + r.width / 2, y: r.top + r.height / 3 });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [deleteSelected, groupSelected, rf]);

  useEffect(() => {
    const onCmd = (e: Event) => {
      const cmd = (e as CustomEvent<string>).detail;
      if (cmd === 'group') groupSelected();
      if (cmd === 'fit') rf.fitView({ padding: 0.15, duration: 300 });
      if (cmd === 'delete') deleteSelected();
    };
    // Palette double-click: drop the node in the middle of the view.
    const onAdd = (e: Event) => {
      const spec = specs[(e as CustomEvent<string>).detail];
      const r = wrapper.current?.getBoundingClientRect();
      if (spec && r) place(spec, { x: r.left + r.width / 2, y: r.top + r.height / 2 });
    };
    window.addEventListener('mg-command', onCmd);
    window.addEventListener('mg-add', onAdd);
    return () => {
      window.removeEventListener('mg-command', onCmd);
      window.removeEventListener('mg-add', onAdd);
    };
  }, [deleteSelected, groupSelected, place, rf, specs]);

  return (
    <div
      className="canvas"
      ref={wrapper}
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      onDoubleClick={(e) => {
        // Double-click on empty canvas (or inside an open nest's body) to add a node.
        const t = e.target as HTMLElement;
        if (t.classList.contains('react-flow__pane') || t.classList.contains('nest-body') || t.closest('.node.nest.expanded') === t) {
          setMenu(null);
          setPopup({ x: e.clientX, y: e.clientY });
        }
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectEnd={onConnectEnd}
        isValidConnection={(c) => checkConnection(c) === null}
        onNodeDragStart={() => checkpoint()}
        onNodeDoubleClick={(e, n) => {
          if (n.type !== 'nest' || (e.target as HTMLElement).classList.contains('nest-body')) return;
          edit(
            (d) => {
              const m = graphOf(d, n.data.key).nodes.find((x) => x.id === n.data.node.id);
              if (m) m.ui = { ...(m.ui ?? {}), expanded: !n.data.expanded };
            },
            { history: false },
          );
          requestRelayout(n.data.path);
        }}
        onPaneClick={() => {
          setMenu(null);
          setPopup(null);
        }}
        onPaneContextMenu={(e) => {
          e.preventDefault();
          setMenu(null);
          setPopup({ x: e.clientX, y: e.clientY });
        }}
        onNodeContextMenu={(e, n) => {
          e.preventDefault();
          setPopup(null);
          const ref = parseRfId(n.id);
          if (!selectedIds.has(n.id)) select([ref]);
          setMenu({ x: e.clientX, y: e.clientY, ref });
        }}
        deleteKeyCode={null}
        zoomOnDoubleClick={false}
        selectionKeyCode="Shift"
        multiSelectionKeyCode={['Control', 'Meta']}
        minZoom={0.05}
        maxZoom={2.5}
        defaultEdgeOptions={{ type: 'default' }}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="var(--grid)" />
        <Controls showInteractive={false} position="bottom-left" />
        <MiniMap pannable zoomable position="bottom-right" nodeColor={(n) => categoryColor((n.data as FlowNode['data']).spec?.category)} maskColor="var(--minimap-mask)" />
      </ReactFlow>

      {popup && <QuickAdd popup={popup} specs={specs} onPick={(s) => { place(s, popup, popup.pending); setPopup(null); }} onClose={() => setPopup(null)} />}
      {menu && <ContextMenu menu={menu} doc={doc} onClose={() => setMenu(null)} onGroup={groupSelected} onDelete={deleteSelected} />}
    </div>
  );
}

function QuickAdd({ popup, specs, onPick, onClose }: { popup: Popup; specs: Record<string, NodeSpec>; onPick: (s: NodeSpec) => void; onClose: () => void }) {
  const [q, setQ] = useState('');
  const [active, setActive] = useState(0);
  const list = useMemo(() => {
    const query = q.toLowerCase();
    return Object.values(specs)
      .filter((s) => !query || s.type.toLowerCase().includes(query) || s.category.toLowerCase().includes(query) || s.description.toLowerCase().includes(query))
      .filter((s) => !popup.pending || (popup.pending.handle.startsWith('o:') ? s.inputs.length || s.dynamicPorts : s.outputs.length || s.dynamicPorts))
      .sort((a, b) => (a.type.toLowerCase().startsWith(query) ? 0 : 1) - (b.type.toLowerCase().startsWith(query) ? 0 : 1) || a.type.localeCompare(b.type));
  }, [q, specs, popup.pending]);
  const left = Math.min(popup.x, window.innerWidth - 320);
  const top = Math.min(popup.y, window.innerHeight - 380);
  return (
    <div className="quick-add" style={{ left, top }} onMouseDown={(e) => e.stopPropagation()}>
      <input
        autoFocus
        placeholder="Add node…"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setActive(0);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Escape') onClose();
          if (e.key === 'ArrowDown') setActive((a) => Math.min(a + 1, list.length - 1));
          if (e.key === 'ArrowUp') setActive((a) => Math.max(a - 1, 0));
          if (e.key === 'Enter' && list[active]) onPick(list[active]);
        }}
      />
      <div className="quick-list">
        {list.map((s, i) => (
          <button key={s.type} className={i === active ? 'active' : ''} onMouseEnter={() => setActive(i)} onClick={() => onPick(s)}>
            <span className="dot" style={{ background: categoryColor(s.category) }} />
            <span className="qa-type">{s.type}</span>
            <span className="qa-cat">{s.category}</span>
          </button>
        ))}
        {!list.length && <div className="empty">No matching nodes</div>}
      </div>
    </div>
  );
}

function ContextMenu({ menu, doc, onClose, onGroup, onDelete }: { menu: Menu; doc: Doc; onClose: () => void; onGroup: () => void; onDelete: () => void }) {
  const { edit, select, toast } = useStore.getState();
  const key = keyOfPath(doc, menu.ref.path);
  const node = key === null ? undefined : graphOf(doc, key).nodes.find((n) => n.id === menu.ref.id);
  if (!node || key === null) return null;
  const nest = isNest(node);
  const run = (fn: () => void) => () => {
    fn();
    onClose();
  };
  const setType = (type: 'Group' | 'Repeat') =>
    edit((d) => {
      const n = graphOf(d, key).nodes.find((m) => m.id === node.id)!;
      n.type = type;
      if (type === 'Repeat') n.attrs.count = n.attrs.count ?? 2;
      else delete n.attrs.count;
    });
  return (
    <div className="context-menu" style={{ left: menu.x, top: menu.y }} onMouseLeave={onClose}>
      <div className="cm-title">{node.name}</div>
      <button onClick={run(onGroup)}>Wrap selection in a nest <kbd>Ctrl G</kbd></button>
      {nest && (
        <button
          onClick={run(() => {
            edit((d) => {
              const n = graphOf(d, key).nodes.find((m) => m.id === node.id)!;
              n.ui = { ...(n.ui ?? {}), expanded: !node.ui?.expanded };
            }, { history: false });
            requestRelayout(menu.ref.path);
          })}
        >
          {node.ui?.expanded ? 'Collapse nest' : 'Expand nest'}
        </button>
      )}
      {node.type === 'Group' && <button onClick={run(() => setType('Repeat'))}>Repeat this nest ×N</button>}
      {node.type === 'Repeat' && <button onClick={run(() => setType('Group'))}>Use a single copy</button>}
      {node.type === 'Group' && (
        <button
          onClick={run(() => {
            edit((d) => ungroup(d, key, node.id));
            select([]);
          })}
        >
          Unwrap nest <kbd>inline its nodes</kbd>
        </button>
      )}
      <button
        onClick={run(() =>
          edit((d) => {
            const g = graphOf(d, key);
            const copy = structuredClone(g.nodes.find((m) => m.id === node.id)!);
            copy.id = Math.max(...g.nodes.map((m) => m.id)) + 1;
            copy.name = uniqueName(g, copy.name);
            copy.pos = [copy.pos[0] + 40, copy.pos[1] + 40];
            g.nodes.push(copy);
            toast(isNest(copy) ? 'Duplicated (the copy shares the same subgraph)' : 'Duplicated');
          }),
        )}
      >
        Duplicate
      </button>
      <button className="danger" onClick={run(onDelete)}>Delete <kbd>Del</kbd></button>
    </div>
  );
}
