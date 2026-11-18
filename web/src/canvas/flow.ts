// Turns the document into React Flow nodes and edges. Expanded nests become
// boxes whose children are the nodes of their subgraph, positioned relative
// to the box; collapsed nests are ordinary nodes.

import { MarkerType, type Edge, type Node } from '@xyflow/react';
import { graphOf, isNest, nestTarget } from '../lib/docops';
import type { AnalyzeResult, Direction, Doc, GNode, Graph, GraphKey, NodeAnalysis, NodeSpec, PortSpec } from '../lib/types';

export const NODE_W = 236;
export const HEADER = 34;
export const ROW = 22;
export const FOOTER = 10;
export const PARAM_ROW = 18;
export const ERROR_H = 44;
export const NEST_PAD = 26;

export interface FlowNodeData extends Record<string, unknown> {
  key: GraphKey;            // graph this node belongs to
  dir: Direction;           // LR: inputs left, outputs right; TB: inputs top, outputs bottom
  path: number[];           // nest ids leading to that graph
  node: GNode;
  spec?: NodeSpec;
  info?: NodeAnalysis;
  error?: string;
  ports: { inputs: PortSpec[]; outputs: PortSpec[] };
  origin: { x: number; y: number };  // doc position = flow position + origin
  // nests only
  expanded?: boolean;
  bodyTop?: number;         // y where the children area starts inside the box
  childOrigin?: { x: number; y: number };  // doc position of a child = its box-relative position + this
  recursive?: boolean;      // the nest would contain itself
}

export type FlowNode = Node<FlowNodeData>;

export function rfId(path: number[], id: number) {
  return `${path.join('/')}:${id}`;
}

export function parseRfId(rf: string): { path: number[]; id: number } {
  const at = rf.lastIndexOf(':');
  const p = rf.slice(0, at);
  return { path: p ? p.split('/').map(Number) : [], id: Number(rf.slice(at + 1)) };
}

export function analysisKey(key: GraphKey) {
  return key === '' ? 'main' : key;
}

// Ports of a nest straight from the document, so they never lag the analysis.
export function subgraphPorts(doc: Doc, name: string): { inputs: PortSpec[]; outputs: PortSpec[] } {
  const g = doc.subgraphs[name];
  if (!g) return { inputs: [], outputs: [] };
  const order = (type: string) =>
    g.nodes
      .filter((n) => n.type === type)
      .sort((a, b) => Number(a.attrs.index ?? 0) - Number(b.attrs.index ?? 0) || a.id - b.id);
  return {
    inputs: order('Input').map((n) => ({ name: n.name, dtype: n.attrs.dtype === 'i64' ? 'i64' : 'f32', optional: false })),
    outputs: order('Output').map((n) => ({ name: n.name, dtype: null, optional: false })),
  };
}

export function portsOf(doc: Doc, node: GNode, spec?: NodeSpec, info?: NodeAnalysis) {
  if (isNest(node)) return subgraphPorts(doc, nestTarget(node));
  if (info) return info.ports;
  return { inputs: spec?.inputs ?? [], outputs: spec?.outputs ?? [] };
}

type Ports = { inputs: PortSpec[]; outputs: PortSpec[] };

export function directionOf(doc: Doc): Direction {
  return doc.layout === 'TB' ? 'TB' : 'LR';
}

// Vertical nodes put all inputs in one row under the header and all outputs
// in one row along the bottom edge.
export function opHeight(ports: Ports, params: boolean, error: boolean, dir: Direction = 'LR') {
  const extra = (params ? PARAM_ROW : 0) + (error ? ERROR_H : 0);
  if (dir === 'TB') return HEADER + (ports.inputs.length ? ROW : 0) + (ports.outputs.length ? ROW : 0) + FOOTER + extra;
  const rows = Math.max(ports.inputs.length, ports.outputs.length, 1);
  return HEADER + rows * ROW + FOOTER + extra;
}

export function nestHeader(ports: Ports, dir: Direction = 'LR') {
  if (dir === 'TB') return HEADER + (ports.inputs.length ? ROW : 0) + 8;
  return HEADER + Math.max(ports.inputs.length, ports.outputs.length, 1) * ROW + 12;
}

interface Built {
  nodes: FlowNode[];
  edges: Edge[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

const DTYPE_COLOR = { f32: 'var(--f32)', i64: 'var(--i64)' } as const;

export function buildFlow(doc: Doc, specs: Record<string, NodeSpec>, analysis: AnalyzeResult | null): { nodes: FlowNode[]; edges: Edge[] } {
  const built = buildGraph(doc, specs, analysis, '', [], new Set());
  return { nodes: built.nodes, edges: built.edges };
}

// The nodes of one graph as they would be drawn (sizes, ports), positioned in
// that graph's own coordinates. Used by the layout engines.
export function graphTopNodes(doc: Doc, specs: Record<string, NodeSpec>, analysis: AnalyzeResult | null, key: GraphKey): FlowNode[] {
  return buildGraph(doc, specs, analysis, key, [], new Set(key ? [key] : [])).nodes.filter((n) => n.parentId === undefined);
}

function buildGraph(doc: Doc, specs: Record<string, NodeSpec>, analysis: AnalyzeResult | null, key: GraphKey, path: number[], stack: Set<string>): Built {
  const g: Graph = graphOf(doc, key);
  const ga = analysis?.graphs[analysisKey(key)];
  const dir = directionOf(doc);
  const out: Built = { nodes: [], edges: [], minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };

  for (const n of g.nodes) {
    const spec = specs[n.type];
    const info = ga?.nodes[String(n.id)];
    const error = ga?.errors[String(n.id)] ?? (spec || !Object.keys(specs).length ? undefined : `unknown node type '${n.type}'`);
    const ports = portsOf(doc, n, spec, info);
    const id = rfId(path, n.id);
    const sub = isNest(n) ? nestTarget(n) : '';
    const recursive = !!sub && stack.has(sub);
    const expanded = isNest(n) && !!n.ui?.expanded && !!doc.subgraphs[sub] && !recursive;

    let width = NODE_W;
    let height = opHeight(ports, !!info?.params, !!error, dir);
    let inner: Built | null = null;
    let bodyTop = 0;

    if (expanded) {
      inner = buildGraph(doc, specs, analysis, sub, [...path, n.id], new Set([...stack, sub]));
      bodyTop = nestHeader(ports, dir) + (error ? ERROR_H : 0);
      const innerW = Number.isFinite(inner.minX) ? inner.maxX - inner.minX : 200;
      const innerH = Number.isFinite(inner.minY) ? inner.maxY - inner.minY : 80;
      width = Math.max(NODE_W + 80, innerW + 2 * NEST_PAD);
      height = bodyTop + innerH + NEST_PAD + (dir === 'TB' && ports.outputs.length ? ROW : 0);
    }

    out.nodes.push({
      id,
      type: isNest(n) ? 'nest' : 'op',
      position: { x: n.pos[0], y: n.pos[1] },
      width,
      height,
      style: { width, height },
      zIndex: path.length,
      data: { key, dir, path, node: n, spec, info, error, ports, origin: { x: 0, y: 0 }, expanded, bodyTop, recursive },
    });
    const self = out.nodes[out.nodes.length - 1];
    out.minX = Math.min(out.minX, n.pos[0]);
    out.minY = Math.min(out.minY, n.pos[1]);
    out.maxX = Math.max(out.maxX, n.pos[0] + width);
    out.maxY = Math.max(out.maxY, n.pos[1] + height);

    if (inner) {
      // Children: positions relative to the box, origin remembered for drags.
      const ox = (Number.isFinite(inner.minX) ? inner.minX : 0) - NEST_PAD;
      const oy = (Number.isFinite(inner.minY) ? inner.minY : 0) - bodyTop;
      self.data.childOrigin = { x: ox, y: oy };
      for (const c of inner.nodes) {
        if (c.parentId === undefined) {
          c.parentId = id;
          c.position = { x: c.position.x - ox, y: c.position.y - oy };
          c.data.origin = { x: ox, y: oy };
        }
        out.nodes.push(c);
      }
      out.edges.push(...inner.edges);

      // Dashed wires from the box's ports to the Input/Output nodes inside.
      const subGraph = doc.subgraphs[sub];
      for (const p of ports.inputs) {
        const target = subGraph.nodes.find((m) => m.type === 'Input' && m.name === p.name);
        if (target) out.edges.push(wire(`${id}|in|${p.name}`, id, `ii:${p.name}`, rfId([...path, n.id], target.id), 'pin'));
      }
      for (const p of ports.outputs) {
        const source = subGraph.nodes.find((m) => m.type === 'Output' && m.name === p.name);
        if (source) out.edges.push(wire(`${id}|out|${p.name}`, rfId([...path, n.id], source.id), 'pout', id, `oo:${p.name}`));
      }
    }
  }

  for (const l of g.links) {
    const from = g.nodes.find((n) => n.id === l.from[0]);
    if (!from) continue;
    const info = ga?.nodes[String(l.from[0])];
    const idx = info?.ports.outputs.findIndex((p) => p.name === l.from[1]) ?? -1;
    const dtype = (idx >= 0 ? info?.outputs?.[idx]?.dtype : undefined) ?? 'f32';
    const shape = idx >= 0 ? info?.outputs?.[idx]?.text : undefined;
    // Use the Tidy route only while both ends are still where Tidy put them.
    const to = g.nodes.find((n) => n.id === l.to[0]);
    const r = l.route;
    const routed = !!r && !!to && r.src[0] === from.pos[0] && r.src[1] === from.pos[1] && r.dst[0] === to.pos[0] && r.dst[1] === to.pos[1];
    out.edges.push({
      id: `${path.join('/')}|${l.from[0]}:${l.from[1]}>${l.to[0]}:${l.to[1]}`,
      type: routed ? 'routed' : 'default',
      source: rfId(path, l.from[0]),
      sourceHandle: `o:${l.from[1]}`,
      target: rfId(path, l.to[0]),
      targetHandle: `i:${l.to[1]}`,
      style: { stroke: DTYPE_COLOR[dtype], strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: DTYPE_COLOR[dtype], width: 16, height: 16 },
      data: { path, link: l, shape, route: routed ? r!.points : undefined },
      zIndex: path.length,
    });
  }
  return out;
}

function wire(id: string, source: string, sourceHandle: string, target: string, targetHandle: string): Edge {
  return {
    id, source, sourceHandle, target, targetHandle,
    className: 'nest-wire', selectable: false, deletable: false, focusable: false, zIndex: 0,
    markerEnd: { type: MarkerType.ArrowClosed, color: 'var(--border-strong)', width: 14, height: 14 },
  };
}

// ---- layout ------------------------------------------------------------------

// Layered layout of one graph in the document's direction, using the sizes
// React Flow would draw (so expanded nests take the room they need).
export function layoutGraph(doc: Doc, specs: Record<string, NodeSpec>, analysis: AnalyzeResult | null, key: GraphKey) {
  const g = graphOf(doc, key);
  const sizes = new Map<number, { w: number; h: number }>();
  const flow = buildGraph(doc, specs, analysis, key, [], new Set(key ? [key] : []));
  for (const n of flow.nodes) {
    if (n.parentId !== undefined) continue;
    sizes.set(n.data.node.id, { w: n.width ?? NODE_W, h: n.height ?? 80 });
  }

  const depth = new Map<number, number>(g.nodes.map((n) => [n.id, 0]));
  for (let pass = 0; pass < g.nodes.length; pass++) {
    let changed = false;
    for (const l of g.links) {
      const d = (depth.get(l.from[0]) ?? 0) + 1;
      if ((depth.get(l.to[0]) ?? 0) < d) {
        depth.set(l.to[0], d);
        changed = true;
      }
    }
    if (!changed) break;
  }
  // Pull sources (inputs, embeddings) right next to their first consumer.
  for (const n of g.nodes) {
    const outs = g.links.filter((l) => l.from[0] === n.id).map((l) => depth.get(l.to[0]) ?? 0);
    if (outs.length && !g.links.some((l) => l.to[0] === n.id)) depth.set(n.id, Math.max(0, Math.min(...outs) - 1));
  }

  const layers = new Map<number, GNode[]>();
  for (const n of g.nodes) {
    const d = depth.get(n.id) ?? 0;
    layers.set(d, [...(layers.get(d) ?? []), n]);
  }

  // Layers advance along the flow axis (x for LR, y for TB); nodes within a
  // layer spread across the other axis, ordered by their predecessors.
  const vertical = directionOf(doc) === 'TB';
  const along = (s: { w: number; h: number }) => (vertical ? s.h : s.w);
  const across = (s: { w: number; h: number }) => (vertical ? s.w : s.h);
  const gapAlong = vertical ? 56 : 70;
  const gapAcross = vertical ? 36 : 40;

  const slotOf = new Map<number, number>();
  let cursor = 0;
  for (const d of [...layers.keys()].sort((a, b) => a - b)) {
    const nodes = layers.get(d)!;
    const order = (n: GNode) => {
      const vs = g.links.filter((l) => l.to[0] === n.id && slotOf.has(l.from[0])).map((l) => slotOf.get(l.from[0])!);
      return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : 0;
    };
    nodes.sort((a, b) => order(a) - order(b));
    const sizeOf = (n: GNode) => sizes.get(n.id) ?? { w: NODE_W, h: 80 };
    const total = nodes.reduce((s, n) => s + across(sizeOf(n)) + gapAcross, -gapAcross);
    let offset = -total / 2;
    let thickness = 0;
    for (const n of nodes) {
      const s = sizeOf(n);
      n.pos = vertical ? [Math.round(offset), cursor] : [cursor, Math.round(offset)];
      slotOf.set(n.id, offset + across(s) / 2);
      offset += across(s) + gapAcross;
      thickness = Math.max(thickness, along(s));
    }
    cursor += thickness + gapAlong;
  }
}

// Lays out every graph, innermost subgraphs first so boxes have final sizes.
export function layoutAll(doc: Doc, specs: Record<string, NodeSpec>, analysis: AnalyzeResult | null) {
  const done = new Set<string>();
  const visit = (key: GraphKey, stack: Set<string>) => {
    if (done.has(key) || stack.has(key)) return;
    const g = graphOf(doc, key);
    for (const n of g.nodes) {
      const sub = isNest(n) ? nestTarget(n) : '';
      if (sub && doc.subgraphs[sub]) visit(sub, new Set([...stack, key]));
    }
    layoutGraph(doc, specs, analysis, key);
    done.add(key);
  };
  for (const name of Object.keys(doc.subgraphs)) visit(name, new Set());
  visit('', new Set());
}

// Graphs (by key) in which two sibling nodes overlap on screen.
export function overlappingGraphs(nodes: FlowNode[]): GraphKey[] {
  const groups = new Map<string, FlowNode[]>();
  for (const n of nodes) groups.set(n.parentId ?? '', [...(groups.get(n.parentId ?? '') ?? []), n]);
  const keys: { key: GraphKey; depth: number }[] = [];
  for (const list of groups.values()) {
    const hit = list.some((a, i) =>
      list.slice(i + 1).some(
        (b) =>
          a.position.x < b.position.x + (b.width ?? 0) &&
          b.position.x < a.position.x + (a.width ?? 0) &&
          a.position.y < b.position.y + (b.height ?? 0) &&
          b.position.y < a.position.y + (a.height ?? 0),
      ),
    );
    if (hit && !keys.some((k) => k.key === list[0].data.key)) keys.push({ key: list[0].data.key, depth: list[0].data.path.length });
  }
  // Innermost first, so outer layouts see the final box sizes.
  return keys.sort((a, b) => b.depth - a.depth).map((k) => k.key);
}
