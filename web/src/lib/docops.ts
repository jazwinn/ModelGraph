// Pure edits on a Doc draft. The store clones the document, applies one of
// these, and pushes the previous version onto the undo stack.

import type { AttrValue, Doc, GLink, GNode, Graph, GraphAnalysis, GraphKey, NodeSpec, TensorType } from './types';

export function graphOf(doc: Doc, key: GraphKey): Graph {
  if (key === '') return doc.graph;
  const g = doc.subgraphs[key];
  if (!g) throw new Error(`subgraph '${key}' does not exist`);
  return g;
}

export function isNest(n: GNode): boolean {
  return n.type === 'Group' || n.type === 'Repeat';
}

export function nestTarget(n: GNode): string {
  return typeof n.attrs.subgraph === 'string' ? n.attrs.subgraph : '';
}

// The graph a path of nest ids leads to ('' = main).
export function graphKeyAt(doc: Doc, path: number[]): GraphKey {
  let key: GraphKey = '';
  for (const id of path) {
    const n = graphOf(doc, key).nodes.find((m) => m.id === id);
    if (!n || !isNest(n)) throw new Error('stale path');
    key = nestTarget(n);
  }
  return key;
}

export function nextId(g: Graph): number {
  return g.nodes.reduce((m, n) => Math.max(m, n.id), 0) + 1;
}

export function uniqueName(g: Graph, stem: string, ignore?: number): string {
  const taken = new Set(g.nodes.filter((n) => n.id !== ignore).map((n) => n.name));
  if (!taken.has(stem)) return stem;
  for (let i = 1; ; i++) if (!taken.has(`${stem}_${i}`)) return `${stem}_${i}`;
}

export function uniqueSubgraphName(doc: Doc, stem: string): string {
  if (!doc.subgraphs[stem]) return stem;
  for (let i = 2; ; i++) if (!doc.subgraphs[`${stem}${i}`]) return `${stem}${i}`;
}

function stemFor(type: string): string {
  return type.charAt(0).toLowerCase() + type.slice(1);
}

export function addNode(g: Graph, spec: NodeSpec, x: number, y: number, extra?: Partial<GNode>): GNode {
  const attrs: Record<string, AttrValue> = {};
  for (const a of spec.attrs) attrs[a.name] = a.default;
  if (spec.role !== 'op' && spec.attrs.some((a) => a.name === 'index')) {
    attrs.index = g.nodes.filter((n) => n.type === spec.type).length;
  }
  const node: GNode = { id: nextId(g), type: spec.type, name: uniqueName(g, stemFor(spec.type)), attrs, pos: [x, y], ...extra };
  g.nodes.push(node);
  return node;
}

export function connect(g: Graph, from: number, fromPort: string, to: number, toPort: string) {
  g.links = g.links.filter((l) => !(l.to[0] === to && l.to[1] === toPort));
  g.links.push({ from: [from, fromPort], to: [to, toPort] });
}

export function removeNodes(g: Graph, ids: Set<number>) {
  g.nodes = g.nodes.filter((n) => !ids.has(n.id));
  g.links = g.links.filter((l) => !ids.has(l.from[0]) && !ids.has(l.to[0]));
}

export function sameLink(a: GLink, b: GLink) {
  return a.from[0] === b.from[0] && a.from[1] === b.from[1] && a.to[0] === b.to[0] && a.to[1] === b.to[1];
}

export function wouldCycle(g: Graph, from: number, to: number): boolean {
  const stack = [to];
  const seen = new Set<number>();
  while (stack.length) {
    const n = stack.pop()!;
    if (n === from) return true;
    if (seen.has(n)) continue;
    seen.add(n);
    for (const l of g.links) if (l.from[0] === n) stack.push(l.to[0]);
  }
  return false;
}

export function subgraphUsers(doc: Doc, name: string): number {
  let count = 0;
  for (const g of [doc.graph, ...Object.values(doc.subgraphs)]) {
    for (const n of g.nodes) if (isNest(n) && nestTarget(n) === name) count++;
  }
  return count;
}

// "B, S, 768" from an analysed type, for the declared shape of an Input node.
export function shapeAttr(t?: TensorType | null): string {
  return t ? t.shape.join(', ') : 'B, S, $d_model';
}

export function emptySubgraph(): Graph {
  return {
    nodes: [
      { id: 1, type: 'Input', name: 'x', attrs: { index: 0, dtype: 'f32', shape: 'B, S, $d_model' }, pos: [0, 0] },
      { id: 2, type: 'Output', name: 'y', attrs: { index: 0 }, pos: [320, 0] },
    ],
    links: [{ from: [1, 'out'], to: [2, 'value'] }],
  };
}

// Wraps `ids` (all in graph `key`) into a new nest. Links that cross the
// boundary become Input/Output nodes inside and ports on the nest outside.
export function groupNodes(doc: Doc, key: GraphKey, ids: Set<number>, analysis?: GraphAnalysis): number {
  const g = graphOf(doc, key);
  const picked = g.nodes.filter((n) => ids.has(n.id));
  if (!picked.length) throw new Error('select the nodes to wrap first');
  if (picked.some((n) => n.type === 'TokenIds' || n.type === 'Logits')) throw new Error('TokenIds and Logits must stay in the main graph');

  const subName = uniqueSubgraphName(doc, 'Nest');
  const sub: Graph = { nodes: picked.map((n) => structuredClone(n)), links: [] };
  let next = nextId(sub);

  const minX = Math.min(...picked.map((n) => n.pos[0]));
  const minY = Math.min(...picked.map((n) => n.pos[1]));
  const maxX = Math.max(...picked.map((n) => n.pos[0]));

  const outer: GLink[] = [];
  const inputs = new Map<string, string>();  // "node:port" outside -> Input node name inside
  const outputs = new Map<string, string>(); // "node:port" inside  -> Output node name inside
  let inputIndex = 0;
  let outputIndex = 0;
  const outputType = (nodeId: number, port: string) => {
    const a = analysis?.nodes[String(nodeId)];
    const i = a?.ports.outputs.findIndex((p) => p.name === port) ?? -1;
    return i >= 0 ? a?.outputs?.[i] : undefined;
  };

  for (const l of g.links) {
    const fromIn = ids.has(l.from[0]);
    const toIn = ids.has(l.to[0]);
    if (fromIn && toIn) {
      sub.links.push(structuredClone(l));
    } else if (!fromIn && toIn) {
      const k = `${l.from[0]}:${l.from[1]}`;
      let name = inputs.get(k);
      if (!name) {
        name = uniqueName(sub, l.to[1] === 'x' || l.to[1] === 'a' || l.to[1] === 'b' ? 'x' : l.to[1]);
        const t = outputType(l.from[0], l.from[1]);
        sub.nodes.push({ id: next++, type: 'Input', name, attrs: { index: inputIndex++, dtype: t?.dtype ?? 'f32', shape: shapeAttr(t) }, pos: [minX - 260, minY + 110 * inputs.size] });
        inputs.set(k, name);
        outer.push({ from: [l.from[0], l.from[1]], to: [-1, name] });
      }
      const inputNode = sub.nodes.find((n) => n.name === name)!;
      sub.links.push({ from: [inputNode.id, 'out'], to: [l.to[0], l.to[1]] });
    } else if (fromIn && !toIn) {
      const k = `${l.from[0]}:${l.from[1]}`;
      let name = outputs.get(k);
      if (!name) {
        name = uniqueName(sub, outputs.size === 0 ? 'y' : `y${outputs.size}`);
        const outNode: GNode = { id: next++, type: 'Output', name, attrs: { index: outputIndex++ }, pos: [maxX + 300, minY + 110 * outputs.size] };
        sub.nodes.push(outNode);
        sub.links.push({ from: [l.from[0], l.from[1]], to: [outNode.id, 'value'] });
        outputs.set(k, name);
      }
      outer.push({ from: [-1, name], to: [l.to[0], l.to[1]] });
    }
  }

  // Replace the picked nodes with the nest.
  const cx = picked.reduce((s, n) => s + n.pos[0], 0) / picked.length;
  const cy = picked.reduce((s, n) => s + n.pos[1], 0) / picked.length;
  removeNodes(g, ids);
  const nest: GNode = { id: nextId(g), type: 'Group', name: uniqueName(g, 'nest'), attrs: { subgraph: subName }, pos: [cx, cy], ui: { expanded: true } };
  g.nodes.push(nest);
  for (const l of outer) {
    g.links.push({ from: [l.from[0] === -1 ? nest.id : l.from[0], l.from[1]], to: [l.to[0] === -1 ? nest.id : l.to[0], l.to[1]] });
  }
  doc.subgraphs[subName] = sub;
  return nest.id;
}

// Inlines a nest's subgraph into its parent graph (the inverse of groupNodes).
export function ungroup(doc: Doc, key: GraphKey, nestId: number) {
  const g = graphOf(doc, key);
  const nest = g.nodes.find((n) => n.id === nestId);
  if (!nest || !isNest(nest)) throw new Error('not a nest');
  if (nest.type === 'Repeat') throw new Error('turn the Repeat back into a single Group before ungrouping');
  const subName = nestTarget(nest);
  const sub = graphOf(doc, subName);

  const idMap = new Map<number, number>();
  let next = nextId(g);
  const io = new Map<number, GNode>();
  for (const n of sub.nodes) {
    if (n.type === 'Input' || n.type === 'Output') {
      io.set(n.id, n);
      continue;
    }
    const copy = structuredClone(n);
    copy.id = next++;
    copy.name = uniqueName(g, n.name);
    copy.pos = [n.pos[0] + nest.pos[0], n.pos[1] + nest.pos[1]];
    idMap.set(n.id, copy.id);
    g.nodes.push(copy);
  }

  const outerIn = g.links.filter((l) => l.to[0] === nestId);
  const outerOut = g.links.filter((l) => l.from[0] === nestId);
  const links = g.links.filter((l) => l.to[0] !== nestId && l.from[0] !== nestId);

  for (const l of sub.links) {
    const src = io.get(l.from[0]);
    const dst = io.get(l.to[0]);
    if (src && dst) continue; // Input wired straight to Output: handled below
    if (src) {
      // Input -> inner node: connect whatever fed that port outside.
      for (const o of outerIn.filter((o) => o.to[1] === src.name)) links.push({ from: o.from, to: [idMap.get(l.to[0])!, l.to[1]] });
    } else if (dst) {
      for (const o of outerOut.filter((o) => o.from[1] === dst.name)) links.push({ from: [idMap.get(l.from[0])!, l.from[1]], to: o.to });
    } else {
      links.push({ from: [idMap.get(l.from[0])!, l.from[1]], to: [idMap.get(l.to[0])!, l.to[1]] });
    }
  }
  for (const l of sub.links) {
    const src = io.get(l.from[0]);
    const dst = io.get(l.to[0]);
    if (!src || !dst) continue;
    for (const i of outerIn.filter((o) => o.to[1] === src.name))
      for (const o of outerOut.filter((o) => o.from[1] === dst.name)) links.push({ from: i.from, to: o.to });
  }
  g.links = links;
  g.nodes = g.nodes.filter((n) => n.id !== nestId);
  if (subgraphUsers(doc, subName) === 0) delete doc.subgraphs[subName];
}

export function emptyDoc(): Doc {
  return {
    format: 'modelgraph',
    version: 1,
    name: 'Untitled',
    variables: { vocab_size: 256, d_model: 64, num_heads: 4, num_layers: 2, ffn_dim: 256, max_seq_len: 256, norm_eps: 1e-5 },
    model: { checkpoint: '', format: 'raw', rename_rules: [], tokenizer: '', chat_template: '', system_prompt: 'You are a helpful assistant.', eos: [], device: 'cpu', context_size: 256 },
    graph: {
      nodes: [
        { id: 1, type: 'TokenIds', name: 'tokens', attrs: {}, pos: [0, 0] },
        { id: 2, type: 'Logits', name: 'logits', attrs: {}, pos: [700, 0] },
      ],
      links: [],
    },
    subgraphs: {},
  };
}

// Numbers typed into a numeric field stay numbers; anything else is kept as
// an expression string such as "$d_model * 4".
export function parseAttrInput(text: string, kind: 'int' | 'float' | 'other'): AttrValue {
  const t = text.trim();
  if (kind !== 'other' && t !== '' && /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(t)) {
    const v = Number(t);
    if (kind === 'int' && Number.isInteger(v) && !/[.eE]/.test(t)) return v;
    if (kind === 'float' || !Number.isInteger(v)) return v;
    return v;
  }
  return text;
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}
