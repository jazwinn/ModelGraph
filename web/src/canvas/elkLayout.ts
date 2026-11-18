// "Tidy" using ELK's layered algorithm: crossing minimisation, room reserved
// for long edges, and orthogonal edge routes that go around nodes instead of
// through them. Each graph is laid out on its own, innermost nests first, so
// an expanded nest is laid out at its final size.

import ELK from 'elkjs/lib/elk.bundled.js';
import type { ElkExtendedEdge, ElkNode, ElkPort } from 'elkjs/lib/elk-api';
import { graphOf, isNest, nestTarget } from '../lib/docops';
import type { AnalyzeResult, Doc, GraphKey, NodeSpec } from '../lib/types';
import { directionOf, graphTopNodes, HEADER, ROW } from './flow';

const elk = new ELK();

function portPositions(dir: 'LR' | 'TB', width: number, height: number, count: number, input: boolean) {
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i < count; i++) {
    if (dir === 'TB') out.push({ x: (width * (i + 0.5)) / count, y: input ? 0 : height });
    else out.push({ x: input ? 0 : width, y: HEADER + i * ROW + ROW / 2 });
  }
  return out;
}

async function layoutGraph(doc: Doc, specs: Record<string, NodeSpec>, analysis: AnalyzeResult | null, key: GraphKey) {
  const g = graphOf(doc, key);
  const dir = directionOf(doc);
  const nodes = graphTopNodes(doc, specs, analysis, key);
  const portIds = new Set<string>();

  const children: ElkNode[] = nodes.map((n) => {
    const id = String(n.data.node.id);
    const w = n.width ?? 236;
    const h = n.height ?? 80;
    const ports: ElkPort[] = [];
    const add = (names: string[], input: boolean) =>
      portPositions(dir, w, h, names.length, input).forEach((p, i) => {
        const pid = `${id}|${input ? 'i' : 'o'}|${names[i]}`;
        portIds.add(pid);
        ports.push({
          id: pid,
          x: p.x,
          y: p.y,
          width: 0,
          height: 0,
          layoutOptions: { 'elk.port.side': dir === 'TB' ? (input ? 'NORTH' : 'SOUTH') : input ? 'WEST' : 'EAST', 'elk.port.index': String(i) },
        });
      });
    add(n.data.ports.inputs.map((p) => p.name), true);
    add(n.data.ports.outputs.map((p) => p.name), false);
    return { id, width: w, height: h, ports, layoutOptions: { 'elk.portConstraints': 'FIXED_POS' } };
  });

  const edges: ElkExtendedEdge[] = [];
  g.links.forEach((l, i) => {
    const s = `${l.from[0]}|o|${l.from[1]}`;
    const t = `${l.to[0]}|i|${l.to[1]}`;
    if (portIds.has(s) && portIds.has(t)) edges.push({ id: String(i), sources: [s], targets: [t] });
  });

  const result = await elk.layout({
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': dir === 'TB' ? 'DOWN' : 'RIGHT',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.spacing.nodeNodeBetweenLayers': dir === 'TB' ? '56' : '80',
      'elk.spacing.nodeNode': '40',
      'elk.layered.spacing.edgeNodeBetweenLayers': '24',
      'elk.spacing.edgeNode': '24',
      'elk.spacing.edgeEdge': '14',
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
      'elk.layered.nodePlacement.bk.fixedAlignment': 'BALANCED',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
    },
    children,
    edges,
  });

  // Write node positions and edge routes back into the document.
  const pos = new Map<number, [number, number]>();
  for (const c of result.children ?? []) pos.set(Number(c.id), [Math.round(c.x ?? 0), Math.round(c.y ?? 0)]);
  for (const n of g.nodes) {
    const p = pos.get(n.id);
    if (p) n.pos = p;
  }
  for (const l of g.links) delete l.route;
  for (const e of (result.edges ?? []) as ElkExtendedEdge[]) {
    const section = e.sections?.[0];
    const link = g.links[Number(e.id)];
    if (!section || !link) continue;
    const src = pos.get(link.from[0]);
    const dst = pos.get(link.to[0]);
    if (!src || !dst) continue;
    const points: [number, number][] = [
      [section.startPoint.x, section.startPoint.y],
      ...(section.bendPoints ?? []).map((b) => [b.x, b.y] as [number, number]),
      [section.endPoint.x, section.endPoint.y],
    ];
    link.route = { points, src, dst };
  }
}

// Lays out every graph in `doc` (mutated), innermost subgraphs first.
export async function tidyAll(doc: Doc, specs: Record<string, NodeSpec>, analysis: AnalyzeResult | null, only?: GraphKey[]) {
  const done = new Set<string>();
  const order: GraphKey[] = [];
  const visit = (key: GraphKey, stack: Set<string>) => {
    if (done.has(key) || stack.has(key)) return;
    for (const n of graphOf(doc, key).nodes) {
      const sub = isNest(n) ? nestTarget(n) : '';
      if (sub && doc.subgraphs[sub]) visit(sub, new Set([...stack, key]));
    }
    done.add(key);
    order.push(key);
  };
  for (const name of Object.keys(doc.subgraphs)) visit(name, new Set());
  visit('', new Set());
  for (const key of order) {
    if (only && !only.includes(key)) continue;
    await layoutGraph(doc, specs, analysis, key);
  }
}
