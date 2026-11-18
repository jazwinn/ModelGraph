import { Handle, Position, type NodeProps } from '@xyflow/react';
import { memo } from 'react';
import { formatCount, graphOf } from '../lib/docops';
import { evaluate } from '../lib/expr';
import { requestRelayout, useStore } from '../lib/store';
import type { PortSpec } from '../lib/types';
import { HEADER, ROW, type FlowNode } from './flow';

export const CATEGORY_COLORS: Record<string, string> = {
  'Inputs & Outputs': '#e5b84b',
  Layers: '#5aa9ff',
  Transformer: '#a685ff',
  Structure: '#ff7ab2',
  Math: '#4fd1a5',
  Tensor: '#62c9d9',
  'Attention parts': '#c79bff',
};

export function categoryColor(category?: string) {
  return (category && CATEGORY_COLORS[category]) || '#ff9f5a';
}

function dtypeClass(p: PortSpec, shape?: string) {
  if (shape?.endsWith('i64') || p.dtype === 'i64') return 'i64';
  return 'f32';
}

function Ports({ data }: { data: FlowNode['data'] }) {
  const { ports, info } = data;
  if (data.dir === 'TB') return <InputRow data={data} />;
  const rows = Math.max(ports.inputs.length, ports.outputs.length, 1);
  return (
    // Rows are static (not positioned) so each Handle anchors to the node box.
    <div className="ports" style={{ height: rows * ROW }}>
      <div className="port-col">
        {ports.inputs.map((p, i) => (
          <div key={`i${p.name}`} className="port in" style={{ height: ROW }}>
            <Handle type="target" position={Position.Left} id={`i:${p.name}`} className={`handle ${dtypeClass(p)}`} style={{ top: HEADER + i * ROW + ROW / 2 }} />
            <span className="port-name">{p.name}{p.optional ? '?' : ''}</span>
          </div>
        ))}
      </div>
      <div className="port-col right">
        {ports.outputs.map((p, i) => {
          const shape = info?.outputs?.[i]?.text;
          return (
            <div key={`o${p.name}`} className="port out" style={{ height: ROW }}>
              {shape && <span className="shape">{shape}</span>}
              <span className="port-name">{p.name}</span>
              <Handle type="source" position={Position.Right} id={`o:${p.name}`} className={`handle ${dtypeClass(p, shape)}`} style={{ top: HEADER + i * ROW + ROW / 2 }} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- vertical (top-to-bottom) ports: one row of inputs, one row of outputs ----

const spread = (i: number, n: number) => `${((i + 0.5) / n) * 100}%`;

function InputRow({ data }: { data: FlowNode['data'] }) {
  const { inputs } = data.ports;
  if (!inputs.length) return null;
  return (
    <>
      <div className="port-row" style={{ height: ROW }}>
        {inputs.map((p) => (
          <span key={p.name} className="port-chip">{p.name}{p.optional ? '?' : ''}</span>
        ))}
      </div>
      {inputs.map((p, i) => (
        <Handle key={`h${p.name}`} type="target" position={Position.Top} id={`i:${p.name}`} className={`handle ${dtypeClass(p)}`} style={{ left: spread(i, inputs.length) }} />
      ))}
    </>
  );
}

function OutputRow({ data, pinned }: { data: FlowNode['data']; pinned?: boolean }) {
  const { outputs } = data.ports;
  if (data.dir !== 'TB' || !outputs.length) return null;
  return (
    <>
      <div className={`port-row out${pinned ? ' pinned' : ''}`} style={{ height: ROW }}>
        {outputs.map((p, i) => {
          const shape = data.info?.outputs?.[i]?.text;
          return (
            <span key={p.name} className="port-chip">
              {p.name}
              {shape && <span className="shape">{shape}</span>}
            </span>
          );
        })}
      </div>
      {outputs.map((p, i) => (
        <Handle key={`h${p.name}`} type="source" position={Position.Bottom} id={`o:${p.name}`} className={`handle ${dtypeClass(p, data.info?.outputs?.[i]?.text)}`} style={{ left: spread(i, outputs.length) }} />
      ))}
    </>
  );
}

function Header({ data, color, children }: { data: FlowNode['data']; color: string; children?: React.ReactNode }) {
  const { node } = data;
  const lowerType = children ? node.name.toLowerCase() : node.type.toLowerCase();   // nests show a badge instead
  return (
    <div className="node-header" style={{ height: HEADER }}>
      <span className="dot" style={{ background: color }} />
      <span className="node-name">{node.name}</span>
      {node.name.toLowerCase() !== lowerType && <span className="node-type">{node.type}</span>}
      {children}
    </div>
  );
}

export const OpNode = memo(function OpNode({ data, selected }: NodeProps<FlowNode>) {
  const color = categoryColor(data.spec?.category);
  const role = data.spec?.role;
  return (
    <div className={`node op${selected ? ' selected' : ''}${data.error ? ' has-error' : ''}`} style={{ ['--accent' as string]: color }}>
      <Header data={data} color={color} />
      <Ports data={data} />
      {!!data.info?.params && <div className="node-foot">{formatCount(data.info.params)} params</div>}
      {data.error && <div className="node-error" title={data.error}>{data.error}</div>}
      <OutputRow data={data} />
      {role === 'input' && data.path.length > 0 &&
        (data.dir === 'TB' ? (
          <Handle type="target" position={Position.Top} id="pin" className="handle hidden" />
        ) : (
          <Handle type="target" position={Position.Left} id="pin" className="handle hidden" style={{ top: HEADER / 2 }} />
        ))}
      {role === 'output' && data.path.length > 0 &&
        (data.dir === 'TB' ? (
          <Handle type="source" position={Position.Bottom} id="pout" className="handle hidden" />
        ) : (
          <Handle type="source" position={Position.Right} id="pout" className="handle hidden" style={{ top: HEADER / 2 }} />
        ))}
    </div>
  );
});

export const NestNode = memo(function NestNode({ data, selected }: NodeProps<FlowNode>) {
  const edit = useStore((s) => s.edit);
  const { node, expanded, ports, info } = data;
  const color = categoryColor('Structure');
  const sub = typeof node.attrs.subgraph === 'string' ? node.attrs.subgraph : '';
  const vars = useStore((s) => s.analysis?.variables);
  let count = node.type === 'Repeat' ? String(node.attrs.count ?? '') : '';
  try {
    if (count && vars) count = String(evaluate(count, vars));
  } catch {
    /* keep the expression text */
  }
  const toggle = () => {
    edit(
      (d) => {
        const n = graphOf(d, data.key).nodes.find((m) => m.id === node.id);
        if (n) n.ui = { ...(n.ui ?? {}), expanded: !expanded };
      },
      { history: false },
    );
    requestRelayout(data.path);
  };

  const badge = (
    <span className="nest-badge">
      {node.type === 'Repeat' && <b>×{count}</b>}
      <span>{sub || 'no subgraph'}</span>
    </span>
  );

  return (
    <div
      className={`node nest${expanded ? ' expanded' : ''}${selected ? ' selected' : ''}${data.error ? ' has-error' : ''}${node.type === 'Repeat' ? ' repeat' : ''}`}
      style={{ ['--accent' as string]: color }}
    >
      <Header data={data} color={color}>
        {badge}
        <button className="nest-toggle nodrag" onClick={toggle} title={expanded ? 'Collapse' : 'Expand'} disabled={data.recursive}>
          {expanded ? '−' : '+'}
        </button>
      </Header>
      <Ports data={data} />
      {!expanded && !!info?.params && <div className="node-foot">{formatCount(info.params)} params</div>}
      {expanded && !!info?.params && <div className="nest-params">{formatCount(info.params)} params</div>}
      {data.error && <div className="node-error" title={data.error}>{data.error}</div>}
      <OutputRow data={data} pinned={expanded} />
      {expanded &&
        ports.inputs.map((p, i) =>
          data.dir === 'TB' ? (
            <Handle key={`ii${p.name}`} type="source" position={Position.Bottom} id={`ii:${p.name}`} className="handle hidden" isConnectable={false} style={{ top: (data.bodyTop ?? 0) - 6, bottom: 'auto', left: spread(i, ports.inputs.length) }} />
          ) : (
            <Handle key={`ii${p.name}`} type="source" position={Position.Left} id={`ii:${p.name}`} className="handle hidden" isConnectable={false} style={{ top: HEADER + i * ROW + ROW / 2, left: 8 }} />
          ),
        )}
      {expanded &&
        ports.outputs.map((p, i) =>
          data.dir === 'TB' ? (
            <Handle key={`oo${p.name}`} type="target" position={Position.Top} id={`oo:${p.name}`} className="handle hidden" isConnectable={false} style={{ top: 'auto', bottom: ROW + 6, left: spread(i, ports.outputs.length) }} />
          ) : (
            <Handle key={`oo${p.name}`} type="target" position={Position.Right} id={`oo:${p.name}`} className="handle hidden" isConnectable={false} style={{ top: HEADER + i * ROW + ROW / 2, right: 8 }} />
          ),
        )}
      {expanded && <div className="nest-body" style={{ top: data.bodyTop, bottom: data.dir === 'TB' && ports.outputs.length ? ROW : 0 }} />}
    </div>
  );
});

export const nodeTypes = { op: OpNode, nest: NestNode };
