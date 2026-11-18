import { BaseEdge, getBezierPath, type EdgeProps } from '@xyflow/react';
import { memo } from 'react';

type Pt = [number, number];

// Rounded orthogonal path through `pts`.
function roundedPath(pts: Pt[], radius = 8): string {
  if (pts.length < 2) return '';
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1];
    const [x, y] = pts[i];
    const [nx, ny] = pts[i + 1];
    const inLen = Math.hypot(x - px, y - py);
    const outLen = Math.hypot(nx - x, ny - y);
    const r = Math.min(radius, inLen / 2, outLen / 2);
    if (r < 0.5) {
      d += ` L ${x} ${y}`;
      continue;
    }
    const ax = x - ((x - px) / inLen) * r;
    const ay = y - ((y - py) / inLen) * r;
    const bx = x + ((nx - x) / outLen) * r;
    const by = y + ((ny - y) / outLen) * r;
    d += ` L ${ax} ${ay} Q ${x} ${y} ${bx} ${by}`;
  }
  const [lx, ly] = pts[pts.length - 1];
  return `${d} L ${lx} ${ly}`;
}

// An edge that follows the route Tidy computed (in graph coordinates). The
// route is shifted onto the live handle positions; if anything looks off it
// falls back to a normal curve.
export const RoutedEdge = memo(function RoutedEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, markerEnd, data } = props;
  const route = (data as { route?: Pt[] } | undefined)?.route;
  const [fallback] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  if (!route || route.length < 2) return <BaseEdge path={fallback} style={style} markerEnd={markerEnd} />;

  const dx = sourceX - route[0][0];
  const dy = sourceY - route[0][1];
  const pts: Pt[] = route.map(([x, y]) => [x + dx, y + dy]);
  const last = pts[pts.length - 1];
  // The two ends can differ by a few pixels from ELK's port positions;
  // snap the end onto the target handle, keeping the last segment straight.
  if (Math.abs(last[0] - targetX) > 40 || Math.abs(last[1] - targetY) > 40) {
    return <BaseEdge path={fallback} style={style} markerEnd={markerEnd} />;
  }
  if (pts.length >= 3) {
    const prev = pts[pts.length - 2];
    if (Math.abs(prev[1] - last[1]) < 0.5) prev[1] = targetY;
    else if (Math.abs(prev[0] - last[0]) < 0.5) prev[0] = targetX;
  }
  pts[pts.length - 1] = [targetX, targetY];
  return <BaseEdge path={roundedPath(pts)} style={style} markerEnd={markerEnd} />;
});

export const edgeTypes = { routed: RoutedEdge };
