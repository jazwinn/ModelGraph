// Per-node activity from the running model (GET /api/activity), turned into
// what the canvas draws:
//   heat:  each node glows by its share of the model's time
//   trace: the server pauses after every node, so the glow walks the graph
// Keys are canvas node ids (flow.ts rfId), which the server uses as well.

import { create } from 'zustand';
import { api } from './api';

export type ActivityMode = 'off' | 'heat' | 'trace';

export interface ActivityNode {
  seconds: number;
  calls: number;
  nest: boolean;
}

export interface ActivityReply {
  mode: ActivityMode;
  traceDelayMs: number;
  nodes: Record<string, ActivityNode>;
  running: string[];
  forwardSeconds: number;
  forwards: number;
  synced: boolean;
}

export interface NodeHeat {
  share: number;      // of all forward-pass time
  intensity: number;  // 0..1, relative to the hottest single node
  msPerPass: number;
  callsPerPass: number;
}

interface ActivityState {
  mode: ActivityMode;
  traceDelayMs: number;
  heat: Record<string, NodeHeat>;
  running: Record<string, true>;
  forwards: number;
  synced: boolean;
  setMode: (mode: ActivityMode, traceDelayMs?: number) => Promise<void>;
  apply: (reply: ActivityReply) => void;
}

const STORAGE_KEY = 'mg-activity';

function loadPrefs(): { mode: ActivityMode; traceDelayMs: number } {
  try {
    const p = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    const mode = ['off', 'heat', 'trace'].includes(p.mode) ? (p.mode as ActivityMode) : 'off';
    const traceDelayMs = Number.isFinite(p.traceDelayMs) ? Number(p.traceDelayMs) : 150;
    return { mode, traceDelayMs };
  } catch {
    return { mode: 'off', traceDelayMs: 150 };
  }
}

function savePrefs(mode: ActivityMode, traceDelayMs: number) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ mode, traceDelayMs }));
  } catch {
    /* ignore */
  }
}

export function computeHeat(r: ActivityReply): Record<string, NodeHeat> {
  const out: Record<string, NodeHeat> = {};
  if (r.forwardSeconds <= 0 || r.forwards <= 0) return out;
  let hottest = 0;
  for (const n of Object.values(r.nodes)) if (!n.nest) hottest = Math.max(hottest, n.seconds);
  if (hottest <= 0) return out;
  for (const [key, n] of Object.entries(r.nodes)) {
    out[key] = {
      share: n.seconds / r.forwardSeconds,
      // Square root keeps cheap nodes faintly visible next to one dominant node.
      intensity: Math.min(1, Math.sqrt(n.seconds / hottest)),
      msPerPass: (n.seconds * 1000) / r.forwards,
      callsPerPass: n.calls / r.forwards,
    };
  }
  return out;
}

const initial = loadPrefs();

export const useActivity = create<ActivityState>((set, get) => ({
  mode: initial.mode,
  traceDelayMs: initial.traceDelayMs,
  heat: {},
  running: {},
  forwards: 0,
  synced: false,

  setMode: async (mode, traceDelayMs = get().traceDelayMs) => {
    set({ mode, traceDelayMs, ...(mode === 'trace' ? {} : { running: {} }) });
    savePrefs(mode, traceDelayMs);
    await api.setActivity(mode, traceDelayMs);
  },

  apply: (r) => {
    const running: Record<string, true> = {};
    for (const k of r.running) running[k] = true;
    set({ heat: computeHeat(r), running, forwards: r.forwards, synced: r.synced });
  },
}));

// Tell the server the saved mode once it is reachable (it starts in 'off').
let synced = false;
export async function syncActivityMode() {
  if (synced) return;
  const { mode, traceDelayMs } = useActivity.getState();
  await api.setActivity(mode, traceDelayMs);
  synced = true;
}

export function resetActivitySync() {
  synced = false;
}

// Called from the status poll. Fetches while the model runs, plus once more
// after it stops so the final heat stays on screen. Starts true so a reloaded
// page shows the last run.
let lastBusy = true;
export async function pollActivity(busy: boolean) {
  const fetchNow = busy || lastBusy;
  lastBusy = busy;
  if (!fetchNow || useActivity.getState().mode === 'off') {
    if (!busy && Object.keys(useActivity.getState().running).length) useActivity.setState({ running: {} });
    return;
  }
  useActivity.getState().apply(await api.activity());
}

// The glow for one canvas node. Returns primitives so nodes re-render only
// when their own value changes.
export function useNodeActivity(id: string) {
  const mode = useActivity((s) => s.mode);
  const intensity = useActivity((s) => (s.mode === 'off' ? 0 : Math.round((s.heat[id]?.intensity ?? 0) * 50) / 50));
  const share = useActivity((s) => (s.mode === 'off' ? 0 : s.heat[id]?.share ?? 0));
  const ms = useActivity((s) => s.heat[id]?.msPerPass ?? 0);
  const calls = useActivity((s) => s.heat[id]?.callsPerPass ?? 0);
  const running = useActivity((s) => s.mode === 'trace' && !!s.running[id]);
  return { mode, intensity, share, ms, calls, running };
}

export function formatShare(share: number) {
  const pct = share * 100;
  return pct >= 10 ? `${pct.toFixed(0)}%` : pct >= 1 ? `${pct.toFixed(1)}%` : `${pct.toFixed(2)}%`;
}

export function activityTitle(a: { share: number; ms: number; calls: number }) {
  if (!a.share) return undefined;
  const calls = a.calls > 1.01 ? ` · runs ${Math.round(a.calls)}× per pass` : '';
  return `${formatShare(a.share)} of model time · ${a.ms.toFixed(a.ms < 1 ? 3 : 2)} ms per forward pass${calls}`;
}
