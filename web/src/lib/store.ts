import { create } from 'zustand';
import { api, baseName } from './api';
import { addRecent } from './recent';
import { graphKeyAt } from './docops';
import type { AnalyzeResult, Doc, GraphKey, NodeRef, NodeSpec, Status } from './types';

interface Toast {
  id: number;
  text: string;
  error: boolean;
}

interface State {
  doc: Doc | null;
  file: string;              // graphs/<file> on the server; '' = unsaved
  dirty: boolean;
  past: Doc[];
  future: Doc[];

  specs: Record<string, NodeSpec>;
  registryVersion: number;
  analysis: AnalyzeResult | null;
  status: Status | null;
  builtRevision: number;
  revision: number;

  selection: NodeRef[];      // selected nodes (all in one graph for grouping)
  loadedAt: number;          // bumps when a document is opened (the canvas re-fits)
  scriptsOpen: boolean;
  scriptFile: string | null;
  toasts: Toast[];

  setDoc: (doc: Doc, file: string) => void;
  edit: (fn: (draft: Doc) => void, opts?: { history?: boolean }) => void;
  checkpoint: () => void;    // push the current doc before a multi-step change (drags)
  undo: () => void;
  redo: () => void;
  select: (refs: NodeRef[]) => void;
  loadSpecs: () => Promise<void>;
  analyze: () => Promise<void>;
  toast: (text: string, error?: boolean) => void;
  openScripts: (file?: string | null) => void;
  closeScripts: () => void;
  markBuilt: () => void;
  setStatus: (s: Status) => void;
  saveAs: (file: string) => Promise<void>;
}

let toastId = 0;

export const useStore = create<State>((set, get) => ({
  doc: null,
  file: '',
  dirty: false,
  past: [],
  future: [],
  specs: {},
  registryVersion: -1,
  analysis: null,
  status: null,
  builtRevision: -1,
  revision: 0,
  selection: [],
  loadedAt: 0,
  scriptsOpen: false,
  scriptFile: null,
  toasts: [],

  setDoc: (doc, file) => {
    set({ doc, file, dirty: false, past: [], future: [], selection: [], revision: get().revision + 1, loadedAt: Date.now() });
    addRecent(file, doc.name);
    void get().analyze();
  },

  edit: (fn, opts) => {
    const { doc, past } = get();
    if (!doc) return;
    const draft = structuredClone(doc);
    try {
      fn(draft);
    } catch (e) {
      get().toast((e as Error).message, true);
      return;
    }
    const history = opts?.history ?? true;
    set({
      doc: draft,
      dirty: true,
      past: history ? [...past.slice(-199), doc] : past,
      future: history ? [] : get().future,
      revision: get().revision + 1,
    });
    scheduleAnalyze();
  },

  checkpoint: () => {
    const { doc, past } = get();
    if (doc) set({ past: [...past.slice(-199), doc], future: [] });
  },

  undo: () => {
    const { past, doc, future } = get();
    if (!past.length || !doc) return;
    set({ doc: past[past.length - 1], past: past.slice(0, -1), future: [doc, ...future], dirty: true, revision: get().revision + 1 });
    scheduleAnalyze();
  },

  redo: () => {
    const { past, doc, future } = get();
    if (!future.length || !doc) return;
    set({ doc: future[0], future: future.slice(1), past: [...past, doc], dirty: true, revision: get().revision + 1 });
    scheduleAnalyze();
  },

  select: (refs) => set({ selection: refs }),

  loadSpecs: async () => {
    const r = await api.nodes();
    const specs: Record<string, NodeSpec> = {};
    for (const s of r.specs) specs[s.type] = s;
    set({ specs, registryVersion: r.version });
  },

  analyze: async () => {
    const { doc, file } = get();
    if (!doc) return;
    try {
      const analysis = await api.analyze(doc, file);
      if (get().doc === doc) set({ analysis });
    } catch (e) {
      get().toast((e as Error).message, true);
    }
  },

  toast: (text, error = false) => {
    const id = ++toastId;
    set({ toasts: [...get().toasts, { id, text, error }] });
    setTimeout(() => set({ toasts: get().toasts.filter((t) => t.id !== id) }), error ? 6000 : 3000);
  },

  openScripts: (file = null) => set({ scriptsOpen: true, scriptFile: file }),
  closeScripts: () => set({ scriptsOpen: false }),
  markBuilt: () => set({ builtRevision: get().revision }),

  setStatus: (s) => {
    const prev = get().status;
    set({ status: s });
    if (s.registryVersion !== get().registryVersion) {
      // A Python node was added, edited or removed: refresh the palette and re-check the graph.
      void get().loadSpecs().then(() => get().analyze());
      if (prev && s.scriptGeneration !== prev.scriptGeneration) get().toast('Python nodes reloaded');
    }
  },

  saveAs: async (file) => {
    const { doc } = get();
    if (!doc) return;
    const name = file.endsWith('.json') ? file : `${file}.json`;
    await api.saveGraph(name, doc);
    set({ file: name, dirty: false });
    addRecent(name, doc.name);
    get().toast(`Saved ${baseName(name)}`);
  },
}));

let analyzeTimer: ReturnType<typeof setTimeout> | undefined;
function scheduleAnalyze() {
  clearTimeout(analyzeTimer);
  analyzeTimer = setTimeout(() => void useStore.getState().analyze(), 120);
}

// Ask the canvas to re-tidy after a nest was expanded or collapsed; `path`
// is the path of the graph that holds that nest.
export function requestRelayout(path: number[]) {
  window.dispatchEvent(new CustomEvent('mg-relayout', { detail: path }));
}

// Helpers used across components.
export function analysisFor(key: GraphKey) {
  return useStore.getState().analysis?.graphs[key === '' ? 'main' : key];
}

export function keyOfPath(doc: Doc, path: number[]): GraphKey | null {
  try {
    return graphKeyAt(doc, path);
  } catch {
    return null;
  }
}
