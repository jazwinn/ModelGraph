// Hugging Face downloads. The server runs one job at a time; this store polls
// it and, when a download finishes, fills in the document's weight settings.

import { create } from 'zustand';
import { api } from './api';
import { useStore } from './store';
import type { HubResult, HubStatus } from './types';

interface HubState {
  status: HubStatus | null;
  start: (repo: string) => Promise<void>;
  cancel: () => Promise<void>;
  resume: () => Promise<void>;   // pick up a job that was running before a page reload
}

let timer: ReturnType<typeof setTimeout> | null = null;
let target: { file: string; job: number } | null = null;   // the document that asked

function apply(r: HubResult) {
  const { file, edit, toast } = useStore.getState();
  if (!target || target.file !== file) {
    toast(`Downloaded ${r.repo} to ${r.folder}. Open the graph you started it from, or set the paths by hand.`);
    return;
  }
  edit((d) => {
    d.model.checkpoint = r.checkpoint;
    d.model.format = r.format;
    if (r.tokenizer) d.model.tokenizer = r.tokenizer;
    if (r.chat_template) d.model.chat_template = r.chat_template;
    if (r.eos.length) d.model.eos = r.eos;
  });
  toast(`Downloaded ${r.repo}. Paths updated; build with "checkpoint" to load the weights.`);
  for (const note of r.notes) toast(note, true);
}

async function poll() {
  timer = null;
  let s: HubStatus;
  try {
    s = await api.hubStatus();
  } catch {
    timer = setTimeout(poll, 1500);
    return;
  }
  useHub.setState({ status: s });
  if (s.busy) {
    timer = setTimeout(poll, 400);
    return;
  }
  if (target && s.job === target.job) {
    if (s.result) apply(s.result);
    else if (s.error && s.error !== 'cancelled') useStore.getState().toast(s.error, true);
    target = null;
  }
}

export const useHub = create<HubState>(() => ({
  status: null,

  start: async (repo) => {
    const file = useStore.getState().file;
    await api.hubDownload(repo, file);
    const s = await api.hubStatus();
    target = { file, job: s.job };
    useHub.setState({ status: s });
    if (!timer) timer = setTimeout(poll, 300);
  },

  cancel: async () => {
    await api.hubCancel();
  },

  resume: async () => {
    if (timer) return;
    try {
      const s = await api.hubStatus();
      useHub.setState({ status: s });
      if (s.busy) timer = setTimeout(poll, 400);
    } catch {
      /* server not up yet */
    }
  },
}));
