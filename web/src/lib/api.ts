import type { AnalyzeResult, Doc, NodeSpec, ScriptStatus, Status } from './types';

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data: unknown;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || res.statusText);
  }
  if (!res.ok) throw new Error((data as { error?: string }).error ?? res.statusText);
  return data as T;
}

export function isAbsolute(file: string) {
  return /^[A-Za-z]:[\\/]/.test(file) || file.startsWith('/') || file.startsWith('\\\\');
}

export function baseName(file: string) {
  return file.split(/[\\/]/).pop() ?? file;
}

export const api = {
  info: () => call<{ root: string; graphs: string; scripts: string; cuda: boolean }>('GET', '/api/info'),
  nodes: () => call<{ version: number; specs: NodeSpec[] }>('GET', '/api/nodes'),
  analyze: (doc: Doc, file: string) => call<AnalyzeResult>('POST', '/api/analyze', { doc, file }),

  listGraphs: () => call<{ files: string[] }>('GET', '/api/graphs'),
  // `file` is a name inside graphs/ or an absolute path chosen in a file dialog.
  loadGraph: (file: string) =>
    isAbsolute(file)
      ? call<Doc>('GET', `/api/file?path=${encodeURIComponent(file)}`)
      : call<Doc>('GET', `/api/graphs/${encodeURIComponent(file)}`),
  saveGraph: (file: string, doc: Doc) =>
    isAbsolute(file)
      ? call<{ saved: string }>('PUT', `/api/file?path=${encodeURIComponent(file)}`, doc)
      : call<{ saved: string }>('PUT', `/api/graphs/${encodeURIComponent(file)}`, doc),
  openDialog: () => call<{ cancelled?: boolean; file?: string; doc?: Doc }>('POST', '/api/dialog/open'),
  saveDialog: (suggested: string) => call<{ cancelled?: boolean; file?: string }>('POST', '/api/dialog/save', { suggested }),
  templates: () => call<{ templates: { id: string; title: string }[] }>('GET', '/api/templates'),
  template: (id: string) => call<Doc>('GET', `/api/templates/${encodeURIComponent(id)}`),

  build: (doc: Doc, file: string, mode: 'fresh' | 'checkpoint' | 'keep') => call('POST', '/api/build', { doc, file, mode }),
  generate: (o: { prompt: string; maxTokens: number; temperature: number; chat: boolean; system?: string }) => call('POST', '/api/generate', o),
  stop: () => call('POST', '/api/stop'),
  status: () => call<Status>('GET', '/api/status'),

  scripts: () => call<{ scripts: ScriptStatus[]; dir: string }>('GET', '/api/scripts'),
  readScript: (file: string) => call<{ file: string; text: string }>('GET', `/api/scripts/${encodeURIComponent(file)}`),
  writeScript: (file: string, text: string) => call('PUT', `/api/scripts/${encodeURIComponent(file)}`, { text }),
  newScript: (name: string) => call<{ file: string }>('POST', '/api/scripts', { name }),
  deleteScript: (file: string) => call('DELETE', `/api/scripts/${encodeURIComponent(file)}`),
};
