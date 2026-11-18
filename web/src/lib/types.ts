// Mirrors the graph JSON written by the C++ side (src/Graph/Document.cpp).

export type AttrValue = boolean | number | string;

export interface GNode {
  id: number;
  type: string;
  name: string;
  attrs: Record<string, AttrValue>;
  pos: [number, number];
  ui?: { expanded?: boolean };
}

export interface GLink {
  from: [number, string];
  to: [number, string];
  // Set by Tidy: the path the edge takes around other nodes, in graph
  // coordinates. Used only while both end nodes stay where Tidy put them.
  route?: {
    points: [number, number][];
    src: [number, number]; // source node position when routed
    dst: [number, number]; // target node position when routed
  };
}

export interface Graph {
  nodes: GNode[];
  links: GLink[];
}

export interface ModelSettings {
  checkpoint: string;
  format: string;
  rename_rules: { from: string; to: string }[];
  tokenizer: string;
  chat_template: string;
  system_prompt?: string;
  eos: number[];
  device: string;
  context_size: number;
}

// Which way data flows on the canvas: left-to-right or top-to-bottom.
export type Direction = 'LR' | 'TB';

export interface Doc {
  format?: string;
  layout?: Direction;
  version?: number;
  name: string;
  variables: Record<string, AttrValue>;
  model: ModelSettings;
  graph: Graph;
  subgraphs: Record<string, Graph>;
}

// ---- registry ----

export interface PortSpec {
  name: string;
  dtype: 'f32' | 'i64' | null;
  optional: boolean;
}

export type AttrKind = 'bool' | 'int' | 'float' | 'string' | 'enum' | 'shape' | 'subgraph' | 'noderef';

export interface AttrSpec {
  name: string;
  kind: AttrKind;
  default: AttrValue;
  options: string[];
  help: string;
}

export interface NodeSpec {
  type: string;
  category: string;
  description: string;
  source: string;
  role: 'op' | 'input' | 'output';
  inputs: PortSpec[];
  outputs: PortSpec[];
  attrs: AttrSpec[];
  dynamicPorts: boolean;
}

// ---- analysis ----

export interface TensorType {
  dtype: 'f32' | 'i64';
  shape: (number | string)[];
  text: string;
}

export interface NodeAnalysis {
  ports: { inputs: PortSpec[]; outputs: PortSpec[] };
  inputs?: (TensorType | null)[];
  outputs?: TensorType[];
  params?: number;
}

export interface GraphAnalysis {
  nodes: Record<string, NodeAnalysis>;
  errors: Record<string, string>;
  graphErrors: string[];
  totalParams: number;
}

export interface AnalyzeResult {
  graphs: Record<string, GraphAnalysis>; // "main" or a subgraph name
  variables: Record<string, number>;
  registryVersion: number;
}

export interface Status {
  busy: boolean;
  ready: boolean;
  phase: string;
  log: string;
  output: string;
  prompt: string;
  tokensPerSecond: number;
  parameters: number;
  buildCount: number;
  registryVersion: number;
  scriptGeneration: number;
}

export interface ScriptStatus {
  file: string;
  type: string;
  ok: boolean;
  error: string;
  reloads: number;
}

// A graph inside the document: "" is the main graph, otherwise a subgraph name.
export type GraphKey = string;
export const MAIN: GraphKey = '';

// Where a node lives on the canvas: the chain of nest node ids from the main
// graph down to its graph, plus its own id. Nests may share a subgraph, so the
// path (not just the graph) identifies an on-screen copy.
export interface NodeRef {
  path: number[];
  id: number;
}

// ---- Hugging Face downloads ----

export interface HubResult {
  repo: string;
  folder: string;
  checkpoint: string;
  format: string;
  modelType: string;
  tokenizer?: string;
  chat_template?: string;
  eos: number[];
  notes: string[];
  config?: Record<string, number>;
}

export interface HubStatus {
  busy: boolean;
  job: number;
  repo?: string;
  phase?: string;
  file?: string;
  fileDone?: number;
  fileTotal?: number;
  done?: number;
  total?: number;
  error?: string | null;
  result?: HubResult | null;
}
