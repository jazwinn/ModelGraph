# ModelGraph

Build and test LLM architectures visually: drag nodes, wire them up, wrap groups of nodes into **nests**, and run the result on a real inference engine without writing C++. When a building block is missing, write it as a **Python node**; saving the file reloads it immediately.

All tensor maths, layers, tokenizers and checkpoint loading live in the sibling [InferenceEngine](../InferenceEngine) repository. ModelGraph is the graph layer and the editor on top.

## What you can do

- **Edit every part of a model.** The GPT-2 and Llama templates are built from primitives (Linear, SplitHeads, RoPE, KVCache, SDPA, Activation, ...) inside nests, so you can open the Attention or MLP box and change anything.
- **Nests.** Select nodes and press `Ctrl G` to wrap them in a box. Inputs and outputs that cross the boundary become the box's ports. Expand or collapse a box in place, unwrap it, or turn it into a `Repeat ×N` (each copy gets its own weights, e.g. `transformer.h.0`, `transformer.h.1`, ...).
- **Live checking.** Every edit re-runs shape inference: tensor shapes on every port (`[B,S,768]`), parameter counts per node and nest, and errors shown on the node that causes them.
- **Variables.** Hyper-parameters such as `d_model` are defined once and used anywhere as expressions (`$d_model * 4`).
- **Run it.** Build with random weights or load a `.safetensors` checkpoint, then generate text on CPU or CUDA. "Rebuild, keep weights" recompiles after an edit and keeps every weight whose name and shape still match, so you can swap one layer and keep the rest pretrained.
- **Python nodes.** Each `.py` file in `scripts/` defines a node type with its ports, attributes, optional weights, an optional shape function and `forward()`. Edit them in the built-in editor. Models that are already built pick up a new `forward()` on the next token.

## Quick start (Windows)

Requirements: Visual Studio 2026 (or 2022), CMake ≥ 3.24, vcpkg at `C:\vcpkg`, CUDA 13 (for the CUDA build), Python ≥ 3.9 with development files (the python.org installer includes them), and Node.js ≥ 18. You do **not** need the InferenceEngine source: the build downloads its latest release (see below).

```powershell
# 1. Native side: engine, graph compiler, server, tests
cmake --preset default -DPython_EXECUTABLE="$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
cmake --build build --config Release

# 2. Web UI
cd web; npm install; npm run build; cd ..

# 3. Model weights (optional: the "tiny" model runs on random weights)
python tools\download_models.py gpt2                   # ~550 MB
python tools\download_models.py llama3.2-1b-instruct   # ~2.5 GB

# 4. Run: opens http://127.0.0.1:8710 in your browser
build\src\Release\mg_server.exe
```

Weights and tokenizers go into `models/`, which is git-ignored. The example graphs in `graphs/` point there (`../models/gpt2/model.safetensors`, ...). The script downloads from Hugging Face ([openai-community/gpt2](https://huggingface.co/openai-community/gpt2), and [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct), an ungated copy of Meta's model under the Llama 3.2 Community License). Set `HF_TOKEN` for models that need a login.

You can also download from the editor: select nothing (so the inspector shows the document), type a repo such as `HuggingFaceTB/SmolLM2-135M-Instruct` under **Weights & tokenizer** and press **Download**. The server fetches `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and the config files into `models/<name>/`, then fills in the checkpoint, name mapping, tokenizer, chat template and stop tokens. Only single-file `model.safetensors` checkpoints are supported, and the graph must still match the model's architecture. For gated repos, set `HF_TOKEN` before starting `mg_server`.

Use the `cpu` preset (`cmake --preset cpu`) to build without CUDA. Pick the Python to embed with `-DPython_EXECUTABLE=...`. Mixing two Python installs (for example a conda Python on `PATH`) is not supported; the build copies the right `python3xx.dll` next to each executable.

For UI development, run `npm run dev` in `web/` (Vite on port 5173 proxies `/api` to a running `mg_server`).

### Where InferenceEngine comes from

By default the build uses the latest [InferenceEngine release](https://github.com/jazwinn/InferenceEngine/releases): the CPU or CUDA package (matching `IE_ENABLE_CUDA`) is downloaded into `build/_deps/`. Before every build it checks GitHub again and installs a newer release if one exists. Offline, the installed copy keeps working.

| Option | Effect |
| --- | --- |
| *(default)* | latest release, re-checked on every build |
| `-DIE_VERSION=v0.1.0` | stay on one release |
| `-DIE_SOURCE_DIR=../InferenceEngine` (or `cmake --preset dev`) | build the engine from a local checkout, for working on both repos |

If GitHub rate-limits the release check (60 requests per hour without a login), set a `GITHUB_TOKEN` environment variable.

## Using the editor

| Action | How |
| --- | --- |
| Add a node | Drag from the palette, double-click the canvas, or press `/` |
| Add into a nest | Drop or double-click inside an expanded box |
| Connect | Drag from an output (right) to an input (left); drop on empty canvas to add a connected node |
| Nest / unwrap | Select nodes, then `Ctrl G`; right-click a nest and choose **Unwrap** |
| Repeat a block | Select a nest, then **Repeat ×N** in the inspector |
| Expand / collapse | The `+`/`−` button on a nest, or double-click it |
| Delete | `Del` / `Backspace` |
| Undo / redo / save | `Ctrl Z` / `Ctrl Y` / `Ctrl S` |
| Horizontal / vertical | The **Horizontal / Vertical** toggle in the top bar re-arranges every graph; the choice is saved with the graph |
| Tidy layout, fit | **Tidy** in the top bar (ELK layered layout: edges are routed around nodes, not through them); `F` to fit |

Weight names come from node names along the nest path: node `query` inside nest `attention`, inside copy 3 of the Repeat `transformer.h`, owns `transformer.h.3.attention.query.weight`. That is how checkpoints map onto graphs. The `gpt2` and `llama` name mappings reuse InferenceEngine's converters; `raw` plus rename rules covers other checkpoints.

## Python nodes

```python
import ie

name = "DyT"                       # node type shown in the palette
inputs = ["x"]
outputs = ["y"]
attrs = {"dim": "$d_model"}        # bool / int / float / str; "$..." are expressions

def params(a):                     # optional learned weights (loaded from checkpoints too)
    return {"weight": {"shape": [a.dim], "init": "ones"}}

def shape(a, x):                   # optional: output shape from input shapes
    return x

def forward(ctx, x):               # ctx.attrs, ctx.params, ctx.start_pos, ctx.device
    return ie.tanh(x) * ctx.params.weight
```

`ie` exposes InferenceEngine tensors (CPU or CUDA): arithmetic and `@`, `reshape`, `transpose`, `mean`, `ie.linear`, `ie.matmul`, `ie.softmax`, activations, `ie.sdpa`, `ie.split_heads`, `ie.merge_heads`, `x.numpy()` and `ie.tensor(array)`. `ie.exp`, `ie.log`, `ie.pow` and `ie.map` fall back to the CPU, so they are for prototyping. Examples are in [scripts/](scripts).

## Layout

```
src/Graph/     graph format, node registry, shape inference, compiler -> nn::Module
src/Script/    embedded Python: Python node specs, file watching, hot reload
src/Session/   build / load weights / generate on a worker thread
src/Server/    mg_server: JSON API + serves web/dist
src/Cli/       mg_cli: check, run and export graphs headlessly
web/           React + React Flow editor
graphs/        example models (gpt2, llama3.2-1b, tiny)
models/        downloaded weights and tokenizers (git-ignored)
tools/         download_models.py
scripts/       Python node definitions
tests/         GoogleTest suite
```

`mg_cli` covers the same pipeline without the UI:

```powershell
build\src\Release\mg_cli.exe check graphs\gpt2.json
build\src\Release\mg_cli.exe run graphs\llama3.2-1b.json --chat --tokens 64 "What is attention?"
build\src\Release\mg_cli.exe template tiny graphs\my-model.json
```

## Tests

```powershell
ctest --test-dir build -C Release --output-on-failure
```

The equivalence tests build the GPT-2-style, Llama-style and post-norm templates (with nested, fully decomposed attention and MLP) and check that their logits match InferenceEngine's hand-written `CausalModel` with and without the KV cache, on CPU and CUDA. Other tests cover shape inference, nests and Repeat, Python nodes (loading, hot reload, parameters, numpy interop) and the document format.
