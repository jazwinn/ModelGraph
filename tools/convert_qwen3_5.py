"""Download Qwen3.5 and convert it for graphs/qwen3.5-4b.json.

    python tools/convert_qwen3_5.py                  # Qwen/Qwen3.5-4B
    python tools/convert_qwen3_5.py --repo Qwen/Qwen3.5-4B --keep

The Hub checkpoint can't be loaded as-is: it is split over several files,
carries a vision tower and a multi-token-prediction head the graph doesn't
use, and its RMSNorms compute x * (1 + w) where the built-in node computes
x * w. This script streams the shards into models/<name>/model.safetensors
with only the language model's tensors, named after the graph's nodes
(layer i lives at layers.<i // 4>.<delta0|delta1|delta2|attn>), and with 1
added to those norm weights. bf16 weights stay bf16; the loader widens them.

It also writes a tokenizer.json without the NFC normaliser (the engine's
tokenizer has none; NFC leaves ordinary text unchanged) and a chat template
with thinking switched off unless --think. The raw download in
models/<name>/hf/ is deleted unless --keep.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_models import ROOT, download  # noqa: E402

FILES = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
LAYER_SLOTS = ["delta0", "delta1", "delta2", "attn"]   # one Repeat copy of the HybridBlock subgraph
SKIP = ("model.visual.", "mtp.")
# Norms Qwen3.5 applies as x * (1 + w); the gated norm inside linear_attn is a plain x * w.
ONE_PLUS = re.compile(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm)\.weight$|^model\.language_model\.norm\.weight$")
DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4}


def graph_name(hf):
    """HF tensor name -> the graph's weight name, or None to drop it."""
    if hf.startswith(SKIP):
        return None
    name = hf.removeprefix("model.language_model.")
    m = re.match(r"layers\.(\d+)\.(.*)", name)
    if not m:
        return name   # embed_tokens.weight, norm.weight
    i, rest = int(m.group(1)), m.group(2)
    rest = (rest.replace("mlp.gate_proj", "mlp.gate").replace("mlp.up_proj", "mlp.up").replace("mlp.down_proj", "mlp.down")
                .replace("linear_attn.conv1d.weight", "linear_attn.conv.weight")
                .replace("linear_attn.A_log", "linear_attn.delta_rule.A_log")
                .replace("linear_attn.dt_bias", "linear_attn.delta_rule.dt_bias"))
    return f"layers.{i // 4}.{LAYER_SLOTS[i % 4]}.{rest}"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def to_f32(raw, dtype):
    if dtype == "F32":
        return np.frombuffer(raw, np.float32)
    if dtype == "BF16":
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, np.float16).astype(np.float32)
    sys.exit(f"unexpected dtype {dtype}")


def convert(hf_dir, out_path):
    index = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    plan = []   # (out name, shard, entry, data start, transform)
    for shard in shards:
        header, start = read_header(os.path.join(hf_dir, shard))
        for hf, entry in header.items():
            name = graph_name(hf)
            if name is None:
                continue
            if entry["dtype"] not in DTYPE_SIZE:
                sys.exit(f"{hf}: unexpected dtype {entry['dtype']}")
            kind = "one_plus" if ONE_PLUS.search(hf) else "copy"
            plan.append((name, shard, entry, start, kind))
    plan.sort(key=lambda p: p[0])

    out_header, offset = {}, 0
    for name, _, entry, _, kind in plan:
        shape = list(entry["shape"])
        if name.endswith("linear_attn.conv.weight"):
            shape = [shape[0], shape[-1]]   # [C, 1, K] -> [C, K]
        dtype = "F32" if kind == "one_plus" else entry["dtype"]
        size = int(np.prod(shape)) * DTYPE_SIZE[dtype]
        out_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    part = out_path + ".part"
    handles = {s: open(os.path.join(hf_dir, s), "rb") for s in shards}
    with open(part, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for i, (name, shard, entry, start, kind) in enumerate(plan):
            f = handles[shard]
            b, e = entry["data_offsets"]
            f.seek(start + b)
            raw = f.read(e - b)
            if kind == "one_plus":
                raw = (to_f32(raw, entry["dtype"]) + 1.0).astype(np.float32).tobytes()
            out.write(raw)
            print(f"\r  writing {i + 1}/{len(plan)} tensors", end="", flush=True)
    print()
    for f in handles.values():
        f.close()
    os.replace(part, out_path)
    return len(plan), offset


def write_tokenizer(hf_dir, folder, think):
    tok = json.load(open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8"))
    if tok.get("normalizer") and tok["normalizer"].get("type") != "NFC":
        sys.exit(f"tokenizer.json has a {tok['normalizer'].get('type')} normaliser; only NFC can be dropped safely")
    tok["normalizer"] = None
    with open(os.path.join(folder, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)

    # The editor can't pass enable_thinking, so bake the choice into the template.
    template = open(os.path.join(hf_dir, "chat_template.jinja"), encoding="utf-8").read()
    if not think:
        template = "{%- set enable_thinking = false %}\n" + template
    with open(os.path.join(folder, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(template)
    config = json.load(open(os.path.join(hf_dir, "tokenizer_config.json"), encoding="utf-8"))
    config["chat_template"] = template
    with open(os.path.join(folder, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    shutil.copyfile(os.path.join(hf_dir, "config.json"), os.path.join(folder, "config.json"))
    ids = {t["content"]: t["id"] for t in tok["added_tokens"]}
    return [ids[t] for t in ("<|im_end|>", "<|endoftext|>") if t in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--keep", action="store_true", help="keep the raw download in models/<name>/hf/")
    p.add_argument("--think", action="store_true", help="leave thinking mode on (the chat template's default)")
    args = p.parse_args()

    folder = os.path.join(ROOT, args.repo.split("/")[-1].lower())
    hf_dir = os.path.join(folder, "hf")
    os.makedirs(hf_dir, exist_ok=True)
    print(f"{args.repo} -> {os.path.relpath(folder, os.path.join(ROOT, '..'))}")

    base = f"https://huggingface.co/{args.repo}/resolve/main/"
    for name in FILES:
        download(base + name, os.path.join(hf_dir, name))
    index = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))
    for shard in sorted(set(index["weight_map"].values())):
        download(base + shard, os.path.join(hf_dir, shard))

    print("converting")
    count, size = convert(hf_dir, os.path.join(folder, "model.safetensors"))
    eos = write_tokenizer(hf_dir, folder, args.think)
    print(f"  {count} tensors, {size / 1e9:.2f} GB; stop tokens {eos}")
    if not args.keep:
        shutil.rmtree(hf_dir)


if __name__ == "__main__":
    main()
