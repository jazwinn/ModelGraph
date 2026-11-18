"""Download model weights and tokenizers into models/ (git-ignored).

    python tools/download_models.py            # list what is available
    python tools/download_models.py gpt2
    python tools/download_models.py llama3.2-1b-instruct
    python tools/download_models.py all

Files come from Hugging Face. Set HF_TOKEN if a repository needs a login.
Existing complete files are skipped, so re-running is cheap.
"""

import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

MODELS = {
    "gpt2": {
        "repo": "openai-community/gpt2",
        "folder": "gpt2",
        "files": ["model.safetensors", "tokenizer.json"],
        "about": "GPT-2 small, 124M parameters (~550 MB)",
    },
    "llama3.2-1b-instruct": {
        # An ungated copy of meta-llama/Llama-3.2-1B-Instruct; Meta's Llama 3.2
        # Community License still applies.
        "repo": "unsloth/Llama-3.2-1B-Instruct",
        "folder": "llama-3.2-1b-instruct",
        "files": ["model.safetensors", "tokenizer.json", "tokenizer_config.json"],
        "about": "Llama 3.2 1B Instruct (~2.5 GB)",
    },
}


def request(url, method="GET"):
    headers = {"User-Agent": "ModelGraph-downloader"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers, method=method)


def remote_size(url):
    try:
        with urllib.request.urlopen(request(url, "HEAD"), timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except urllib.error.URLError:
        return 0


def download(url, dest):
    size = remote_size(url)
    if os.path.exists(dest) and (size == 0 or os.path.getsize(dest) == size):
        print(f"  ok       {os.path.relpath(dest, ROOT)}")
        return
    part = dest + ".part"
    with urllib.request.urlopen(request(url), timeout=60) as r, open(part, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  fetching {os.path.relpath(dest, ROOT)}  {done * 100 // total:3d}%  "
                      f"{done / 1e6:,.0f}/{total / 1e6:,.0f} MB", end="", flush=True)
    print()
    os.replace(part, dest)


def fetch(name):
    spec = MODELS[name]
    folder = os.path.join(ROOT, spec["folder"])
    os.makedirs(folder, exist_ok=True)
    print(f"{name}: {spec['about']}  <- huggingface.co/{spec['repo']}")
    for f in spec["files"]:
        url = f"https://huggingface.co/{spec['repo']}/resolve/main/{f}"
        try:
            download(url, os.path.join(folder, f))
        except urllib.error.HTTPError as e:
            hint = " (set HF_TOKEN and accept the model's license on Hugging Face)" if e.code in (401, 403) else ""
            sys.exit(f"  failed   {f}: HTTP {e.code}{hint}")


def main():
    names = sys.argv[1:]
    if not names:
        print(__doc__.strip().splitlines()[0])
        for n, s in MODELS.items():
            print(f"  {n:22} {s['about']}")
        return
    if names == ["all"]:
        names = list(MODELS)
    for n in names:
        if n not in MODELS:
            sys.exit(f"unknown model '{n}'; choose from: {', '.join(MODELS)}")
        fetch(n)


if __name__ == "__main__":
    main()
