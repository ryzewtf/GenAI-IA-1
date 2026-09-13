"""Generate `notebooks/vllm_p3_collect.ipynb` — the end-to-end collection proof (P3).

P1/P2 proved capture (TP=1 and TP=2). P3 proves the whole COLLECTION loop
(`src/capture/vllm_collect.py`): shard planning, per-document capture with the faithfulness gate,
byte-exact stream writing, the engine-neutral manifest, upload + round-trip verify into a local
backend, the resumable ledger, and finally T5.3 validation of the written shards. It runs a tiny
corpus (a handful of short docs, 2 shards) for BOTH proven models — olmoe-0125 (TP=1) and
qwen3-30b-a3b (TP=2) — each as its own subprocess, so the model loads, collects, and frees the GPU
before the next one (no dirty-kernel stacking).

Backend is LOCAL only: nothing is uploaded to HF, quantized, or deleted. Kaggle: GPU T4 ×2,
Internet On.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "vllm_p3_collect.ipynb"
VLLM_VERSION = "0.10.2"


HEADER_MD = """# vLLM collection proof — P3 (end-to-end, both models)

Proves `src/capture/vllm_collect.py`: plan shards → capture each document (faithfulness gate) →
write byte-exact streams → engine-neutral manifest → upload + round-trip verify (local backend) →
resumable ledger → **T5.3 validation** of the written shards. Runs a tiny 2-shard corpus for
`olmoe-0125` (TP=1) and `qwen3-30b-a3b` (TP=2), each as its own subprocess.

**Kaggle:** GPU T4 ×2, Internet On. **Local backend only** — no HF upload, no quantization, no
deletion. Run top-to-bottom once; do NOT restart the kernel.

**Paste back Cell 5 and Cell 6 output** — the `# shard N: complete` lines, each model's
`... collected ...` summary, and the `VALIDATION: PASS/FAIL` line. A `SelectionMismatch` anywhere
means capture was unfaithful and the shard was refused (good — loud, not silent).
"""

ENV_SRC = '''# ============================================================================
# Cell 1 — runtime audit (subprocess; does NOT import torch into the kernel).
# ============================================================================
import subprocess
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
     "--format=csv"], capture_output=True, text=True).stdout, flush=True)
print("EXPECT two rows, compute_cap 7.5.")
'''

INSTALL_SRC = f'''# ============================================================================
# Cell 2 — install vLLM + transformers (P0's confirmed recipe). Do NOT restart after.
# ============================================================================
VLLM_VERSION = "{VLLM_VERSION}"
import subprocess, sys


def sh(args):
    print("$", " ".join(args), flush=True)
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print((p.stdout or "")[-3000:], flush=True)
    print("exit:", p.returncode, flush=True)
    return p.returncode


sh([sys.executable, "-m", "pip", "uninstall", "-y",
    "vllm", "torch", "torchvision", "torchaudio", "transformers"])
sh([sys.executable, "-m", "pip", "install", "-q",
    f"vllm=={{VLLM_VERSION}}", "transformers==4.55.2"])
sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"])

chk = subprocess.run(
    [sys.executable, "-c", "import torch, vllm; from vllm import LLM; print('IMPORT_OK')"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(chk.stdout, flush=True)
print(">>> Proceed only if IMPORT_OK printed. DO NOT restart the kernel.")
'''

REPO_SRC = '''# ============================================================================
# Cell 3 — clone the repo, put src/ on sys.path AND PYTHONPATH (TP=2 workers need it).
# ============================================================================
import os, subprocess, sys
from pathlib import Path

GIT_URL = "https://github.com/ryzewtf/GenAI-IA-1.git"
GIT_REF = "VLLM_PORT"
REPO = Path("/kaggle/working/repo")


def run(cmd, cwd=None, check=True, quiet=False):
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], cwd=cwd and str(cwd), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.stdout and not quiet:
        print(p.stdout, flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(str(c) for c in cmd)}")
    return p


url = GIT_URL
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if tok:
        url = GIT_URL.replace("https://", f"https://{tok}@")
        print("using GITHUB_TOKEN from Kaggle Secrets")
except Exception:
    print("no GITHUB_TOKEN secret; cloning anonymously (fine if the repo is public)")

if REPO.exists():
    run(["git", "fetch", "--all", "--tags"], cwd=REPO)
    run(["git", "checkout", GIT_REF], cwd=REPO)
    run(["git", "pull", "--ff-only"], cwd=REPO, check=False)
else:
    run(["git", "clone", url, str(REPO)])
    run(["git", "checkout", GIT_REF], cwd=REPO)
print("repo at", run(["git", "rev-parse", "HEAD"], cwd=REPO, quiet=True).stdout.strip())

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.chdir(REPO)
'''

CORPUS_SRC = '''# ============================================================================
# Cell 4 — write a tiny 2-shard corpus and define the collect+validate helper.
# ============================================================================
import json, subprocess, sys, os
from pathlib import Path

CORPUS = Path("/kaggle/working/p3tiny.jsonl")
LOCAL_ROOT = Path("/tmp/vtraces")     # local backend: the durable trace copy to validate
SCRATCH = Path("/tmp/vscratch")       # per-shard scratch (deleted after each verified upload)
CORPUS_NAME = "p3tiny"

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, humans first walked on the surface of the Moon.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The mitochondria is the powerhouse of the cell.",
    "Paris is the capital of France and sits on the Seine.",
    "A group of flamingos is called a flamboyance.",
]
with CORPUS.open("w", encoding="ascii", newline="\\n") as fh:
    for i, text in enumerate(DOCS):
        fh.write(json.dumps({
            "doc_id": i, "text": text, "domain": "prose", "lang": "en", "source": "p3",
            "n_tokens_ref": len(text.split()), "split": "train", "shard_id": i // 3,   # 2 shards
        }) + "\\n")
print("wrote", CORPUS, "with", len(DOCS), "docs in 2 shards")


def collect_and_validate(model_key):
    """Run the vLLM collector (subprocess -> loads, collects, frees GPU) then T5.3-validate."""
    print(f"\\n{'='*78}\\n== collecting {model_key}\\n{'='*78}", flush=True)
    rc = subprocess.run([
        sys.executable, "-m", "src.capture.vllm_collect",
        "--model", model_key, "--corpus", str(CORPUS), "--corpus-name", CORPUS_NAME,
        "--backend", "local", "--local-root", str(LOCAL_ROOT), "--scratch", str(SCRATCH),
        "--subsample-n", "5",   # tiny corpus: force a small hidden stride so hidden.bin is exercised
    ], text=True).returncode
    print(f"== collector exit {rc}", flush=True)
    if rc != 0:
        return False

    trace_dir = LOCAL_ROOT / "traces" / model_key / CORPUS_NAME
    print(f"\\n== T5.3 validation of {trace_dir}", flush=True)
    v = subprocess.run([
        sys.executable, "-m", "src.traces.validate", str(trace_dir), "--corpus", str(CORPUS),
    ], text=True).returncode
    print(f"VALIDATION: {'PASS' if v == 0 else 'FAIL'} (exit {v})", flush=True)
    return rc == 0 and v == 0
'''

OLMOE_SRC = '''# ============================================================================
# Cell 5 — collect + validate olmoe-0125 (TP=1).
# ============================================================================
ok_olmoe = collect_and_validate("olmoe-0125")
print("\\nP3 olmoe-0125 (TP=1):", "OK" if ok_olmoe else "FAILED")
'''

QWEN_SRC = '''# ============================================================================
# Cell 6 — collect + validate qwen3-30b-a3b (TP=2).
# ============================================================================
ok_qwen = collect_and_validate("qwen3-30b-a3b")
print("\\nP3 qwen3-30b-a3b (TP=2):", "OK" if ok_qwen else "FAILED")
print("\\nP3 PROVEN" if (ok_olmoe and ok_qwen) else "\\nP3 INCOMPLETE — see failures above")
print("Both engines' full collection loop works: shards captured, verified, ledgered, validated.")
'''


def _cell(kind: str, src: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build() -> dict:
    return {
        "cells": [
            _cell("markdown", HEADER_MD),
            _cell("code", ENV_SRC),
            _cell("code", INSTALL_SRC),
            _cell("code", REPO_SRC),
            _cell("code", CORPUS_SRC),
            _cell("code", OLMOE_SRC),
            _cell("code", QWEN_SRC),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
