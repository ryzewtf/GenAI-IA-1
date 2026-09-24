"""Generate `notebooks/vllm_campaign_collect.ipynb` — the real per-model collection run.

P3 proved the loop on a tiny local corpus. This notebook runs the SAME `src.capture.vllm_collect`
against the renumbered corpus (Kaggle Dataset `moe-corpus-v2`, `mixed-v2.jsonl` — doc_id = write-order
line index so hidden_index ascends across shards, T2.3) and uploads shards to `Ryze242005/vllm-traces-v2`.
The old `Ryze242005/vllm-traces` (broken mixed-v1 shards) and the llama.cpp traces
(`Ryze242005/moe-traces`) are LEFT UNTOUCHED (D3: no deletions). The vLLM run hash is 1fd87056….

ONE model per session (set MODEL_KEY). The ledger is resumable — a re-run skips shards already
uploaded+verified. Kaggle: GPU T4 ×2, Internet On, and ATTACH the corpus dataset (Add Input →
`ryzewtf/moe-corpus`). HF write token in Kaggle Secrets as HF_TOKEN.

Tier-1 order (arch + harness proven): olmoe-0125 → olmoe-0125-instruct → olmoe-0924 → qwen3-30b-a3b.
Tier 2 (deepseek-v2-lite, gpt-oss-20b) needs the quant / router-bias work first. Gemma-4 is NOT
collected here — it stays on the llama.cpp engine (decision D1).

Do NOT cram >1–2 models into one clock-hour: HF caps dataset commits at 128/hour and the uploader
makes ~21 commits/model.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "vllm_campaign_collect.ipynb"
VLLM_VERSION = "0.10.2"
TRACE_REPO = "Ryze242005/vllm-traces-v2"  # v2: mixed-v2 corpus; old vllm-traces (broken mixed-v1) left as-is


HEADER_MD = f"""# vLLM trace collection — campaign run (one model per session)

Runs `src.capture.vllm_collect` against the canonical corpus and uploads to the **new** vLLM traces
repo `{TRACE_REPO}` (the old llama.cpp traces at `Ryze242005/moe-traces` are left untouched — D3).

**Before running:**
1. GPU **T4 ×2**, **Internet On**.
2. **Add Input → the `mixed-v2` corpus dataset** (doc_id renumbered to write order so traces are
   readable across shards — T2.3; see scripts/renumber_corpus.py). Do NOT let anything re-fetch it —
   HF streaming is unpinned and would draw different documents, breaking T4.3 byte-identity.
3. Kaggle **Secrets**: `HF_TOKEN` (write scope) and `GITHUB_TOKEN`.
4. Set **`MODEL_KEY`** in the collect cell. One model per session.

Run top-to-bottom once; do **not** restart the kernel (Kaggle restart reverts the pip installs).
The ledger is resumable — re-running continues where an interrupted session stopped. Paste back the
`… collected …` summary line.
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
# Cell 4 — locate the canonical corpus (mounted dataset) and export HF_TOKEN.
# ============================================================================
import os
from pathlib import Path

# mixed-v2: doc_id renumbered to write-order line index so hidden_index (doc_id*n_ctx+pos) ascends
# across shards (T2.3) — mixed-v1 numbered doc_id in fetch order and produced unreadable traces.
# The corpus MUST be the mounted Kaggle dataset — never re-fetched (T4.3 byte-identity).
CORPUS = Path("/kaggle/input/moe-corpus-v2/mixed-v2.jsonl")
CORPUS_NAME = "mixed-v2"
assert CORPUS.exists(), (
    f"{CORPUS} not found — Add Input -> the mixed-v2 dataset (Internet On won't fetch it for you)")
print("corpus:", CORPUS, f"({CORPUS.stat().st_size/1e6:.1f} MB)")

# Guard: mixed-v2 MUST have doc_id == line index (write-order). Mounting mixed-v1 (fetch-order) or
# any non-renumbered file here would produce traces no reader can open — and we would not find out
# until after the paid GPU session. This check is a few ms and dependency-free.
import json as _json
with CORPUS.open(encoding="utf-8") as _fh:
    for _i, _line in enumerate(_fh):
        if not _line.strip():
            continue
        _did = _json.loads(_line)["doc_id"]
        if _did != _i:
            raise SystemExit(
                f"{CORPUS} is NOT renumbered: line {_i} has doc_id={_did} (expected {_i}). "
                "Mount the mixed-v2 dataset (scripts/renumber_corpus.py), not mixed-v1.")
print(f"corpus is renumbered (doc_id == line index) for {_i + 1} docs — mixed-v2 OK")

# HF write token from Kaggle Secrets -> env for HFBackend. NEVER inline the token.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("WARNING: no HF_TOKEN secret -", e, "(the upload will fail without it)")
'''

COLLECT_SRC = f'''# ============================================================================
# Cell 5 — collect ONE model against the full corpus and upload to the new vLLM repo.
# ============================================================================
# Tier-1 order: olmoe-0125 -> olmoe-0125-instruct -> olmoe-0924 -> qwen3-30b-a3b.
# Tier-2 (same notebook, by MODEL_KEY): gpt-oss-20b (run notebooks/gptoss_probe FIRST — MXFP4-on-T4
# is unproven) and deepseek-v2-lite (run notebooks/quantize_deepseek FIRST to build the W4A16 repo).
MODEL_KEY = "olmoe-0125"     # <<< set this per session
TRACE_REPO = "{TRACE_REPO}"  # new repo; old llama.cpp traces untouched (D3)

import subprocess, sys

SCRATCH = "/tmp/vscratch"    # ledger + per-shard scratch (deleted after each verified upload)

argv = [
    sys.executable, "-m", "src.capture.vllm_collect",
    "--model", MODEL_KEY, "--corpus", str(CORPUS), "--corpus-name", CORPUS_NAME,
    "--backend", "hf", "--repo-id", TRACE_REPO, "--scratch", SCRATCH,
    "--public",  # vllm-traces-v2 is created PUBLIC (8.7TB public quota, not the 100GB private one)
    # To resume an interrupted model, add:  "--shards", "10-20"
]
print("$", " ".join(argv), flush=True)
rc = subprocess.run(argv, text=True).returncode
print(f"\\n== collector exit {{rc}}", flush=True)
print(f"{{MODEL_KEY}}:", "OK — shards on HF" if rc == 0 else "FAILED (see above)")
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
            _cell("code", COLLECT_SRC),
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
