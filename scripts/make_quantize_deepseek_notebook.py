"""Generate ``notebooks/quantize_deepseek.ipynb`` — the one-time DeepSeek-V2-Lite W4A16 quant job.

Runs ``scripts/quantize_deepseek.py`` on Kaggle T4x2: self-quantize the routed experts to 4-bit
(W4A16) while keeping the router (.mlp.gate) in fp16, verify no router gate was quantized, and push
the result to ``Ryze242005/DeepSeek-V2-Lite-w4a16-gptq`` (private). After this exists, set
models.yaml deepseek vllm.model_sha256 (collector resolves it) and collect via the campaign notebook.

Kaggle: GPU **T4 x2**, **Internet On**, Add Input -> the mixed-v2 corpus (calibration), HF write
token in Secrets as HF_TOKEN. Run top-to-bottom once; do not restart the kernel.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "quantize_deepseek.ipynb"
REPO_ID = "Ryze242005/DeepSeek-V2-Lite-w4a16-gptq"

HEADER_MD = f"""# DeepSeek-V2-Lite -> W4A16 (GPTQ, router fp16) — one-time quantization

Self-quantizes the routed experts to 4-bit and keeps the router in fp16, then pushes to
`{REPO_ID}` (private). See scripts/quantize_deepseek.py and configs/models.yaml (deepseek vllm block).

**Before running:** GPU **T4 x2**, **Internet On**, **Add Input -> the mixed-v2 corpus dataset**
(calibration), and Kaggle **Secrets**: `HF_TOKEN` (write) + `GITHUB_TOKEN`. One-time; paste back the
`verify OK` + `pushed to ...` lines.
"""

ENV_SRC = '''# Cell 1 — runtime audit (subprocess; does not import torch into the kernel).
import subprocess
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
     "--format=csv"], capture_output=True, text=True).stdout, flush=True)
print("EXPECT two rows, compute_cap 7.5.")
'''

INSTALL_SRC = '''# Cell 2 — install GPTQModel + deps. Do NOT restart the kernel after.
import subprocess, sys


def sh(args):
    print("$", " ".join(args), flush=True)
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print((p.stdout or "")[-3000:], flush=True)
    print("exit:", p.returncode, flush=True)
    return p.returncode


sh([sys.executable, "-m", "pip", "install", "-q", "-U", "gptqmodel", "--no-build-isolation"])
sh([sys.executable, "-m", "pip", "install", "-q", "transformers==4.55.2", "accelerate", "datasets"])
chk = subprocess.run(
    [sys.executable, "-c", "import gptqmodel, torch; print('IMPORT_OK', gptqmodel.__version__)"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(chk.stdout, flush=True)
print(">>> Proceed only if IMPORT_OK printed. DO NOT restart the kernel.")
'''

REPO_SRC = '''# Cell 3 — clone the repo, put it on sys.path + PYTHONPATH.
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
except Exception:
    print("no GITHUB_TOKEN secret; cloning anonymously")

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

CORPUS_SRC = '''# Cell 4 — locate the mixed-v2 corpus (calibration) and export HF_TOKEN.
import os
from pathlib import Path

CORPUS = Path("/kaggle/input/moe-corpus-v2/mixed-v2.jsonl")
assert CORPUS.exists(), f"{CORPUS} not found — Add Input -> the mixed-v2 dataset"
print("calib corpus:", CORPUS, f"({CORPUS.stat().st_size/1e6:.1f} MB)")
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("WARNING: no HF_TOKEN secret -", e, "(the push will fail without it)")
'''

QUANT_SRC = f'''# Cell 5 — quantize (router fp16) + verify + push to {REPO_ID} (private).
import subprocess, sys

argv = [
    sys.executable, "-m", "scripts.quantize_deepseek",
    "--model-id", "deepseek-ai/DeepSeek-V2-Lite",
    "--out", "/tmp/deepseek-w4a16",
    "--repo-id", "{REPO_ID}",
    "--calib-corpus", str(CORPUS),
    "--calib-samples", "256",
    "--push",
]
print("$", " ".join(argv), flush=True)
rc = subprocess.run(argv, text=True).returncode
print(f"\\n== quantize exit {{rc}}", flush=True)
print("OK — quantized model pushed" if rc == 0 else "FAILED (see above)")
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
            _cell("code", QUANT_SRC),
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
