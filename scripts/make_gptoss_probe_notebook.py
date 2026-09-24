"""Generate ``notebooks/gptoss_probe.ipynb`` — the cheap MXFP4-on-T4 go/no-go for gpt-oss-20b.

gpt-oss ships native MXFP4 on the MoE weights. transformers gates MXFP4 at compute capability
>= (7,5); a T4 is exactly (7,5), and vLLM's own recipe lists only H100/H200/etc. — so whether
gpt-oss loads and captures on a Kaggle T4 is an OPEN empirical question, not a known-good path. This
notebook answers it for a few cents of GPU: load the model once (MXFP4, TP=1, enforce_eager) and run
ONE document through VLLMCaptureEngine, confirming (a) it loads on sm_75, (b) discover finds 24
`.mlp.router` modules, and (c) the SelectionMismatch gate passes (recompute top-k over the captured
router output == vLLM's select_experts — the proof the router bias is handled right). No upload.

If this passes, collect gpt-oss via notebooks/vllm_campaign_collect.ipynb (MODEL_KEY="gpt-oss-20b").
If MXFP4 will not run on T4, fall back to the llama.cpp GGUF panel (decision point — tell the user).

Kaggle: GPU **T4 x2**, **Internet On**. No corpus mount needed (the probe builds a 1-doc corpus).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "gptoss_probe.ipynb"
VLLM_VERSION = "0.10.2"

HEADER_MD = """# gpt-oss-20b — MXFP4-on-T4 probe (go/no-go, no upload)

Loads gpt-oss-20b (native MXFP4) on a Kaggle T4 and captures ONE document. Success = model loads on
sm_75, `.mlp.router` x24 discovered, and the SelectionMismatch gate passes. **Before running:** GPU
**T4 x2**, **Internet On**. Run top-to-bottom; do not restart the kernel. Paste back the final
`PROBE OK` / failure line.
"""

ENV_SRC = '''# Cell 1 — runtime audit.
import subprocess
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
     "--format=csv"], capture_output=True, text=True).stdout, flush=True)
print("EXPECT two rows, compute_cap 7.5.")
'''

INSTALL_SRC = f'''# Cell 2 — install vLLM + the MXFP4 kernel deps (triton>=3.4, kernels). No restart after.
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
# MXFP4 needs a recent triton + the kernels package; without them transformers dequantizes to bf16
# (~40 GB) and OOMs a T4. Install and record whether they are present.
sh([sys.executable, "-m", "pip", "install", "-q", "-U", "triton>=3.4", "kernels"])
sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"])

chk = subprocess.run(
    [sys.executable, "-c",
     "import torch, vllm, triton; "
     "print('IMPORT_OK vllm', vllm.__version__, 'triton', triton.__version__)"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(chk.stdout, flush=True)
print(">>> Proceed only if IMPORT_OK printed. DO NOT restart the kernel.")
'''

REPO_SRC = '''# Cell 3 — clone the repo, put src/ on sys.path AND PYTHONPATH.
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

PROBE_SRC = '''# Cell 4 — load gpt-oss once, discover .mlp.router, capture ONE doc, fire the gate.
import numpy as np
from src.runtime.runner import load_model_meta
from src.capture.vllm_collect import VLLMCaptureEngine, spec_and_gating_for
from src.capture.vllm_trace import DocumentTrace, discover_router_and_experts

meta = load_model_meta("configs/models.yaml", "gpt-oss-20b")
vllm = meta["vllm"]
spec, gating = spec_and_gating_for(meta)
print("spec:", (spec.n_moe_layers, spec.n_experts, spec.top_k, spec.hidden_dim),
      "| softmax", gating.softmax, "has_router_bias", gating.has_router_bias)

engine = VLLMCaptureEngine(
    model_id=vllm["model_id"], tensor_parallel_size=1, spec=spec, gating=gating,
    router_suffix=vllm["router_suffix"], experts_suffix=vllm["experts_suffix"],
    max_model_len=2048, gpu_memory_utilization=float(vllm.get("gpu_memory_utilization", 0.90)),
    dtype="float16", seed=0,
)
engine.load()  # <-- the MXFP4-on-T4 moment: fails here if sm_75 cannot load the model
try:
    model = engine.llm.llm_engine.model_executor.driver_worker.model_runner.model
    _, _, gnames, enames = discover_router_and_experts(
        model, router_suffix=vllm["router_suffix"], experts_suffix=vllm["experts_suffix"],
        n_expected=spec.n_moe_layers)
    print(f"discovered {len(gnames)} routers, e.g. {gnames[0]} .. {gnames[-1]}")

    ids = engine.tokenize("The mixture-of-experts router selects a few experts per token.")
    captured = engine.capture_document(ids)
    n = len(ids)
    trace = DocumentTrace(spec, n_tokens=n, capture_mask=[False] * n, gating=gating)
    for L in range(spec.n_moe_layers):
        logits, topk, router_input = captured[L]
        trace.put_layer(L, logits=logits, vllm_topk=topk, router_input=router_input)  # gate fires here
    print(f"SelectionMismatch gate PASSED over {n} tokens x {spec.n_moe_layers} layers")
    print("PROBE OK — gpt-oss loads and captures faithfully on T4; safe to run the campaign notebook")
finally:
    engine.remove()
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
            _cell("code", PROBE_SRC),
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
