"""Generate `notebooks/vllm_p2_tp2_capture.ipynb` — the P2 capture proof (TP=2, Qwen3-30B).

Why this exists
---------------
P1 (notebooks/vllm_p1_capture.ipynb) proved the three router streams can be captured from a vLLM
forward pass on ONE card, with topk.bin matching vLLM's own selection. P2 proves the SAME on the
hard path: at tensor_parallel_size=2, vLLM runs each rank in its own PROCESS, so a RouterCapture
built in the driver reaches nothing. This notebook injects the capture INTO the workers via
`Executor.collective_rpc` (which cloudpickles a callable to every worker), runs the prompts, and
drains rank 0 — whose capture is complete because the router (`.mlp.gate`) is a ReplicatedLinear,
so router_logits and select_experts' topk_ids are the FULL, global selection on every rank.

What it proves, in order
------------------------
1. The P0 recipe loads Qwen3-30B-A3B-GPTQ-Int4 at TP=2 (moe_wna16 INT4 MoE on Turing).
2. collective_rpc reaches the workers: worker_install_capture discovers 48 `.mlp.gate` +
   `.mlp.experts` modules per rank and registers the hooks + the class-level select_experts patch
   IN the worker process. The per-rank acks are printed (the T1.4 analogue, confirmed on the box).
3. Draining rank 0 after each prefill yields all three streams; recomputed top-8 equals vLLM's
   selection on every token (the faithfulness gate — a SelectionMismatch halts). This is also what
   catches the vllm-ascend #15451 failure mode: if the captured IDs were TP-LOCAL rather than
   global expert IDs, the set comparison against a recompute over all 128 experts would fail here.
4. The staged buffers match src.traces.format's size arithmetic exactly.

This notebook does NOT upload, quantize, or delete anything. Kaggle: Accelerator GPU T4 ×2 (BOTH
cards used this time), Internet On. Attach the GPTQ weights as a dataset to skip the ~150s download.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "vllm_p2_tp2_capture.ipynb"

VLLM_VERSION = "0.10.2"   # P0/P1-confirmed: loads INT4 MoE on T4, moe_wna16 loader fixed.


HEADER_MD = """# vLLM capture proof — P2 (TP=2, Qwen3-30B)

**Question:** at `tensor_parallel_size=2`, where each rank runs in its own PROCESS, can we still
capture the three router streams with **topk.bin matching vLLM's own selection**? Driver-side
Python hooks can't cross the worker boundary, so we inject the capture into the workers via
`collective_rpc` and drain rank 0 (the router is replicated, so its capture is the full selection).

**Kaggle Settings:** Accelerator **GPU T4 ×2** (BOTH cards used now), Internet **On**. Attaching the
`Qwen/Qwen3-30B-A3B-GPTQ-Int4` weights as a dataset skips the ~150s download.

**Run top-to-bottom ONCE, do NOT restart the kernel.** Cell 3 clones the repo AND puts it on
`PYTHONPATH` so the spawned workers can import `src`. Cell 4 loads the model at TP=2 and installs
the capture on every worker (printing each rank's discovered module count). Cell 5 drains rank 0
per document and runs the gate.

**Paste back Cell 4's per-rank install acks and Cell 5's `MATCH:`/`SIZES:` lines** (or any
traceback). Watch especially for a `SelectionMismatch` — that would mean the captured expert IDs
are TP-local, not global (vllm-ascend #15451), and we'd fix the capture point before trusting it.
"""

ENV_SRC = '''# ============================================================================
# Cell 1 — runtime audit (subprocess; does NOT import torch into the kernel).
# ============================================================================
import subprocess, sys
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
     "--format=csv"], capture_output=True, text=True).stdout, flush=True)
print("EXPECT two rows, compute_cap 7.5. BOTH cards are used at TP=2.")
'''

INSTALL_SRC = f'''# ============================================================================
# Cell 2 — install vLLM + transformers (P0's confirmed recipe). Do NOT restart after.
# ============================================================================
VLLM_VERSION = "{VLLM_VERSION}"   # P0-confirmed: loads INT4 MoE on T4, moe_wna16 loader fixed.
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

print("\\n--- verifying import in a clean subprocess ---", flush=True)
chk = subprocess.run(
    [sys.executable, "-c",
     "import torch, vllm; print('torch', torch.__version__); "
     "print('vllm', vllm.__version__); from vllm import LLM; print('IMPORT_OK')"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(chk.stdout, flush=True)
print(">>> Proceed to Cell 3 only if IMPORT_OK printed. DO NOT restart the kernel.")
'''

REPO_SRC = '''# ============================================================================
# Cell 3 — clone the repo, put src/ on sys.path AND PYTHONPATH (workers need it too).
# ============================================================================
# At TP=2 vLLM SPAWNS worker processes. collective_rpc resolves our capture functions by importing
# src.capture.vllm_trace in each worker, so the repo must be on PYTHONPATH *before* the LLM is
# built (Cell 4) -- a sys.path insertion in this process does not reach spawned children.
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

HEAD = run(["git", "rev-parse", "HEAD"], cwd=REPO, quiet=True).stdout.strip()
print(f"\\nrepo at {HEAD}  (ref {GIT_REF})")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# The critical extra step for TP>1: make src importable in SPAWNED worker processes.
os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")
import src.capture.vllm_trace as _vt  # noqa: F401
import src.traces.format as _fmt       # noqa: F401
print("src importable from", REPO, "and exported on PYTHONPATH for workers")
'''

LOAD_SRC = '''# ============================================================================
# Cell 4 — load Qwen3-30B at TP=2 and INSTALL capture on every worker via collective_rpc.
# ============================================================================
import os
os.environ["VLLM_USE_V1"] = "0"                    # Turing needs the V0 engine (P0-confirmed).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Free any GPU left resident by a prior failed load in this kernel (see P1 notebook for why).
import gc, sys, torch
for _attr in ("last_traceback", "last_value", "last_type"):
    try:
        setattr(sys, _attr, None)
    except Exception:
        pass
try:
    del llm                                        # noqa: F821
except NameError:
    pass
gc.collect(); gc.collect()
torch.cuda.empty_cache()

MODEL_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"          # 128 experts, top-8, 48 MoE layers, GPTQ-Int4
N_MOE_LAYERS = 48                                   # models.yaml: qwen3-30b-a3b

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,                         # BOTH T4s; capture is injected into workers
    enforce_eager=True,                             # required on Turing; also the hook precondition
    dtype="float16",
    max_model_len=2048,
    max_num_seqs=1,
    gpu_memory_utilization=0.90,                    # P0: GPTQ weights ~7.8 GiB/GPU -> headroom
)

import src.capture.vllm_trace as vt

# The executor is the handle to all workers. collective_rpc cloudpickles our function to each rank
# and calls it with the worker as the first arg. worker_install_capture discovers the router/experts
# modules ON that worker and registers the capture there. Returns one ack per rank.
executor = llm.llm_engine.model_executor
acks = executor.collective_rpc(
    vt.worker_install_capture,
    kwargs=dict(router_suffix=".mlp.gate", experts_suffix=".mlp.experts",
                n_moe_layers=N_MOE_LAYERS),
)
print("per-rank install acks (expect n_gates ==", N_MOE_LAYERS, "on every rank):")
for a in acks:
    print("  ", a)
'''

CAPTURE_SRC = '''# ============================================================================
# Cell 5 — drain rank 0 per document, run the faithfulness gate, check byte sizes.
# ============================================================================
import numpy as np
from src.capture.vllm_trace import DocumentTrace, gating_from_config
from src.traces.format import TraceSpec, expected_file_sizes

# qwen3-30b-a3b card (models.yaml): 128 experts, top-8, hidden 2048, logit_tensor_used=ffn_moe_probs
SPEC = TraceSpec(n_moe_layers=N_MOE_LAYERS, n_experts=128, top_k=8, hidden_dim=2048)
GATING = gating_from_config({"logit_tensor_used": "ffn_moe_probs", "has_router_bias": False})

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, humans first walked on the surface of the Moon.",
    "Photosynthesis converts light energy into chemical energy in plants.",
]
tok = llm.get_tokenizer()

matches = 0
try:
    for doc_id, prompt in enumerate(PROMPTS):
        executor.collective_rpc(vt.worker_reset_capture)     # clear every worker's per-doc buffers
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        n = len(ids)
        llm.generate([{"prompt_token_ids": ids}],
                     SamplingParams(max_tokens=1, temperature=0.0))

        # Drain ALL ranks; rank 0's capture is complete (router is replicated). collective_rpc
        # returns results in rank order, so index 0 is rank 0.
        drained = executor.collective_rpc(vt.worker_drain_capture)
        r0 = drained[0]
        if r0 is None:
            raise RuntimeError("rank 0 returned no capture -- worker_install_capture did not run?")
        for stream, got in (("gate output", r0["outputs"]), ("gate input", r0["inputs"]),
                            ("select_experts topk", r0["topk_ids"])):
            if sorted(got) != list(range(N_MOE_LAYERS)):
                raise RuntimeError(
                    f"{stream} fired for layers {sorted(got)}, expected 0..{N_MOE_LAYERS-1}")

        doc = DocumentTrace(SPEC, n_tokens=n, capture_mask=[True] * n, gating=GATING)
        for L in range(N_MOE_LAYERS):
            logits = r0["outputs"][L][-n:]
            router_input = r0["inputs"][L][-n:]
            vllm_topk = r0["topk_ids"][L][-n:]
            doc.put_layer(L, logits=logits, vllm_topk=vllm_topk, router_input=router_input)
        bufs = doc.to_buffers(token_ids=ids, doc_id=doc_id, global_token_base=doc_id * 4096)
        want = expected_file_sizes(SPEC, n, doc.n_captured)
        for name, blob in bufs.items():
            assert len(blob) == want[name], f"{name}: {len(blob)} != {want[name]}"
        matches += 1
        print(f"doc {doc_id}: {n} tokens, selection gate PASSED (recomputed == vLLM), sizes exact")
finally:
    executor.collective_rpc(vt.worker_remove_capture)

print(f"\\nMATCH: {matches}/{len(PROMPTS)} documents -- recomputed top-8 == vLLM select_experts")
print("SIZES: OK" if matches == len(PROMPTS) else "SIZES: FAILED")
print("\\nP2 (TP=2) PROVEN: capture injected into workers, rank 0 drain is the full global selection,")
print("topk.bin matches vLLM, bytes match the frozen format. The two big models are collectable.")
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
            _cell("code", LOAD_SRC),
            _cell("code", CAPTURE_SRC),
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
