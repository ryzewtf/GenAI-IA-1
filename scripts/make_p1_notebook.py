"""Generate `notebooks/vllm_p1_capture.ipynb` — the P1 capture proof (TP=1, OLMoE).

Why this exists
---------------
P0 (notebooks/vllm_probe.ipynb) proved an INT4 MoE loads and generates on a Kaggle T4. P1 proves
the CAPTURE works: that we can pull the three router streams (§1.6) out of a vLLM forward pass and
write them in the frozen trace format, with topk.bin matching vLLM's own expert selection.

Per the user's "smallest first" rule, this runs the EASY path only: OLMoE-1B-7B-0125's fp16 weights
(~12.9 GiB) fit on ONE T4, so tensor_parallel_size=1 and the router modules live in the driver
process where Python forward-hooks can reach them. The two TP=2 models (Qwen3-30B, Gemma-4) need vLLM's in-worker
capturer and are a separate step; this notebook validates the streams themselves before that.

What it proves, in order
------------------------
1. The env recipe from P0 still loads OLMoE at TP=1 (fp16 — OLMoE is not quantized; the 7B fp16
   weights are ~12.9 GiB, which fills most of one T4, so util is pushed high and ctx kept short).
2. The vLLM module tree actually contains a router per MoE layer at the path models.yaml predicts
   (model.layers.{i}.mlp.gate). This is PRINTED, not assumed — vLLM's internal names differ from
   HuggingFace's and must be confirmed on the box (the analogue of llama.cpp's T1.4 node scan).
3. Hooks capture router input + output; recomputed top-8 equals vLLM's selection on every token
   (100%% match, or SelectionMismatch halts).
4. The staged buffers match src.traces.format's size arithmetic exactly, so the bytes are the same
   layout every downstream reader already parses.

This notebook does NOT upload, quantize, or delete anything. It writes a tiny trace under /tmp and
checks it. Kaggle settings: Accelerator = GPU T4 x2 (only one card used), Internet = On.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "vllm_p1_capture.ipynb"

# Same pins P0 confirmed on 2026-09-12: vllm 0.10.2 (moe_wna16 loader fix), let vLLM pull its own
# torch (2.8.0+cu128 landed), transformers 4.55.2 (no torch.compiler.disable(reason=) skew).
VLLM_VERSION = "0.10.2"


HEADER_MD = """# vLLM capture proof — P1 (TP=1, OLMoE)

**Question:** can we pull the three router streams (topk / logits / router-input, plan §1.6) out
of a vLLM forward pass and write them in the frozen trace format, with **topk.bin matching vLLM's
own selection**?

This is the EASY path: OLMoE fits one T4, so `tensor_parallel_size=1` and Python hooks reach the
router. The big TP=2 models are a later step.

**Kaggle Settings:** Accelerator **GPU T4 ×2** (only one card is used), Internet **On**, no dataset
needed.

**Run top-to-bottom ONCE, do NOT restart the kernel** (a restart reverts the pip installs to the
base image). Cell 1 audits via subprocess without importing torch; Cell 2 installs; Cell 3 clones
the repo for `src/`; Cell 4 loads OLMoE and PRINTS the router module tree; Cell 5 captures a few
docs and runs the gate.

**Paste back Cell 4's module-tree print and Cell 5's `MATCH:`/`SIZES:` lines** (or any traceback).
"""

ENV_SRC = '''# ============================================================================
# Cell 1 — runtime audit (subprocess; does NOT import torch into the kernel).
# ============================================================================
import subprocess, sys
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
     "--format=csv"], capture_output=True, text=True).stdout, flush=True)
print("EXPECT two rows, compute_cap 7.5. Only card 0 is used (TP=1).")
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


# Purge first (any preinstalled version), then let vLLM pull its own matching torch.
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
# Cell 3 — clone the repo at GIT_REF and put src/ on the path (same logic as moe_session.ipynb).
# ============================================================================
# The pipeline convention is: no logic in notebooks. src/capture/vllm_trace.py and
# src/traces/format.py do the work; this cell just clones the repo and makes them importable.
import os, subprocess, sys
from pathlib import Path

GIT_URL = "https://github.com/ryzewtf/GenAI-IA-1.git"
GIT_REF = "VLLM_PORT"                       # the vLLM-port branch (holds src/capture/vllm_trace.py)
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


# A private repo needs a token; a public one must not be handed one. Kaggle Secrets is the only
# place a token belongs -- an inline PAT is committed the moment the notebook is saved.
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
import src.capture.vllm_trace as _vt  # noqa: F401
import src.traces.format as _fmt       # noqa: F401
print("src importable from", REPO)
'''

LOAD_SRC = '''# ============================================================================
# Cell 4 — load OLMoE at TP=1 and PRINT the router module tree (the T1.4 analogue).
# ============================================================================
import os
os.environ["VLLM_USE_V1"] = "0"                    # Turing needs the V0 engine (P0-confirmed).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # curbs fragmentation

# --- free GPU left over from a PRIOR failed load in this kernel -------------------------------
# We do NOT restart the kernel (a Kaggle restart reverts Cell 2's pip installs). But a failed
# LLM(...) leaves ~13 GiB of OLMoE weights resident on card 0, and re-running this cell would then
# OOM on top of them. IPython also pins the dead engine alive via its stored traceback, so clear
# that too, then drop any prior `llm` and empty the CUDA cache. Safe to run even on a clean card.
import gc, sys, torch
for _attr in ("last_traceback", "last_value", "last_type"):
    try:
        setattr(sys, _attr, None)
    except Exception:
        pass
try:
    del llm                                        # noqa: F821 -- may not exist yet
except NameError:
    pass
gc.collect(); gc.collect()
torch.cuda.empty_cache()
_free, _total = torch.cuda.mem_get_info()
print(f"GPU0 free before load: {_free/2**30:.2f} GiB / {_total/2**30:.2f} GiB")
if _free < 13.3 * 2**30:
    print("  WARNING: <13.3 GiB free -- a prior load is still resident. If the load below OOMs,")
    print("  do Kaggle 'Restart & clear cell outputs', then re-run from Cell 2 (reinstall ~2 min).")

MODEL_ID = "allenai/OLMoE-1B-7B-0125"              # 6.9B params -> ~12.9 GiB fp16 on ONE T4.
N_MOE_LAYERS = 16                                   # models.yaml: olmoe-0125

# OLMoE is a 7B model: unquantized fp16 weights are ~12.9 GiB, and a 14.56 GiB T4 has almost no
# room left for KV cache. At util 0.85 the KV budget goes NEGATIVE (weights > budget) and vLLM
# raises "No available memory for the cache blocks". Fix, staying at TP=1 (Python hooks must reach
# the router in-process): give the card almost entirely to vLLM (0.95) and shrink the context to
# what this 3-short-prompt probe actually needs. 0.95*14.56 = 13.83 GiB - 12.9 weights - ~0.5
# activation leaves ~0.4 GiB KV, i.e. a few thousand tokens at 1024 ctx -- ample here.
import torch
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=1,                         # single card -> hooks reach the router
    enforce_eager=True,                             # required on Turing; also needed for hooks
    dtype="float16",
    max_model_len=1024,                             # probe prompts are short; frees KV headroom
    max_num_seqs=1,                                 # one doc at a time -> minimal KV footprint
    gpu_memory_utilization=0.95,                    # 0.85 left the KV cache NEGATIVE for this 7B
)

# Reach the underlying torch nn.Module. On the V0 engine this is the model runner's `model`.
runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = runner.model
print("top-level model type:", type(model).__name__)

# Find the router (gate) module for each MoE layer. models.yaml predicts model.layers.{i}.mlp.gate
# for OLMoE, but vLLM may name/wrap it differently -- so DISCOVER it and print what we found.
named = dict(model.named_modules())
import re
def layer_of(name):
    m = re.search(r"layers\\.(\\d+)\\.", name)
    return int(m.group(1)) if m else -1

# The router (raw logits) is the ReplicatedLinear `.gate`; the selection happens inside the
# FusedMoE `.experts`. models.yaml predicts model.layers.{i}.mlp.gate for OLMoE -- DISCOVER both
# and PRINT them so we confirm the vLLM names on the box before trusting any captured bytes.
gates = sorted([n for n in named if n.endswith(".mlp.gate") and layer_of(n) >= 0], key=layer_of)
experts = sorted([n for n in named if n.endswith(".mlp.experts") and layer_of(n) >= 0],
                 key=layer_of)
print(f"\\nfound {len(gates)} .mlp.gate and {len(experts)} .mlp.experts modules")
for label, names in (("gate", gates[:3]), ("experts", experts[:3])):
    for n in names:
        print(f"   {label:8s} {n} -> {type(named[n]).__name__}")

assert len(gates) == N_MOE_LAYERS, f"expected {N_MOE_LAYERS} gates, found {len(gates)}: {gates}"
assert len(experts) == N_MOE_LAYERS, f"expected {N_MOE_LAYERS} experts, found {len(experts)}"
ROUTER_MODULES = [named[n] for n in gates]
EXPERTS_MODULES = [named[n] for n in experts]
# Sanity: the experts module should expose select_experts (vLLM's kernel-side selection).
have_sel = hasattr(EXPERTS_MODULES[0], "select_experts")
print(f"\\nexperts[0] has select_experts: {have_sel}  ({type(EXPERTS_MODULES[0]).__name__})")
print("If False, paste the experts module type -- the topk capture point differs by vLLM version.")
'''

CAPTURE_SRC = '''# ============================================================================
# Cell 5 — capture a few docs, run the faithfulness gate, check byte sizes.
# ============================================================================
# This exercises the WHOLE harness on real vLLM tensors. The selection gate (recomputed top-k ==
# vLLM's own) is the thing that proves capture is faithful; a mismatch raises SelectionMismatch.
import numpy as np, torch
from src.capture.vllm_trace import DocumentTrace, RouterCapture, gating_from_config
from src.traces.format import TraceSpec, expected_file_sizes

# OLMoE-0125 card (models.yaml). logit_tensor_used=ffn_moe_probs -> softmax; no router bias.
SPEC = TraceSpec(n_moe_layers=N_MOE_LAYERS, n_experts=64, top_k=8, hidden_dim=2048)
GATING = gating_from_config({"logit_tensor_used": "ffn_moe_probs", "has_router_bias": False})

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, humans first walked on the surface of the Moon.",
    "Photosynthesis converts light energy into chemical energy in plants.",
]
tok = llm.get_tokenizer()

# Capture BOTH the router (logits + input) AND vLLM's own select_experts topk_ids. The topk_ids
# come from vLLM's kernel path, INDEPENDENTLY of our recomputation -- that independence is what
# makes the gate inside DocumentTrace.put_layer a real faithfulness test (the T1.4 analogue), not
# a tautology. If they disagree on any token, put_layer raises SelectionMismatch.
cap = RouterCapture(ROUTER_MODULES, experts_modules=EXPERTS_MODULES)
cap.register()
matches = 0
try:
    for doc_id, prompt in enumerate(PROMPTS):
        cap.reset()
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        n = len(ids)
        # Prefill only, greedy; we read the hooks, not the generated text.
        llm.generate([{"prompt_token_ids": ids}],
                     SamplingParams(max_tokens=1, temperature=0.0))

        for stream, got in (("gate output", cap.outputs), ("gate input", cap.inputs),
                            ("select_experts topk", cap.topk_ids)):
            if sorted(got) != list(range(N_MOE_LAYERS)):
                raise RuntimeError(
                    f"{stream} fired for layers {sorted(got)}, expected 0..{N_MOE_LAYERS-1}")

        doc = DocumentTrace(SPEC, n_tokens=n,
                            capture_mask=[True] * n,   # capture all tokens in this tiny probe
                            gating=GATING)
        for L in range(N_MOE_LAYERS):
            # prefill packs this prompt's tokens contiguously; take the last n rows of each stream.
            logits = cap.outputs[L][-n:]
            router_input = cap.inputs[L][-n:]
            vllm_topk = cap.topk_ids[L][-n:]           # vLLM's OWN selection -> the gate's other side
            doc.put_layer(L, logits=logits, vllm_topk=vllm_topk, router_input=router_input)
        bufs = doc.to_buffers(token_ids=ids, doc_id=doc_id, global_token_base=doc_id * 4096)
        want = expected_file_sizes(SPEC, n, doc.n_captured)
        for name, blob in bufs.items():
            assert len(blob) == want[name], f"{name}: {len(blob)} != {want[name]}"
        matches += 1
        print(f"doc {doc_id}: {n} tokens, selection gate PASSED (recomputed == vLLM), sizes exact")
finally:
    cap.remove()

print(f"\\nMATCH: {matches}/{len(PROMPTS)} documents -- recomputed top-8 == vLLM select_experts")
print("SIZES: OK" if matches == len(PROMPTS) else "SIZES: FAILED")
print("\\nP1 (TP=1) PROVEN: three streams captured, topk.bin matches vLLM's kernel selection, bytes")
print("match the frozen format. Next: the TP=2 in-worker capturer for Qwen3-30B / Gemma-4.")
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
