"""Generate `notebooks/vllm_probe.ipynb` — the P0 vLLM-on-Turing viability probe.

Why this exists (and why it is separate from moe_session.ipynb)
--------------------------------------------------------------
The panel is being ported off llama.cpp onto vLLM to kill the 30-min per-session build. That port
has ONE make-or-break unknown that no amount of local reasoning settles: **does an INT4 MoE model
actually run on a Kaggle T4 (sm_75) under vLLM at all?** Every INT4 path we found targets sm_80+:

  * AWQ W4A16 native kernels need sm_80 (vllm #1063).
  * llm-compressor AWQ/GPTQ emit compressed-tensors -> Marlin; the Marlin *MoE* W4A16 kernel fails
    even on A100/sm_80 and needs Ada/Hopper (vllm #35922).
  * vLLM auto-converts classic GPTQ to gptq_marlin at load (vllm #23631); the plain gptq_gemm
    kernel is flagged buggy/deprecated (vllm #34118).
  * The ONLY kernel that could run INT4 experts on Turing is the `moe_wna16` Triton path, and its
    sm_75 support is unverified.

So before we quantize anything, delete any Kaggle upload, or write a line of capture code, we spend
ONE short session answering exactly that. The target is `Qwen/Qwen3-30B-A3B-GPTQ-Int4`: it IS our
panel's hardest model (Qwen3-30B-A3B, 128 experts), already GPTQ-Int4 quantized by Qwen, public,
16.9 GB, and it needs both T4s (tensor_parallel_size=2). If it loads and emits a token, INT4-MoE on
T4 is viable and we have a recipe to copy. If it dies with `no kernel image` / a Marlin error, the
pivot cannot serve the two big models on T4 and we reconsider — having spent ~15 min and deleted
nothing.

What the probe deliberately does NOT do
---------------------------------------
* No capture. Hooks cannot cross the TP=2 worker-subprocess boundary anyway; capture is P1, using
  vLLM's built-in routed-experts capturer for TP>1 and Python hooks only for TP=1 models.
* No quantization, no uploads, no repo clone, no secrets. The model is public.
* No commitment to a vLLM version beyond a single editable pin — discovering the working version
  is part of what the probe is for.

Kaggle settings for this notebook: Accelerator = GPU T4 x2, Internet = On, no dataset attached.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "vllm_probe.ipynb"


HEADER_MD = """# vLLM-on-Turing viability probe (P0)

**Question this session answers:** does an INT4 MoE model run on a Kaggle **T4 (sm_75)** under vLLM?

**Kaggle Settings — set these before running:**

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 ×2** |
| Internet | **On** |
| Dataset / Model attached | none needed |

**Run order — run top to bottom ONCE, and do NOT restart the kernel.** A kernel restart on Kaggle
reverts the pip installs back to the base image (torch 2.10), which reintroduces the version skew.
Cell 1 deliberately does not import torch, so after Cell 2 swaps in torch 2.7.1 the kernel is still
torch-free and Cell 3's import is the first — it picks up the freshly installed 2.7.1. Just run
Cell 1 → Cell 2 → (wait for `IMPORT_OK`) → Cell 3 → Cell 4, in order, in one fresh session.

**Paste back the FULL output of Cell 2** (the pip resolution + subprocess version lines) **and
Cell 3** — especially Cell 3's `torch:` / `transformers:` version lines and the vLLM load log line
naming the quantization kernel (`gptq_marlin`, `moe_wna16`, `compressed-tensors`, …), which is the
whole point.

**Decision:**
- Load cell prints `OUTPUT: '...'` and `SUCCESS` → INT4-MoE on T4 is viable; the format is
  **GPTQ-Int4** and we build the capture harness on top of it.
- Load cell raises `no kernel image is available` / a Marlin / `AssertionError: Only SiLU` error →
  INT4-MoE on T4 is not viable for this path; stop and report the traceback. Nothing was quantized,
  uploaded, or deleted.

**If the engine fails to *initialise* (not a kernel error, but a crash during engine startup):**
set `USE_V1 = False` in the load cell and re-run all — some Turing setups need vLLM's V0 engine.
"""

ENV_SRC = '''# ============================================================================
# Cell 1 — runtime audit. Confirms this is really a T4 x2 (sm_75) box.
# ============================================================================
# CRITICAL: this cell does NOT `import torch` into the kernel. If it did, the kernel would hold
# Kaggle's BASE torch in memory; Cell 2 then swaps torch on disk, but the kernel keeps the stale
# one and every later `import torch` returns it -- which is exactly the version skew that broke the
# vLLM import before. So the audit runs in a CHILD process and the kernel stays torch-free until
# Cell 3 (which does the first in-kernel import, AFTER Cell 2's install).
import subprocess, sys

print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
     "--format=csv"], capture_output=True, text=True).stdout, flush=True)

chk = subprocess.run([sys.executable, "-c", (
    "import torch;"
    "print('torch', torch.__version__, '| cuda', torch.version.cuda);"
    "print('devices', torch.cuda.device_count());"
    "print('capability', [torch.cuda.get_device_capability(i)"
    " for i in range(torch.cuda.device_count())]);"
    "print('bf16', torch.cuda.is_bf16_supported(), '(expect False on Turing -> use fp16)')"
)], capture_output=True, text=True)
print(chk.stdout, chk.stderr, flush=True)
print("EXPECT capability [(7, 5), (7, 5)]. The torch above is Kaggle's BASE build; Cell 2 replaces")
print("it. This kernel has deliberately NOT imported torch.")
'''

INSTALL_SRC = '''# ============================================================================
# Cell 2 — install vLLM + a consistent torch/transformers, verify import. Do NOT restart after.
# ============================================================================
# vLLM bundles its own torch; installing WITH deps is intentional here (unlike the main pipeline's
# --no-deps rule) because vLLM's compiled kernels must match their torch. This can churn the Kaggle
# image's torch -- fine for a throwaway probe VM, never do it in a collection session.
#
# VLLM_VERSION is a single editable pin. If this version refuses to install or won't init on sm_75,
# change it and re-run. Finding the version that works is part of the probe.
VLLM_VERSION = "0.10.2"   # 0.10.1.1 selects the moe_wna16 Turing kernel correctly but has a loader
# bug (#22961: moe_wna16_weight_loader missing `return_success`); PR #22797 fixes it in 0.10.2.

import subprocess, sys


def sh(args):
    print("$", " ".join(args), flush=True)
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print((p.stdout or "")[-3000:], flush=True)
    print("exit:", p.returncode, flush=True)
    return p.returncode


# The Kaggle image ships bleeding-edge torch + transformers that fight vLLM. Two skews bit earlier,
# both cured by a clean, consistent install:
#   * Frankenstein torch (torch.compiler vs _dynamo mismatch) -> `disable() got unexpected kwarg
#     'reason'`, from piecemeal installs replacing torch only partially.
#   * torchvision ABI skew -> `operator torchvision::nms does not exist`.
#
# 0) PURGE first, regardless of version, so nothing preinstalled is left to create a half-replaced
#    (Frankenstein) torch. Some may already be absent -- pip returns non-zero for "not installed",
#    which is harmless here.
sh([sys.executable, "-m", "pip", "uninstall", "-y",
    "vllm", "torch", "torchvision", "torchaudio", "transformers"])

# 1) Install vLLM and let IT pull its OWN matching torch in one command (0.10.2 wants a different
#    torch than 0.10.1.1 did, so hard-pinning torch would conflict -- let the resolver choose it,
#    which also guarantees an internally consistent torch). transformers 4.55.2 is a Qwen3-MoE-
#    capable family that does not pass `reason` to torch.compiler.disable.
sh([sys.executable, "-m", "pip", "install", "-q",
    f"vllm=={VLLM_VERSION}", "transformers==4.55.2"])

# 2) Drop torchvision/torchaudio: a text LLM needs neither, and their absence makes transformers
#    skip the broken image import entirely (sidesteps the nms skew, whatever torch landed).
sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"])

# Verify `import vllm` in a FRESH interpreter (this kernel may hold a broken partial import from an
# earlier attempt). Print the versions it actually loaded, so a failure tells us what to change.
print("\\n--- verifying import in a clean subprocess ---", flush=True)
chk = subprocess.run(
    [sys.executable, "-c",
     "import torch, vllm; print('torch', torch.__version__); "
     "print('vllm', vllm.__version__); from vllm import LLM; print('IMPORT_OK')"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(chk.stdout, flush=True)

if "IMPORT_OK" in (chk.stdout or ""):
    print("\\n" + "=" * 70)
    print(">>> vLLM imports cleanly. Proceed straight to Cell 3 -- DO NOT restart the kernel. <<<")
    print(">>> (A restart reverts these pip installs to Kaggle's base image. Because Cell 1 never")
    print(">>>  imported torch, Cell 3's import is the FIRST one and picks up torch 2.7.1.)")
    print("=" * 70)
else:
    print("\\n>>> Import STILL failing above. Do NOT run Cell 3 yet -- paste the subprocess")
    print(">>> traceback back. If it names another torch/ABI/transformers skew, we adjust a pin;")
    print(">>> no GPU time is spent until the import is clean.")
'''

LOAD_SRC = '''# ============================================================================
# Cell 3 — THE DECISIVE TEST. Load Qwen3-30B-A3B-GPTQ-Int4 on T4 x2 and emit one token.
# ============================================================================
# Env must be set BEFORE vllm is imported.
# USE_V1 = False is REQUIRED on Turing: vLLM's V1 engine hard-refuses compute capability < 8.0
# ("NotImplementedError: VLLM_USE_V1=1 is not supported with Compute Capability < 8.0"), confirmed
# on this T4 box. Setting VLLM_USE_V1=0 selects the V0 engine, which supports sm_75.
import os
USE_V1 = False
os.environ["VLLM_USE_V1"] = "1" if USE_V1 else "0"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet stalls at 0%%/99%% in managed notebooks

MODEL_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"   # our panel's hardest model, already GPTQ-Int4, public

# Diagnostics FIRST, so if the import still fails the paste tells us the real versions in THIS
# kernel (not the subprocess). A consistent torch has matching compiler/_dynamo signatures.
import torch, transformers, inspect
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
try:
    print("disable sig :", inspect.signature(torch.compiler.disable))
except Exception as _e:
    print("disable sig : <error>", _e)

from vllm import LLM, SamplingParams

# tensor_parallel_size=2  -> both T4s; the 16.9 GB weights do not fit on one 15 GB card.
# enforce_eager=True      -> no CUDA-graph capture (needed on Turing and for P1's hooks later).
# dtype="float16"         -> Turing has no bf16 tensor cores.
# max_model_len / gpu_memory_utilization kept small; we only need one token.
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,
    enforce_eager=True,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
)

out = llm.generate(["The capital of France is"],
                   SamplingParams(max_tokens=8, temperature=0.0))
print("\\nOUTPUT:", repr(out[0].outputs[0].text))
print("\\n=== INT4 MoE ON T4 (sm_75): SUCCESS -> GPTQ-Int4 is a viable format ===")
'''

INSPECT_SRC = '''# ============================================================================
# Cell 4 — (only if Cell 3 succeeded) record which quant kernel vLLM actually used.
# ============================================================================
# The load log above already names it; this is a belt-and-braces readback. Best-effort: vLLM's
# internal layout shifts between versions, so a failure here is not a probe failure -- the SUCCESS
# line in Cell 3 is what matters.
try:
    qc = llm.llm_engine.vllm_config.quant_config
    print("quant_config:", type(qc).__name__, getattr(qc, "__dict__", qc))
except Exception as e:
    print("could not read quant_config directly:", repr(e))
    print("Read the Cell 3 log instead for the 'Using <kernel>' line.")

print("\\nProbe complete. Paste back: Cell 1 caps, Cell 2 vllm version, and the FULL Cell 3 log "
      "(the quant-kernel line + either OUTPUT/SUCCESS or the traceback).")
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
            _cell("code", LOAD_SRC),
            _cell("code", INSPECT_SRC),
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
