#!/usr/bin/env python3
"""Generate the Kaggle CPU half of the converged lookahead (Phase 10 refit) and F5 seed runs (audit L7).

Writes notebooks/audit_lookahead_kaggle.ipynb and audit_lookahead_kaggle.zip (the code the notebook
runs). Kaggle takes Qwen3 and Gemma (largest traces); the other five models run locally. Per model the
notebook pulls the trace streams from HF (no logits.bin), runs

  1. ``scripts.phase10_lookahead`` with the converged probe settings (L2 from the convergence study), then
  2. ``scripts.audit_seeds run`` (F5, seeds 1 and 2, on the convergence-study layers),

deletes the streams, and packs everything into /kaggle/working/lookahead_seeds_results.zip. Unzip it
into a fresh directory, then copy ``lookahead_conv_kaggle.json`` to results/phase10/ and the ``seeds/s*``
directories into E:/temp2/seeds.

    python -m scripts.make_lookahead_notebook
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "audit_lookahead_kaggle.ipynb"
ZIP = REPO / "audit_lookahead_kaggle.zip"
KAGGLE_MODELS = ["qwen3-30b-a3b", "gemma-4-26b-a4b"]
L2 = {"qwen3-30b-a3b": 0.001, "gemma-4-26b-a4b": 0.001}  # val-selected in results/audit/convergence.json


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src}


INTRO = f"""# Audit — converged lookahead + F5 seeds (Kaggle CPU half)

1. **Lookahead:** router input of layer ℓ−j → expert set of layer ℓ, j ∈ {{1, 2, 4, 8}}, targets at
   relative depth 0.3/0.5/0.7/0.9, with the converged softmax probe (2000 L-BFGS iterations, L2 from the
   convergence study). v0.1 used the 150-iteration probe.
2. **F5 seeds 1 and 2** on the convergence-study layers (seed spread for the MLP probe).

Models here: {", ".join(KAGGLE_MODELS)} (the other five models run locally).

**Settings:** Accelerator **None (CPU)** · Internet **On** · Secret `HF_TOKEN` (read access to the
trace repos). Upload `audit_lookahead_kaggle.zip` (as a Dataset or into the notebook; Kaggle
auto-extracts it, the next cell handles both).

Run All. Per model: download 8–14 GB (~5 min), lookahead ~1–2 h, seeds ~30–60 min. Resumable within a
session: finished cells are skipped on a rerun. Download **`/kaggle/working/lookahead_seeds_results.zip`**
at the end. To split across two sessions, set `MODELS` differently in each; set `RUN_SEEDS = False`
to do only the lookahead.
"""

PARAMS = f"""# ============================================================================
# PARAMETERS
# ============================================================================
MODELS = {KAGGLE_MODELS!r}
L2 = {L2!r}
MAXITER = 2000
RUN_LOOKAHEAD = True
RUN_SEEDS = True
SEEDS = "1,2"
THREADS = "4"
"""

LOCATE = """# --- locate the code (Kaggle auto-extracts uploaded zips; also handles a raw zip) -> writable dir ---
import glob, zipfile, os, shutil
CODE = '/kaggle/working/code'
shutil.rmtree(CODE, ignore_errors=True)
hits = glob.glob('/kaggle/input/**/scripts/audit_seeds.py', recursive=True)
if hits:
    src = os.path.dirname(os.path.dirname(hits[0]))
    shutil.copytree(src, CODE)
    print("copied extracted code", src, "->", CODE)
else:
    zips = glob.glob('/kaggle/input/**/audit_lookahead_kaggle.zip', recursive=True) + \\
           glob.glob('/kaggle/working/*.zip')
    assert zips, "code not found: expected scripts/audit_seeds.py or audit_lookahead_kaggle.zip under /kaggle/input"
    zipfile.ZipFile(zips[0]).extractall(CODE)
    print("extracted", zips[0], "->", CODE)
for need in ["scripts/phase10_lookahead.py", "scripts/audit_seeds.py", "scripts/phase7_sweep.py",
             "src/probes/linear.py", "corpora/mixed-v2.splits.json"]:
    assert os.path.exists(os.path.join(CODE, need)), f"missing {need} in {CODE}"
assert '"--probe-kwargs"' in open(os.path.join(CODE, 'scripts/phase10_lookahead.py')).read(), \\
    "old code: phase10_lookahead has no --probe-kwargs"
import subprocess, sys
for mod in (["scripts.phase10_lookahead", "--help"], ["scripts.audit_seeds", "run", "--help"]):
    subprocess.run([sys.executable, "-m", *mod], cwd=CODE, check=True,
                   env=dict(os.environ, PYTHONPATH=CODE), stdout=subprocess.DEVNULL)
print("code OK")
"""

RUN = """# --- per model: download streams (no logits), lookahead, seeds, free the disk ---
import os, sys, json, shutil, subprocess, time
from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("HF_TOKEN")
CODE = '/kaggle/working/code'; ROOT = '/kaggle/temp/traces'; OUT = '/kaggle/working/out'
os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, PYTHONPATH=CODE, PYTHONUTF8="1", OMP_NUM_THREADS=THREADS,
           OPENBLAS_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, NUMEXPR_NUM_THREADS=THREADS)
sys.path.insert(0, CODE)
from scripts.phase6_metrics import ENGINE_BY_MODEL
REPO = {"llamacpp": "Ryze242005/llamacpp-traces-v2", "vllm": "Ryze242005/vllm-traces-v2"}
status = {}
for m in MODELS:
    eng = ENGINE_BY_MODEL[m]; dst = f"{ROOT}/{eng}"
    t0 = time.time()
    print(f"\\n=== {m} ({eng}) : download ===", flush=True)
    snapshot_download(REPO[eng], repo_type="dataset", local_dir=dst, token=tok,
                      allow_patterns=[f"traces/{m}/mixed-v2/**"], ignore_patterns=["*logits.bin"])
    print(f"=== {m} : downloaded in {time.time()-t0:.0f}s ===", flush=True)
    if RUN_LOOKAHEAD:
        pkw = json.dumps({"lbfgs_maxiter": MAXITER, "l2": L2[m]})
        rc = subprocess.run([sys.executable, "-m", "scripts.phase10_lookahead", "--models", m, "--root", ROOT,
                             "--probe-kwargs", pkw, "--out", f"{OUT}/lookahead_conv_kaggle.json"],
                            cwd=CODE, env=env).returncode
        status[(m, "lookahead")] = rc
        print(f"=== {m} : lookahead rc={rc} after {(time.time()-t0)/60:.0f} min ===", flush=True)
    if RUN_SEEDS:
        rc = subprocess.run([sys.executable, "-m", "scripts.audit_seeds", "run", "--models", m, "--seeds", SEEDS,
                             "--root", ROOT, "--out", f"{OUT}/seeds"], cwd=CODE, env=env).returncode
        status[(m, "seeds")] = rc
        print(f"=== {m} : seeds rc={rc} after {(time.time()-t0)/60:.0f} min ===", flush=True)
    shutil.rmtree(f"{dst}/traces/{m}", ignore_errors=True)
print("\\nALL DONE", status)
"""

PACK = """# --- summary + package ---
import json, glob, os, math, shutil
OUT = '/kaggle/working/out'
p = f"{OUT}/lookahead_conv_kaggle.json"
if os.path.exists(p):
    for m, cells in json.load(open(p)).items():
        for key, c in sorted(cells.items(), key=lambda kv: (kv[1]['target_layer'], kv[1]['lag'])):
            print(f"lookahead {m:<18} l={c['target_layer']:<3} j={c['lag']} att={c['mi_attainable']:.3f} "
                  f"conv={c['converged']} it={c['n_iter']}")
for f in sorted(glob.glob(OUT + '/seeds/s*/*.json')):
    d = json.load(open(f))
    for L, r in sorted(d['layers'].items(), key=lambda kv: int(kv[0])):
        if r.get('status') != 'ok':
            continue
        b = r['metrics']['family_b']
        att = b['mi_bits'] / (b['entropy_bits'] - math.log2(r['fit']['top_k']))
        print(f"seeds {os.path.basename(os.path.dirname(f)):>3} {d['model']:<18} L{L:<3} att={att:.4f}")
shutil.make_archive('/kaggle/working/lookahead_seeds_results', 'zip', OUT)
print("\\npackaged -> /kaggle/working/lookahead_seeds_results.zip  (download this)")
"""


def build_notebook() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": [md(INTRO), code(PARAMS), code("!pip -q install -U huggingface_hub"),
                    code(LOCATE), code(RUN), code(PACK)]}
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def build_zip() -> int:
    files = [p for d in ("src", "scripts", "configs") for p in (REPO / d).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    files += [REPO / "corpora" / "mixed-v2.splits.json", NB]
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(REPO).as_posix())
    return len(files)


if __name__ == "__main__":
    build_notebook()
    n = build_zip()
    print(f"wrote {NB.relative_to(REPO)} and {ZIP.name} ({n} files, {ZIP.stat().st_size / 1e6:.1f} MB)")
