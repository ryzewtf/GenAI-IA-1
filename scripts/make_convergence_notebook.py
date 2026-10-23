#!/usr/bin/env python3
"""Generate the Kaggle CPU half of the F4/FV convergence study (audit L6).

Writes notebooks/audit_convergence_kaggle.ipynb and audit_convergence_kaggle.zip (the code the
notebook runs). Kaggle takes Qwen3 and Gemma (largest traces); the other five models run locally. The notebook
pulls one model's trace streams from HF at a time (no logits.bin: F4/FV read hidden/topk/tokens
only), runs ``scripts.audit_convergence run`` for it, deletes the streams, and packs the per-L2
result JSONs into /kaggle/working/convergence_results.zip. Unzip that into E:/temp2/conv and run
``python -m scripts.audit_convergence report``.

    python -m scripts.make_convergence_notebook
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "audit_convergence_kaggle.ipynb"
ZIP = REPO / "audit_convergence_kaggle.zip"
KAGGLE_MODELS = ["qwen3-30b-a3b", "gemma-4-26b-a4b"]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src}


INTRO = f"""# Audit L6 — F4/FV convergence study (Kaggle CPU half)

Refits the softmax router-input probes F4 and FV at three relative depths per model (≈0.25/0.5/0.75)
for L2 ∈ {{1e-4, 1e-3, 1e-2}} with up to 2000 L-BFGS iterations. Models here: {", ".join(KAGGLE_MODELS)}
(the other five models run locally).

**Settings:** Accelerator **None (CPU)** · Internet **On** · Secret `HF_TOKEN` (read access to the
trace repos). Upload `audit_convergence_kaggle.zip` (as a Dataset or into the notebook; Kaggle
auto-extracts it, the next cell handles both).

Run All. Each model: download 8–14 GB (~5 min), fits ~30–90 min. Resumable within a session: finished
layers are skipped on a rerun. Download **`/kaggle/working/convergence_results.zip`** at the end.
To split across two sessions, set `MODELS` in the parameters cell differently in each.
"""

PARAMS = f"""# ============================================================================
# PARAMETERS
# ============================================================================
MODELS = {KAGGLE_MODELS!r}
MAXITER = 2000
THREADS = "4"
"""

LOCATE = """# --- locate the code (Kaggle auto-extracts uploaded zips; also handles a raw zip) -> writable dir ---
import glob, zipfile, os, shutil
CODE = '/kaggle/working/code'
shutil.rmtree(CODE, ignore_errors=True)
hits = glob.glob('/kaggle/input/**/scripts/audit_convergence.py', recursive=True)
if hits:
    src = os.path.dirname(os.path.dirname(hits[0]))
    shutil.copytree(src, CODE)
    print("copied extracted code", src, "->", CODE)
else:
    zips = glob.glob('/kaggle/input/**/audit_convergence_kaggle.zip', recursive=True) + \\
           glob.glob('/kaggle/working/*.zip')
    assert zips, "code not found: expected scripts/audit_convergence.py or audit_convergence_kaggle.zip under /kaggle/input"
    zipfile.ZipFile(zips[0]).extractall(CODE)
    print("extracted", zips[0], "->", CODE)
for need in ["scripts/audit_convergence.py", "scripts/phase7_sweep.py", "src/probes/linear.py",
             "corpora/mixed-v2.splits.json"]:
    assert os.path.exists(os.path.join(CODE, need)), f"missing {need} in {CODE}"
assert '"--out"' in open(os.path.join(CODE, 'scripts/audit_convergence.py')).read(), "old code: --out missing"
import subprocess, sys
subprocess.run([sys.executable, "-m", "scripts.audit_convergence", "run", "--help"], cwd=CODE, check=True,
               env=dict(os.environ, PYTHONPATH=CODE), stdout=subprocess.DEVNULL)
print("code OK")
"""

RUN = """# --- per model: download streams (no logits), run the study, free the disk ---
import os, sys, shutil, subprocess, time
from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("HF_TOKEN")
CODE = '/kaggle/working/code'; ROOT = '/kaggle/temp/traces'; OUT = '/kaggle/working/conv'
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
    print(f"=== {m} : downloaded in {time.time()-t0:.0f}s; fitting ===", flush=True)
    rc = subprocess.run([sys.executable, "-m", "scripts.audit_convergence", "run", "--models", m,
                         "--root", ROOT, "--maxiter", str(MAXITER), "--out", OUT],
                        cwd=CODE, env=env).returncode
    status[m] = rc
    print(f"=== {m} : rc={rc} after {(time.time()-t0)/60:.0f} min; freeing disk ===", flush=True)
    shutil.rmtree(f"{dst}/traces/{m}", ignore_errors=True)
print("\\nSTUDY COMPLETE", status)
"""

PACK = """# --- summary + package ---
import json, glob, os, math, shutil
OUT = '/kaggle/working/conv'
for f in sorted(glob.glob(OUT + '/l2_*/*.json')):
    d = json.load(open(f))
    for L, r in sorted(d['layers'].items(), key=lambda kv: int(kv[0])):
        if r.get('status') != 'ok':
            print(f, L, r.get('status')); continue
        b = r['metrics']['family_b']; hp = r['fit']['hyperparams']
        att = b['mi_bits'] / (b['entropy_bits'] - math.log2(r['fit']['top_k']))
        print(f"{os.path.basename(os.path.dirname(f)):>10} {d['model']:<18} {d['feature']:<3} L{L:<3} "
              f"att={att:.3f} val_ce={r['fit']['best_val_ce_bits']:.4f} conv={hp['converged']} it={hp['n_iter']}")
shutil.make_archive('/kaggle/working/convergence_results', 'zip', OUT)
print("\\npackaged -> /kaggle/working/convergence_results.zip  (download this)")
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
