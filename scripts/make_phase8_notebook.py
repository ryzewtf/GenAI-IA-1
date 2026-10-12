#!/usr/bin/env python3
"""Generate notebooks/phase8_ladder.ipynb -- the T8.1/T8.2 OLMoE precision ladder in ONE session.

One Kaggle T4x2 session collects the same mixed-v2 shards of OLMoE-0125 under llama.cpp at three
precisions (F16 reference, Q8_0, Q4_K_M), compares them (T8.2/T8.3), and also compares the F16 rung
against the vLLM fp16 panel trace of the same tokens (engine equivalence: the panel's Pair G and
Pair T each straddle llama.cpp and vLLM). The bootstrap cell is copied from moe_session.ipynb so the
proven clone/install path cannot drift.

Why each guard exists:
  - every rung gets its own --scratch, so its shard ledger is its own (a shared ledger would mark
    rungs 2-3 complete before they ran and silently capture nothing);
  - every rung passes --quant, so the manifest names the file actually loaded (models.yaml only
    knows the Q4_K_M file; inheriting it would label the F16 trace Q4_K_M);
  - the router audit (T1.2) runs on the F16 and Q8_0 files too -- a quantized router invalidates
    every margin T8.2 bins on.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "phase8_ladder.ipynb"
SESSION_NB = REPO / "notebooks" / "moe_session.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src}


INTRO = """# Phase 8 — OLMoE precision ladder (T8.1–T8.3) + engine equivalence

**Settings:** Accelerator **GPU T4 ×2** · Internet **On** · Secrets `GITHUB_TOKEN` (if the repo is
private) and `HF_TOKEN`.
**Add Input:** model `ryzewtf/moe-panel/gguf/olmoe-0125-f16` (the F16 reference), model
`ryzewtf/moe-panel/gguf/olmoe-0125` (Q4_K_M), dataset `moe-corpus-v2` (mixed-v2.jsonl).

Run All. Roughly: build ~25 min, Q8_0 quantize ~5 min, three rungs × ~3 min, compare ~5 min.
Download **`/kaggle/working/phase8_results.zip`** at the end (reports + the three rung traces).
"""

PARAMS = """# ============================================================================
# PARAMETERS
# ============================================================================
GIT_REF = "VLLM_PORT"
MODEL = "olmoe-0125"
CORPUS_NAME = "mixed-v2"
SHARDS = "0-1"   # ~150k tokens: shard 0 = 76k, shard 1 ~ 75k. Plan T8.1 asks for >= 100k.
VLLM_REPO = "Ryze242005/vllm-traces-v2"   # the panel's vLLM fp16 trace of the same tokens
LADDER = "/kaggle/temp/ladder"
"""

LOCATE = """# ============================================================================
# Locate inputs BEFORE the slow build: a missing attachment should cost seconds, not 25 minutes.
# ============================================================================
from pathlib import Path
import json as _json

def _one(pattern):
    hits = sorted(Path("/kaggle/input").rglob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {pattern} under /kaggle/input, found {hits}. "
                         "Check Add Input (see the first cell).")
    return hits[0]

F16 = _one("olmoe-0125-F16.gguf")
Q4 = _one("olmoe-0125-Q4_K_M.gguf")
CORPUS = _one("mixed-v2.jsonl")
with open(CORPUS, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if line.strip() and _json.loads(line)["doc_id"] != i:
            raise SystemExit(f"{CORPUS} line {i} doc_id != line index: not the mixed-v2 corpus")
print("F16   ", F16, f"{F16.stat().st_size/2**30:.1f} GiB")
print("Q4_K_M", Q4, f"{Q4.stat().st_size/2**30:.1f} GiB")
print("corpus", CORPUS, f"{i+1} docs, renumbered OK")
"""

SETUP = """# ============================================================================
# Build (moe_trace + llama-quantize, CUDA sm_75), T1.2 router audit and T1.4 node gate on Q4_K_M.
# ============================================================================
run([sys.executable, "scripts/kaggle_setup.py", "--model", MODEL,
     "--extra-targets", "llama-quantize"])

def _built(name):
    hits = [p for root in ("/tmp", "/kaggle/temp") for p in Path(root).rglob(name)
            if p.is_file() and "CMakeFiles" not in str(p)]
    if not hits:
        raise SystemExit(f"{name} was not built")
    return hits[0]

MOE_TRACE = _built("moe_trace")
QUANTIZE = _built("llama-quantize")
print("moe_trace     ", MOE_TRACE)
print("llama-quantize", QUANTIZE)
"""

QUANT = """# ============================================================================
# Q8_0 rung: quantize from the F16 reference (CPU-only work, a few minutes), then audit routers.
# ============================================================================
Path(LADDER).mkdir(parents=True, exist_ok=True)
Q8 = Path(LADDER) / "olmoe-0125-Q8_0.gguf"
if not Q8.exists():
    if run([str(QUANTIZE), str(F16), str(Q8), "Q8_0", "4"], check=False).returncode != 0:
        Q8.unlink(missing_ok=True)   # a partial file must not be collected as a rung
HAVE_Q8 = Q8.exists()
print("Q8_0", Q8, f"{Q8.stat().st_size/2**30:.1f} GiB" if HAVE_Q8 else "QUANTIZE FAILED -- rung dropped")

# T1.2 on the files models.yaml does not cover. A quantized router in the F16 reference
# invalidates the whole ladder, so that one halts (before any GPU capture); Q8_0 drops its rung.
run([sys.executable, "-m", "src.capture.router_audit", str(F16),
     "-o", f"results/router_dtype_audit_ladder_{F16.stem}.csv"])
if HAVE_Q8 and run([sys.executable, "-m", "src.capture.router_audit", str(Q8),
                    "-o", f"results/router_dtype_audit_ladder_{Q8.stem}.csv"],
                   check=False).returncode != 0:
    print("Q8_0 router audit FAILED -- rung dropped")
    HAVE_Q8 = False
"""

COLLECT = """# ============================================================================
# Collect the three rungs. Own scratch (=> own shard ledger) and --quant per rung.
# ============================================================================
import shutil, glob
REPORTS = Path("/kaggle/working/phase8"); REPORTS.mkdir(parents=True, exist_ok=True)

def root_of(name):   # the dir holding <model>/<corpus>/shard_*
    return next((Path(LADDER) / name / "out").rglob("shard_00000")).parents[2]

def save_rung(name):
    # /kaggle/temp dies with the session: each rung goes to /kaggle/working THE MOMENT it exists,
    # so a later failure (quantize, another rung, HF download) cannot take finished traces with it.
    shutil.copytree(root_of(name), REPORTS / "traces" / name, dirs_exist_ok=True)
    v = Path(f"results/validate_{MODEL}_{CORPUS_NAME}.json")
    if v.exists():
        shutil.copy(v, REPORTS / f"validate_{name}.json")   # overwritten by the next rung

RUNGS = [("f16", F16, "F16"), ("q8_0", Q8, "Q8_0"), ("q4_k_m", Q4, "Q4_K_M")]
FAILED = [] if HAVE_Q8 else ["q8_0"]
RUNGS = [r for r in RUNGS if r[0] not in FAILED]
for name, gguf, quant in RUNGS:
    rung = Path(LADDER) / name
    rc = run([sys.executable, "scripts/kaggle_collect.py", "--model", MODEL,
              "--corpus", str(CORPUS), "--corpus-name", CORPUS_NAME,
              "--gguf", str(gguf), "--quant", quant, "--binary", str(MOE_TRACE),
              "--variant-name", f"ladder-{name}", "--shards", SHARDS,
              "--backend", "local", "--scratch", str(rung), "--local-root", str(rung / "out"),
              "--preflight-gb", "0"], check=False).returncode
    try:
        save_rung(name)
        m = _json.loads(next((rung / "out").rglob("manifest.json")).read_text())
        ok = rc == 0 and m["quant"] == quant
        print(f"== {name}: rc={rc} manifest quant={m['quant']} sha={m['gguf_sha256'][:12]} "
              f"{'OK' if ok else 'PROBLEM'}")
    except StopIteration:
        ok = False
        print(f"== {name}: rc={rc}, NO TRACE WRITTEN")
    if not ok:
        FAILED.append(name)
print("\\nrungs failed:", FAILED or "none")
"""

COMPARE = """# ============================================================================
# T8.2/T8.3 ladder (F16 reference) + engine equivalence against the vLLM fp16 panel trace.
# ============================================================================
import traceback
if "f16" not in FAILED:
    for cand in ("q8_0", "q4_k_m"):
        if cand in FAILED:
            continue
        run([sys.executable, "-m", "src.analysis.precision",
             "--reference-root", str(root_of("f16")), "--candidate-root", str(root_of(cand)),
             "--model", MODEL, "--corpus", CORPUS_NAME,
             "--report", str(REPORTS / f"ladder_f16_vs_{cand}.json")], check=False)

    # Engine equivalence: vLLM fp16 panel trace of the same shards vs the llama.cpp F16 rung.
    # Never allowed to stop the packaging cell -- the F16 trace is already saved, so this
    # comparison can be redone on the workstation if anything here fails.
    try:
        from huggingface_hub import snapshot_download
        lo, hi = (int(x) for x in SHARDS.split("-"))
        vroot = Path(LADDER) / "vllm"
        snapshot_download(VLLM_REPO, repo_type="dataset", local_dir=str(vroot),
                          token=os.environ.get("HF_TOKEN"),
                          allow_patterns=[f"traces/{MODEL}/{CORPUS_NAME}/shard_{s:05d}/*"
                                          for s in range(lo, hi + 1)])
        # Diagnose a token-count refusal from the log alone (BOS handling differs by engine?).
        import numpy as np
        for label, root in (("vllm", vroot / "traces"), ("llama.cpp", root_of("f16"))):
            for sd in sorted((root / MODEL / CORPUS_NAME).glob("shard_*")):
                mf = _json.loads((sd / "manifest.json").read_text())
                tok = np.fromfile(sd / "tokens.bin", dtype=np.uint32).reshape(-1, 4)
                print(f"{label:9} {sd.name} n_tokens={mf['n_tokens']} first ids={tok[:6, 0].tolist()}")
        run([sys.executable, "-m", "src.analysis.precision",
             "--reference-root", str(vroot / "traces"), "--candidate-root", str(root_of("f16")),
             "--model", MODEL, "--corpus", CORPUS_NAME, "--cross-engine",
             "--report", str(REPORTS / "engine_vllm_fp16_vs_llamacpp_f16.json")], check=False)
    except Exception:
        traceback.print_exc()
        print("engine-equivalence step failed; F16 trace is saved, redo it on the workstation")
else:
    print("F16 reference rung failed: no comparison possible this session")
"""

PACK = """# ============================================================================
# Package: reports + the three rung traces + router audits + collection log.
# ============================================================================
for f in glob.glob("results/router_dtype_audit_ladder_*.csv") + ["results/collection_log.csv"]:
    if Path(f).exists():
        shutil.copy(f, REPORTS)
shutil.make_archive("/kaggle/working/phase8_results", "zip", REPORTS)
print(f"\\n== packaged /kaggle/working/phase8_results.zip "
      f"({Path('/kaggle/working/phase8_results.zip').stat().st_size/2**20:.0f} MiB) "
      f"in {(time.time()-T0)/60:.1f} min")
"""


def main() -> None:
    session = json.loads(SESSION_NB.read_text(encoding="utf-8"))
    bootstrap = "".join(session["cells"][2]["source"])
    if "GITHUB_TOKEN" not in bootstrap or "def run(" not in bootstrap:
        raise SystemExit("moe_session.ipynb cell 2 is no longer the bootstrap cell; update this script")
    cells = [md(INTRO), code(PARAMS), code(bootstrap), code(LOCATE), code(SETUP), code(QUANT),
             code(COLLECT), code(COMPARE), code(PACK)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
