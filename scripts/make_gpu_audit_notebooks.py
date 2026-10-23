#!/usr/bin/env python3
"""Generate the two Kaggle notebooks for the GPU audit checks (CHECKLISTS [H]; audit L5, C10, C13, T8.5).

1. ``notebooks/audit_convert_deepseek_cpu.ipynb`` — **CPU session, no GPU quota.** Rebuilds
   DeepSeek-V2-Lite from the HF safetensors with the panel's own recipe (``scripts/kaggle_convert.py``:
   F16 intermediate, then Q4_K_M), checks that the regenerated Q4_K_M is tensor-for-tensor identical
   to the panel's file (so the new rung comes from the same F16 as the collected one), quantizes the
   same F16 to **Q8_0**, audits its router dtype, and leaves the Q8_0 GGUF in ``/kaggle/working``.
2. ``notebooks/audit_gpu_checks.ipynb`` — **GPU T4×2.** Attach notebook 1's output. Collects, each
   rung in its own scratch and saved the moment it exists:

   * DeepSeek Q8_0 and Q4_K_M on the same shards → the precision effect on a second model (T8.4);
   * Gemma 4 Q4_K_M at ``tensor_split`` 0.5/0.5 and 0.35/0.65 → device-split equivalence (T8.5);
   * OLMoE-0125-Instruct Q4_K_M, prefill vs one-token-at-a-time decode → the T3.8 check on GPU,
     and optionally OLMoE-0125 F16 the same way (is the gap a quantized-kernel effect?).

   Every comparison runs in-session against a same-session reference, so nothing depends on traces
   from another session. Traces are packed without ``hidden.bin`` (the comparisons read only
   ``topk``/``logits``/``tokens``).

    python -m scripts.make_gpu_audit_notebooks
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SESSION_NB = REPO / "notebooks" / "moe_session.ipynb"
NB_CPU = REPO / "notebooks" / "audit_convert_deepseek_cpu.ipynb"
NB_GPU = REPO / "notebooks" / "audit_gpu_checks.ipynb"
NB_DEC = REPO / "notebooks" / "audit_gpu_decode.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


# --------------------------------------------------------------------------------------------------
# Notebook 1: CPU conversion
# --------------------------------------------------------------------------------------------------

CPU_INTRO = """# Audit — DeepSeek-V2-Lite Q8_0 rung (CPU session, no GPU quota)

**Settings:** Accelerator **None (CPU)** · Internet **On** · Secrets `HF_TOKEN` and `GITHUB_TOKEN`
(if the repo is private).
**Add Input:** model `ryzewtf/moe-panel/gguf/deepseek-v2-lite` (the panel's Q4_K_M, for the identity check).

Run All. Roughly: converter environment + tokenizer check ~1 min (fails fast if the converter would
not recognise the tokenizer), build llama-quantize ~10 min, download 31 GB ~10 min, convert to F16 ~20 min,
Q4_K_M ~15 min, Q8_0 ~10 min, identity check ~10 min. Output: `deepseek-v2-lite-Q8_0.gguf` (~16 GB)
plus `convert_report.json` in `/kaggle/working`. Then attach **this notebook's output** to
`audit_gpu_checks.ipynb`.

`convert_report.json` → `q4_identical_to_panel` must be `true` before the GPU session: it proves the
Q8_0 rung is quantized from the same F16 as the trace already collected.
"""

CPU_PARAMS = """# ============================================================================
# PARAMETERS
# ============================================================================
GIT_REF = "VLLM_PORT"
MODEL = "deepseek-v2-lite"
PANEL_Q4_SHA = "967ab33ee0d61ebe38bd974fdc31e4c9a4eea13fd9c72bcec43d7afc9f2bbef1"   # configs/models.yaml
"""

CPU_LOCATE = """# ============================================================================
# Locate the attached panel Q4_K_M and check disk BEFORE the slow steps.
# ============================================================================
import json as _json, shutil as _sh
hits = sorted(Path("/kaggle/input").rglob("deepseek-v2-lite-Q4_K_M.gguf"))
if len(hits) != 1:
    raise SystemExit(f"attach ryzewtf/moe-panel/gguf/deepseek-v2-lite (found {hits})")
PANEL_Q4 = hits[0]
for d in ("/kaggle/temp", "/tmp", "/kaggle/working"):
    if Path(d).is_dir():
        print(f"{d:16s} {_sh.disk_usage(d).free/2**30:7.1f} GiB free")
print("panel Q4_K_M", PANEL_Q4, f"{PANEL_Q4.stat().st_size/2**30:.1f} GiB")
"""

# convert_hf_to_gguf.py identifies the pre-tokenizer by hashing the tokenizer's encoding of this
# string (llama.cpp 7077abbe, conversion/base.py `chktxt`). Kept ASCII-escaped: it holds runs of
# quotes that would end the cell's triple-quoted source.
CHKTXT = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) \U0001f636‍\U0001f32b️ (multiple emojis concatenated) ✅ \U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច\U0001f601 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

CPU_CONVENV = """# ============================================================================
# Converter environment. The converter names the pre-tokenizer from a hash of the tokenizer's
# output, and that output depends on the transformers version: 4.57.6 (the panel conversion,
# results/convert_deepseek-v2-lite.json) gives 049ecf76..., which the pinned tree knows; the image's
# transformers 5.x gives 99ecacb4..., which it does not, and the converter stops ("BPE pre-tokenizer
# was not recognized"). The first run of this notebook died there, after the 31 GB download.
# So the converter runs from the image's interpreter with the recorded transformers/tokenizers
# shadowing the image's through PYTHONPATH. --no-deps leaves the image's numpy and torch alone.
# The hash is checked here, before anything slow.
# ============================================================================
CONV_SITE = Path("/tmp/convsite")
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(CONV_SITE),
     "transformers==4.57.6", "tokenizers==0.22.2", "huggingface_hub==0.36.0"])
CONV_PY = Path("/tmp/convpy")
CONV_PY.write_text(f'#!/bin/sh\\nPYTHONPATH={CONV_SITE}${{PYTHONPATH:+:$PYTHONPATH}} exec {sys.executable} "$@"\\n')
CONV_PY.chmod(0o755)

from huggingface_hub import snapshot_download
TOK_DIR = snapshot_download("deepseek-ai/DeepSeek-V2-Lite", local_dir="/tmp/ds_tok",
                            allow_patterns=["tokenizer*.json", "config.json"])
EXPECTED_CHKHSH = "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754"
CHKTXT_FILE = Path("/tmp/chktxt.txt")   # a file, not argv: the text is multi-line and full of quotes
CHKTXT_FILE.write_text(@CHKTXT@, encoding="utf-8", newline="")
probe = run([str(CONV_PY), "-c",
             "import sys, transformers, tokenizers; from hashlib import sha256; "
             "from transformers import AutoTokenizer; "
             "t = AutoTokenizer.from_pretrained(sys.argv[1]); "
             "s = open(sys.argv[2], encoding='utf-8', newline='').read(); "
             "print(transformers.__version__, tokenizers.__version__, "
             "sha256(str(t.encode(s)).encode()).hexdigest())",
             TOK_DIR, str(CHKTXT_FILE)], quiet=True, check=False)
got = probe.stdout.strip().splitlines()[-1].split() if probe.stdout.strip() else []
print("converter env:", " ".join(got) or probe.stdout)
if probe.returncode != 0 or got[:2] != ["4.57.6", "0.22.2"] or got[2:] != [EXPECTED_CHKHSH]:
    raise SystemExit("converter environment is not the recorded one; nothing was downloaded")
"""

CPU_CONVERT = """# ============================================================================
# The panel recipe: HF safetensors -> F16 -> Q4_K_M at the pinned llama.cpp commit (CPU build).
# --keep-f16 keeps the F16 for the Q8_0 rung; --prune drops the safetensors once converted.
# --python: the converter runs under the environment checked above.
# ============================================================================
run([sys.executable, "scripts/kaggle_convert.py", "--model", MODEL, "--keep-f16", MODEL, "--prune",
     "--python", str(CONV_PY)])
from src.runtime.setup_kaggle import _pick_scratch
GGUF_DIR = _pick_scratch() / "gguf"
F16 = GGUF_DIR / f"{MODEL}-F16.gguf"
Q4_NEW = GGUF_DIR / f"{MODEL}-Q4_K_M.gguf"
for p in (F16, Q4_NEW):
    if not p.is_file():
        raise SystemExit(f"missing {p}; see the conversion log above")
    print(p, f"{p.stat().st_size/2**30:.1f} GiB")
"""

CPU_Q8 = """# ============================================================================
# Q8_0 from the same F16, then the T1.2 router-dtype audit on it.
# ============================================================================
QUANTIZE = next(p for root in ("/tmp", "/kaggle/temp") for p in Path(root).rglob("llama-quantize")
                if p.is_file() and "CMakeFiles" not in str(p))
Q8 = Path("/kaggle/working") / f"{MODEL}-Q8_0.gguf"
if run([str(QUANTIZE), str(F16), str(Q8), "Q8_0", str(os.cpu_count() or 4)], check=False).returncode != 0:
    Q8.unlink(missing_ok=True)
    raise SystemExit("llama-quantize Q8_0 failed")
audit = run([sys.executable, "-m", "src.capture.router_audit", str(Q8),
             "-o", f"/kaggle/working/router_dtype_audit_{Q8.stem}.csv"], check=False)
ROUTER_OK = audit.returncode == 0
print("Q8_0", Q8, f"{Q8.stat().st_size/2**30:.1f} GiB", "router audit", "PASS" if ROUTER_OK else "FAIL")
"""

CPU_IDENTITY = """# ============================================================================
# Identity check: the regenerated Q4_K_M against the panel's, file hash and per-tensor data hash.
# A file-hash mismatch alone can be metadata (e.g. a converter-env string); identical tensor data is
# what makes the Q8_0 rung a same-source comparison.
# ============================================================================
import hashlib
from gguf import GGUFReader

def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def tensor_hashes(p):
    r = GGUFReader(str(p))
    import numpy as np
    return {t.name: (str(t.tensor_type), hashlib.sha256(np.ascontiguousarray(t.data)).hexdigest())
            for t in r.tensors}

new_sha, panel_sha = file_sha(Q4_NEW), file_sha(PANEL_Q4)
a, b = tensor_hashes(Q4_NEW), tensor_hashes(PANEL_Q4)
differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
report = {
    "model": MODEL, "q8_file": Q8.name, "q8_sha256": file_sha(Q8), "q8_size_bytes": Q8.stat().st_size,
    "q8_router_audit_pass": ROUTER_OK,
    "q4_regenerated_sha256": new_sha, "q4_panel_sha256": panel_sha,
    "q4_panel_sha_matches_models_yaml": panel_sha == PANEL_Q4_SHA,
    "q4_file_identical": new_sha == panel_sha,
    "q4_n_tensors": len(a), "q4_tensors_differing": differ[:50], "q4_n_tensors_differing": len(differ),
    "q4_identical_to_panel": not differ and set(a) == set(b),
}
Path("/kaggle/working/convert_report.json").write_text(_json.dumps(report, indent=2))
for rec in Path("results").rglob("*conversion*.json"):
    shutil.copy(rec, "/kaggle/working/")
print(_json.dumps(report, indent=2))
if not report["q4_identical_to_panel"] or not ROUTER_OK:
    print("\\n!! DO NOT start the GPU session on this output: see the report above")
# The clone is not output. Left in /kaggle/working it is attached to the GPU notebook along with the
# Q8_0 (where `_one(...)` must find exactly one of each input), counts against the output cap, and
# makes `kaggle kernels output` page through thousands of files.
os.chdir("/kaggle/working")
shutil.rmtree(REPO, ignore_errors=True)
print(sorted(p.name for p in Path("/kaggle/working").iterdir()))
print(f"\\ndone in {(time.time()-T0)/60:.1f} min")
"""

# --------------------------------------------------------------------------------------------------
# Notebook 2: GPU checks
# --------------------------------------------------------------------------------------------------

GPU_INTRO = """# Audit — GPU checks: DeepSeek precision rung, Gemma device split, GPU decode vs prefill

**Settings:** Accelerator **GPU T4 ×2** · Internet **On** · Secrets `GITHUB_TOKEN` (if private), `HF_TOKEN`.
**Add Input:**
* the output of `audit_convert_deepseek_cpu` (`deepseek-v2-lite-Q8_0.gguf`, `convert_report.json`)
* models `ryzewtf/moe-panel/gguf/deepseek-v2-lite`, `.../gemma-4-26b-a4b`, `.../olmoe-0125-instruct`,
  `.../olmoe-0125-f16`
* dataset `moe-corpus-v2` (`mixed-v2.jsonl`)

Run All. Roughly: build ~25 min; DeepSeek Q8_0 + Q4_K_M on 2 shards ~15 min; Gemma two splits on 2
shards ~20 min; OLMoE-Instruct prefill + decode on 1 shard ~10 + 30–60 min; optional F16 decode
another 30–60 min (skipped automatically if under `RESERVE_H` hours remain). Every rung is saved to
`/kaggle/working/audit_gpu/` as soon as it finishes; the comparisons are in `reports/`.
"""

GPU_PARAMS = """# ============================================================================
# PARAMETERS
# ============================================================================
GIT_REF = "VLLM_PORT"
CORPUS_NAME = "mixed-v2"
SHARDS = "0-1"            # ~150k tokens, the Phase 8 ladder's shards
DECODE_SHARDS = "0"       # one shard (~76k tokens) for the slow one-token-at-a-time legs
ALT_SPLIT = [0.35, 0.65]  # T8.5 candidate; the pinned collection split is [0.5, 0.5]
RUN_F16_DECODE = True     # OLMoE F16 prefill vs decode (kernel/quantization question)
SESSION_H = 12.0
RESERVE_H = 1.5           # skip optional legs when less than this is left
WORK = "/tmp/audit"       # /tmp is the ~1 TB scratch disk; /kaggle/temp does not exist on Kaggle
"""

GPU_LOCATE = """# ============================================================================
# Locate every input BEFORE the build: a missing attachment costs seconds, not 25 minutes.
# ============================================================================
import json as _json, glob

# Hardware first: the Gemma split and the 15.6 GiB Q8_0 file both need two T4s (sm_75, as built).
_gpus = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False, quiet=True).stdout.strip().splitlines()
print("GPUs:", _gpus)
if len(_gpus) != 2 or not all("T4" in g for g in _gpus):
    raise SystemExit(f"need Accelerator 'GPU T4 x2', found {_gpus or 'no GPU'}")
if shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").exists():
    raise SystemExit("nvcc not found: this image cannot build moe_trace")
Path(WORK).mkdir(parents=True, exist_ok=True)
_free = shutil.disk_usage(WORK).free / 2**30
print(f"{WORK}: {_free:.0f} GiB free")
if _free < 25:
    raise SystemExit(f"{WORK} has {_free:.0f} GiB free; the rungs need ~10 GiB plus the build")

def _one(pattern):
    hits = sorted(Path("/kaggle/input").rglob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {pattern} under /kaggle/input, found {hits}")
    return hits[0]

DS_Q8 = _one("deepseek-v2-lite-Q8_0.gguf")
DS_Q4 = _one("deepseek-v2-lite-Q4_K_M.gguf")
GEMMA = _one("gemma-4-26b-a4b-Q4_K_M.gguf")
OLMI_Q4 = _one("olmoe-0125-instruct-Q4_K_M.gguf")
OLM_F16 = _one("olmoe-0125-F16.gguf")
CORPUS = _one("mixed-v2.jsonl")
rep = _json.loads(_one("convert_report.json").read_text())
if not (rep.get("q4_identical_to_panel") and rep.get("q8_router_audit_pass")):
    raise SystemExit(f"convert_report.json does not clear the Q8_0 rung: {rep}")
with open(CORPUS, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if line.strip() and _json.loads(line)["doc_id"] != i:
            raise SystemExit(f"{CORPUS} line {i}: not the renumbered mixed-v2 corpus")
for p in (DS_Q8, DS_Q4, GEMMA, OLMI_Q4, OLM_F16):
    print(f"{p.name:36s} {p.stat().st_size/2**30:5.1f} GiB  {p}")
print("corpus", CORPUS, f"{i+1} docs OK; Q8_0 sha {rep['q8_sha256'][:12]}")
"""

GPU_SETUP = """# ============================================================================
# Build moe_trace (CUDA sm_75) with the T1.2/T1.4 gates on DeepSeek; router audit on the Q8_0 file.
# The node specs for Gemma and OLMoE are committed and were gated on these exact files earlier.
# ============================================================================
run([sys.executable, "scripts/kaggle_setup.py", "--model", "deepseek-v2-lite"])
MOE_TRACE = next(p for root in ("/tmp", "/kaggle/temp") for p in Path(root).rglob("moe_trace")
                 if p.is_file() and "CMakeFiles" not in str(p))
print("moe_trace", MOE_TRACE)
run([sys.executable, "-m", "src.capture.router_audit", str(DS_Q8),
     "-o", "results/router_dtype_audit_ds_q8.csv"])
"""

GPU_COLLECT = """# ============================================================================
# Rungs. Own scratch (=> own shard ledger) per rung; --quant per rung; each saved the moment it exists.
# ============================================================================
import re, zipfile
OUT = Path("/kaggle/working/audit_gpu"); (OUT / "reports").mkdir(parents=True, exist_ok=True)
RUN_YAML = Path("configs/run.yaml")
RUN_YAML_TEXT = RUN_YAML.read_text()
assert RUN_YAML_TEXT.count("tensor_split: [0.5, 0.5]") == 1, "run.yaml no longer pins tensor_split [0.5, 0.5] once"
DONE, FAILED = {}, []

def hours_left():
    return SESSION_H - (time.time() - T0) / 3600

def root_of(name):
    return next((Path(WORK) / name / "out").rglob("shard_00000")).parents[2]

def save_slim(name):
    # topk/logits/tokens/manifests only: the comparisons never read hidden.bin, and it is most of the bytes
    root = root_of(name)
    with zipfile.ZipFile(OUT / f"trace_{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in root.rglob("*"):
            if f.is_file() and f.name != "hidden.bin":
                z.write(f, f.relative_to(root.parent))

def rung(name, model, gguf, quant, shards, *, split=None, decode="off", timeout_s=None):
    val = Path(f"results/validate_{model}_{CORPUS_NAME}.json")
    val.unlink(missing_ok=True)   # a failed rung must not inherit the previous rung's report
    if split is not None:
        RUN_YAML.write_text(RUN_YAML_TEXT.replace("tensor_split: [0.5, 0.5]",
                                                  f"tensor_split: [{split[0]}, {split[1]}]"))
    try:
        cmd = [sys.executable, "scripts/kaggle_collect.py", "--model", model,
               "--corpus", str(CORPUS), "--corpus-name", CORPUS_NAME,
               "--gguf", str(gguf), "--quant", quant, "--binary", str(MOE_TRACE),
               "--variant-name", name, "--shards", shards, "--decode-mode", decode,
               "--backend", "local", "--scratch", str(Path(WORK) / name),
               "--local-root", str(Path(WORK) / name / "out"), "--preflight-gb", "0"]
        if timeout_s:
            cmd += ["--timeout-s", str(timeout_s)]
        rc = run(cmd, check=False).returncode
    finally:
        RUN_YAML.write_text(RUN_YAML_TEXT)   # never leave the pinned split edited
    # kaggle_collect writes results/validate_<model>_<corpus>.json, so a second rung of the same
    # model would overwrite the first: keep a per-rung copy.
    if val.exists():
        shutil.copy(val, OUT / "reports" / f"validate_{name}.json")
    try:
        m = _json.loads(next((Path(WORK) / name / "out").rglob("manifest.json")).read_text())
        save_slim(name)
        ok = rc == 0 and m["quant"] == quant
        print(f"== {name}: rc={rc} quant={m['quant']} split={m.get('device_plan', {}).get('tensor_split')} "
              f"decode={m.get('capture_variant', {}).get('decode_mode')} {'OK' if ok else 'PROBLEM'}")
    except StopIteration:
        ok = False
        print(f"== {name}: rc={rc}, NO TRACE WRITTEN")
    if ok:
        DONE[name] = model
    else:
        FAILED.append(name)
    return ok

rung("ds_q8", "deepseek-v2-lite", DS_Q8, "Q8_0", SHARDS)
rung("ds_q4", "deepseek-v2-lite", DS_Q4, "Q4_K_M", SHARDS)
rung("gemma_base", "gemma-4-26b-a4b", GEMMA, "Q4_K_M", SHARDS)
rung("gemma_split", "gemma-4-26b-a4b", GEMMA, "Q4_K_M", SHARDS, split=ALT_SPLIT)
rung("olmi_prefill", "olmoe-0125-instruct", OLMI_Q4, "Q4_K_M", DECODE_SHARDS)
rung("olmi_decode", "olmoe-0125-instruct", OLMI_Q4, "Q4_K_M", DECODE_SHARDS, decode="full",
     timeout_s=int(max(0.5, hours_left() - RESERVE_H) * 3600))
if RUN_F16_DECODE and hours_left() > RESERVE_H + 1.0:
    rung("olm16_prefill", "olmoe-0125", OLM_F16, "F16", DECODE_SHARDS)
    rung("olm16_decode", "olmoe-0125", OLM_F16, "F16", DECODE_SHARDS, decode="full",
         timeout_s=int(max(0.5, hours_left() - 0.5) * 3600))
else:
    print("F16 decode legs skipped:", f"{hours_left():.1f} h left")
print("\\nrungs ok:", sorted(DONE), " failed:", FAILED or "none")
"""

GPU_COMPARE = """# ============================================================================
# Comparisons, all against a same-session reference. None of them can stop the packaging cell.
# ============================================================================
def compare(ref, cand, mode, report, extra=()):
    if ref not in DONE or cand not in DONE:
        print(f"skip {report}: missing {ref if ref not in DONE else cand}")
        return
    run([sys.executable, "-m", "src.analysis.precision",
         "--reference-root", str(root_of(ref)), "--candidate-root", str(root_of(cand)),
         "--model", DONE[ref], "--corpus", CORPUS_NAME, "--mode", mode,
         "--report", str(OUT / "reports" / report), *extra], check=False)

compare("ds_q8", "ds_q4", "ladder", "deepseek_q8_vs_q4.json")
compare("gemma_base", "gemma_split", "T8.5", "gemma_split_T8.5.json")
compare("olmi_prefill", "olmi_decode", "T3.8", "olmoe_instruct_q4_decode_T3.8.json")
compare("olm16_prefill", "olm16_decode", "T3.8", "olmoe_f16_decode_T3.8.json")
"""

GPU_PACK = """# ============================================================================
# Package: reports, router audits, validation and collection logs (traces are already zipped).
# ============================================================================
for f in glob.glob("results/router_dtype_audit_*.csv") + glob.glob("results/validate_*.json") + \\
         ["results/collection_log.csv"]:
    if Path(f).exists():
        shutil.copy(f, OUT / "reports")
(OUT / "reports" / "session.json").write_text(_json.dumps(
    {"done": DONE, "failed": FAILED, "hours": (time.time() - T0) / 3600, "git_head": HEAD}, indent=2))
shutil.make_archive("/kaggle/working/audit_gpu_reports", "zip", OUT / "reports")
for f in sorted(OUT.glob("*.zip")) + [Path("/kaggle/working/audit_gpu_reports.zip")]:
    print(f"{f.name:40s} {f.stat().st_size/2**20:8.0f} MiB")
# The clone must not be saved as output: listing it pages through thousands of files (HTTP 429).
os.chdir("/kaggle/working"); shutil.rmtree(REPO, ignore_errors=True)
print(sorted(p.name for p in Path("/kaggle/working").iterdir()))
print(f"done in {(time.time()-T0)/60:.1f} min")
"""


# --------------------------------------------------------------------------------------------------
# Notebook 3: GPU decode only (T3.8 re-run). Notebook 2's decode legs were refused by the runner's
# prefill-only ubatch bracket (fixed in src/runtime/runner.py on 2026-10-23); its other rungs stand.
# Reuses notebook 2's helpers verbatim so the rungs are collected exactly the same way.
# --------------------------------------------------------------------------------------------------

DEC_INTRO = """# Audit — GPU decode vs prefill (T3.8), OLMoE-0125-Instruct Q4_K_M and OLMoE-0125 F16

**Settings:** Accelerator **GPU T4 ×2** · Internet **On**.
**Add Input:** models `ryzewtf/moe-panel/gguf/olmoe-0125-instruct`, `.../olmoe-0125-f16`;
dataset `moe-corpus-v2` (`mixed-v2.jsonl`).

Run All. Build ~25 min; per model a prefill leg (~1–5 min) and a one-token-at-a-time decode leg
(~8–16 min) on one shard (~76k tokens). Stops within a minute if the cloned branch predates the
decode-aware capture check.
"""

DEC_PARAMS = """# ============================================================================
# PARAMETERS
# ============================================================================
GIT_REF = "VLLM_PORT"
CORPUS_NAME = "mixed-v2"
DECODE_SHARDS = "0"       # one shard (~76k tokens) for the slow one-token-at-a-time legs
RUN_F16_DECODE = True     # OLMoE F16 prefill vs decode (kernel/quantization question)
SESSION_H = 12.0
RESERVE_H = 1.5           # skip optional legs when less than this is left
WORK = "/tmp/audit"       # /tmp is the ~1 TB scratch disk; /kaggle/temp does not exist on Kaggle
"""

DEC_LOCATE_INPUTS = """
# The 2026-10-23 run lost both decode legs to a prefill-only capture check. Refuse a clone without
# the fix before spending 25 minutes on the build.
if 'mode == "full"' not in Path("src/runtime/runner.py").read_text(encoding="utf-8"):
    raise SystemExit(f"{GIT_REF} lacks the decode-aware nodes_captured check: push it first")

OLMI_Q4 = _one("olmoe-0125-instruct-Q4_K_M.gguf")
OLM_F16 = _one("olmoe-0125-F16.gguf")
CORPUS = _one("mixed-v2.jsonl")
with open(CORPUS, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if line.strip() and _json.loads(line)["doc_id"] != i:
            raise SystemExit(f"{CORPUS} line {i}: not the renumbered mixed-v2 corpus")
for p in (OLMI_Q4, OLM_F16):
    print(f"{p.name:36s} {p.stat().st_size/2**30:5.1f} GiB  {p}")
print("corpus", CORPUS, f"{i+1} docs OK")
"""

DEC_SETUP = """# ============================================================================
# Build moe_trace (CUDA sm_75); the OLMoE node specs are committed and gated on these files.
# ============================================================================
run([sys.executable, "scripts/kaggle_setup.py", "--model", "olmoe-0125-instruct"])
MOE_TRACE = next(p for root in ("/tmp", "/kaggle/temp") for p in Path(root).rglob("moe_trace")
                 if p.is_file() and "CMakeFiles" not in str(p))
print("moe_trace", MOE_TRACE)
"""

DEC_RUNGS = """
rung("olmi_prefill", "olmoe-0125-instruct", OLMI_Q4, "Q4_K_M", DECODE_SHARDS)
rung("olmi_decode", "olmoe-0125-instruct", OLMI_Q4, "Q4_K_M", DECODE_SHARDS, decode="full",
     timeout_s=int(max(0.5, hours_left() - RESERVE_H) * 3600))
if RUN_F16_DECODE and hours_left() > RESERVE_H + 1.0:
    rung("olm16_prefill", "olmoe-0125", OLM_F16, "F16", DECODE_SHARDS)
    rung("olm16_decode", "olmoe-0125", OLM_F16, "F16", DECODE_SHARDS, decode="full",
         timeout_s=int(max(0.5, hours_left() - 0.5) * 3600))
else:
    print("F16 decode legs skipped:", f"{hours_left():.1f} h left")
print("\\nrungs ok:", sorted(DONE), " failed:", FAILED or "none")
"""

DEC_COMPARES = """
compare("olmi_prefill", "olmi_decode", "T3.8", "olmoe_instruct_q4_decode_T3.8.json")
compare("olm16_prefill", "olm16_decode", "T3.8", "olmoe_f16_decode_T3.8.json")
"""


def _head(text: str, marker: str) -> str:
    """The part of a notebook-2 cell before ``marker`` (its shared helpers)."""
    if text.count(marker) != 1:
        raise SystemExit(f"cell no longer contains {marker!r} exactly once; update this script")
    return text.split(marker)[0]


def _write(path: Path, cells: list[dict]) -> None:
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    session = json.loads(SESSION_NB.read_text(encoding="utf-8"))
    bootstrap = "".join(session["cells"][2]["source"])
    if "GITHUB_TOKEN" not in bootstrap or "def run(" not in bootstrap or "HF_TOKEN" not in bootstrap:
        raise SystemExit("moe_session.ipynb cell 2 is no longer the bootstrap cell; update this script")
    _write(NB_CPU, [md(CPU_INTRO), code(CPU_PARAMS), code(bootstrap), code(CPU_LOCATE),
                    code(CPU_CONVENV.replace("@CHKTXT@", ascii(CHKTXT))),
                    code(CPU_CONVERT), code(CPU_Q8), code(CPU_IDENTITY)])
    _write(NB_GPU, [md(GPU_INTRO), code(GPU_PARAMS), code(bootstrap), code(GPU_LOCATE),
                    code(GPU_SETUP), code(GPU_COLLECT), code(GPU_COMPARE), code(GPU_PACK)])
    locate = _head(GPU_LOCATE, 'DS_Q8 = _one(') + DEC_LOCATE_INPUTS
    _write(NB_DEC, [md(DEC_INTRO), code(DEC_PARAMS), code(bootstrap), code(locate),
                    code(DEC_SETUP), code(_head(GPU_COLLECT, 'rung("ds_q8"') + DEC_RUNGS),
                    code(_head(GPU_COMPARE, 'compare("ds_q8"') + DEC_COMPARES), code(GPU_PACK)])


if __name__ == "__main__":
    main()
