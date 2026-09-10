"""Generate `notebooks/moe_session.ipynb`, the single Kaggle session notebook.

Why a generator instead of a checked-in `.ipynb` edited by hand
--------------------------------------------------------------
A notebook is JSON with source split into per-line string lists. Hand-editing it produces diffs
nobody can read, and the failure mode is a session that dies three cells in because a cell was
edited in the Kaggle UI and never came home. Keeping the source here means the notebook is
reviewable as Python, regenerable with one command, and the UI copy is always a build artifact
rather than the original.

Why ONE notebook for four kinds of session
------------------------------------------
A Kaggle session is a notebook plus its settings, so the obvious layout is four notebooks: gates,
corpus, collect, ladder. They would share a bootstrap -- clone the repo at a pinned ref, install
deps, audit the runtime, harvest the results -- and that shared part is the part that must not
drift, for the same reason `kaggle_collect.py` delegates to `runner.main` instead of re-assembling
the S.3 contract. Four copies of the bootstrap is four places for the pinned commit to go stale.
So: one notebook, a `STAGE` switch in the first cell, and the settings that genuinely differ
(accelerator, internet, attached model) documented per stage in the header rather than encoded in
four files.

What the notebook is NOT allowed to do
--------------------------------------
* **No pipeline logic.** Every stage shells out to a script under `scripts/` or a module under
  `src/` that already has tests. Code that only ever runs inside a Kaggle session is code that is
  only ever tested by a Kaggle session, and those cost 12 hours and a quota slot to run.
* **No commits or pushes.** The user commits. The harvest cell writes a git patch to
  `/kaggle/working` instead, which is also the only form that survives the session being killed --
  a push needs a token, a token needs a secret, and a secret is one more thing to be wrong at hour
  eleven.
* **Nothing large in `/kaggle/working`** (invariant I9: ~20 GB and a ~500 file commit cap). Builds,
  weights and traces go to scratch; only the patch and the JSON records come home.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "moe_session.ipynb"

GIT_URL = "https://github.com/ryzewtf/GenAI-IA-1.git"


# --------------------------------------------------------------------------------------------
# Cell sources. Written as plain strings; `_cell` handles the line-splitting nbformat wants.
# --------------------------------------------------------------------------------------------

HEADER_MD = """# MoE Routing Predictability — Kaggle session

One notebook, four session types. Set `STAGE` in the next cell, then match the Kaggle
**Settings** panel to the row below. The settings are not cosmetic: a gate session on a GPU
burns quota for a CPU workload, and a collect session without the GPU silently falls back to
a run that will not finish inside the 12 h cap.

| `STAGE` | Accelerator | Internet | Attach | Roughly |
|---|---|---|---|---|
| `audit` | None | On | — | 2 min |
| `gates` | None | On | the model's variation (optional) | 20–60 min |
| `corpus` | None | **On** | — | 1–3 h |
| `collect` | **GPU T4 ×2** | On | the model's variation (recommended) | up to 12 h |
| `ladder` | **GPU T4 ×2** | On | `olmoe-0125-f16` | 2–4 h |

**Attaching is optional but preferred.** `acquire_gguf` resolves in this order: an attached copy
under `/kaggle/input`, then this session's scratch, then `kagglehub.model_download` of the
published handle, then the HF repo. Attaching costs no session time and no bandwidth, and a
pinned model version is *immutable* — which is what T3.6's cross-session byte-identity gate
actually wants. A per-session re-download is a trace whose weights can change under it.

The panel is private, so a session without credentials sees these models as **missing**, not as
forbidden. If a download 404s, check the token before concluding the model is gone.

**Nothing is committed or pushed from here.** The last cell writes a git patch to
`/kaggle/working`; download it, `git apply` it on the workstation, and commit there.
"""

PARAMS_SRC = '''# ============================================================================
# PARAMETERS — the only cell you normally edit.
# ============================================================================

STAGE = "gates"        # audit | gates | corpus | collect | ladder
MODEL = "olmoe-0125"   # model key in configs/models.yaml; ignored by audit/corpus

# --- repo -------------------------------------------------------------------
# A branch name tracks; a commit SHA pins. Pin for anything whose output goes in
# the paper: `run_config_sha256` records the llama.cpp commit, but nothing else
# records which version of the *analysis* code produced a number.
GIT_REF = "main"

# --- gates ------------------------------------------------------------------
RESCAN_NODES = False   # redo T1.4 even if a spec already exists

# --- corpus -----------------------------------------------------------------
CORPUS_SPEC = "mixed-v1"   # or mixed-v1-scale for the T5.4 4M run
TARGET_TOKENS = None       # None = the spec's own target; set an int if Gate Q1 fired

# --- collect / ladder -------------------------------------------------------
CORPUS_FILE = None     # None = corpora/<CORPUS_SPEC>.jsonl, built by a corpus session
# The T8.1 ladder collects the SAME model at a second precision, so it cannot use the GGUF that
# models.yaml records -- that entry names one file per model and the ladder's second rung is not a
# second model. Point this at the F16 (attach `moe-panel/gguf/olmoe-0125-f16` and give its path
# under /kaggle/input) for STAGE='ladder'; leave it None everywhere else.
GGUF_PATH = None
SHARDS = None          # None = all; else "0-19" or "3,7,11" to resume a killed run
BACKEND = None         # None = run.yaml default; "hf" uploads shards, "local" keeps them
HF_REPO_ID = None      # required by --backend hf
DECODE_MODE = None     # T3.8 leg: off | full | tail
OVERRIDE_TENSOR = None # T3.7 leg
VARIANT_NAME = None    # REQUIRED once DECODE_MODE or OVERRIDE_TENSOR is set

assert STAGE in ("audit", "gates", "corpus", "collect", "ladder"), STAGE
'''

BOOTSTRAP_SRC = '''# ============================================================================
# Bootstrap: clone the repo at GIT_REF, install what the image lacks.
# ============================================================================
import os, subprocess, sys, time, shutil
from pathlib import Path

GIT_URL = "%(git_url)s"
REPO = Path("/kaggle/working/repo")
T0 = time.time()


def run(cmd, cwd=None, check=True, quiet=False):
    """Run a command, stream it, and fail loudly.

    `check=True` by default on purpose: a step that fails and lets the notebook carry on
    produces a session whose later cells operate on the previous step's leftovers, which is
    the expensive kind of wrong here -- it looks like it worked.
    """
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

os.chdir(REPO)
sys.path.insert(0, str(REPO))

# --no-deps: the Kaggle image pins torch/numpy against each other, and letting pip resolve
# transitive deps here is how a session ends up with a numpy the image's torch was not built
# against. Same rule as PIP_PACKAGES in src/runtime/setup_kaggle.py.
run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "huggingface_hub", "datasets", "sentencepiece", "gguf", "kagglehub"], check=False)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet hangs at 0%%/99%% in managed notebooks
os.environ.setdefault("PYTHONUNBUFFERED", "1")
''' % {"git_url": GIT_URL}

AUDIT_SRC = '''# ============================================================================
# Runtime audit (T0.1). Cheap, and it is the record of what this session was.
# ============================================================================
run([sys.executable, "-m", "src.runtime.diagnose"], check=False)
'''

DISPATCH_SRC = '''# ============================================================================
# Stage dispatch. Every branch shells out to something with tests behind it.
# ============================================================================

def setup(model, cpu):
    """T0.2/T0.3 build + T1.1 fetch + the T1.2/T1.4 gates, for one model.

    Idempotent by design, so re-running it after a killed session is cheap. The gates run here
    rather than at collection time because a quantized router (T1.2) invalidates every margin the
    study reads, and a wrong node spec (T1.4) produces a full-size, plausible, WRONG trace (I13).
    Both are better found before the model is loaded.
    """
    cmd = [sys.executable, "scripts/kaggle_setup.py", "--model", model]
    if cpu:
        cmd.append("--cpu")
    if RESCAN_NODES:
        cmd.append("--rescan-nodes")
    run(cmd)


if STAGE == "audit":
    print("audit only -- nothing else to do.")

elif STAGE == "gates":
    # CPU build on purpose: llama-eval-callback runs at -ngl 0 -c 512, so a GPU session here
    # spends an accelerator slot on a workload that never touches the GPU.
    setup(MODEL, cpu=True)

elif STAGE == "corpus":
    cmd = [sys.executable, "-m", "src.corpus.fetch", "--spec", CORPUS_SPEC]
    if TARGET_TOKENS:
        cmd += ["--target-tokens", str(TARGET_TOKENS)]
    run(cmd)

elif STAGE in ("collect", "ladder"):
    setup(MODEL, cpu=False)

    corpus = CORPUS_FILE or f"corpora/{CORPUS_SPEC}.jsonl"
    if not Path(corpus).exists():
        raise SystemExit(
            f"{corpus} does not exist. Corpora are gitignored and are NOT carried by the repo -- "
            "run a STAGE='corpus' session first and attach or rebuild its output. Collecting "
            "against a corpus built ad hoc in this session would make the traces incomparable "
            "with every other model's.")

    cmd = [sys.executable, "scripts/kaggle_collect.py", "--model", MODEL, "--corpus", corpus,
           "--corpus-name", CORPUS_SPEC]
    if GGUF_PATH:
        cmd += ["--gguf", GGUF_PATH]
    elif STAGE == "ladder":
        raise SystemExit(
            "STAGE='ladder' with GGUF_PATH=None would collect the same Q4_K_M file the baseline "
            "already used, and T8.2 would then compare a trace against itself and report perfect "
            "agreement. Attach the F16 variation and set GGUF_PATH.")
    if VARIANT_NAME is None and STAGE == "ladder":
        raise SystemExit(
            "STAGE='ladder' needs VARIANT_NAME so the manifest records which rung this is. Two "
            "precisions written under one variant name are an unanalysable mixture.")
    if SHARDS:
        cmd += ["--shards", SHARDS]
    if BACKEND:
        cmd += ["--backend", BACKEND]
    if HF_REPO_ID:
        cmd += ["--repo-id", HF_REPO_ID]
    if DECODE_MODE:
        cmd += ["--decode-mode", DECODE_MODE]
    if OVERRIDE_TENSOR:
        cmd += ["--override-tensor", OVERRIDE_TENSOR]
    if VARIANT_NAME:
        cmd += ["--variant-name", VARIANT_NAME]
    run(cmd)

print(f"\\n== {STAGE} finished in {(time.time() - T0) / 60:.1f} min")
'''

HARVEST_SRC = '''# ============================================================================
# Harvest. Everything this session learned, in a form that survives the session.
# ============================================================================
# The session writes into a clone that dies with the VM. What must come home is small and
# textual: the gate results and manifests under results/, the fields written back into
# configs/models.yaml, and the generated node specs. A git patch carries all three with their
# provenance intact, applies cleanly on the workstation, and -- unlike a push -- needs no token
# and leaves the commit to the user.
OUT = Path("/kaggle/working")
tag = f"{STAGE}-{MODEL}" if STAGE in ("gates", "collect", "ladder") else STAGE

run(["git", "add", "-AN", "."], cwd=REPO, check=False)   # include new files in the diff
# --full-index so `git apply --3way` can find the base blobs by unabbreviated sha; --binary so a
# record that turns out not to be text survives the trip instead of becoming "Binary files differ".
patch = run(["git", "diff", "--full-index", "--binary", "HEAD"], cwd=REPO, quiet=True).stdout

if patch.strip():
    p = OUT / f"session-{tag}.patch"
    p.write_text(patch, encoding="utf-8")
    print(f"wrote {p}  ({len(patch.splitlines())} lines)")
    print("\\napply on the workstation with:")
    # --3way, not a plain apply. This patch is generated on Linux against an LF working tree;
    # the workstation has core.autocrlf=true, so its working tree is CRLF while the index is LF.
    # A plain `git apply` matches context against the WORKING TREE and can fail on that mismatch
    # alone. --3way applies against the blob objects in the index instead, which are LF on both
    # machines, and degrades to a real merge with conflict markers instead of refusing outright.
    print(f"    git apply --3way session-{tag}.patch")
else:
    print("no repo changes to carry home (expected for STAGE='audit').")

# Copy the JSON/CSV records out separately too. The patch is the thing to apply, but a loose
# copy is what you want when the patch conflicts and you need to read the numbers anyway.
dest = OUT / f"results-{tag}"
if Path("results").exists():
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree("results", dest,
                    ignore=shutil.ignore_patterns("*.bin", "*.gguf", "traces", "smoke"))
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"\\ncopied {n} record files to {dest}")

# I9: /kaggle/working is ~20 GB with a ~500 file commit cap. If this session collected traces
# they are in scratch and are uploaded by the collector's own backend -- they must not be
# copied here, and a size check is cheaper than finding out at commit time.
size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2**20
files = sum(1 for f in OUT.rglob("*") if f.is_file())
print(f"\\n/kaggle/working: {files} files, {size_mb:.1f} MiB")
if files > 400 or size_mb > 4096:
    print("  WARNING: approaching the I9 commit caps (~500 files / 20 GB).")
'''


def _cell(kind: str, src: str) -> dict:
    """nbformat wants source as a list of lines that each keep their trailing newline."""
    lines = src.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build() -> dict:
    return {
        "cells": [
            _cell("markdown", HEADER_MD),
            _cell("code", PARAMS_SRC),
            _cell("code", BOOTSTRAP_SRC),
            _cell("code", AUDIT_SRC),
            _cell("code", DISPATCH_SRC),
            _cell("code", HARVEST_SRC),
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
