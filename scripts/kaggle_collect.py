#!/usr/bin/env python3
"""Load the model and collect traces — the second half of a Kaggle session.

    !python scripts/kaggle_collect.py --model qwen3-30b-a3b --corpus corpora/mixed_v1.jsonl

Assumes :mod:`scripts.kaggle_setup` has already run: a CUDA ``moe_trace``, the GGUF on scratch, a
verified node spec.

**This script finds things and checks things; it does not re-implement the shard loop.** The loop
lives in :mod:`src.runtime.runner` — resumable, ledgered, byte-exact on re-collection (T3.6) — and
this delegates to its CLI rather than assembling `plan_shards` / `ShardState` / a backend by hand.
A second assembly of that machinery would be a second place for the S.3 contract to drift, and the
contract is the reason a killed session is recoverable at all.

What it adds is the session-shaped wrapper:

* **Resolves paths.** The GGUF is wherever setup put it — an attached Kaggle Dataset under
  ``/kaggle/input``, or the scratch models dir. The binary is under ``<scratch>/build/sm75``.
  Neither location is worth remembering at 2am.
* **Refuses an unverified node spec.** A spec that has not passed T1.4 is the llama.cpp-master
  *hypothesis*, not a result. Capturing against a wrong node produces a full-size, plausible,
  wrong trace (invariant I13) — the failure mode with no downstream symptom.
* **Preflight before the model loads** (T0.6 plus T0.1's findings): free space, sustained write
  throughput, and an HF token when the upload backend is ``hf``. Kaggle Secrets are opt-in, so
  "unset" is the default state, and finding out after twelve hours of GPU quota loses the run —
  scratch does not outlive the session (invariant I9).
* **Sets ``GGML_CUDA_DISABLE_FUSION=1``** for every capture run. The CUDA backend fuses
  softmax+argsort+top-k into one kernel and the capture design needs ``ffn_moe_topk`` to be a
  materialised tensor the callback can read. (``runner.capture_env`` sets this too; it is exported
  here as well so a manual ``moe_trace`` invocation in the same session inherits it.)
* **Runs T5.3 on what was just collected**, while the session still has time to react.

The variant flags collect the *candidate* leg of a Phase 3 gate: ``--decode-mode`` for T3.8,
``--override-tensor`` for T3.7. Compare against a baseline collection with::

    python -m src.analysis.precision --mode T3.8 \\
        --reference-root <baseline traces> --candidate-root <variant traces> \\
        --model <key> --corpus <name>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capture.nodespec import parse_spec  # noqa: E402
from src.runtime import runner as runner_mod  # noqa: E402
from src.runtime.config import RunConfig  # noqa: E402
from src.runtime.preflight import PreflightError, check_upload_credentials  # noqa: E402
from src.runtime.preflight import from_config as run_preflight_from_config  # noqa: E402
from src.runtime.runner import RunnerError, load_model_meta  # noqa: E402

MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"
RUN_CONFIG = REPO_ROOT / "configs" / "run.yaml"
SPEC_DIR = REPO_ROOT / "configs" / "nodes"
RESULTS = REPO_ROOT / "results"


def _say(message: str) -> None:
    print(message, flush=True)


def find_gguf(model_key: str, filename: str, scratch: Path) -> Path:
    """Locate the GGUF wherever setup left it, preferring an attached dataset.

    Preference order matters: an attached Kaggle Dataset is immutable and costs no session time,
    while a scratch copy is whatever this session downloaded. If both exist they should be the
    same bytes — and if they are not, the SHA256 in the manifest is what will say so.
    """
    inputs = Path("/kaggle/input")
    candidates: list[Path] = []
    if inputs.is_dir():
        candidates += sorted(inputs.rglob(filename))
    candidates.append(scratch / "models" / model_key / filename)
    if (scratch / "models").is_dir():
        candidates += sorted((scratch / "models").rglob(filename))
    candidates.append(REPO_ROOT / "models" / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RunnerError(
        f"cannot find {filename}. Run `python scripts/kaggle_setup.py --model {model_key}` first, "
        "or pass --gguf explicitly."
    )


def find_binary(scratch: Path, arch: str, explicit: Path | None) -> Path:
    if explicit:
        if not explicit.exists():
            raise RunnerError(f"--binary {explicit} does not exist")
        return explicit
    for candidate in (
        scratch / "build" / f"sm{arch}" / "moe_trace",
        scratch / "build" / f"sm{arch}" / "moe_trace.exe",
        REPO_ROOT / "build" / "cpu" / "moe_trace.exe",
        REPO_ROOT / "build" / "cpu" / "moe_trace",
    ):
        if candidate.exists():
            return candidate
    raise RunnerError(
        f"no moe_trace binary under {scratch / 'build' / f'sm{arch}'}. Run scripts/kaggle_setup.py."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect traces for one model (T5.2)")
    parser.add_argument("--model", required=True, help="model key in configs/models.yaml")
    parser.add_argument("--corpus", required=True, type=Path, help="corpus JSONL (T4.2)")
    parser.add_argument("--corpus-name", default=None)
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--shards", default=None, help="restrict to '0-19' or '3,7,11'")
    parser.add_argument("--backend", choices=("local", "hf"), default=None,
                        help="default: unhashed.upload.backend")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo for --backend hf")
    parser.add_argument("--local-root", type=Path, default=None,
                        help="--backend local target; default <scratch>/traces")
    parser.add_argument("--preflight-gb", type=float, default=2.0, help="0 disables (T0.6)")
    parser.add_argument("--timeout-s", type=float, default=None, help="per-shard subprocess cap")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan the shards and print the command lines; load nothing")
    parser.add_argument("--skip-validate", action="store_true", help="skip T5.3 afterwards")
    # --- Phase 3 variant legs -------------------------------------------------------------
    parser.add_argument("--variant-name", default="baseline")
    parser.add_argument("--decode-mode", choices=("off", "full", "tail"), default="off",
                        help="T3.8 candidate leg")
    parser.add_argument("--decode-prefix", type=int, default=512)
    parser.add_argument("--override-tensor", default=None,
                        help=r"T3.7 candidate leg, e.g. 'blk\.\d+\.ffn_gate_inp\.weight=CPU'")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = RunConfig.load(RUN_CONFIG)
    unhashed = config.unhashed or {}
    scratch = args.scratch or Path((unhashed.get("paths") or {}).get("scratch", "/tmp/moe"))
    arch = str(config.hashed["build"].get("cuda_architectures", "75"))
    corpus_name = args.corpus_name or args.corpus.stem
    backend = args.backend or str((unhashed.get("upload") or {}).get("backend", "local"))

    try:
        meta = load_model_meta(MODELS_CONFIG, args.model)
        gguf_meta = meta.get("gguf") or {}
        if not gguf_meta.get("file"):
            raise RunnerError(
                f"{args.model} has no gguf.file in configs/models.yaml (T1.1). Nothing to load."
            )
        gguf = args.gguf or find_gguf(args.model, str(gguf_meta["file"]), scratch)
        binary = find_binary(scratch, arch, args.binary)

        spec_path = args.spec or (SPEC_DIR / f"{args.model}.spec")
        if not spec_path.exists():
            raise RunnerError(
                f"no node spec at {spec_path}. T1.4 produces it — run scripts/kaggle_setup.py. "
                "moe_trace cannot capture without one, and a spec naming the wrong node produces "
                "a full-size, plausible, wrong trace (invariant I13)."
            )
        spec = parse_spec(spec_path.read_text(encoding="utf-8"))
        if not spec.verified:
            raise RunnerError(
                f"{spec_path} is UNVERIFIED — the llama.cpp-master hypothesis, not a T1.4 result. "
                f"Run `python scripts/kaggle_setup.py --model {args.model} --rescan-nodes` to "
                "confirm it against this checkpoint before collecting."
            )
    except (RunnerError, KeyError) as exc:
        _say(f"COLLECT FAILED: {exc}")
        return 2

    _say(f"model    : {args.model}")
    _say(f"gguf     : {gguf}")
    _say(f"spec     : {spec_path}  ({spec.n_moe_layers} MoE layers, T1.4-verified)")
    _say(f"binary   : {binary}")
    _say(f"corpus   : {args.corpus}  (as '{corpus_name}')")
    _say(f"scratch  : {scratch}")
    _say(f"backend  : {backend}")
    _say(f"config   : run_config_sha256 {config.sha256[:16]}")
    if args.decode_mode != "off" or args.override_tensor:
        _say(f"VARIANT  : {args.variant_name}  decode={args.decode_mode} "
             f"override_tensor={args.override_tensor!r}")
        _say("           This is a Phase 3 gate leg, NOT a collection shard. It is recorded in "
             "every manifest so it cannot be merged with baseline traces by accident.")

    if not args.dry_run:
        try:
            check_upload_credentials(config, backend=backend)
            if args.preflight_gb > 0:
                _say("\n== preflight (T0.6)")
                result = run_preflight_from_config(config, report_path=RESULTS / "preflight.json")
                _say(f"  {result.write_mbps:.0f} MB/s sustained, "
                     f"{result.free_gb_before:.0f} GB free at {result.scratch_path}")
        except PreflightError as exc:
            _say(f"\nPREFLIGHT FAILED: {exc}")
            return 2

    # Exported so a manual moe_trace invocation in this same session inherits it too.
    if config.inference.get("disable_cuda_fusion", True):
        os.environ["GGML_CUDA_DISABLE_FUSION"] = "1"

    runner_argv = [
        "--model", args.model,
        "--model-path", str(gguf),
        "--corpus", str(args.corpus),
        "--corpus-name", corpus_name,
        "--spec", str(spec_path),
        "--binary", str(binary),
        "--models", str(MODELS_CONFIG),
        "--run-config", str(RUN_CONFIG),
        "--scratch", str(scratch),
        "--backend", backend,
        "--preflight-gb", "0",          # already run above, before the model was located
        "--variant-name", args.variant_name,
        "--decode-mode", args.decode_mode,
        "--decode-prefix", str(args.decode_prefix),
    ]
    if backend == "hf":
        if not args.repo_id:
            _say("--backend hf requires --repo-id")
            return 2
        runner_argv += ["--repo-id", args.repo_id]
    else:
        runner_argv += ["--local-root", str(args.local_root or (scratch / "traces"))]
    if args.override_tensor:
        runner_argv += ["--override-tensor", args.override_tensor]
    if args.shards:
        runner_argv += ["--shards", args.shards]
    if args.timeout_s:
        runner_argv += ["--timeout-s", str(args.timeout_s)]
    if args.dry_run:
        runner_argv.append("--dry-run")

    session = unhashed.get("session") or {}
    _say(f"\n== collecting (session cap {int(session.get('wall_limit_s', 43200)) / 3600:.0f} h, "
         f"reserve {int(session.get('reserve_s', 1800)) / 60:.0f} min)")

    started = time.perf_counter()
    rc = runner_mod.main(runner_argv)
    _say(f"\n== collection returned {rc} after {(time.perf_counter() - started) / 60:.1f} min")
    if rc != 0 or args.dry_run or args.skip_validate:
        return rc

    # LocalDirBackend writes under its root at the SAME remote prefix the hf backend uses,
    # i.e. "traces/<model>/<corpus>", so the local tree carries that segment too. Resolved by
    # searching rather than by assuming, because guessing wrong here silently skips T5.3 -- and a
    # skipped validation reads exactly like a passed one in the session log.
    local_root = Path(args.local_root or (scratch / "traces"))
    trace_dir = next(
        (d for d in (local_root / "traces" / args.model / corpus_name,
                     local_root / args.model / corpus_name) if d.is_dir()),
        local_root / "traces" / args.model / corpus_name,
    )
    if backend == "hf":
        _say("\n== T5.3 skipped: shards were uploaded and deleted from scratch (I9). Validate "
             "them from the Hub copy in a CPU session.")
        return 0
    if not trace_dir.is_dir():
        _say(f"\n== T5.3 skipped: no local trace dir at {trace_dir}")
        return 0

    _say(f"\n== T5.3 validation of {trace_dir}")
    proc = subprocess.run(
        [sys.executable, "-m", "src.traces.validate", str(trace_dir),
         "--corpus", str(args.corpus),
         "--json", str(RESULTS / f"validate_{args.model}_{corpus_name}.json")],
        cwd=str(REPO_ROOT), check=False,
    )
    if proc.returncode != 0:
        _say("\nT5.3 reported problems above. Do not analyse these shards until they are "
             "explained — a trace that fails validation is not a partial result.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
