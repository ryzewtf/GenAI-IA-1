#!/usr/bin/env python3
"""One command to make a fresh Kaggle session ready to collect one model.

    !python scripts/kaggle_setup.py --model qwen3-30b-a3b

Env, deps, pinned llama.cpp checkout, CUDA build, GGUF download, SHA256, node spec. Idempotent:
every step checks whether its work is already done, so re-running after a killed session is cheap
and safe. The heavy steps are the CUDA build (~10-20 min once per session) and the download.

This is a thin, opinionated driver over :mod:`src.runtime.setup_kaggle` plus the two Phase 1 gates
that must pass before a single token is collected. It exists because the general script has six
steps and eight flags, and the thing you actually want on Kaggle is "get me ready to run THIS
model" — with the Kaggle-specific defaults already chosen rather than remembered.

What it adds over the general script, each for a measured reason:

* **Scratch is chosen by free space.** T0.1 found ``/kaggle/temp`` does not exist on the current
  image, and ``/kaggle/working`` is 19.5 GiB with a ~500-file commit cap (invariant I9). ``/tmp``
  had 1026.8 GiB free. A GGUF must never land on the working mount.
* **``HF_HUB_DISABLE_XET=1`` by default.** HF's Xet transfer path is the default in current
  ``huggingface_hub`` and there are open reports of it hanging at 0% or 99% specifically inside
  managed notebooks including Kaggle. The plain HTTPS path is slower in theory and finishes in
  practice. Its known failure mode is files over ~50 GB; the largest here is 17.3 GiB.
* **An attached Kaggle Dataset wins over downloading.** If the GGUF is already mounted read-only
  under ``/kaggle/input``, use it: it costs no session time, no bandwidth, and a pinned dataset
  version is *immutable*, which is what T3.6's cross-session byte-identity gate actually wants.
  A re-download from the Hub every session is a trace whose weights could change under it.
* **SHA256 is computed from the bytes on this disk, never copied from the Hub or from
  ``models.yaml``.** It is written into every shard manifest as the claim about which weights
  produced the trace. If it disagrees with a previously recorded hash the run stops: that is a
  real finding about the artifact, not a formality to wave through.
* **T1.2 and T1.4 run here, not at collection time.** The router-dtype audit (T1.2) is a gate —
  §1.6 requires an F32 router, and a quantized one invalidates the margins the whole study reads.
  Node discovery (T1.4) needs ``llama-eval-callback`` and produces the ``.spec`` without which
  ``moe_trace`` cannot start. Both are cheap and both are better discovered now than after the
  model is loaded.

Then run :mod:`scripts.kaggle_collect`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime.setup_kaggle import (  # noqa: E402
    MIN_SCRATCH_FREE_GB,
    SetupContext,
    SetupError,
    _load_yaml,
    _pick_scratch,
    step_build,
    step_deps,
    step_env,
    step_llama,
)

MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"
RUN_CONFIG = REPO_ROOT / "configs" / "run.yaml"
SPEC_DIR = REPO_ROOT / "configs" / "nodes"
RESULTS = REPO_ROOT / "results"

#: Read in 8 MB blocks: big enough that the syscall overhead vanishes, small enough that hashing a
#: 17 GB GGUF never holds more than 8 MB resident.
_HASH_CHUNK = 8 * 1024 * 1024


def _say(message: str) -> None:
    print(message, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def find_attached_gguf(filename: str) -> Path | None:
    """Look for ``filename`` anywhere under ``/kaggle/input``.

    Searched by basename rather than by an expected dataset slug, because whoever built the
    dataset chose the slug and the folder layout, and requiring them to match a string in this
    file would make a correct dataset look like a missing one.
    """
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for candidate in root.rglob(filename):
        if candidate.is_file():
            return candidate
    return None


def resolve_model(model_key: str) -> dict[str, Any]:
    raw = _load_yaml(MODELS_CONFIG)
    models = raw.get("models") or {}
    if model_key not in models:
        raise SetupError(f"unknown model {model_key!r}; have {sorted(models)}")
    meta = dict(raw.get("defaults") or {})
    meta.update(models[model_key] or {})
    gguf = meta.get("gguf") or {}
    if not gguf.get("file"):
        raise SetupError(
            f"{model_key} has no gguf.file in configs/models.yaml. T1.1 fills it in; guessing a "
            "filename here would silently fetch the wrong quantization."
        )
    # `repo: null` with a filename is not a half-filled entry -- it is what
    # `scripts/kaggle_convert.py --write` records for a GGUF this project converted itself, which
    # by definition lives in no HF repo. Such a model must be found on disk (an attached Kaggle
    # Dataset or the scratch mount); `acquire_gguf` enforces that rather than downloading.
    return meta


def acquire_gguf(
    model_key: str, meta: dict[str, Any], ctx: SetupContext, *, verify_hash: bool = True
) -> Path:
    """Return a local path to the model's GGUF, downloading only if it is not already here."""
    gguf = meta["gguf"]
    filename = str(gguf["file"])
    # NOT str(...): `str(None)` is "None", which is truthy, and would send a converted model down
    # the download path to fetch a repo literally named None.
    repo = gguf.get("repo") or ""

    attached = find_attached_gguf(filename)
    if attached is not None:
        _say(f"  using attached Kaggle Dataset copy: {attached}")
        path = attached
    else:
        dest_dir = ctx.models_dir / model_key
        path = dest_dir / filename
        if path.exists():
            _say(f"  already downloaded: {path}")
        elif not repo:
            scratch_copy = next(iter((ctx.scratch / "gguf").glob(filename)), None)
            if scratch_copy is None:
                raise SetupError(
                    f"{model_key} records gguf.repo: null, meaning this GGUF was converted by "
                    f"scripts/kaggle_convert.py rather than downloaded -- so {filename} has to be "
                    "on disk already. It is not under /kaggle/input, "
                    f"{dest_dir}, or {ctx.scratch / 'gguf'}. Either attach the Kaggle Dataset "
                    "holding the converted panel, or re-run the conversion in a CPU session."
                )
            _say(f"  using converted copy from this session: {scratch_copy}")
            path = scratch_copy
        else:
            free_gb = shutil.disk_usage(ctx.scratch).free / 2**30
            _say(f"  downloading {repo}/{filename} -> {dest_dir}  ({free_gb:.0f} GiB free)")
            dest_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["hf", "download", repo, filename, "--local-dir", str(dest_dir)],
                check=False,
            )
            if proc.returncode != 0:
                raise SetupError(
                    f"hf download {repo} {filename} exited {proc.returncode}. If it hung rather "
                    "than failed, this is very likely the Xet transfer path — HF_HUB_DISABLE_XET "
                    "is set by this script, so check it survived into the child environment."
                )
            if not path.exists():
                # `hf download` mirrors the repo layout, so the file may be one level down.
                found = next(iter(dest_dir.rglob(filename)), None)
                if found is None:
                    raise SetupError(f"{filename} not found under {dest_dir} after download")
                path = found

    size = path.stat().st_size
    declared_size = gguf.get("size_bytes")
    if declared_size and int(declared_size) != size:
        raise SetupError(
            f"{path} is {size} bytes but models.yaml records {declared_size}. These are different "
            "artifacts; a trace collected from this file would carry a manifest that describes "
            "another one."
        )

    if not verify_hash:
        _say(f"  size {size:,} bytes; SHA256 skipped (--skip-hash)")
        return path

    _say(f"  hashing {size / 2**30:.1f} GiB ...")
    started = time.perf_counter()
    digest = _sha256(path)
    _say(f"  sha256 {digest}  ({time.perf_counter() - started:.0f}s)")

    declared = gguf.get("sha256")
    if declared and declared != digest:
        raise SetupError(
            f"SHA256 MISMATCH for {path}.\n  models.yaml: {declared}\n  this file:   {digest}\n"
            "The bytes differ from the ones a previous run recorded. Stopping: this is a real "
            "finding about the artifact (a re-uploaded file, a different quantization, a truncated "
            "download), not a formality. Do not collect until it is explained."
        )
    if not declared:
        _say(f"  NOTE: models.yaml has no sha256 for {model_key}. Record it:")
        _say(f"        gguf.sha256: {digest}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"gguf_sha256_{model_key}.json").write_text(
        json.dumps(
            {"model": model_key, "repo": repo, "file": filename, "path": str(path),
             "size_bytes": size, "sha256": digest, "source": "attached" if attached else "download"},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def run_router_audit(model_key: str, gguf_path: Path) -> None:
    """T1.2 — the router must be F32. This is a gate, not a report."""
    _say("\n== T1.2 router dtype audit")
    out = RESULTS / "router_dtype_audit.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "src.capture.router_audit", str(gguf_path), "-o", str(out)],
        cwd=str(REPO_ROOT), check=False,
    )
    if proc.returncode != 0:
        raise SetupError(
            f"T1.2 FAILED for {model_key} (exit {proc.returncode}). §1.6 reads margins off the "
            "router logits; a quantized router makes the k-th/(k+1)-th gap an artifact of the "
            "quantization rather than a property of the model, and every Family B number "
            "downstream inherits it. Do not collect."
        )


def run_nodescan(model_key: str, gguf_path: Path, ctx: SetupContext, *, force: bool) -> Path:
    """T1.4 — discover the real node names, then write the .spec moe_trace needs."""
    spec_path = SPEC_DIR / f"{model_key}.spec"
    if spec_path.exists() and not force:
        _say(f"\n== T1.4 node spec already present: {spec_path}")
        return spec_path

    _say("\n== T1.4 node discovery (llama-eval-callback)")
    eval_cb = next(
        (p for p in (ctx.build_dir / "bin" / "llama-eval-callback",
                     ctx.build_dir / "bin" / "llama-eval-callback.exe",
                     ctx.build_dir / "llama-eval-callback") if p.exists()),
        None,
    )
    if eval_cb is None:
        raise SetupError(
            f"llama-eval-callback not found under {ctx.build_dir}. The build step turns it on via "
            "-DMOE_TRACE_UPSTREAM_TOOLS=ON; if the build was skipped as already-done, it may "
            "predate that flag. Delete the build dir and re-run --steps build."
        )

    log = RESULTS / f"nodescan_{model_key}.log"
    proc = subprocess.run(
        [sys.executable, "-m", "src.capture.nodescan", model_key,
         "--run", "--gguf", str(gguf_path), "--binary", str(eval_cb),
         "--models", str(MODELS_CONFIG), "--save-log", str(log), "--write"],
        cwd=str(REPO_ROOT), check=False,
    )
    if proc.returncode != 0:
        raise SetupError(
            f"T1.4 FAILED for {model_key} (exit {proc.returncode}); dump at {log}. Without a "
            "verified node spec moe_trace has nothing to capture — and a spec that names the "
            "wrong node produces a full-size, plausible, wrong trace (invariant I13)."
        )

    proc = subprocess.run(
        [sys.executable, "-m", "src.capture.nodespec", "--model", model_key,
         "--models", str(MODELS_CONFIG), "--out-dir", str(SPEC_DIR)],
        cwd=str(REPO_ROOT), check=False,
    )
    if proc.returncode != 0:
        raise SetupError(f"writing {spec_path} failed (exit {proc.returncode})")
    _say(f"  wrote {spec_path}")
    return spec_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a Kaggle session for one model")
    parser.add_argument("--model", required=True, help="model key in configs/models.yaml")
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 4))
    parser.add_argument("--skip-build", action="store_true", help="reuse an existing build tree")
    parser.add_argument("--skip-hash", action="store_true",
                        help="skip the SHA256 (only for a throwaway smoke run)")
    parser.add_argument("--skip-gates", action="store_true",
                        help="skip T1.2/T1.4; the collector will refuse to start without them")
    parser.add_argument("--rescan-nodes", action="store_true", help="redo T1.4 even if a spec exists")
    parser.add_argument("--cpu", action="store_true", help="build without CUDA (CPU-only session)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Set before anything imports huggingface_hub or spawns `hf`.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    run_config = _load_yaml(RUN_CONFIG)
    build = run_config["hashed"]["build"]
    commit = build.get("llama_cpp_commit")
    if not commit:
        _say("configs/run.yaml has build.llama_cpp_commit: null — pin it before building.")
        return 2

    scratch = Path(args.scratch) if args.scratch else _pick_scratch()
    ctx = SetupContext(
        scratch=scratch,
        dry_run=False,
        jobs=args.jobs,
        cuda_arch=str(build.get("cuda_architectures", "75")),
        llama_commit=str(commit),
        quant=str(_load_yaml(MODELS_CONFIG)["defaults"]["quant"]),
        models=(args.model,),
        hf_token_present=bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
    )

    free_gb = shutil.disk_usage(scratch.parent if scratch.parent.exists() else "/").free / 2**30
    _say(f"model    : {args.model}")
    _say(f"scratch  : {scratch}  ({free_gb:.0f} GiB free, want >= {MIN_SCRATCH_FREE_GB})")
    _say(f"arch     : sm_{ctx.cuda_arch}{' (CUDA OFF)' if args.cpu else ''}")
    _say(f"commit   : {ctx.llama_commit[:12]}")
    _say(f"hf token : {'set' if ctx.hf_token_present else 'UNSET'}")

    try:
        for step in (step_env, step_deps, step_llama):
            _say(f"\n== {step.__name__.replace('step_', '')}")
            result = step(ctx)
            _say(f"  -> {result.status}: {result.detail[:300]}")
            if result.failed:
                raise SetupError(result.detail)

        if args.skip_build:
            _say("\n== build (skipped)")
        else:
            _say("\n== build (this is the slow one)")
            if args.cpu:
                # Only for a CPU session used to prepare artifacts; collection needs CUDA.
                os.environ["MOE_TRACE_FORCE_CPU"] = "1"
            result = step_build(ctx)
            _say(f"  -> {result.status}: {result.detail[:300]}")
            if result.failed:
                raise SetupError(result.detail)

        _say(f"\n== model: {args.model}")
        meta = resolve_model(args.model)
        gguf_path = acquire_gguf(args.model, meta, ctx, verify_hash=not args.skip_hash)

        spec_path = SPEC_DIR / f"{args.model}.spec"
        if args.skip_gates:
            _say("\n== T1.2 / T1.4 skipped (--skip-gates)")
        else:
            run_router_audit(args.model, gguf_path)
            spec_path = run_nodescan(args.model, gguf_path, ctx, force=args.rescan_nodes)

    except SetupError as exc:
        _say(f"\nSETUP FAILED: {exc}")
        return 1

    binary = next(
        (p for p in (ctx.build_dir / "moe_trace", ctx.build_dir / "moe_trace.exe") if p.exists()),
        ctx.build_dir / "moe_trace",
    )
    _say("\n" + "=" * 78)
    _say("READY. Next:")
    _say(f"  !python scripts/kaggle_collect.py --model {args.model} \\")
    _say(f"      --gguf {gguf_path} \\")
    _say(f"      --spec {spec_path} \\")
    _say(f"      --binary {binary} \\")
    _say("      --corpus corpora/mixed_v1.jsonl")
    _say("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
