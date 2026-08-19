#!/usr/bin/env python3
"""Convert the whole panel from HF safetensors to GGUF with ONE recipe.

    !python scripts/kaggle_convert.py --model olmoe-0924
    !python scripts/kaggle_convert.py --all --prune

Run this in a **CPU session**. Conversion is CPU- and I/O-bound, so it costs zero GPU quota, and
the 30 h/week GPU budget (Gate Q1) is the binding constraint on this study — spending any of it on
a job that never touches a GPU is the most expensive mistake available here.

Why convert at all, when five of the seven have a published GGUF
----------------------------------------------------------------
Because a published GGUF is somebody else's conversion. The panel currently mixes a first-party
allenai build, two `unsloth` builds, one `mradermacher` build and one `wantsleep` build, each made
at an unknown llama.cpp commit with unknown flags. Every cross-model number this study reports
would then be a comparison across five converters as well as across five models, and there is no
way afterwards to tell those two apart. Converting all seven at the pinned commit
(`hashed.build.llama_cpp_commit`) with identical flags removes that confound completely.

It also buys three concrete things that were otherwise unavailable:

* **Pair A becomes collectable.** `olmoe-0924` and `olmoe-0125` publish no GGUF at all, and Pair A
  is the panel's *only* strictly controlled pair (§1.5) — the one comparison where F1 conditions on
  an identical tokenizer. Without conversion it is not a missing nicety, it is a missing arm.
* **GPT-OSS stops being a requantization.** The converter detects `quant_method: mxfp4` on the
  source and promotes the file type to `MOSTLY_MXFP4_MOE`, repacking the 4-bit codes losslessly
  (`conversion/base.py::repack_mxfp4_blocks`). That is exactly §0b's "native MXFP4, do not
  requantize", reached automatically — so this script simply does not quantize that checkpoint.
* **T1.2 passes by construction.** See below; this turned out not to need a flag.

The router stays F32 without being asked
----------------------------------------
Checked in the pinned tree rather than assumed, because the first plausible recipe was wrong:

* `conversion/base.py` forces `FFN_GATE_INP` and `FFN_GATE_INP_SHEXP` to F32 **regardless of
  `--outtype`**. So the F16 intermediate already holds an F32 router.
* `src/llama-quant.cpp:308` refuses to quantize any tensor whose name contains
  `ffn_gate_inp.weight`, and `llama_tensor_get_type` returns `tensor->type` for such a tensor
  *before* it consults `--tensor-type` overrides. The router therefore inherits the F16 file's
  dtype, which the previous point has already fixed at F32.

The consequence worth writing down: `--tensor-type ffn_gate_inp=f32` is **inert** — it is skipped
by the early return, not applied. Passing it would have produced a run that looked like it enforced
the §1.6 precondition while enforcing nothing. T1.2 (`router_audit`) is run here on the finished
artifact for that reason: the guarantee is a property of two files in llama.cpp, and this script
verifies it rather than trusting this comment.

Precision
---------
"Same precision" across the panel means **the same recipe**, not F16 at collection time. F16
inference does not fit: Qwen3-30B is ~57 GiB and Gemma-26B ~52 GiB in F16 against 30 GiB of total
VRAM on 2xT4. So every checkpoint takes the identical path

    safetensors --(convert_hf_to_gguf.py --outtype f16)--> F16 --(llama-quantize Q4_K_M)--> Q4_K_M

with GPT-OSS diverging only where the source format forces it. The F16 intermediate is kept for
`--keep-f16` models because T8.1's precision ladder needs an uncompressed reference.

Disk
----
Peak is source + F16 + Q4_K_M, worst case Qwen3 at ~131 GiB. T0.1 measured 1026.8 GiB free on
`/tmp`, which is where `_pick_scratch()` lands. `--prune` drops the safetensors and (unless
`--keep-f16`) the F16 as soon as the next stage has consumed them, which keeps a whole-panel run
inside the largest single model's peak.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capture.nodescan import set_model_fields  # noqa: E402
from src.runtime.setup_kaggle import (  # noqa: E402
    SetupContext,
    built_target,
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
RESULTS = REPO_ROOT / "results"

_HASH_CHUNK = 8 * 1024 * 1024

#: Never needed for routing, and each one is a large download. `original/` and `*.pth` are
#: duplicate weights in a second format; `*.gguf` would fetch somebody else's conversion, which is
#: the exact thing this script exists to stop using.
DOWNLOAD_EXCLUDE = ("*.pth", "*.bin", "original/*", "*.gguf", "*.onnx")

#: Panel order for `--all`: cheapest first, so a session that dies late still leaves the small
#: checkpoints converted, and Pair A (the only controlled pair) is completed before anything else.
PANEL_ORDER = (
    "olmoe-0924",
    "olmoe-0125",
    "olmoe-0125-instruct",
    "deepseek-v2-lite",
    "gpt-oss-20b",
    "gemma-4-26b-a4b",
    "qwen3-30b-a3b",
)


class ConvertError(RuntimeError):
    """A conversion step failed. Always fatal: a half-converted GGUF is not a usable artifact."""


def find_llama_tree(ctx: SetupContext) -> Path:
    """The llama.cpp tree whose converter this run will use.

    The session checkout wins over the vendored one whenever it exists. Both are supposed to be the
    pinned commit, but only one of them is *verifiable* here -- ``step_llama`` fetched it by sha and
    it still has its ``.git``, whereas the vendored copy is a plain directory whose provenance is
    whatever the last person to update it made it. So the checkout is preferred and checked, and the
    vendored tree is the fallback for a workstation run where no session checkout exists.

    An existing-but-empty ``llama_dir`` is an error rather than a reason to fall back. In clones
    made before the vendored tree was committed, ``.vendor/llama_cpp_pull`` was a gitlink with no
    ``.gitmodules``, so quietly reaching for it is how a run ends up converting with a converter
    nobody chose -- and a converter at another commit is another recipe, which is the exact confound
    this script exists to remove.
    """
    session = ctx.llama_dir
    if session.exists():
        if not (session / "convert_hf_to_gguf.py").exists():
            raise ConvertError(
                f"{session} exists but holds no convert_hf_to_gguf.py. In a clone made before the "
                "vendored tree was committed, .vendor/llama_cpp_pull is an empty gitlink; run the "
                "'llama' step so the pinned commit is fetched into the session checkout."
            )
        proc = subprocess.run(["git", "-C", str(session), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
        head = (getattr(proc, "stdout", "") or "").strip()
        # A tree that cannot answer (no .git, so no output) is accepted; a tree that ANSWERS with
        # the wrong commit is not.
        if head and head != ctx.llama_commit:
            raise ConvertError(
                f"{session} is at {head[:12]} but configs/run.yaml pins {ctx.llama_commit[:12]}. "
                "Converting there would produce a GGUF this project cannot describe."
            )
        return session

    vendored = REPO_ROOT / ".vendor" / "llama_cpp_pull"
    if (vendored / "convert_hf_to_gguf.py").exists():
        return vendored
    raise ConvertError(
        f"no llama.cpp tree with a converter: neither {session} nor {vendored}. In a clone made "
        "before the vendored tree was committed the latter is an empty gitlink; run the 'llama' "
        "step to fetch the pinned commit."
    )



def _say(message: str) -> None:
    print(message, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 2**30


# -- planning --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionPlan:
    """What one model's conversion will do, decided before anything is downloaded."""

    model_key: str
    hf_repo: str
    outtype: str
    quantize_to: str | None
    """``None`` means "ship the converter's output as-is" -- GPT-OSS's native MXFP4."""
    reason: str

    @property
    def requantizes(self) -> bool:
        return self.quantize_to is not None


def plan_conversion(model_key: str, meta: dict[str, Any], *, outtype: str = "f16") -> ConversionPlan:
    """Decide the recipe for one model from its config alone.

    Split out from the doing so the panel-wide decision can be read (and tested) without a
    checkpoint on disk: `--dry-run` prints exactly this table.
    """
    repo = meta.get("hf_repo")
    if not repo:
        raise ConvertError(
            f"{model_key} has no hf_repo in configs/models.yaml. There is nothing to convert from, "
            "and guessing a repo id would silently convert a different checkpoint."
        )

    quant = str(meta.get("quant", "Q4_K_M"))
    if quant.upper() == "MXFP4":
        # Not a special case this script invents: the converter reads `quant_method: mxfp4` off the
        # source config and promotes the file type itself, losslessly. Quantizing on top of that
        # would be a lossy transform of an already-lossy format, which §0b forbids by name.
        return ConversionPlan(
            model_key=model_key,
            hf_repo=str(repo),
            outtype=outtype,
            quantize_to=None,
            reason="source is MXFP4; converter repacks it natively (plan §0b: do not requantize)",
        )
    return ConversionPlan(
        model_key=model_key,
        hf_repo=str(repo),
        outtype=outtype,
        quantize_to=quant,
        reason=f"panel recipe: {outtype} intermediate -> {quant}",
    )


def resolve_model(model_key: str, models_path: Path = MODELS_CONFIG) -> dict[str, Any]:
    raw = _load_yaml(models_path)
    models = raw.get("models") or {}
    if model_key not in models:
        raise ConvertError(f"unknown model {model_key!r}; have {sorted(models)}")
    meta = dict(raw.get("defaults") or {})
    meta.update(models[model_key] or {})
    return meta


# -- steps -----------------------------------------------------------------------------------


def _run(cmd: Sequence[str], *, what: str, cwd: Path | None = None) -> None:
    _say(f"    $ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=False)
    if proc.returncode != 0:
        raise ConvertError(f"{what} exited {proc.returncode}")


def download_source(plan: ConversionPlan, dest: Path) -> Path:
    """Fetch the safetensors repo. Idempotent: `hf download` skips files already present."""
    if (dest / "config.json").exists():
        _say(f"  source already present: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    _say(f"  downloading {plan.hf_repo} -> {dest}  ({_free_gb(dest):.0f} GiB free)")
    # One --exclude followed by every pattern: the flag is nargs="*", so repeating it would keep
    # only the last group and quietly re-enable the duplicate-format downloads this list exists to
    # skip.
    cmd = ["hf", "download", plan.hf_repo, "--local-dir", str(dest), "--exclude", *DOWNLOAD_EXCLUDE]
    try:
        _run(cmd, what=f"hf download {plan.hf_repo}")
    except ConvertError as exc:
        raise ConvertError(
            f"{exc}\nIf this was 401/403, the repo is gated and needs an accepted licence plus "
            "HF_TOKEN in the session. google/gemma-4-26B-A4B is the one gated repo in this panel; "
            "the others were checked and are open."
        ) from None
    if not (dest / "config.json").exists():
        raise ConvertError(
            f"{dest} has no config.json after download. convert_hf_to_gguf.py reads the model "
            "class from it, so this is not a warning."
        )
    return dest


def convert_to_gguf(plan: ConversionPlan, source: Path | str, outfile: Path, *,
                    llama_tree: Path, remote: bool) -> Path:
    """Run the pinned `convert_hf_to_gguf.py`. This is the step the whole script exists for."""
    if outfile.exists():
        _say(f"  {outfile.name} already converted ({outfile.stat().st_size / 2**30:.1f} GiB)")
        return outfile
    converter = llama_tree / "convert_hf_to_gguf.py"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(converter), str(source),
           "--outfile", str(outfile), "--outtype", plan.outtype]
    if remote:
        # Streams the safetensors straight from the Hub. Saves the source copy entirely (~57 GiB
        # for Qwen3) at the cost of needing the network for the whole conversion.
        cmd.append("--remote")
    # cwd is the tree itself: the converter does `from conversion import ...`, a sibling package,
    # and `import gguf` from the gguf-py it inserts on sys.path relative to its own location.
    _run(cmd, what="convert_hf_to_gguf.py", cwd=llama_tree)
    if not outfile.exists():
        raise ConvertError(f"converter reported success but {outfile} does not exist")
    return outfile


def quantize(plan: ConversionPlan, f16: Path, outfile: Path, binary: Path, *, jobs: int) -> Path:
    """`llama-quantize` at the pinned commit.

    No `--tensor-type ffn_gate_inp=f32`: see the module docstring. That override is unreachable for
    the router, and passing it would advertise a guarantee it does not provide. The real guarantee
    is verified afterwards by `verify_router_dtype`.
    """
    if outfile.exists():
        _say(f"  {outfile.name} already quantized ({outfile.stat().st_size / 2**30:.1f} GiB)")
        return outfile
    assert plan.quantize_to is not None
    _run([str(binary), str(f16), str(outfile), plan.quantize_to, str(jobs)],
         what=f"llama-quantize {plan.quantize_to}")
    if not outfile.exists():
        raise ConvertError(f"llama-quantize reported success but {outfile} does not exist")
    return outfile


def verify_router_dtype(model_key: str, gguf_path: Path) -> None:
    """T1.2, run on the artifact this script just produced.

    Not a formality here even though the recipe is supposed to make it automatic: that "supposed
    to" rests on two separate files in llama.cpp agreeing, and this is the cheap check that they
    still do at the pinned commit. §1.6 reads the k-th/(k+1)-th margin off the router logits, so a
    quantized router makes every Family B number an artifact of the quantization.
    """
    _say("  T1.2 router dtype audit")
    out = RESULTS / f"router_dtype_audit_{model_key}.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "src.capture.router_audit", str(gguf_path), "-o", str(out)],
        cwd=str(REPO_ROOT), check=False,
    )
    if proc.returncode != 0:
        raise ConvertError(
            f"T1.2 FAILED on the freshly converted {gguf_path.name} (exit {proc.returncode}). "
            "That is a finding about the pinned llama.cpp, not about this checkpoint: the "
            "converter is supposed to force ffn_gate_inp to F32 and the quantizer is supposed to "
            "skip it. Do not collect until it is explained."
        )


# -- record ----------------------------------------------------------------------------------


@dataclass
class ConversionRecord:
    model_key: str
    plan: ConversionPlan
    gguf_path: Path
    size_bytes: int
    sha256: str
    llama_commit: str
    f16_path: Path | None = None
    f16_sha256: str | None = None
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_key,
            "hf_repo": self.plan.hf_repo,
            "outtype": self.plan.outtype,
            "quantize_to": self.plan.quantize_to,
            "recipe_reason": self.plan.reason,
            # The provenance chain the manifests need: a shard manifest records only the GGUF's
            # sha256, and this file is what maps that hash back to a source repo and a converter.
            "llama_cpp_commit": self.llama_commit,
            "converter": "convert_hf_to_gguf.py",
            "file": self.gguf_path.name,
            "path": str(self.gguf_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "f16_path": str(self.f16_path) if self.f16_path else None,
            "f16_sha256": self.f16_sha256,
            "elapsed_s": round(self.elapsed_s, 1),
            "notes": self.notes,
        }

    def yaml_block(self) -> str:
        """The `gguf:` mapping to paste (or `--write`) into configs/models.yaml.

        `repo: null` is deliberate and true: this artifact came from no GGUF repo. The collector
        resolves such a model by filename from an attached Kaggle Dataset, which is also the
        immutable-artifact property T3.6's cross-session byte-identity gate wants.
        """
        return (
            f"    gguf: {{repo: null,\n"
            f"           file: {self.gguf_path.name},\n"
            f"           sha256: {self.sha256},\n"
            f"           size_bytes: {self.size_bytes}}}"
        )


def write_record(record: ConversionRecord) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"convert_{record.model_key}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def register(record: ConversionRecord, *, force: bool) -> list[str]:
    """Record the converted artifact in configs/models.yaml."""
    return set_model_fields(
        MODELS_CONFIG,
        record.model_key,
        {
            "gguf": (
                f"{{repo: null, file: {record.gguf_path.name}, "
                f"sha256: {record.sha256}, size_bytes: {record.size_bytes}}}"
            )
        },
        force=force,
    )


# -- driver ----------------------------------------------------------------------------------


def convert_one(
    model_key: str,
    ctx: SetupContext,
    args: argparse.Namespace,
    *,
    quantize_binary: Path,
    llama_tree: Path,
) -> ConversionRecord:
    meta = resolve_model(model_key)
    plan = plan_conversion(model_key, meta, outtype=args.outtype)
    started = time.perf_counter()

    _say(f"\n{'=' * 78}\n== {model_key}: {plan.reason}\n{'=' * 78}")

    source_dir = ctx.scratch / "hf" / model_key
    gguf_dir = ctx.scratch / "gguf"
    keep_f16 = model_key in set(args.keep_f16)

    if args.remote:
        source: Path | str = plan.hf_repo
        _say(f"  reading {plan.hf_repo} remotely (--remote); no local safetensors copy")
    else:
        source = download_source(plan, source_dir)

    f16 = convert_to_gguf(
        plan, source, gguf_dir / f"{model_key}-{plan.outtype.upper()}.gguf",
        llama_tree=llama_tree, remote=args.remote,
    )

    if args.prune and not args.remote and source_dir.exists():
        _say(f"  pruning {source_dir} ({_free_gb(ctx.scratch):.0f} GiB free before)")
        shutil.rmtree(source_dir, ignore_errors=True)

    notes: list[str] = []
    if plan.requantizes:
        final = quantize(
            plan, f16, gguf_dir / f"{model_key}-{plan.quantize_to}.gguf",
            quantize_binary, jobs=args.jobs,
        )
    else:
        final = f16
        keep_f16 = True  # it IS the artifact
        notes.append(
            "not quantized: source is MXFP4 and the converter repacked it losslessly (§0b)"
        )

    verify_router_dtype(model_key, final)

    size = final.stat().st_size
    _say(f"  hashing {size / 2**30:.1f} GiB ...")
    digest = _sha256(final)
    _say(f"  sha256 {digest}")

    record = ConversionRecord(
        model_key=model_key,
        plan=plan,
        gguf_path=final,
        size_bytes=size,
        sha256=digest,
        llama_commit=ctx.llama_commit,
        f16_path=f16 if keep_f16 else None,
        elapsed_s=time.perf_counter() - started,
        notes=notes,
    )

    if args.prune and not keep_f16 and f16 != final and f16.exists():
        _say(f"  pruning {f16.name}")
        f16.unlink()
    elif keep_f16 and f16 != final:
        notes.append("F16 kept for the T8.1 precision ladder")

    out = write_record(record)
    _say(f"  wrote {out}")
    _say(f"  {model_key} done in {record.elapsed_s / 60:.1f} min -> {final}")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert the panel to GGUF with one recipe")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="model key in configs/models.yaml")
    target.add_argument("--all", action="store_true", help=f"convert, in order: {', '.join(PANEL_ORDER)}")
    parser.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32"],
                        help="intermediate precision; MXFP4 sources ignore it (default: f16)")
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 4))
    parser.add_argument("--remote", action="store_true",
                        help="stream safetensors from the Hub instead of downloading them")
    parser.add_argument("--prune", action="store_true",
                        help="delete the safetensors and the F16 once consumed")
    parser.add_argument("--keep-f16", action="append", default=["olmoe-0125"],
                        help="models whose F16 to keep (T8.1 ladder); repeatable")
    parser.add_argument("--skip-build", action="store_true", help="reuse an existing build tree")
    parser.add_argument("--write", action="store_true",
                        help="record the result in configs/models.yaml")
    parser.add_argument("--force", action="store_true",
                        help="with --write, overwrite a DIFFERENT recorded gguf block")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the recipe table and exit; downloads nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    keys = list(PANEL_ORDER) if args.all else [args.model]

    if args.dry_run:
        _say(f"{'model':22s} {'outtype':8s} {'quantize':10s} reason")
        for key in keys:
            try:
                plan = plan_conversion(key, resolve_model(key), outtype=args.outtype)
            except ConvertError as exc:
                _say(f"{key:22s} {'-':8s} {'-':10s} CANNOT: {exc}")
                continue
            _say(f"{key:22s} {plan.outtype:8s} {str(plan.quantize_to or 'none'):10s} {plan.reason}")
        return 0

    run_config = _load_yaml(RUN_CONFIG)
    build = run_config["hashed"]["build"]
    commit = build.get("llama_cpp_commit")
    if not commit:
        _say("configs/run.yaml has build.llama_cpp_commit: null — pin it before converting.")
        return 2

    scratch = Path(args.scratch) if args.scratch else _pick_scratch()
    ctx = SetupContext(
        scratch=scratch,
        dry_run=False,
        jobs=args.jobs,
        cuda_arch=str(build.get("cuda_architectures", "75")),
        llama_commit=str(commit),
        quant=str(_load_yaml(MODELS_CONFIG)["defaults"]["quant"]),
        models=tuple(keys),
        hf_token_present=bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        # This session quantizes; it never infers. A Kaggle CPU session has no nvcc at all, so
        # asking for CUDA does not degrade gracefully -- it stops at configure time.
        cuda=False,
    )

    _say(f"models  : {', '.join(keys)}")
    _say(f"scratch : {scratch}  ({_free_gb(scratch):.0f} GiB free)")
    _say(f"commit  : {ctx.llama_commit[:12]}")
    _say(f"hf token: {'set' if ctx.hf_token_present else 'UNSET (gated repos will 401)'}")

    try:
        for step in (step_env, step_deps, step_llama):
            _say(f"\n== {step.__name__.replace('step_', '')}")
            result = step(ctx)
            _say(f"  -> {result.status}: {result.detail[:300]}")
            if result.failed:
                raise ConvertError(result.detail)

        if args.skip_build:
            _say("\n== build (skipped)")
        else:
            # CPU build: this session quantizes, it does not infer. A CUDA build here would cost
            # ten minutes and buy nothing -- llama-quantize is CPU-only.
            _say("\n== build (CPU, llama-quantize only)")
            # Only llama-quantize: moe_trace cannot run in a session with no GPU anyway, and
            # building it would spend minutes of a 4-vCPU box on a binary nothing here invokes.
            result = step_build(ctx, targets=("llama-quantize",))
            _say(f"  -> {result.status}: {result.detail[:300]}")
            if result.failed:
                raise ConvertError(result.detail)

        quantize_binary = built_target(ctx, "llama-quantize")
        if quantize_binary is None:
            raise ConvertError(
                f"llama-quantize not found under {ctx.build_dir}. It comes from LLAMA_BUILD_TOOLS, "
                "which src/capture/CMakeLists.txt ties to MOE_TRACE_UPSTREAM_TOOLS (ON by "
                "default). A build tree that predates that flag needs deleting and rebuilding."
            )

        llama_tree = find_llama_tree(ctx)
        _say(f"\nconverter: {llama_tree / 'convert_hf_to_gguf.py'}")

        records: list[ConversionRecord] = []
        for key in keys:
            records.append(convert_one(key, ctx, args, quantize_binary=quantize_binary,
                                       llama_tree=llama_tree))

        if args.write:
            _say("\n== recording in configs/models.yaml")
            for record in records:
                changes = register(record, force=args.force)
                _say(f"  {record.model_key}: {changes or 'unchanged'}")

    except ConvertError as exc:
        _say(f"\nCONVERSION FAILED: {exc}")
        return 1

    _say("\n" + "=" * 78)
    _say("CONVERTED:")
    for record in records:
        _say(f"  {record.model_key:22s} {record.size_bytes / 2**30:6.1f} GiB  {record.sha256[:16]}…")
    if not args.write:
        _say("\nPaste into configs/models.yaml (or re-run with --write):")
        for record in records:
            _say(f"\n  {record.model_key}:\n{record.yaml_block()}")
    _say("\nNext: upload " + str(ctx.scratch / "gguf") + " as a Kaggle Dataset, then")
    _say("  !python scripts/kaggle_setup.py --model <key>   (in a GPU session)")
    _say("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
