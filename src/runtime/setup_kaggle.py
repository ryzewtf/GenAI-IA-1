"""Session setup - plan T0.2 / T0.3, and the T1.1 fetch.

One idempotent, step-based bootstrap for a fresh Kaggle session. Run the whole thing or one step:

    !python -m src.runtime.setup_kaggle                     # every step, in order
    !python -m src.runtime.setup_kaggle --steps build       # just the llama.cpp build
    !python -m src.runtime.setup_kaggle --dry-run           # print the plan, touch nothing

Design rules, each of which is a lesson the plan already paid for:

* **Detect, never assume.** The Kaggle base image changes without notice and the accelerator, RAM,
  and internet switch are all per-session settings. Every step first checks whether its work is
  already done and skips if so, so a re-run after a killed session is cheap and safe. What the image
  actually provides is a question for `src/runtime/diagnose.py`, not for a hardcoded assumption here.
* **The build commit is pinned and non-negotiable.** `configs/run.yaml:build.llama_cpp_commit` is
  inside `run_config_sha256`, and it is load-bearing for CORRECTNESS, not just reproducibility:
  `ffn_moe_topk` is a strided view under `ggml_argsort_top_k` and a contiguous op under the older
  `ggml_top_k`. This script refuses to build a different commit rather than "helpfully" taking HEAD.
* **Patches are pinned the same way the commit is.** `build.llama_cpp_patches` names the patch
  files applied on top of the pinned tree, each with the sha256 of its own bytes, and the whole
  block is inside `run_config_sha256`. A patch is a change to the code that produces the numbers,
  so a tree that is "the pinned commit plus something" is not the pinned commit and must not hash
  as though it were. Applying is verified, idempotent, and refuses on any surprise: a patch that
  neither applies nor is already applied stops the run rather than building an unknown tree.
* **`GGML_NATIVE=OFF` is mandatory.** Kaggle does not guarantee the same host CPU between sessions,
  and a `-march=native` binary that selects different SIMD paths across sessions breaks T3.6's
  cross-session byte-identity gate. The step refuses to proceed if asked to turn it on.
* **Nothing large is written to `/kaggle/working`** (invariant I9: 20 GB and a ~500 file commit cap).
  Builds and weights go to scratch or to an attached Dataset.
* **No step silently succeeds at half its job.** Each returns a `StepResult` with an explicit
  `status`, and `main` exits non-zero if any step failed.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "configs" / "run.yaml"
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"

LLAMA_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

# Candidates for the build tree plus GGUFs, chosen BY FREE SPACE rather than by order. T0.1
# (2026-08-18) found /kaggle/temp absent from the current Kaggle image entirely while /tmp had
# 1026.8 GiB free, so a fixed preference list is exactly the thing that goes stale here. Includes a
# local path so the same script is runnable on the workstation for a dry run.
SCRATCH_CANDIDATES = ("/kaggle/temp", "/tmp")
#: Below this, a mount is the small overlay (T0.1 measured /kaggle/working at 19.5 GiB), not the
#: scratch disk. Kept well under `unhashed.preflight.min_free_gb` -- setup needs room for a build
#: tree and a couple of GGUFs, and refusing here would block work that genuinely fits.
MIN_SCRATCH_FREE_GB = 40

# Installed with --no-deps on purpose: the Kaggle image already pins torch/numpy/etc, and letting pip
# resolve transitive dependencies here is how a session ends up with a different numpy than the one
# the image's torch was built against.
PIP_PACKAGES = ("huggingface_hub", "datasets", "sentencepiece", "gguf")

STEPS = ("env", "deps", "llama", "build", "models", "verify")


class SetupError(Exception):
    """A step cannot proceed. Always fatal for that step; main decides whether to continue."""


@dataclass
class StepResult:
    name: str
    status: str  # "ok" | "skipped" | "failed" | "dry-run"
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def failed(self) -> bool:
        return self.status == "failed"


@dataclass
class SetupContext:
    scratch: Path
    dry_run: bool
    jobs: int
    cuda_arch: str
    llama_commit: str
    quant: str
    models: tuple[str, ...]
    hf_token_present: bool
    #: ``(path_relative_to_repo_root, sha256)`` per patch, in application order. Order is part of
    #: the definition: two patches touching one file do not commute.
    #:
    #: Defaults to empty because that is what an absent `build.llama_cpp_patches` means, and
    #: `_read_patch_pins` reads it the same way -- the two must agree or a config without the key
    #: would behave differently depending on which path constructed the context. What guards
    #: against an accidentally-unpatched build is the pin in run.yaml and `_apply_patches`
    #: refusing anything it cannot verify, not the absence of a default here.
    llama_patches: tuple[tuple[str, str], ...] = ()
    cuda: bool = True
    """False for a CPU session -- conversion, quantization, corpus building.

    Not cosmetic: a Kaggle CPU session has no nvcc at all, so `-DGGML_CUDA=ON` fails at configure
    time rather than falling back. It also changes :attr:`build_dir`, because a CPU and a CUDA
    build must not share one directory: CMake would reuse the cached CUDA settings and the second
    configure would fail for reasons that have nothing to do with the command that was run.
    """

    @property
    def llama_dir(self) -> Path:
        return self.scratch / "llama.cpp"

    @property
    def build_dir(self) -> Path:
        return self.scratch / "build" / (f"sm{self.cuda_arch}" if self.cuda else "cpu")

    @property
    def models_dir(self) -> Path:
        return self.scratch / "models"

    @property
    def cuda_toolkit_root(self) -> str:
        """The CUDA toolkit root, from the environment the image sets, else the conventional path.
        Passed as CUDAToolkit_ROOT so FindCUDAToolkit searches the toolkit we actually built nvcc
        from rather than guessing among several."""
        return os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"

    @property
    def cuda_driver_lib_dir(self) -> str:
        """Directory holding a linkable `libcuda.so`, put on CMAKE_LIBRARY_PATH so FindCUDAToolkit
        can DEFINE the CUDA::cuda_driver imported target that ggml-cuda links (see step_build).

        The Kaggle CUDA 12.8 image ships NO toolkit driver stub (`<root>/.../stubs/libcuda.so` is
        absent) and no `libcuda.so` on the default linker path, so FindCUDAToolkit cannot create the
        target and configure fails at generate time -- before any GPU work. The real driver library
        IS on the image, just in dirs the linker does not search by default: `/usr/local/nvidia/lib64`
        (the loaded driver) and `/usr/local/cuda/compat` (the compat driver), each with a proper
        `libcuda.so` dev symlink. Probe the known locations and return the first that has one; prefer
        the loaded driver over the compat build. Verified on the image: adding this dir makes the
        configure that otherwise errors with 'CUDA::cuda_driver ... target was not found' succeed.

        POSIX strings, not Path arithmetic: these are Linux image paths (the build runs on Kaggle);
        Path on a Windows generator/test host would join them with backslashes."""
        root = self.cuda_toolkit_root.rstrip("/")
        candidates = [
            "/usr/local/nvidia/lib64",              # the loaded driver (matches diagnose)
            f"{root}/compat",                        # CUDA 12.x forward-compat driver
            f"{root}/targets/x86_64-linux/lib/stubs",  # toolkit stub, if a future image ships one
            f"{root}/lib64/stubs",
        ]
        for d in candidates:
            if os.path.exists(f"{d}/libcuda.so"):
                return d
        return candidates[0]


# -- helpers -----------------------------------------------------------------------------------


def _run(
    cmd: Sequence[str], *, cwd: Path | None = None, timeout: float = 3600.0, dry_run: bool = False
) -> tuple[int, str]:
    """Run a command. Never shell=True: paths on these mounts contain no metacharacters, and a
    shell would only add a quoting failure mode."""
    printable = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"    [dry-run] {printable}")
        return 0, ""
    print(f"    $ {printable}")
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SetupError(f"{cmd[0]!r} not found on PATH: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SetupError(f"{printable} timed out after {timeout}s") from exc
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        # The tail, not the head: the useful part of a failed cmake/nvcc log is at the end.
        raise SetupError(f"{printable} exited {proc.returncode}\n--- last 40 lines ---\n"
                         + "\n".join(out.strip().splitlines()[-40:]))
    return proc.returncode, out


def _pick_scratch() -> Path:
    roomiest: tuple[float, str] | None = None
    for candidate in SCRATCH_CANDIDATES:
        if not os.path.isdir(candidate):
            continue
        try:
            free_gb = shutil.disk_usage(candidate).free / 2**30
        except OSError:
            continue
        if roomiest is None or free_gb > roomiest[0]:
            roomiest = (free_gb, candidate)
    if roomiest is not None:
        free_gb, candidate = roomiest
        if free_gb < MIN_SCRATCH_FREE_GB:
            # Not fatal -- a dry run or a small single-model session is legitimate -- but the
            # silent version of this is a build that dies half way through linking.
            print(f"    WARNING: roomiest scratch candidate {candidate} has only {free_gb:.1f} GiB "
                  f"free (want >= {MIN_SCRATCH_FREE_GB}). Weights and a build tree may not fit.")
        return Path(candidate) / "moe"
    # Not a Kaggle session. Keep it out of the repo so a stray build tree cannot be committed.
    return Path(os.environ.get("TEMP", "/tmp")) / "moe"


def _read_patch_pins(build: dict) -> tuple[tuple[str, str], ...]:
    """Read `build.llama_cpp_patches` as an ordered list of (path, sha256).

    Absent means none, which is the correct reading for every model but Gemma 4. A malformed entry
    is an error rather than a skip: silently ignoring a patch entry is exactly the failure this
    whole mechanism exists to prevent.
    """
    entries = build.get("llama_cpp_patches") or []
    pins: list[tuple[str, str]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "file" not in entry or "sha256" not in entry:
            raise SetupError(
                f"build.llama_cpp_patches[{i}] must be a mapping with `file` and `sha256`; got "
                f"{entry!r}. The hash is not optional -- it is what puts the patch inside "
                "run_config_sha256.")
        pins.append((str(entry["file"]), str(entry["sha256"])))
    return tuple(pins)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is in the image and in requirements
        raise SetupError("pyyaml is required to read the pinned config") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# -- steps -------------------------------------------------------------------------------------


def step_env(ctx: SetupContext) -> StepResult:
    """Report the environment and refuse the configurations that cannot work."""
    t0 = time.perf_counter()
    detail = []

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        gpus = [r.strip() for r in proc.stdout.strip().splitlines() if r.strip()]
    except FileNotFoundError:
        gpus = []

    caps = sorted({r.split(",")[1].strip().replace(".", "") for r in gpus if len(r.split(",")) > 1})
    detail.append(f"{len(gpus)} GPU(s) {gpus}")

    # A capability mismatch against the pinned cuda_arch is not a warning. platform.gpu_arch is
    # inside run_config_sha256 (I3), so building for the wrong arch produces shards that must never
    # be merged with the rest -- and nothing downstream would notice.
    #
    # Both checks are about what this session will COLLECT, so ctx.cuda gates them: a conversion or
    # corpus session compiles no kernels and emits no shards, and whatever GPU happens to be in the
    # workstation is not a fact about the experiment. Refusing there would only teach the operator
    # to edit the pin, which is the one thing I3 exists to prevent.
    if ctx.cuda:
        if caps and ctx.cuda_arch not in caps:
            raise SetupError(
                f"pinned cuda_architectures={ctx.cuda_arch} but this session's GPU(s) report "
                f"{caps}. Either switch the accelerator or change configs/run.yaml deliberately -- "
                "platform.gpu_arch is inside run_config_sha256 (invariant I3), so a mismatch makes "
                "these shards a different experiment."
            )
        if len(caps) > 1:
            raise SetupError(f"mixed GPU architectures {caps}; kernel selection differs per arch")
    elif caps and ctx.cuda_arch not in caps:
        detail.append(f"CPU session; ignoring sm_{ctx.cuda_arch} pin vs local {caps}")

    free_gb = shutil.disk_usage(ctx.scratch.parent if ctx.scratch.parent.exists() else "/").free / 2**30
    detail.append(f"{free_gb:.1f} GiB free on scratch")

    if not ctx.dry_run:
        ctx.scratch.mkdir(parents=True, exist_ok=True)

    return StepResult("env", "dry-run" if ctx.dry_run else "ok", "; ".join(detail),
                      {"gpus": gpus, "compute_caps": caps, "scratch_free_gb": round(free_gb, 1)},
                      time.perf_counter() - t0)


def step_deps(ctx: SetupContext) -> StepResult:
    """Install only what the image lacks, with --no-deps."""
    t0 = time.perf_counter()
    missing = []
    for pkg in PIP_PACKAGES:
        module = {"huggingface_hub": "huggingface_hub", "datasets": "datasets",
                  "sentencepiece": "sentencepiece", "gguf": "gguf"}[pkg]
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return StepResult("deps", "skipped", "all present", {"missing": []},
                          time.perf_counter() - t0)
    _run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", *missing],
         dry_run=ctx.dry_run, timeout=900)
    return StepResult("deps", "dry-run" if ctx.dry_run else "ok", f"installed {missing}",
                      {"missing": missing}, time.perf_counter() - t0)


def step_llama(ctx: SetupContext) -> StepResult:
    """Fetch llama.cpp at the pinned commit, and only that commit."""
    t0 = time.perf_counter()
    target = ctx.llama_commit

    if (ctx.llama_dir / ".git").is_dir():
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ctx.llama_dir),
                              capture_output=True, text=True, check=False)
        head = proc.stdout.strip()
        if head == target:
            # The tree is at the right commit, but that says nothing about patches: a prior model
            # in the same session with no patches of its own left HEAD here without applying this
            # model's. _apply_patches is idempotent (it reverse-checks before applying), so run it
            # on this path too rather than returning a tree that is the commit minus the patch.
            applied = _apply_patches(ctx)
            detail = f"already at {head[:12]}"
            if applied:
                detail += f" +{len(applied)} patch(es)"
            return StepResult("llama", "skipped", detail,
                              {"commit": head, "patches": applied}, time.perf_counter() - t0)
        # Re-point rather than re-clone: the fetch is the expensive part.
        _run(["git", "fetch", "--depth", "1", "origin", target], cwd=ctx.llama_dir,
             dry_run=ctx.dry_run)
        _run(["git", "checkout", "--detach", target], cwd=ctx.llama_dir, dry_run=ctx.dry_run)
    else:
        # A shallow clone cannot check out an arbitrary commit, so init + fetch that one object.
        if not ctx.dry_run:
            ctx.llama_dir.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q"], cwd=ctx.llama_dir, dry_run=ctx.dry_run)
        _run(["git", "remote", "add", "origin", LLAMA_REPO_URL], cwd=ctx.llama_dir,
             dry_run=ctx.dry_run)
        _run(["git", "fetch", "--depth", "1", "origin", target], cwd=ctx.llama_dir,
             dry_run=ctx.dry_run, timeout=1800)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=ctx.llama_dir, dry_run=ctx.dry_run)

    applied = _apply_patches(ctx)

    detail = f"at {target[:12]}"
    if applied:
        detail += f" +{len(applied)} patch(es)"
    return StepResult("llama", "dry-run" if ctx.dry_run else "ok", detail,
                      {"commit": target, "dir": str(ctx.llama_dir), "patches": applied},
                      time.perf_counter() - t0)


def _apply_patches(ctx: SetupContext) -> list[dict[str, str]]:
    """Apply `build.llama_cpp_patches` to the checked-out tree, verifying each one first.

    Three properties, each of which is a way this could go wrong quietly:

    * **The patch bytes are checked against the sha256 in run.yaml before anything is applied.**
      That sha is inside `run_config_sha256`, so an edited patch file changes the run config hash
      and every manifest written afterwards says so. Without the check, editing the patch would
      change the binary while the recorded hash kept insisting nothing had moved.
    * **Idempotent, because `step_llama` is.** A re-run after a killed session re-enters here with
      the patch already applied. `git apply --reverse --check` succeeding is the reliable test for
      that, and it is checked BEFORE attempting to apply, so the normal resume path is a skip
      rather than a failure that has to be interpreted.
    * **A patch that neither applies nor is already applied is fatal.** The tempting fallback is to
      warn and build anyway; that produces a binary that is missing a capture node, and the first
      symptom is a node the harness cannot find in a session that has already spent its setup time.
    """
    if not ctx.llama_patches:
        return []

    applied: list[dict[str, str]] = []
    for rel, want_sha in ctx.llama_patches:
        patch_path = REPO_ROOT / rel
        if not patch_path.exists():
            raise SetupError(
                f"{rel} is listed in build.llama_cpp_patches but does not exist. The patch is part "
                "of the build definition, not an optional extra -- a tree without it is a "
                "different tree than run_config_sha256 claims.")
        blob = patch_path.read_bytes()
        got_sha = hashlib.sha256(blob).hexdigest()
        if got_sha != want_sha:
            raise SetupError(
                f"{rel} hashes {got_sha[:16]} but run.yaml pins {want_sha[:16]}. Either the patch "
                "was edited without updating the pin -- in which case every manifest since is "
                "wrong about what produced it -- or this is the wrong file. Fix the pin "
                "deliberately; do not paste the new hash in to make the message go away.")

        if ctx.dry_run:
            applied.append({"patch": rel, "sha256": got_sha, "status": "dry-run"})
            continue

        # Already applied? Then the reverse patch is what applies cleanly.
        reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_path)],
                                 cwd=str(ctx.llama_dir), capture_output=True, text=True)
        if reverse.returncode == 0:
            applied.append({"patch": rel, "sha256": got_sha, "status": "already-applied"})
            continue

        forward = subprocess.run(["git", "apply", "--check", str(patch_path)],
                                 cwd=str(ctx.llama_dir), capture_output=True, text=True)
        if forward.returncode != 0:
            raise SetupError(
                f"{rel} neither applies to nor is already applied to {ctx.llama_dir}: "
                f"{forward.stderr.strip()}. This tree is not the tree the patch was cut against, "
                "so building it would produce a binary nothing in the config describes.")

        _run(["git", "apply", str(patch_path)], cwd=ctx.llama_dir, dry_run=False)
        applied.append({"patch": rel, "sha256": got_sha, "status": "applied"})
        print(f"  applied {rel}")

    return applied


def built_target(ctx: SetupContext, target: str) -> Path | None:
    """Where `target` landed, or None. moe_trace sits at the build root; upstream tools in bin/."""
    for candidate in (ctx.build_dir / target, ctx.build_dir / f"{target}.exe",
                      ctx.build_dir / "bin" / target, ctx.build_dir / "bin" / f"{target}.exe"):
        if candidate.exists():
            return candidate
    return None


def step_build(ctx: SetupContext, *, targets: Sequence[str] = ("moe_trace",)) -> StepResult:
    """Configure and build the requested targets for the pinned architecture.

    `targets` exists because a CPU session wants `llama-quantize` and nothing else: building
    moe_trace there would spend minutes on a binary that cannot run without a GPU, and -- worse --
    the old unconditional `--target moe_trace` meant llama-quantize was never built at all, so the
    conversion script looked for a file the build had not been asked to produce.
    """
    t0 = time.perf_counter()
    binary = ctx.build_dir / targets[0]

    existing = {t: built_target(ctx, t) for t in targets}
    if all(existing.values()):
        return StepResult("build", "skipped", f"{sorted(targets)} present",
                          {"binary": str(existing[targets[0]])}, time.perf_counter() - t0)
    if not ctx.llama_dir.joinpath("include", "llama.h").exists() and not ctx.dry_run:
        raise SetupError(f"no llama.cpp source at {ctx.llama_dir}; run the 'llama' step first")

    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    _run(
        [
            "cmake", "-B", str(ctx.build_dir), "-S", str(REPO_ROOT / "src" / "capture"),
            *generator,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLLAMA_CPP_DIR={ctx.llama_dir}",
            # A CPU session has no CUDA Toolkit, so ON is not a harmless over-request: ggml-cuda's
            # CMakeLists raises "CUDA Toolkit not found" and configure stops.
            #
            # CUDAToolkit_ROOT + a dir holding libcuda.so on CMAKE_LIBRARY_PATH: on the Kaggle CUDA
            # 12.8 image FindCUDAToolkit finds nvcc and the headers but fails to DEFINE the
            # CUDA::cuda_driver imported target that ggml-cuda/CMakeLists.txt links, because there is
            # no toolkit driver stub and no libcuda.so on the default linker path. The target is only
            # created when the library is found, so configure fails at generate, before any GPU work.
            # The real driver library is on the image under a dir the linker does not search by
            # default; cuda_driver_lib_dir finds it. Verified on the image: this turns the failing
            # configure into rc 0.
            *(["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={ctx.cuda_arch}",
               f"-DCUDAToolkit_ROOT={ctx.cuda_toolkit_root}",
               f"-DCMAKE_LIBRARY_PATH={ctx.cuda_driver_lib_dir}"]
              if ctx.cuda else ["-DGGML_CUDA=OFF"]),
            # Mandatory, see the module docstring: a native build breaks T3.6 across sessions.
            "-DGGML_NATIVE=OFF",
            "-DGGML_CUDA_F16=OFF",
            # T1.4 needs llama-eval-callback and T0.5 needs llama-bench. Both are gated upstream on
            # LLAMA_BUILD_COMMON, which defaults OFF for a subdirectory build -- our CMakeLists
            # turns the group on together, which is what MOE_TRACE_UPSTREAM_TOOLS controls.
            "-DMOE_TRACE_UPSTREAM_TOOLS=ON",
        ],
        dry_run=ctx.dry_run, timeout=1800,
    )
    build_cmd = ["cmake", "--build", str(ctx.build_dir)]
    for target in targets:
        build_cmd += ["--target", target]
    _run([*build_cmd, "-j", str(ctx.jobs)], dry_run=ctx.dry_run, timeout=5400)

    return StepResult("build", "dry-run" if ctx.dry_run else "ok", str(binary),
                      {"binary": str(binary), "cuda_arch": ctx.cuda_arch},
                      time.perf_counter() - t0)


def step_models(ctx: SetupContext) -> StepResult:
    """Download the GGUFs named in models.yaml (T1.1), skipping any already present.

    Deliberately last and deliberately skippable: it is the slowest step, it is the one most likely
    to be replaced by an attached Kaggle Dataset, and nothing else here depends on it.
    """
    t0 = time.perf_counter()
    config = _load_yaml(MODELS_CONFIG)["models"]
    wanted = ctx.models or tuple(config)

    unresolved, planned, present = [], [], []
    for key in wanted:
        entry = config.get(key)
        if entry is None:
            raise SetupError(f"unknown model key {key!r}; have {sorted(config)}")
        gguf = entry.get("gguf") or {}
        repo, fname = gguf.get("repo"), gguf.get("file")
        if not repo or not fname:
            # T1.1 fills these in. Guessing a filename here would download the wrong quant silently.
            unresolved.append(key)
            continue
        dest = ctx.models_dir / key / fname
        (present if dest.exists() else planned).append((key, repo, fname, dest))

    if unresolved:
        print(f"    NOTE: no gguf.repo/gguf.file recorded for {unresolved} -- T1.1 fills these in; "
              "not guessing a filename.")

    for key, repo, fname, dest in planned:
        if not ctx.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["hf", "download", repo, fname, "--local-dir", str(dest.parent)],
             dry_run=ctx.dry_run, timeout=5400)

    status = "dry-run" if ctx.dry_run else ("skipped" if not planned else "ok")
    return StepResult(
        "models", status,
        f"{len(present)} present, {len(planned)} fetched, {len(unresolved)} unresolved",
        {"present": [k for k, *_ in present], "fetched": [k for k, *_ in planned],
         "unresolved": unresolved},
        time.perf_counter() - t0,
    )


def step_verify(ctx: SetupContext) -> StepResult:
    """Prove the built binary runs and the repo's own tests pass before any quota is spent."""
    t0 = time.perf_counter()
    checks = {}

    binary = next((p for p in (ctx.build_dir / "moe_trace", ctx.build_dir / "moe_trace.exe")
                   if p.exists()), None)
    if binary is None:
        checks["binary"] = "absent"
    else:
        # No args must be a clean usage error (rc=2), not a crash: that exercises argv parsing and
        # proves the dynamic loader found every dependency, which is the failure that otherwise
        # shows up only once a model is loading.
        proc = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
        checks["binary"] = f"rc={proc.returncode} ({'ok' if proc.returncode == 2 else 'UNEXPECTED'})"

    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                          cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    tail = (proc.stdout or "").strip().splitlines()
    checks["pytest"] = tail[-1] if tail else "no output"

    failed = proc.returncode != 0 or checks.get("binary", "").endswith("UNEXPECTED)")
    return StepResult("verify", "failed" if failed else "ok", json.dumps(checks), checks,
                      time.perf_counter() - t0)


_HANDLERS: dict[str, Callable[[SetupContext], StepResult]] = {
    "env": step_env,
    "deps": step_deps,
    "llama": step_llama,
    "build": step_build,
    "models": step_models,
    "verify": step_verify,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a session (T0.2 / T0.3 / T1.1).")
    parser.add_argument("--steps", nargs="*", default=list(STEPS), choices=list(STEPS))
    parser.add_argument("--scratch", default=None, help="override the scratch root")
    parser.add_argument("--models", nargs="*", default=(), help="model keys; default is all")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=None, help="write the step report as JSON")
    parser.add_argument("--keep-going", action="store_true",
                        help="run later steps even after one fails")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_config = _load_yaml(RUN_CONFIG)
    build = run_config["hashed"]["build"]
    commit = build.get("llama_cpp_commit")
    if not commit:
        # assert_collection_ready() would refuse this later; refusing now saves the whole build.
        print("setup: configs/run.yaml has build.llama_cpp_commit: null. Pin it before building - "
              "the commit is inside run_config_sha256 and is load-bearing for correctness.",
              file=sys.stderr)
        return 2
    if build.get("ggml_native"):
        print("setup: build.ggml_native is true. A -march=native binary is tied to the host CPU "
              "that compiled it, and Kaggle does not guarantee the same CPU between sessions, "
              "which breaks T3.6's cross-session byte-identity gate.", file=sys.stderr)
        return 2

    ctx = SetupContext(
        scratch=Path(args.scratch) if args.scratch else _pick_scratch(),
        dry_run=args.dry_run,
        jobs=args.jobs,
        cuda_arch=str(build.get("cuda_architectures", "75")),
        llama_commit=str(commit),
        llama_patches=_read_patch_pins(build),
        quant=str(_load_yaml(MODELS_CONFIG)["defaults"]["quant"]),
        models=tuple(args.models),
        hf_token_present=bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
    )

    print(f"scratch={ctx.scratch}  arch=sm_{ctx.cuda_arch}  commit={ctx.llama_commit[:12]}  "
          f"jobs={ctx.jobs}  hf_token={'set' if ctx.hf_token_present else 'unset'}")

    results: list[StepResult] = []
    for name in args.steps:
        print(f"\n== {name}")
        try:
            result = _HANDLERS[name](ctx)
        except SetupError as exc:
            result = StepResult(name, "failed", str(exc))
        results.append(result)
        print(f"  -> {result.status}: {result.detail[:400]}")
        if result.failed and not args.keep_going:
            print(f"\nsetup: stopping at failed step {name!r} (use --keep-going to continue)",
                  file=sys.stderr)
            break

    if args.report:
        Path(args.report).write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )

    failures = [r.name for r in results if r.failed]
    print(f"\n{len(results)} step(s) run; " + (f"FAILED: {failures}" if failures else "all ok"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
