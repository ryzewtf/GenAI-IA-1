"""Runtime audit - plan T0.1.

Prints one report describing the machine this is running on. Standard library only, no third-party
imports, no writes outside a temp probe, no model loads: it is meant to run as the very first cell
of a fresh Kaggle session, *before* anything has been installed, and to be safe to paste anywhere.

    python -m src.runtime.diagnose          # from the repo
    !python diagnose.py                     # or paste the file into a Kaggle cell

Every question it answers is one the plan later depends on:

  * GPU count, name, compute capability, VRAM  -> run.yaml platform.gpu_arch / n_devices, and
    whether ``tensor_split: [0.5, 0.5]`` describes reality (T0.1, I3)
  * host RAM                                    -> T6.1's 4 GB analysis budget and O1 (15 vs 32 GB)
  * free space and inode headroom per mount     -> T0.6 preflight, I9 (20 GB / ~500 file cap on
    /kaggle/working), and which mount should hold traces
  * measured write throughput                   -> preflight.min_write_mbps
  * toolchain presence and versions             -> T0.2 (nvcc/cmake/ninja/gcc for the llama.cpp
    build) and T0.3 (requirements.lock)
  * outbound network                            -> whether a session can git-pull and reach HF, or
    whether everything must arrive as an attached Dataset
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

SEP = "=" * 78
# Mounts worth reporting on Kaggle plus the local equivalents, so one script serves both hosts.
CANDIDATE_MOUNTS = ("/kaggle/working", "/kaggle/temp", "/kaggle/input", "/tmp", ".")
# Packages the plan actually imports. Absence is information, not an error.
PACKAGES = (
    "numpy", "scipy", "yaml", "pytest", "torch", "transformers",
    "huggingface_hub", "safetensors", "gguf", "triton", "psutil",
    # T3.2's GPT-OSS path turns on these two together: plan C11 corrects the claim that
    # transformers gates MXFP4 at compute capability >= 9.0 (it gates at >= (7, 5), and a T4 is
    # exactly (7, 5)), leaving Triton >= 3.4 AND the `kernels` package as the real requirement.
    # Without them transformers dequantizes to bf16 -- ~40 GB, which will not fit. The first
    # T0.1 run reported triton 3.6.0 but never checked `kernels`, so the question stayed open.
    "kernels", "accelerate",
    # Named in setup_kaggle.PIP_PACKAGES; knowing whether the image already has them tells you
    # what the deps step will actually do.
    "datasets", "sentencepiece",
)
TOOLS = (
    ("nvcc", ("nvcc", "--version")),
    ("cmake", ("cmake", "--version")),
    ("ninja", ("ninja", "--version")),
    ("gcc", ("gcc", "--version")),
    ("g++", ("g++", "--version")),
    ("git", ("git", "--version")),
    ("hf", ("hf", "version")),
)
PROBE_BYTES = 256 * 1024 * 1024  # 256 MB: enough to see past page cache, small enough to be quick


def _run(cmd: tuple[str, ...] | list[str], timeout: float = 30.0) -> tuple[int, str]:
    """Run a command, never raise. Returns (rc, combined output); rc 127 means not found."""
    try:
        p = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, check=False
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except OSError as exc:  # permissions, exec format, ...
        return 126, f"{type(exc).__name__}: {exc}"


def head(title: str) -> None:
    print(f"\n{SEP}\n== {title}\n{SEP}")


def section_host() -> None:
    head("HOST")
    print(f"platform      : {platform.platform()}")
    print(f"machine/arch  : {platform.machine()}  ({platform.processor() or 'processor unknown'})")
    print(f"python        : {sys.version.split()[0]}  ({sys.executable})")
    print(f"cwd           : {os.getcwd()}")

    # os.cpu_count() is the machine's core count; the *usable* count is what matters under a cgroup.
    total = os.cpu_count()
    try:
        usable = len(os.sched_getaffinity(0))  # Linux only
    except AttributeError:
        usable = total
    print(f"cpu           : {usable} usable of {total} total")

    # run.yaml pins n_threads: 4 to Kaggle's vCPU count; flag any mismatch rather than assume.
    if usable is not None and usable != 4:
        print(f"  NOTE: run.yaml pins inference.n_threads=4; this host offers {usable}")

    ram = None
    try:  # Linux: the only reliable source without psutil
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    print(f"ram available : {int(line.split()[1]) * 1024 / 2**30:.1f} GiB")
    except OSError:
        pass
    if ram is None:
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            pass
    if ram is None and sys.platform == "win32":
        # Only reached on the dev workstation. Kept because the same 4 GB analysis budget applies
        # when probes are run locally, and "unknown" is a worse answer than one ctypes call.
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = int(status.ullTotalPhys)
                print(f"ram available : {status.ullAvailPhys / 2**30:.1f} GiB")
        except Exception:  # ctypes on a non-standard build; the report continues either way
            pass
    print(f"ram total     : {ram / 2**30:.1f} GiB" if ram else "ram total     : unknown")
    if ram and ram / 2**30 < 20:
        print("  NOTE: under 20 GiB - plan O1 / T6.1 assume a 4 GB analysis budget fits here")

    # cgroup limits bind before physical RAM does, and a Kaggle OOM kill looks like a crash.
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw and raw != "max":
                print(f"cgroup limit  : {int(raw) / 2**30:.1f} GiB  ({path})")
            break
        except (OSError, ValueError):
            continue


def section_gpu() -> None:
    head("GPU")
    rc, out = _run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version,persistence_mode",
            "--format=csv,noheader",
        )
    )
    if rc != 0:
        print(f"nvidia-smi unavailable (rc={rc}): {out.strip()[:400]}")
        print("  -> no GPU visible. On Kaggle: Settings > Accelerator must be set before the")
        print("     session starts; switching it restarts the VM and clears /kaggle/temp.")
        return

    rows = [r for r in out.strip().splitlines() if r.strip()]
    print(f"{len(rows)} device(s):")
    caps = set()
    for row in rows:
        print(f"  {row.strip()}")
        parts = [p.strip() for p in row.split(",")]
        if len(parts) > 2:
            caps.add(parts[2])

    # platform.gpu_arch is inside run_config_sha256 (I3): the wrong value silently merges two
    # different experiments, so surface the mapping rather than leaving it to be inferred.
    if caps:
        arches = sorted(c.replace(".", "") for c in caps)
        print(f"\ncompute cap -> run.yaml platform.gpu_arch: {', '.join(arches)}")
        if len(arches) > 1:
            print("  WARNING: mixed architectures. Kernel selection differs per arch - collect on")
            print("           one arch only, or the shards are not comparable.")
    print(f"platform.n_devices should read: {len(rows)}")
    if len(rows) != 2:
        print("  NOTE: run.yaml pins tensor_split: [0.5, 0.5], which assumes exactly 2 devices")

    rc, out = _run(("nvidia-smi", "--query", "--display=COMPUTE"), timeout=20)
    if rc == 0:
        for line in out.splitlines():
            if "Compute Mode" in line or "MIG" in line:
                print(line.strip())


def section_mounts() -> None:
    head("STORAGE")
    seen = set()
    for mount in CANDIDATE_MOUNTS:
        if not os.path.isdir(mount):
            # Say so rather than skipping. The first T0.1 run silently omitted /kaggle/temp,
            # which was this repo's configured `unhashed.paths.scratch` at the time; an absent
            # line reads as "not applicable" when it actually meant "your scratch path is gone",
            # and preflight would have created the directory on the small overlay and passed.
            print(f"{mount:17s} ABSENT")
            continue
        real = os.path.realpath(mount)
        if real in seen:
            print(f"{mount:17s} -> same filesystem as an entry above ({real})")
            continue
        seen.add(real)
        try:
            usage = shutil.disk_usage(mount)
        except OSError as exc:
            print(f"{mount:17s} unreadable: {exc}")
            continue
        print(
            f"{mount:17s} {usage.free / 2**30:7.1f} GiB free of {usage.total / 2**30:7.1f} GiB"
            f"   (realpath {real})"
        )
        # I9: /kaggle/working is output-committed and file-count limited; traces must not live there.
        if mount == "/kaggle/working":
            try:
                n = sum(len(files) for _, _, files in os.walk(mount))
                print(f"{'':17s} {n} files present; the output commit caps at roughly 500")
            except OSError:
                pass

    print("\nKaggle input datasets:")
    if os.path.isdir("/kaggle/input"):
        entries = sorted(os.listdir("/kaggle/input"))
        print(f"  {entries if entries else 'none attached'}")
        for entry in entries:
            path = os.path.join("/kaggle/input", entry)
            try:
                files = sorted(os.listdir(path))[:8]
                print(f"    {entry}: {files}{' ...' if len(files) == 8 else ''}")
            except OSError:
                pass
    else:
        print("  /kaggle/input absent - not a Kaggle session")


def section_write_speed() -> None:
    head("WRITE THROUGHPUT")
    # preflight.min_write_mbps is 50. Measure where traces will actually be written, and fsync:
    # without it this measures the page cache and reports a number three orders of magnitude high.
    targets = [m for m in ("/kaggle/temp", "/kaggle/working", "/tmp", ".") if os.path.isdir(m)]
    chunk = b"\0" * (8 * 1024 * 1024)
    for target in targets:
        try:
            free = shutil.disk_usage(target).free
        except OSError:
            continue
        if free < 4 * PROBE_BYTES:
            print(f"{target:17s} skipped: only {free / 2**30:.1f} GiB free")
            continue
        fd, path = tempfile.mkstemp(dir=target, prefix=".diag_write_")
        try:
            written = 0
            t0 = time.perf_counter()
            while written < PROBE_BYTES:
                written += os.write(fd, chunk)
            os.fsync(fd)
            elapsed = time.perf_counter() - t0
            mbps = (written / 1e6) / elapsed
            flag = "" if mbps >= 50 else "   BELOW preflight.min_write_mbps=50"
            print(f"{target:17s} {mbps:8.1f} MB/s  ({written / 2**20:.0f} MiB, fsynced){flag}")
        except OSError as exc:
            print(f"{target:17s} failed: {exc}")
        finally:
            os.close(fd)
            try:
                os.unlink(path)
            except OSError:
                pass


def section_toolchain() -> None:
    head("TOOLCHAIN")
    for name, cmd in TOOLS:
        rc, out = _run(cmd, timeout=25)
        first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        if name == "nvcc" and rc == 0:
            # The CUDA toolkit version, not the driver's - this is what compiles ggml-cuda.
            first = next((ln.strip() for ln in out.splitlines() if "release" in ln), first)
        print(f"{name:6s} {'OK ' if rc == 0 else f'rc={rc}':4s} {first[:110]}")
    for var in ("CUDA_HOME", "CUDA_PATH", "CC", "CXX", "CMAKE_CUDA_ARCHITECTURES", "PATH"):
        val = os.environ.get(var)
        if val:
            print(f"env {var:24s} = {val if var != 'PATH' else val[:300] + ' ...'}")


def section_packages() -> None:
    head("PYTHON PACKAGES")
    import importlib.metadata as md  # stdlib since 3.8

    for name in PACKAGES:
        dist = {"yaml": "PyYAML"}.get(name, name)
        try:
            print(f"{name:16s} {md.version(dist)}")
        except md.PackageNotFoundError:
            print(f"{name:16s} ABSENT")
    print("\npip / venv:")
    for cmd in (("python", "-m", "pip", "--version"),):
        rc, out = _run(cmd, timeout=40)
        print(f"  rc={rc} {out.strip().splitlines()[0][:120] if out.strip() else ''}")


def section_network() -> None:
    head("NETWORK")
    # A Kaggle session with the internet switch off can neither git-pull nor reach HF, which would
    # move every input to an attached Dataset and change the whole workflow. Check, do not assume.
    import socket

    for host, port in (("github.com", 443), ("huggingface.co", 443), ("pypi.org", 443)):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=8):
                print(f"{host:20s} reachable in {1000 * (time.perf_counter() - t0):.0f} ms")
        except OSError as exc:
            print(f"{host:20s} UNREACHABLE ({type(exc).__name__}: {exc})")
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE"):
        val = os.environ.get(var)
        # Never print a token value; presence is the only fact needed.
        shown = "set" if "TOKEN" in var else val
        print(f"env {var:24s} = {shown if val else 'unset'}")


def main(argv: list[str] | None = None) -> int:
    print(SEP)
    print("MoE routing study - runtime audit (plan T0.1). Paste this whole output back.")
    print(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(SEP)
    for fn in (
        section_host,
        section_gpu,
        section_mounts,
        section_toolchain,
        section_packages,
        section_network,
        section_write_speed,  # last: it is the only section that writes, and the slowest
    ):
        try:
            fn()
        except Exception as exc:  # a diagnostic must never fail before printing the rest
            print(f"\n!! section {fn.__name__} raised {type(exc).__name__}: {exc}")
    print(f"\n{SEP}\nend of report\n{SEP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
