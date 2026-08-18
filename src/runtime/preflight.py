"""Scratch preflight — plan T0.6.

Run at the top of every capture session, before a single token is processed.

The plan is explicit about why this measures rather than queries: ``os.statvfs`` on Kaggle
reports the read-only Docker layer, not your writable allocation, so a "free space" number
there is not evidence of anything. The only way to know you can write 40 GB of trace is to
write some of it.

Failing here costs a minute. Failing four hours into a capture run costs the session.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

__all__ = [
    "PreflightResult",
    "PreflightError",
    "run_preflight",
    "check_upload_credentials",
]

_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB writes; large enough to amortize syscall overhead


class PreflightError(RuntimeError):
    """Scratch is unusable or too slow to sustain a capture run."""


@dataclass
class PreflightResult:
    scratch_path: str
    probe_bytes: int
    write_seconds: float
    write_mbps: float
    fsync_seconds: float
    free_gb_before: float
    free_gb_after: float
    min_write_mbps: float
    passed: bool
    detail: str = ""
    min_free_gb: float = 0.0
    #: True when the probe had to create the scratch directory. On Kaggle that is a red flag
    #: rather than a convenience -- see the note in ``run_preflight``.
    created_scratch: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except OSError:
        return float("nan")


def run_preflight(
    scratch: Path | str,
    probe_bytes: int = 2 * 1024**3,
    min_write_mbps: float = 50.0,
    *,
    min_free_gb: float = 0.0,
    report_path: Path | str | None = None,
    raise_on_fail: bool = True,
) -> PreflightResult:
    """Write ``probe_bytes`` to ``scratch``, fsync, measure, delete.

    The fsync is timed separately and counted toward throughput. Without it the measurement
    reports page-cache bandwidth, which on a 32 GB host means a 2 GB probe never touches the
    disk and always "passes".

    Parameters
    ----------
    scratch:
        Directory to probe. Created if absent.
    probe_bytes:
        Probe size. The default matches ``run.yaml``'s ``preflight.scratch_probe_bytes``.
    min_write_mbps:
        Sustained throughput floor, including the fsync.
    min_free_gb:
        Free-space floor for the WHOLE collection, not just the probe. Added after T0.1: the
        probe alone cannot distinguish the big scratch disk from the small overlay, because
        2 GB fits comfortably on both. T0.1 measured /kaggle/working at 19.5 GiB and /tmp at
        1026.8 GiB free, and a trace campaign does not fit on the former.
    raise_on_fail:
        Raise :class:`PreflightError` rather than returning a failed result.
    """
    scratch = Path(scratch)
    # Whether the directory already existed is evidence, not bookkeeping. T0.1 found that
    # /kaggle/temp -- the path this repo shipped as `unhashed.paths.scratch` -- does not exist
    # in the current Kaggle image at all. mkdir() would have created it happily, on the 19.5 GiB
    # overlay, and every check below would have passed. Record it and say so in the detail.
    created_scratch = not scratch.exists()
    scratch.mkdir(parents=True, exist_ok=True)

    def _fail(detail: str, free_before: float) -> PreflightResult:
        return _finish(
            PreflightResult(
                scratch_path=str(scratch),
                probe_bytes=probe_bytes,
                write_seconds=0.0,
                write_mbps=0.0,
                fsync_seconds=0.0,
                free_gb_before=free_before,
                free_gb_after=free_before,
                min_write_mbps=min_write_mbps,
                passed=False,
                detail=detail,
                min_free_gb=min_free_gb,
                created_scratch=created_scratch,
            ),
            report_path,
            raise_on_fail,
        )

    free_before = _disk_free_gb(scratch)

    if min_free_gb > 0 and free_before == free_before and free_before < min_free_gb:
        note = (
            " The directory did not exist and was created, so this is very likely the wrong "
            "mount rather than a full one -- check the configured paths.scratch."
            if created_scratch
            else ""
        )
        return _fail(
            f"{scratch} has {free_before:.1f} GB free, below the {min_free_gb:.1f} GB floor a "
            f"collection needs.{note}",
            free_before,
        )

    if free_before == free_before and free_before * 1e9 < probe_bytes * 1.1:
        return _fail(
            f"only {free_before:.1f} GB free at {scratch}, probe needs "
            f"{probe_bytes / 1e9:.1f} GB",
            free_before,
        )

    chunk = b"\0" * min(_CHUNK_BYTES, probe_bytes)
    fd, tmp_name = tempfile.mkstemp(dir=str(scratch), prefix="preflight_", suffix=".probe")
    tmp_path = Path(tmp_name)

    write_seconds = fsync_seconds = 0.0
    detail = ""
    passed = False

    try:
        written = 0
        start = time.perf_counter()
        with os.fdopen(fd, "wb", buffering=0) as handle:
            while written < probe_bytes:
                block = chunk if probe_bytes - written >= len(chunk) else chunk[: probe_bytes - written]
                handle.write(block)
                written += len(block)
            write_seconds = time.perf_counter() - start

            fsync_start = time.perf_counter()
            handle.flush()
            os.fsync(handle.fileno())
            fsync_seconds = time.perf_counter() - fsync_start

        total_seconds = write_seconds + fsync_seconds
        write_mbps = (probe_bytes / 1e6) / total_seconds if total_seconds > 0 else 0.0
        passed = write_mbps >= min_write_mbps
        if not passed:
            detail = (
                f"sustained write {write_mbps:.1f} MB/s is below the {min_write_mbps:.1f} MB/s "
                "floor; a capture run would spend more time flushing traces than inferring"
            )

    except OSError as exc:
        detail = f"write failed: {exc}"
        write_mbps = 0.0
    finally:
        tmp_path.unlink(missing_ok=True)

    result = PreflightResult(
        scratch_path=str(scratch),
        probe_bytes=probe_bytes,
        write_seconds=round(write_seconds, 3),
        write_mbps=round(write_mbps, 1),
        fsync_seconds=round(fsync_seconds, 3),
        free_gb_before=round(free_before, 1),
        free_gb_after=round(_disk_free_gb(scratch), 1),
        min_write_mbps=min_write_mbps,
        passed=passed,
        detail=detail,
        min_free_gb=min_free_gb,
        created_scratch=created_scratch,
    )
    return _finish(result, report_path, raise_on_fail)


def _finish(
    result: PreflightResult, report_path: Path | str | None, raise_on_fail: bool
) -> PreflightResult:
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    if not result.passed and raise_on_fail:
        raise PreflightError(f"scratch preflight failed at {result.scratch_path}: {result.detail}")
    return result


def check_upload_credentials(run_config: Any, *, backend: str | None = None) -> None:
    """Fail now if the configured upload backend has no credentials.

    Added after T0.1, which reported ``HF_TOKEN`` and ``HUGGINGFACE_HUB_TOKEN`` both unset on a
    fresh Kaggle session while ``unhashed.upload.backend`` is ``hf``. Kaggle Secrets are opt-in
    per notebook, so "unset" is the default state, not an accident. Without this check the
    missing token surfaces in :mod:`src.runtime.upload` after the traces exist -- twelve hours
    of GPU quota into a session whose scratch disk does not survive it (invariant I9).
    """
    unhashed = getattr(run_config, "unhashed", None) or {}
    # The EFFECTIVE backend, which a caller may have overridden on the command line. Reading only
    # the config would refuse a deliberate `--backend local` run for want of a token it will never
    # use -- and the first instinct then is to disable the check, which is exactly the check you
    # want alive for the real hf run.
    if backend is None:
        backend = str((unhashed.get("upload") or {}).get("backend", ""))
    if str(backend).lower() != "hf":
        return
    from .upload import HF_TOKEN_ENV_VARS

    if any(os.environ.get(name) for name in HF_TOKEN_ENV_VARS):
        return
    raise PreflightError(
        "unhashed.upload.backend is 'hf' but neither "
        f"{' nor '.join(HF_TOKEN_ENV_VARS)} is set. Attach the token as a Kaggle Secret and "
        "export it before collecting -- traces live on scratch, which does not outlive the "
        "session, so an upload that cannot authenticate loses the whole run."
    )


def from_config(run_config: Any, *, report_path: Path | str | None = None) -> PreflightResult:
    """Run the preflight using the ``unhashed.preflight`` / ``unhashed.paths`` blocks."""
    unhashed = run_config.unhashed or {}
    settings = unhashed.get("preflight", {}) or {}
    paths = unhashed.get("paths", {}) or {}
    check_upload_credentials(run_config)
    return run_preflight(
        scratch=paths.get("scratch", tempfile.gettempdir()),
        probe_bytes=int(settings.get("scratch_probe_bytes", 2 * 1024**3)),
        min_write_mbps=float(settings.get("min_write_mbps", 50.0)),
        min_free_gb=float(settings.get("min_free_gb", 0.0)),
        report_path=report_path,
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Scratch preflight (plan T0.6)")
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--gb", type=float, default=2.0)
    parser.add_argument("--min-mbps", type=float, default=50.0)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    outcome = run_preflight(
        args.scratch,
        probe_bytes=int(args.gb * 1024**3),
        min_write_mbps=args.min_mbps,
        min_free_gb=args.min_free_gb,
        report_path=args.report,
        raise_on_fail=False,
    )
    print(json.dumps(outcome.to_json(), indent=2))
    raise SystemExit(0 if outcome.passed else 1)
