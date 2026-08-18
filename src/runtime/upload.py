"""Shard upload with round-trip verification — plan S.3 step d, T5.3, I9.

A shard is not "collected" when the capture binary exits. It is collected when its bytes are
readable from durable storage. Everything in this module exists because of one failure mode the
plan calls out twice (T5.3, and the risk table's "upload truncated, session ends, shard lost"):
**an upload that returns success while storing truncated bytes.** Nothing later in the pipeline
can detect that — the reader would happily memmap a short ``logits.bin`` and
:func:`src.traces.format.check_file_sizes` runs against the *local* copy, which is fine. So the
check has to happen here, against the bytes that came back off the wire, before the local copy
is deleted and before the ledger says the shard is done.

Ordering is therefore fixed and not negotiable (mirrors :mod:`src.runtime.state`)::

    size-check local -> hash local -> upload -> download back -> re-hash -> mark complete -> delete

Storage is behind :class:`StorageBackend` so that the round-trip logic is exercised offline by
:class:`LocalDirBackend` in the tests — the same code path production runs, not a mock of it.
``huggingface_hub`` is imported lazily inside :class:`HFBackend`; the analysis venv does not have
it and every test here must pass with it absent.

Why traces go to a backend at all rather than to ``/kaggle/working``: invariant I9 — working is
capped at 20 GB and ~500 files, and one model's hidden states alone exceed that. Upload and
delete per shard (T4.4) keeps scratch bounded at roughly one shard.

Resumability semantics (chosen deliberately; see :func:`upload_shard`)
----------------------------------------------------------------------
``upload_shard`` **always re-uploads and re-verifies**. It never skips a file because the remote
already has it, because remote *presence* is exactly what an interrupted upload leaves behind —
``exists()`` returning True is not evidence of integrity, and a truncated remote file is the
thing this module exists to catch. Re-uploading is idempotent (same bytes, same paths), so a
session killed mid-upload is retried simply by running the shard again. A caller that wants to
avoid paying the bandwidth twice checks :func:`verify_remote_shard` first and skips the shard
only if the remote round-trip already matches the local hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ..traces.format import (
    MANIFEST_NAME,
    STREAM_FILES,
    HIDDEN_INDEX_DTYPE,
    FormatError,
    TraceSpec,
    check_file_sizes,
    read_manifest,
)
from .state import ShardRecord, ShardState

__all__ = [
    "UploadError",
    "StorageBackend",
    "LocalDirBackend",
    "HFBackend",
    "ShardUploadResult",
    "sha256_file",
    "upload_shard",
    "verify_remote_shard",
    "load_remote_manifest",
]

#: Env vars consulted for an HF write token, in order. Never a constructor default: a token
#: passed as an argument ends up in tracebacks, ``repr()`` output and notebook cell history.
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")

CHUNK_BYTES = 1 << 20


class UploadError(RuntimeError):
    """An upload, or its round-trip verification, failed. The shard is NOT complete."""


# --------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------


def sha256_file(path: Path | str, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Streaming SHA256. Never reads the whole file — ``hidden.bin`` is up to 500 MB (T4.4),
    and a Kaggle session has 15 GB of RAM shared with the model."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# backend contract
# --------------------------------------------------------------------------------------


@runtime_checkable
class StorageBackend(Protocol):
    """The four operations the round-trip needs. Deliberately no delete and no rename.

    ``remote_path`` is always a POSIX-style relative path inside the backend's own root
    (``traces/<model>/<corpus>/shard_00007/topk.bin``), so the same prefix works for an HF
    dataset repo and for a local directory.

    Implementations must overwrite an existing ``remote_path`` rather than erroring, because
    retrying an interrupted shard re-uploads every file (see module docstring).
    """

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Store ``local_path`` at ``remote_path``, overwriting."""

    def download_file(self, remote_path: str, dest_path: Path) -> Path:
        """Fetch ``remote_path`` to ``dest_path`` and return the path actually written."""

    def exists(self, remote_path: str) -> bool:
        """True if ``remote_path`` is present. Presence is NOT integrity — see the module
        docstring; nothing in this module treats it as such."""

    def list_files(self, prefix: str) -> list[str]:
        """Remote paths under ``prefix``, sorted. Used for diagnostics, not for verification."""


class LocalDirBackend:
    """Filesystem backend. A real, shipped implementation — not a test double.

    Two uses: the offline test suite (so the tests drive the production round-trip code), and a
    dry run against a scratch directory to time the upload path without touching the network.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _resolve(self, remote_path: str) -> Path:
        # Reject traversal: a manifest-derived prefix should never escape the root, and finding
        # out that it did by overwriting something outside it is not a good way to find out.
        parts = [p for p in str(remote_path).replace("\\", "/").split("/") if p and p != "."]
        if any(p == ".." for p in parts):
            raise UploadError(f"remote path escapes the backend root: {remote_path!r}")
        return self.root.joinpath(*parts)

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        target = self._resolve(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)

    def download_file(self, remote_path: str, dest_path: Path) -> Path:
        source = self._resolve(remote_path)
        if not source.exists():
            raise UploadError(f"{remote_path}: not present in {self.root}")
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest_path)
        return dest_path

    def exists(self, remote_path: str) -> bool:
        return self._resolve(remote_path).is_file()

    def list_files(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.is_dir():
            return []
        return sorted(
            f"{str(prefix).rstrip('/')}/{p.relative_to(base).as_posix()}"
            for p in base.rglob("*")
            if p.is_file()
        )

    def __repr__(self) -> str:
        return f"LocalDirBackend(root={str(self.root)!r})"


class HFBackend:
    """HF Hub dataset repo — the production target (S.1 rule 3: traces live on the Hub).

    ``huggingface_hub`` is imported inside the methods, not at module scope, so that this module
    imports in the analysis venv (which does not have it) and so that the offline test suite
    covers the round-trip logic without the dependency.

    The token is read from the environment on demand and is never stored on the instance, never
    passed as a constructor argument, and never interpolated into an error message or ``repr``.
    Kaggle notebook output is retained per version; a leaked write token there is a leaked write
    token in a published artifact.
    """

    def __init__(self, repo_id: str, repo_type: str = "dataset", *, create: bool = False) -> None:
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._create = bool(create)
        self._api: Any = None
        self._ensured = False

    # -- lazy plumbing ------------------------------------------------------------------

    @staticmethod
    def _token() -> str:
        for name in HF_TOKEN_ENV_VARS:
            value = os.environ.get(name)
            if value:
                return value
        raise UploadError(
            "no Hugging Face token in the environment. Set one of "
            f"{' or '.join(HF_TOKEN_ENV_VARS)} to a token with write access to the trace repo. "
            "Do not pass it as an argument — it would end up in tracebacks and notebook output."
        )

    def _hub(self) -> Any:
        try:
            import huggingface_hub  # noqa: PLC0415 - lazy on purpose, see class docstring
        except ImportError as exc:
            raise UploadError(
                "huggingface_hub is not installed, so traces cannot be uploaded. "
                "`pip install huggingface_hub` in the collection environment, or pass a "
                "LocalDirBackend to keep the trace on disk. This venv deliberately does not "
                "have it: the analysis side never touches the network."
            ) from exc
        return huggingface_hub

    def _client(self) -> Any:
        if self._api is None:
            hub = self._hub()
            self._api = hub.HfApi(token=self._token())
            if self._create and not self._ensured:
                self._api.create_repo(
                    repo_id=self.repo_id, repo_type=self.repo_type, exist_ok=True, private=True
                )
                self._ensured = True
        return self._api

    # -- StorageBackend -----------------------------------------------------------------

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        api = self._client()
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
        except UploadError:
            raise
        except Exception as exc:  # hub raises a wide family; the caller only needs the path
            raise UploadError(f"upload of {remote_path} to {self.repo_id} failed: {exc}") from exc

    def download_file(self, remote_path: str, dest_path: Path) -> Path:
        api = self._client()
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # local_dir + no symlinks: the round-trip must read bytes that actually travelled,
            # not a symlink into a cache entry that was never re-fetched.
            fetched = api.hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                filename=remote_path,
                local_dir=str(dest_path.parent),
            )
        except Exception as exc:
            raise UploadError(
                f"download-back of {remote_path} from {self.repo_id} failed: {exc}"
            ) from exc
        fetched_path = Path(fetched)
        if fetched_path != dest_path:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fetched_path, dest_path)
        return dest_path

    def exists(self, remote_path: str) -> bool:
        api = self._client()
        try:
            return bool(
                api.file_exists(
                    repo_id=self.repo_id, repo_type=self.repo_type, filename=remote_path
                )
            )
        except Exception as exc:
            raise UploadError(f"existence check for {remote_path} failed: {exc}") from exc

    def list_files(self, prefix: str) -> list[str]:
        api = self._client()
        try:
            names = api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
        except Exception as exc:
            raise UploadError(f"listing {self.repo_id} failed: {exc}") from exc
        pref = str(prefix).rstrip("/") + "/"
        return sorted(n for n in names if n.startswith(pref))

    def __repr__(self) -> str:
        # No token, not even a redacted placeholder that might later be filled in by accident.
        return f"HFBackend(repo_id={self.repo_id!r}, repo_type={self.repo_type!r})"


# --------------------------------------------------------------------------------------
# result record
# --------------------------------------------------------------------------------------


@dataclass
class ShardUploadResult:
    """Evidence that one shard survived the round-trip. Feeds the ledger and T5.3's report."""

    shard_id: int
    remote_prefix: str
    files: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {size, sha256}
    verified: bool = False
    bytes_uploaded: int = 0
    elapsed_s: float = 0.0

    @property
    def file_sha256(self) -> dict[str, str]:
        """Stream checksums only — the mapping the ledger and the manifest both key on.

        ``manifest.json`` is excluded on purpose: it carries ``collected_utc`` and the session
        id, so including it would make a legitimate bit-identical recollection of a shard look
        like a checksum conflict and trip T3.6's hard error for the wrong reason.
        """
        return {
            name: meta["sha256"]
            for name, meta in self.files.items()
            if name != MANIFEST_NAME
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "remote_prefix": self.remote_prefix,
            "files": {k: dict(v) for k, v in sorted(self.files.items())},
            "verified": self.verified,
            "bytes_uploaded": self.bytes_uploaded,
            "elapsed_s": round(self.elapsed_s, 3),
        }


# --------------------------------------------------------------------------------------
# shard inspection helpers
# --------------------------------------------------------------------------------------


def _shard_dir_of(shard_dir_or_manifest: Path | str) -> Path:
    path = Path(shard_dir_or_manifest)
    return path.parent if path.name == MANIFEST_NAME else path


def _stream_files(shard_dir: Path) -> list[str]:
    """Stream files actually present, in a fixed order.

    ``hidden.bin`` / ``hidden_index.bin`` are legitimately absent when the subsample is off, so
    absence is not an error here; a *missing declared* stream is caught by
    :func:`check_file_sizes` against the manifest arithmetic instead.
    """
    return [name for name in STREAM_FILES.values() if (shard_dir / name).is_file()]


def _n_captured(shard_dir: Path) -> int:
    """Rows in ``hidden.bin``, taken from ``hidden_index.bin``'s size rather than the manifest —
    the file is the authority on what was written, which is what makes the size check a check."""
    path = shard_dir / STREAM_FILES["hidden_index"]
    if not path.is_file():
        return 0
    size = path.stat().st_size
    if size % HIDDEN_INDEX_DTYPE.itemsize:
        raise UploadError(f"{path}: size {size} is not a whole number of uint32 indices")
    return size // HIDDEN_INDEX_DTYPE.itemsize


def _check_local_sizes(shard_dir: Path, manifest: Mapping[str, Any]) -> int:
    """Refuse to upload a locally-truncated shard. Returns n_captured.

    Cheap, and it separates two diagnoses that look identical from the far end of a round-trip:
    the capture binary wrote a short file, versus the transfer dropped bytes.
    """
    spec = TraceSpec.from_manifest(manifest)
    n_captured = _n_captured(shard_dir)
    try:
        check_file_sizes(shard_dir, spec, int(manifest["n_tokens"]), n_captured)
    except FormatError as exc:
        raise UploadError(
            f"shard {manifest.get('shard_id')}: local files do not match manifest arithmetic, "
            f"refusing to upload — {exc}"
        ) from exc
    return n_captured


def _remote_path(remote_prefix: str, name: str) -> str:
    return f"{str(remote_prefix).strip('/')}/{name}"


def _round_trip(
    backend: StorageBackend,
    remote_prefix: str,
    name: str,
    expect: Mapping[str, Any],
    scratch: Path,
) -> None:
    """Download one file back and compare size then hash. Raises :class:`UploadError`.

    Size is compared first because it names the failure precisely: a size mismatch is a
    truncated or padded transfer, while equal sizes with unequal hashes is corruption in
    flight. Both are fatal, but they point at different things to go fix.
    """
    remote = _remote_path(remote_prefix, name)
    dest = scratch / name
    backend.download_file(remote, dest)

    got_size = dest.stat().st_size
    if got_size != int(expect["size"]):
        raise UploadError(
            f"round-trip FAILED for {remote}: uploaded {expect['size']} B, read back "
            f"{got_size} B (delta {got_size - int(expect['size']):+d}). The remote copy is "
            "truncated or padded; the shard is not complete and must be re-uploaded."
        )

    got_hash = sha256_file(dest)
    if got_hash != expect["sha256"]:
        raise UploadError(
            f"round-trip FAILED for {remote}: sizes match at {got_size} B but the bytes "
            f"differ.\n  local  sha256 = {expect['sha256']}\n  remote sha256 = {got_hash}\n"
            "The remote copy is corrupt; the shard is not complete and must be re-uploaded."
        )


# --------------------------------------------------------------------------------------
# the two public operations
# --------------------------------------------------------------------------------------


def upload_shard(
    shard_dir: Path | str,
    backend: StorageBackend,
    *,
    remote_prefix: str,
    verify: bool = True,
    delete_local_on_success: bool = False,
    state: ShardState | None = None,
) -> ShardUploadResult:
    """Upload one shard, verify the round-trip, then (optionally) record and delete it.

    The step order is the whole point of plan S.3 step d: nothing is marked complete and nothing
    is deleted until bytes that came *back* off the wire hash to the same value as the bytes that
    went out. ``verify=False`` exists only for a bandwidth-measurement dry run and is refused
    together with ``delete_local_on_success`` — deleting the only good copy of a shard on the
    strength of an unverified upload is precisely the loss this module prevents.

    Failure leaves: nothing in ``state``, every local file intact, and possibly a partial remote
    copy that the next attempt overwrites (see the module docstring on resumability).
    """
    started = time.monotonic()
    shard_dir = _shard_dir_of(shard_dir)
    manifest = read_manifest(shard_dir)
    shard_id = int(manifest["shard_id"])

    if delete_local_on_success and not verify:
        raise UploadError(
            f"shard {shard_id}: delete_local_on_success requires verify=True. Deleting the "
            "local copy on an unverified upload is the exact failure S.3 step d exists to stop."
        )

    n_captured = _check_local_sizes(shard_dir, manifest)
    names = _stream_files(shard_dir) + [MANIFEST_NAME]

    result = ShardUploadResult(shard_id=shard_id, remote_prefix=str(remote_prefix).strip("/"))

    # 1. hash locally first, so the comparison baseline is taken before anything is transferred.
    for name in names:
        path = shard_dir / name
        result.files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}

    # 2. upload. The manifest goes last: a reader that finds a manifest can then assume the
    #    streams beside it were at least fully transmitted once.
    for name in names:
        backend.upload_file(shard_dir / name, _remote_path(result.remote_prefix, name))
        result.bytes_uploaded += int(result.files[name]["size"])

    # 3. verify, into a scratch dir that is removed whether or not verification passes.
    if verify:
        scratch = Path(tempfile.mkdtemp(prefix=f"moe-verify-{shard_id:05d}-"))
        try:
            for name in names:
                _round_trip(backend, result.remote_prefix, name, result.files[name], scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        result.verified = True

    result.elapsed_s = time.monotonic() - started

    # 4. only now may the ledger call the shard done, and only then may the bytes go.
    if state is not None:
        if not result.verified:
            raise UploadError(
                f"shard {shard_id}: refusing to record an unverified upload in the ledger "
                "(plan S.3 step d)"
            )
        doc_range = manifest.get("shard_doc_range") or [0, 0]
        state.mark_complete(
            ShardRecord(
                shard_id=shard_id,
                n_tokens=int(manifest["n_tokens"]),
                n_captured=n_captured,
                file_sha256=result.file_sha256,
                doc_range=(int(doc_range[0]), int(doc_range[1])),
                upload_verified=True,
            )
        )

    if delete_local_on_success and result.verified:
        # I9 / T4.4: scratch holds one shard at a time. ~500 MB of hidden state per shard times
        # 20 shards does not fit anywhere on a Kaggle session, so this is not housekeeping.
        _delete_local(shard_dir, names)

    return result


def _delete_local(shard_dir: Path, names: list[str]) -> None:
    for name in names:
        (shard_dir / name).unlink(missing_ok=True)
    try:
        shard_dir.rmdir()
    except OSError:
        # Something unexpected is still in there. Leave it — a stray file is worth a look, and
        # silently recursive-deleting a directory we do not fully understand is not worth it.
        pass


def verify_remote_shard(
    shard_dir_or_manifest: Path | str,
    backend: StorageBackend,
    *,
    remote_prefix: str,
) -> dict[str, Any]:
    """Re-verify an already-uploaded shard without re-uploading it.

    Two callers. T3.6's cross-session check, which re-validates shards a previous session
    recorded; and resume after a session was killed mid-upload, where this decides whether the
    remote copy is already good (skip) or has to be sent again (:func:`upload_shard`).

    Returns a report rather than raising, because "this shard needs re-uploading" is a normal
    control-flow answer during resume, not an error. ``report["verified"]`` is the verdict and
    ``report["failures"]`` names every file and both hashes.
    """
    shard_dir = _shard_dir_of(shard_dir_or_manifest)
    manifest = read_manifest(shard_dir)
    prefix = str(remote_prefix).strip("/")
    names = _stream_files(shard_dir) + [MANIFEST_NAME]

    report: dict[str, Any] = {
        "shard_id": int(manifest["shard_id"]),
        "remote_prefix": prefix,
        "verified": False,
        "files": {},
        "missing": [],
        "failures": [],
    }

    scratch = Path(tempfile.mkdtemp(prefix="moe-reverify-"))
    try:
        for name in names:
            local = shard_dir / name
            expect = {"size": local.stat().st_size, "sha256": sha256_file(local)}
            report["files"][name] = dict(expect)

            if not backend.exists(_remote_path(prefix, name)):
                report["missing"].append(name)
                continue
            try:
                _round_trip(backend, prefix, name, expect, scratch)
            except UploadError as exc:
                report["failures"].append({"file": name, "reason": str(exc)})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    report["verified"] = not report["missing"] and not report["failures"]
    return report


def load_remote_manifest(
    backend: StorageBackend, *, remote_prefix: str
) -> dict[str, Any] | None:
    """Read back a remote shard's manifest, or None if there is not one there yet.

    Lets a resuming session learn what a previous one uploaded when the local scratch copy is
    already gone (it dies with the session), so T3.6's checksum comparison has both sides.
    """
    prefix = str(remote_prefix).strip("/")
    remote = _remote_path(prefix, MANIFEST_NAME)
    if not backend.exists(remote):
        return None
    scratch = Path(tempfile.mkdtemp(prefix="moe-manifest-"))
    try:
        path = backend.download_file(remote, scratch / MANIFEST_NAME)
        return json.loads(Path(path).read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
