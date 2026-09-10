#!/usr/bin/env python3
"""T1.1, publication half: put the converted panel on Kaggle so collection sessions can reach it.

The GGUFs are produced on whatever machine has the disk (here: a workstation) and consumed on
Kaggle, so an artifact that exists only locally blocks every downstream task. `corpora/` and the
weights are gitignored by design -- the transport is Kaggle, not the repo.

Kaggle Models, not Kaggle Datasets
----------------------------------
Kaggle bills the two separately: 200 GB of private models AND 200 GB of private datasets, two
pools. The panel is 80 GiB. Spent from the dataset pool it would take 40% of the same budget the
T4 corpora and the T3/T8 trace shards have to come out of; spent from the model pool it costs
those nothing. `gguf` is also a first-class framework slug rather than a workaround -- Kaggle's own
naming docs use ``google/gemma-2/gguf/2.0-27b-it/1`` -- and the four-part handle
``owner/model/framework/variation`` says what the panel actually is: one conversion recipe applied
to seven checkpoints, under one model page, each variation versioned on its own.

One variation per artifact, not one for the panel
--------------------------------------------------
Eight variations of ``<owner>/moe-panel/gguf/``, one per model key, plus ``olmoe-0125-f16`` for the
F16 the T8.1 precision ladder needs. Three reasons, all of them things that bite the single-blob
layout:

* kagglehub resumes a broken transfer *within* a file (GCS resumable sessions, ``MAX_RETRIES=5``)
  but the session URI lives in memory only. A dead process re-uploads from zero, so the blast
  radius of one bad night should be one checkpoint, not 80 GiB.
* A collection session attaches only the checkpoint it runs. Mounting the whole panel to trace
  olmoe-0924 would drag 80 GiB into a session that needs 3.9.
* Re-converting one model bumps one variation's version. In a single blob every re-upload creates
  a new version of *everything*, and `gguf.sha256` in models.yaml then describes a version number
  that no longer means what it did.

Kaggle's own limits are not the binding constraint here: 100 GB per variation against a largest
artifact of 17.3 GiB, and 50 top-level files against one.

Verification
------------
This script does not download the artifact back. `scripts/kaggle_setup.py` already hashes whatever
GGUF it acquires and compares against ``gguf.sha256``, so the round trip is checked against the
bytes collection actually reads, on the machine that reads them. Re-hashing 80 GiB here would test
the wrong copy at the wrong time.

Models are created **private** (``_create_model`` sets ``is_private = True``, as the dataset path
does). These are derived weights; gemma-4's licence is Google's to enforce and republishing it is
not this project's business. For the same reason no ``license_name`` is sent -- asserting a licence
on someone else's weights is not this script's call, and the upstream licence is named in the
version notes where a human reads it.

Usage
-----
    python scripts/kaggle_publish.py --all --gguf-dir D:\\moe\\gguf --dry-run
    python scripts/kaggle_publish.py --all --gguf-dir D:\\moe\\gguf --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capture.nodescan import set_model_fields  # noqa: E402

RESULTS = REPO_ROOT / "results"
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"

#: The model page every artifact is a variation of, and the framework slug under it. Kaggle slugs
#: are lowercase, 6-50 chars, letters/digits/dashes.
MODEL_SLUG = "moe-panel"
FRAMEWORK = "gguf"

#: Upstream licence per checkpoint, for the version notes. Not sent to Kaggle as `license_name`:
#: these are derived weights and the licence is the original publisher's to assert, not ours.
UPSTREAM_LICENCE = {
    "olmoe-0924": "Apache-2.0",
    "olmoe-0125": "Apache-2.0",
    "olmoe-0125-instruct": "Apache-2.0",
    "deepseek-v2-lite": "DeepSeek Model License",
    "gpt-oss-20b": "Apache-2.0",
    "gemma-4-26b-a4b": "Gemma Terms of Use",
    "qwen3-30b-a3b": "Apache-2.0",
}

#: The F16 intermediates worth publishing, keyed by the model they belong to. Only olmoe-0125's is
#: kept (T8.1 reads the same checkpoint at two precisions); the rest are pruned at conversion.
F16_EXTRAS = {"olmoe-0125": "olmoe-0125-F16.gguf"}


class PublishError(RuntimeError):
    """Publication failed. Never partial-success: a dataset version is all files or none."""


def _say(msg: str) -> None:
    print(msg, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Artifact:
    """One file destined for one dataset."""

    model_key: str
    slug_suffix: str          # the variation slug: usually the model key; "-f16" variants differ
    path: Path
    sha256: str | None        # None when the conversion record never hashed it (the F16s)
    size_bytes: int
    llama_commit: str
    transformers: str | None
    is_f16: bool = False

    @property
    def handle_slug(self) -> str:
        """Filesystem-safe name for the staging directory, not part of the Kaggle handle."""
        return f"{MODEL_SLUG}-{self.slug_suffix}"

    def handle(self, owner: str) -> str:
        return f"{owner}/{MODEL_SLUG}/{FRAMEWORK}/{self.slug_suffix}"

    def version_notes(self) -> str:
        """What a human opening the dataset page needs to reconnect it to the run that made it."""
        parts = [f"{self.path.name}", f"llama.cpp {self.llama_commit[:12]}"]
        if self.transformers:
            parts.append(f"transformers {self.transformers}")
        licence = UPSTREAM_LICENCE.get(self.model_key)
        if licence:
            parts.append(f"upstream licence {licence}")
        if self.sha256:
            parts.append(f"sha256 {self.sha256}")
        return "; ".join(parts)


def load_artifacts(keys: Sequence[str], gguf_dir: Path) -> list[Artifact]:
    """Build the publication list from the conversion records, not from a directory listing.

    The records are the provenance chain: they name the file, its sha256, the llama.cpp commit and
    the converter environment. A directory listing would happily publish a stray file nobody has a
    record for, and an artifact with no record is an artifact nothing downstream can verify.
    """
    artifacts: list[Artifact] = []
    for key in keys:
        record_path = RESULTS / f"convert_{key}.json"
        if not record_path.exists():
            raise PublishError(
                f"no conversion record at {record_path}. Publication is driven by the records, so "
                f"{key} has to be converted (or its record restored) before it can be published."
            )
        rec = json.loads(record_path.read_text(encoding="utf-8"))
        env = rec.get("converter_env") or {}
        path = gguf_dir / rec["file"]
        if not path.exists():
            raise PublishError(f"{path} is named by {record_path.name} but is not on disk")
        artifacts.append(Artifact(
            model_key=key, slug_suffix=key, path=path, sha256=rec.get("sha256"),
            size_bytes=int(rec["size_bytes"]), llama_commit=str(rec.get("llama_cpp_commit") or ""),
            transformers=env.get("transformers"),
        ))

        extra = F16_EXTRAS.get(key)
        if extra and (gguf_dir / extra).exists():
            f16 = gguf_dir / extra
            artifacts.append(Artifact(
                model_key=key, slug_suffix=f"{key}-f16", path=f16,
                # The conversion record leaves f16_sha256 null -- the F16 is an intermediate that
                # usually gets deleted, so nothing hashed it. It is a published artifact now, so
                # it gets hashed here.
                sha256=rec.get("f16_sha256"), size_bytes=f16.stat().st_size,
                llama_commit=str(rec.get("llama_cpp_commit") or ""),
                transformers=env.get("transformers"), is_f16=True,
            ))
    return artifacts


def check_local(artifact: Artifact, *, rehash: bool, dry_run: bool = False) -> str:
    """Confirm the local file still matches its record, and return its sha256.

    Size is checked always because it is free and catches the interesting case: a file truncated by
    a full disk or a killed conversion. The hash is opt-in -- it was computed at conversion time on
    these same bytes, and re-reading 80 GiB to confirm the disk has not rotted is a deliberate
    choice, not a default.
    """
    size = artifact.path.stat().st_size
    if size != artifact.size_bytes:
        raise PublishError(
            f"{artifact.path.name} is {size} bytes but its record says {artifact.size_bytes}. "
            "Publishing it would put a file on Kaggle that models.yaml cannot describe."
        )
    if artifact.sha256 and not rehash:
        return artifact.sha256
    if dry_run:
        # Reading 13 GiB to print a plan is not a plan.
        return artifact.sha256 or ""
    _say(f"  hashing {size / 2**30:.1f} GiB ...")
    digest = _sha256(artifact.path)
    if artifact.sha256 and digest != artifact.sha256:
        raise PublishError(
            f"{artifact.path.name} hashes to {digest[:16]} but its record says "
            f"{artifact.sha256[:16]}. The local copy changed after conversion; do not publish it."
        )
    return digest


def stage(artifact: Artifact, staging_root: Path) -> Path:
    """Put the one file in a directory of its own, since kagglehub uploads directories.

    A hard link rather than a copy: same volume, so it costs nothing and 17 GiB of copying to
    upload 17 GiB is a strange way to spend a minute. Falls back to a copy across volumes.
    """
    staging = staging_root / artifact.handle_slug
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    dest = staging / artifact.path.name
    try:
        os.link(artifact.path, dest)
    except OSError:
        _say(f"  (hard link unavailable, copying {artifact.size_bytes / 2**30:.1f} GiB)")
        shutil.copy2(artifact.path, dest)
    return staging


def already_published(artifact: Artifact, handle: str, digest: str) -> bool:
    """True when this exact file already went up under this handle.

    Local state, deliberately. Kaggle would happily accept the same bytes again as a new version,
    and a re-run of `--all` after one model failed should not re-send the seven that worked.
    """
    path = RESULTS / f"publish_{artifact.slug_suffix}.json"
    if not path.exists():
        return False
    prior = json.loads(path.read_text(encoding="utf-8"))
    return prior.get("handle") == handle and prior.get("sha256") == digest


def write_publish_record(artifact: Artifact, handle: str, digest: str, elapsed: float) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"publish_{artifact.slug_suffix}.json"
    path.write_text(json.dumps({
        "model": artifact.model_key,
        "handle": handle,
        "url": f"https://www.kaggle.com/models/{handle}",
        "file": artifact.path.name,
        "size_bytes": artifact.size_bytes,
        "sha256": digest,
        "llama_cpp_commit": artifact.llama_commit,
        "transformers": artifact.transformers,
        "is_f16_intermediate": artifact.is_f16,
        "elapsed_s": round(elapsed, 1),
    }, indent=2), encoding="utf-8")
    return path


def register(artifact: Artifact, handle: str, digest: str, *, force: bool) -> list[str]:
    """Record the handle in models.yaml so the collector can resolve the model without asking.

    The F16 extras are skipped: models.yaml has one `gguf:` block per model, and T8.1's second
    precision is not a second model. Its handle lives in results/publish_*.json until the ladder
    task decides where it belongs.
    """
    if artifact.is_f16:
        return []
    return set_model_fields(
        MODELS_CONFIG,
        artifact.model_key,
        {"gguf": (f"{{repo: null, kaggle: {handle}, file: {artifact.path.name}, "
                  f"sha256: {digest}, size_bytes: {artifact.size_bytes}}}")},
        force=force,
    )


def publish_one(artifact: Artifact, owner: str, args: argparse.Namespace,
                staging_root: Path) -> tuple[str, str]:
    handle = artifact.handle(owner)
    _say(f"\n{'=' * 78}\n== {artifact.path.name} -> {handle}\n{'=' * 78}")
    digest = check_local(artifact, rehash=args.rehash, dry_run=args.dry_run)

    if already_published(artifact, handle, digest) and not args.force:
        _say(f"  already published (results/publish_{artifact.slug_suffix}.json); --force to resend")
        return handle, digest

    if args.dry_run:
        _say(f"  DRY RUN: would upload {artifact.size_bytes / 2**30:.1f} GiB")
        _say(f"           notes: {artifact.version_notes()}")
        return handle, digest

    import kagglehub

    staging = stage(artifact, staging_root)
    started = time.perf_counter()
    _say(f"  uploading {artifact.size_bytes / 2**30:.1f} GiB ...")
    try:
        kagglehub.model_upload(handle, str(staging), version_notes=artifact.version_notes())
    except Exception as exc:
        raise PublishError(f"upload of {handle} failed: {type(exc).__name__}: {exc}") from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    elapsed = time.perf_counter() - started
    rate = artifact.size_bytes / 2**20 / max(elapsed, 1e-9)
    _say(f"  uploaded in {elapsed / 60:.1f} min ({rate:.1f} MiB/s)")
    _say(f"  wrote {write_publish_record(artifact, handle, digest, elapsed)}")
    return handle, digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the converted panel to Kaggle Models")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="model key in configs/models.yaml")
    target.add_argument("--all", action="store_true", help="every model with a conversion record")
    parser.add_argument("--gguf-dir", type=Path, required=True, help="where the GGUFs live")
    parser.add_argument("--owner", default=None, help="Kaggle username (default: whoami)")
    parser.add_argument("--staging", type=Path, default=None,
                        help="directory for the per-file staging links (default: alongside --gguf-dir)")
    parser.add_argument("--rehash", action="store_true",
                        help="re-read every file and confirm it still matches its record")
    parser.add_argument("--force", action="store_true", help="re-upload even if already published")
    # Overwriting a `gguf:` block and re-sending 80 GiB are different decisions, and the common case
    # after a successful upload run is the first without the second: the panel is on Kaggle, and
    # models.yaml still records the third-party HF builds these conversions replace. One flag for
    # both would price a one-line YAML edit at a full re-upload.
    parser.add_argument("--force-register", action="store_true",
                        help="overwrite an existing gguf: block in models.yaml (invariant I13) "
                             "without re-uploading anything")
    parser.add_argument("--write", action="store_true", help="record handles in configs/models.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and upload nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.all:
        keys = sorted(p.stem[len("convert_"):] for p in RESULTS.glob("convert_*.json"))
    else:
        keys = [args.model]

    owner = args.owner
    if owner is None:
        if args.dry_run:
            owner = "<owner>"
        else:
            import kagglehub
            try:
                owner = kagglehub.whoami()["username"]
            except Exception as exc:
                _say(f"not authenticated with Kaggle ({type(exc).__name__}). Put your API token at "
                     f"~/.kaggle/access_token (UTF-8, no BOM -- PowerShell's `>` writes UTF-16 and "
                     f"kagglehub reads that as garbage) or set KAGGLE_API_TOKEN.")
                return 2

    try:
        artifacts = load_artifacts(keys, args.gguf_dir)
    except PublishError as exc:
        _say(f"PUBLICATION FAILED: {exc}")
        return 1

    total = sum(a.size_bytes for a in artifacts)
    _say(f"owner    : {owner}")
    _say(f"gguf dir : {args.gguf_dir}")
    _say(f"artifacts: {len(artifacts)}, {total / 2**30:.1f} GiB total")

    staging_root = args.staging or (args.gguf_dir.parent / "publish-staging")
    published: list[tuple[Artifact, str, str]] = []
    try:
        for artifact in artifacts:
            handle, digest = publish_one(artifact, owner, args, staging_root)
            published.append((artifact, handle, digest))

        if args.write and not args.dry_run:
            _say("\n== recording in configs/models.yaml")
            for artifact, handle, digest in published:
                changes = register(artifact, handle, digest,
                                   force=args.force or args.force_register)
                _say(f"  {artifact.slug_suffix}: {changes or 'unchanged or not applicable'}")
    except PublishError as exc:
        _say(f"\nPUBLICATION FAILED: {exc}")
        return 1
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    _say("\n" + "=" * 78)
    _say("PUBLISHED:")
    for artifact, handle, _ in published:
        _say(f"  {artifact.slug_suffix:24s} {artifact.size_bytes / 2**30:6.1f} GiB  {handle}")
    if not args.write:
        _say("\nRun again with --write to record these handles in configs/models.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
