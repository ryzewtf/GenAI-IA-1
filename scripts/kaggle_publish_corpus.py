#!/usr/bin/env python3
"""Publish a built corpus as a private Kaggle Dataset.

Why a Dataset and why publish at all
-------------------------------------
The corpus is the fixed input every collection session reads (T5.2), and T3.6 is a
cross-session *byte-identity* gate: two collect runs of the same model must see the same
corpus bytes or their traces are not comparable. A corpus is gitignored and dies with the
Kaggle VM that built it, so the only way a later session gets the *identical* bytes is to
attach an immutable artifact -- which is exactly what a pinned Kaggle Dataset version is.
Re-fetching instead is doubly wrong: the streaming sources are unpinned (a fresh CulturaX
stream draws different documents), and the fetch's HF background threads abort the process
at teardown even on success. Publish once, attach forever.

Datasets, not Models: Kaggle bills private Models and private Datasets against two separate
200 GB pools (see kaggle_publish.py). The 7 MiB corpus belongs in the dataset pool; it costs
the model panel's budget nothing.

Privacy: kagglehub creates a *new* dataset private by default -- there is no is_private
argument on dataset_upload in this version, and the corpus (derived from gated sources) must
stay private. A *first* publish is therefore private; a later --version push inherits the
existing dataset's visibility, so never flip the dataset public in the Kaggle UI.

What goes up
------------
The .jsonl, its .fetch.json report, and a manifest.json this script writes recording the
sha256 and token count. The manifest is what makes an attached copy self-describing: a
collect session can assert the bytes it is about to read match the ones a human blessed here,
without reaching back to this workstation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_SLUG = "moe-corpus"
RESULTS_DIR = REPO_ROOT / "results"


def _say(msg: str) -> None:
    print(msg, flush=True)


class CorpusPublishError(Exception):
    """Publication refused. Never partial: a dataset version is all files or none."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(jsonl: Path, report: Path) -> dict:
    """Refuse to publish a corpus that does not parse or disagrees with its own report.

    The whole point of publishing is that later sessions trust these bytes without re-checking.
    That trust has to be earned exactly once, here, before the bytes leave the workstation.
    """
    if not jsonl.exists():
        raise CorpusPublishError(f"{jsonl} does not exist. Build it with a STAGE='corpus' session.")
    if not report.exists():
        raise CorpusPublishError(
            f"{report} does not exist. The corpus and its fetch report travel together; a corpus "
            "without the report cannot record what it is (shares, substitutions, shortfalls).")

    n = 0
    with jsonl.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusPublishError(
                    f"{jsonl} line {i} does not parse ({exc}). A corpus written by a process that "
                    "crashed mid-write must be rebuilt, not published.") from exc
            n += 1

    rep = json.loads(report.read_text(encoding="utf-8"))
    declared = rep.get("n_docs")
    if declared is not None and declared != n:
        raise CorpusPublishError(
            f"{jsonl} holds {n} docs but {report.name} claims {declared}. The file and its report "
            "disagree, so one of them is from a different run -- do not publish the pair.")
    if rep.get("ok") is False:
        raise CorpusPublishError(
            f"{report.name} records ok=false with problems {rep.get('problems')!r}. Fix the fetch "
            "before publishing.")
    return rep


def build_manifest(jsonl: Path, report: Path, digest: str, rep: dict) -> dict:
    realized = rep.get("realized", {})
    return {
        "artifact": "moe-corpus",
        "source": "scripts/kaggle_publish_corpus.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_file": jsonl.name,
        "report_file": report.name,
        "sha256": digest,
        "size_bytes": jsonl.stat().st_size,
        "n_docs": rep.get("n_docs"),
        "total_tokens": realized.get("total_tokens"),
        "seed": rep.get("seed"),
        "note": "Private. Attach this dataset and set CORPUS_FILE to the .jsonl under "
                "/kaggle/input; the collect session asserts sha256 against this manifest (T3.6).",
    }


def stage(jsonl: Path, report: Path, manifest: dict, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jsonl, staging / jsonl.name)
    shutil.copy2(report, staging / report.name)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_publish_record(handle: str, manifest: dict, elapsed: float) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(manifest)
    record["handle"] = handle
    record["url"] = f"https://www.kaggle.com/datasets/{handle}"
    record["upload_seconds"] = round(elapsed, 1)
    out = RESULTS_DIR / "publish_corpus.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a built corpus as a private Kaggle Dataset")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "mixed-v1.jsonl",
                        help="path to the .jsonl (default: ./mixed-v1.jsonl)")
    parser.add_argument("--report", type=Path, default=None,
                        help="fetch report (default: <corpus>.fetch.json)")
    parser.add_argument("--owner", default=None, help="Kaggle username (default: whoami)")
    parser.add_argument("--slug", default=DATASET_SLUG, help=f"dataset slug (default: {DATASET_SLUG})")
    parser.add_argument("--notes", default=None, help="version notes recorded on Kaggle")
    parser.add_argument("--dry-run", action="store_true", help="verify and stage, upload nothing")
    args = parser.parse_args(argv)

    jsonl = args.corpus.resolve()
    report = (args.report or Path(str(jsonl) + ".fetch.json")).resolve()

    try:
        rep = verify(jsonl, report)
    except CorpusPublishError as exc:
        _say(f"PUBLICATION FAILED: {exc}")
        return 1

    digest = sha256_file(jsonl)
    manifest = build_manifest(jsonl, report, digest, rep)

    owner = args.owner
    if owner is None:
        if args.dry_run:
            owner = "<owner>"
        else:
            import kagglehub
            try:
                owner = kagglehub.whoami()["username"]
            except Exception as exc:  # noqa: BLE001 -- surface any auth failure as the same hint
                _say(f"not authenticated with Kaggle ({type(exc).__name__}). Put your API token at "
                     f"~/.kaggle/access_token (UTF-8, no BOM -- PowerShell's `>` writes UTF-16 and "
                     f"kagglehub reads that as garbage) or set KAGGLE_API_TOKEN.")
                return 2

    handle = f"{owner}/{args.slug}"
    notes = args.notes or f"{manifest['corpus_file']} | {manifest['n_docs']} docs | " \
                          f"{manifest['total_tokens']} tokens | sha256 {digest[:16]}"

    _say(f"owner    : {owner}")
    _say(f"handle   : {handle}")
    _say(f"corpus   : {jsonl}  ({manifest['size_bytes'] / 2**20:.2f} MiB)")
    _say(f"docs     : {manifest['n_docs']}   tokens: {manifest['total_tokens']}")
    _say(f"sha256   : {digest}")
    _say(f"private  : yes (kagglehub creates a new dataset private by default)")

    with tempfile.TemporaryDirectory(prefix="moe-corpus-stage-") as tmp:
        staging = Path(tmp) / args.slug
        stage(jsonl, report, manifest, staging)
        files = sorted(p.name for p in staging.iterdir())
        _say(f"staged   : {files}")

        if args.dry_run:
            _say("\ndry run -- nothing uploaded. Re-run without --dry-run to publish.")
            return 0

        import kagglehub
        _say("\nuploading ...")
        t0 = time.time()
        try:
            kagglehub.dataset_upload(handle, str(staging), version_notes=notes)
        except Exception as exc:  # noqa: BLE001
            _say(f"PUBLICATION FAILED during upload: {type(exc).__name__}: {exc}")
            return 3
        elapsed = time.time() - t0

    rec = write_publish_record(handle, manifest, elapsed)
    _say(f"\ndone in {elapsed:.1f}s")
    _say(f"url      : https://www.kaggle.com/datasets/{handle}")
    _say(f"record   : {rec}")
    _say("\nAttach this dataset to collect/ladder sessions and set CORPUS_FILE to the .jsonl "
         "under /kaggle/input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
