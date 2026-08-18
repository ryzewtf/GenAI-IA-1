"""Shard completion state — plan S.3.

``state.json`` is the resumption ledger for one (model, corpus) trace. It answers exactly one
question: *which shards are finished and safe to build on?*

The definition of "finished" is strict, and the strictness is the point (plan S.3 step d):

    captured -> lockstep asserted -> checksummed -> uploaded -> round-trip verified -> complete

A shard is recorded only after the final step. A run killed at any earlier point leaves the
shard absent from the ledger, so the next session recollects it. That is cheap; merging a
half-uploaded shard is not recoverable after the fact.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["ShardState", "ShardRecord", "StateError"]

SCHEMA_VERSION = 1


class StateError(RuntimeError):
    """The shard ledger is malformed, or conflicts with the current run config."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ShardRecord:
    """One completed shard. Every field is evidence that the shard is trustworthy."""

    shard_id: int
    n_tokens: int
    n_captured: int
    file_sha256: dict[str, str]
    doc_range: tuple[int, int]
    completed_utc: str = field(default_factory=_utc_now)
    upload_verified: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "n_tokens": self.n_tokens,
            "n_captured": self.n_captured,
            "file_sha256": dict(self.file_sha256),
            "doc_range": list(self.doc_range),
            "completed_utc": self.completed_utc,
            "upload_verified": self.upload_verified,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ShardRecord":
        doc_range = payload.get("doc_range") or [0, 0]
        return cls(
            shard_id=int(payload["shard_id"]),
            n_tokens=int(payload["n_tokens"]),
            n_captured=int(payload["n_captured"]),
            file_sha256=dict(payload.get("file_sha256", {})),
            doc_range=(int(doc_range[0]), int(doc_range[1])),
            completed_utc=payload.get("completed_utc", ""),
            upload_verified=bool(payload.get("upload_verified", False)),
        )


class ShardState:
    """Load/modify/save the shard ledger for one (model, corpus) trace."""

    def __init__(
        self,
        path: Path | str,
        model: str,
        corpus: str,
        run_config_sha256: str,
        shards: dict[int, ShardRecord] | None = None,
    ) -> None:
        self.path = Path(path)
        self.model = model
        self.corpus = corpus
        self.run_config_sha256 = run_config_sha256
        self._shards: dict[int, ShardRecord] = dict(shards or {})

    # -- construction --------------------------------------------------------------------

    @classmethod
    def load_or_create(
        cls, path: Path | str, model: str, corpus: str, run_config_sha256: str
    ) -> "ShardState":
        """Resume an existing ledger, or start a fresh one.

        Refuses to resume a ledger written under a different run config. Continuing would
        append shards that cannot legally be concatenated with the ones already there — the
        exact silent corruption plan S.3 is written to prevent.
        """
        path = Path(path)
        if not path.exists():
            return cls(path, model, corpus, run_config_sha256)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateError(f"{path}: malformed state.json — {exc}") from exc

        for key, want in (("model", model), ("corpus", corpus)):
            got = payload.get(key)
            if got != want:
                raise StateError(f"{path}: state is for {key}={got!r}, expected {want!r}")

        got_hash = payload.get("run_config_sha256")
        if got_hash != run_config_sha256:
            raise StateError(
                f"{path}: existing shards were collected under run_config_sha256="
                f"{got_hash!r}, current config is {run_config_sha256!r}.\n"
                "These cannot be merged (plan S.3). Either restore the original run.yaml, or "
                "collect into a new output root."
            )

        shards = {
            int(sid): ShardRecord.from_json(rec)
            for sid, rec in (payload.get("shards") or {}).items()
        }
        return cls(path, model, corpus, run_config_sha256, shards)

    # -- queries -------------------------------------------------------------------------

    def completed_ids(self) -> set[int]:
        return set(self._shards)

    def is_complete(self, shard_id: int) -> bool:
        return shard_id in self._shards

    def pending(self, all_shard_ids: Iterable[int]) -> list[int]:
        """Incomplete shards, in fixed sorted order so resumption is deterministic."""
        return sorted(set(all_shard_ids) - self.completed_ids())

    def record(self, shard_id: int) -> ShardRecord | None:
        return self._shards.get(shard_id)

    @property
    def n_tokens(self) -> int:
        return sum(r.n_tokens for r in self._shards.values())

    def __len__(self) -> int:
        return len(self._shards)

    def __contains__(self, shard_id: object) -> bool:
        return shard_id in self._shards

    # -- mutation ------------------------------------------------------------------------

    def mark_complete(self, record: ShardRecord, *, save: bool = True) -> None:
        """Record a shard as finished. Only call this after the upload round-trip verified."""
        if not record.upload_verified:
            raise StateError(
                f"shard {record.shard_id}: refusing to mark complete before the upload "
                "round-trip is verified (plan S.3 step d). A truncated upload that silently "
                "succeeds is the most likely way to lose a session's work."
            )
        if not record.file_sha256:
            raise StateError(f"shard {record.shard_id}: no file checksums recorded")

        existing = self._shards.get(record.shard_id)
        if existing is not None and existing.file_sha256 != record.file_sha256:
            raise StateError(
                f"shard {record.shard_id} is already complete with different checksums.\n"
                f"  existing: {existing.file_sha256}\n"
                f"  new:      {record.file_sha256}\n"
                "Recollecting a shard should be bit-identical; if it is not, an unpinned "
                "variable is in play (see T3.6)."
            )

        self._shards[record.shard_id] = record
        if save:
            self.save()

    # -- persistence ---------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "corpus": self.corpus,
            "run_config_sha256": self.run_config_sha256,
            "n_shards_complete": len(self._shards),
            "n_tokens_complete": self.n_tokens,
            "updated_utc": _utc_now(),
            "shards": {str(sid): rec.to_json() for sid, rec in sorted(self._shards.items())},
        }

    def save(self) -> Path:
        """Write atomically — a ledger truncated by a session kill is worse than a stale one.

        ``os.replace`` is atomic on both POSIX and Windows, and the fsync before it means the
        contents are durable before the rename makes them visible.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_json(), indent=2)

        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return self.path

    def __repr__(self) -> str:
        return (
            f"ShardState(model={self.model!r}, corpus={self.corpus!r}, "
            f"complete={len(self._shards)}, tokens={self.n_tokens})"
        )
