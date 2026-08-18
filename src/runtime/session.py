"""Session wall-clock budget — plan S.3.

Kaggle batch commits are killed hard at 12 hours. A capture run that discovers this by being
killed mid-shard leaves a partially written trace that is indistinguishable from a good one
without the size and checksum checks. The budget exists so the shard loop stops *voluntarily*,
with time left to flush, checksum and upload.

``reserve_s`` is that headroom. It must cover the slowest plausible flush-plus-upload of one
shard, not the average one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["SessionBudget"]


@dataclass
class SessionBudget:
    """Tracks elapsed wall time against a hard session cap.

    >>> budget = SessionBudget(wall_limit_s=10, reserve_s=4)
    >>> budget.should_stop()          # nothing elapsed yet
    False
    >>> budget.usable_s
    6
    """

    wall_limit_s: float = 12 * 3600
    reserve_s: float = 1800
    _start: float = field(default_factory=time.monotonic, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reserve_s >= self.wall_limit_s:
            raise ValueError(
                f"reserve_s ({self.reserve_s}) must be smaller than wall_limit_s "
                f"({self.wall_limit_s}); there would be no time to do any work"
            )

    @classmethod
    def from_config(cls, run_config: Any) -> "SessionBudget":
        """Build from the ``unhashed.session`` block of ``run.yaml``."""
        session: Mapping[str, Any] = (run_config.unhashed or {}).get("session", {}) or {}
        return cls(
            wall_limit_s=float(session.get("wall_limit_s", 12 * 3600)),
            reserve_s=float(session.get("reserve_s", 1800)),
        )

    # -- clock --------------------------------------------------------------------------

    def elapsed(self) -> float:
        """Seconds since construction. Monotonic, so it is immune to clock adjustment."""
        return time.monotonic() - self._start

    def remaining(self) -> float:
        """Seconds until the hard cap. May be negative."""
        return self.wall_limit_s - self.elapsed()

    @property
    def usable_s(self) -> float:
        """Working time before the reserve begins."""
        return self.wall_limit_s - self.reserve_s

    def should_stop(self) -> bool:
        """True once the reserve window is entered — flush, upload, and exit cleanly."""
        return self.elapsed() >= self.usable_s

    def fraction_used(self) -> float:
        return min(self.elapsed() / self.wall_limit_s, 1.0)

    # -- reporting ----------------------------------------------------------------------

    def summary(self) -> dict[str, float | bool]:
        return {
            "elapsed_s": round(self.elapsed(), 1),
            "remaining_s": round(self.remaining(), 1),
            "usable_s": self.usable_s,
            "should_stop": self.should_stop(),
        }

    def __str__(self) -> str:
        mins = self.elapsed() / 60
        left = self.remaining() / 60
        return f"SessionBudget({mins:.1f} min elapsed, {left:.1f} min to cap)"
