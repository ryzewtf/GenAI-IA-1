"""Tie-aware rank statistics, shared by T3.2 and Phase 8.

Both tasks compare two orderings of the same router logits and both hit the same problem: after
the fp16 down-cast, a third to a half of real (token, layer) rows contain at least one tied pair.
Ordinal ranking would invent an arbitrary order for those and then charge the resulting rank
difference to whatever is under test — the PyTorch implementation in T3.2, the quantization level
in T8.2. Averaging ties is the standard fix and it is what ``scipy.stats.rankdata`` does; this
module hand-rolls it so the metric core keeps a single dependency, and the tests check it against
scipy rather than against constants written to match the implementation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["average_ranks", "spearman_rows"]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Row-wise ranks with ties averaged, matching ``scipy.stats.rankdata(..., 'average')``.

    Hand-rolled rather than imported so the metric core keeps a single dependency, and because
    the tie handling is the whole point: ordinal ranks would break the 33-56% of real rows that
    contain a tied pair in an arbitrary direction, and then charge the resulting rank difference
    to whatever is under test.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    n_rows, n = values.shape
    order = np.argsort(values, axis=1, kind="stable")
    ordinal = np.empty_like(order)
    np.put_along_axis(ordinal, order, np.broadcast_to(np.arange(n), (n_rows, n)), axis=1)

    ranks = (ordinal + 1).astype(np.float64)
    sorted_values = np.take_along_axis(values, order, axis=1)
    # Boundaries of runs of equal values, per row.
    same = np.zeros((n_rows, n + 1), dtype=bool)
    same[:, 1:n] = sorted_values[:, 1:] == sorted_values[:, :-1]
    for r in range(n_rows):
        starts = np.flatnonzero(~same[r, :n])
        ends = np.append(starts[1:], n)
        for start, end in zip(starts, ends):
            if end - start > 1:
                ranks[r, order[r, start:end]] = (start + end + 1) / 2.0
    return ranks


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Spearman rho between two (rows, n) arrays.

    Returns NaN for a row where either side has zero rank variance -- every value tied. That is
    genuinely undefined rather than zero, and averaging a fabricated 0.0 into the layer score
    would turn a degenerate row into evidence of disagreement.
    """
    ra, rb = average_ranks(a), average_ranks(b)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (ra * rb).sum(axis=1) / denom
    return np.where(denom > 0, rho, np.nan)
