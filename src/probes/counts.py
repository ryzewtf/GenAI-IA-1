"""F1 — token-identity predictor and frequency stratification — plan T7.3.

The plan specifies "embedding table ``V × n_experts`` → softmax" and notes it is *equivalent to a
smoothed conditional count table*. It is worth being precise about why, because the equivalence is
exact and it removes an entire optimizer from the critical path:

The model is ``q(e | v) = softmax(W[v])`` with one free parameter row per token type. Rows are
independent and an unconstrained softmax row can represent any distribution over experts, so the
minimiser of the slot-level cross-entropy of §1.2 is, row by row, the empirical conditional slot
frequency. **The count table is not an approximation of the trained embedding — it is that model's
closed-form optimum.** So F1 is computed in one pass over train, is bit-for-bit reproducible with
no seed, and cannot be under-trained. What the gradient version *would* still need is a way to
handle unseen token types, which here is the backoff weight ``alpha`` selected on val.

Memory (plan T7.3)
------------------
Qwen3: V ≈ 152k, ``n_experts`` = 128 → a dense fp32 table is 78 MB per layer and 3.7 GB across 48
layers. Counts are held as int32 and rows are normalised on demand for the requested batch only,
so a fitted F1 costs one table, never two, and the sweep fits **layer by layer**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .base import FitReport

__all__ = ["CountTablePredictor", "frequency_buckets", "UNSEEN_BUCKET"]

UNSEEN_BUCKET = -1
"""Bucket label for token types never seen on train — not a decile of the frequency scale."""

DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


@dataclass
class CountTablePredictor:
    """``q(e | token_id)`` as a train-split count table backed off to the train marginal.

    ``X`` is a ``(n,)`` array of token ids. ``y`` is ``(n, top_k)`` expert sets.

    Smoothing::

        q(e | v) = (count[v, e] + alpha * k * m_train[e]) / (rowsum[v] + alpha * k)

    The pseudo-count mass is distributed *along the train marginal* rather than uniformly, so an
    unseen token type falls back to F0 exactly and a rare type is shrunk toward F0 rather than
    toward a flat distribution the router never produces. ``alpha`` is chosen on **val** — this is
    the closed-form counterpart of the plan's "early-stop on val", and it is the only fitted
    hyperparameter F1 has.
    """

    n_experts: int
    vocab_size: int
    layer: int = -1
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID
    counts: np.ndarray | None = field(default=None, init=False)
    row_sums: np.ndarray | None = field(default=None, init=False)
    train_marginal: np.ndarray | None = field(default=None, init=False)
    alpha: float = field(default=float("nan"), init=False)
    top_k: int = field(default=0, init=False)
    report: FitReport | None = field(default=None, init=False)

    name = "count_table"

    # -- fitting ---------------------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        ids = np.asarray(X_train).ravel().astype(np.intp)
        sets = np.asarray(y_train)
        if sets.ndim != 2 or not sets.size:
            raise ValueError(f"y_train must be a non-empty (n, k) array, got {sets.shape}")
        if ids.shape[0] != sets.shape[0]:
            raise ValueError(f"row mismatch: X_train {ids.shape[0]}, y_train {sets.shape[0]}")
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(
                f"token ids span [{ids.min()}, {ids.max()}], outside [0, {self.vocab_size})"
            )

        self.top_k = int(sets.shape[1])
        flat = (
            np.repeat(ids, self.top_k) * self.n_experts + sets.astype(np.intp).ravel()
        )
        table = np.bincount(flat, minlength=self.vocab_size * self.n_experts)
        self.counts = table.reshape(self.vocab_size, self.n_experts).astype(np.int32)
        self.row_sums = self.counts.sum(axis=1, dtype=np.int64)

        col = self.counts.sum(axis=0, dtype=np.int64).astype(np.float64)
        self.train_marginal = (col + 1.0) / (col.sum() + self.n_experts)

        history: list[float] = []
        best = (float("inf"), float(self.alpha_grid[0]))
        if X_val is not None and y_val is not None and np.asarray(y_val).size:
            for a in self.alpha_grid:
                self.alpha = float(a)
                ce = self._slot_ce(np.asarray(X_val), np.asarray(y_val))
                history.append(ce)
                if ce < best[0]:
                    best = (ce, float(a))
        else:
            history.append(float("nan"))
        self.alpha = best[1]

        self.report = FitReport(
            feature="F1",
            predictor=self.name,
            layer=self.layer,
            n_train_rows=int(sets.shape[0]),
            n_val_rows=int(np.asarray(y_val).shape[0]) if y_val is not None else 0,
            top_k=self.top_k,
            n_experts=self.n_experts,
            design_width=int(self.vocab_size),
            epochs_run=len(history),
            best_epoch=int(np.argmin(history)) if np.isfinite(history[0]) else -1,
            best_val_ce_bits=best[0] if np.isfinite(best[0]) else float("nan"),
            val_ce_history=history,
            hyperparams={
                "alpha": self.alpha,
                "alpha_grid": [float(a) for a in self.alpha_grid],
                "vocab_size": int(self.vocab_size),
                "closed_form": True,
            },
            blocks=[{"block": "token_id", "kind": "count_table", "width": int(self.vocab_size)}],
        )

    def _slot_ce(self, ids: np.ndarray, sets: np.ndarray) -> float:
        """Per-slot CE in bits on the given rows, batched so no dense V-table is materialised."""
        ids = np.asarray(ids).ravel().astype(np.intp)
        sets = np.asarray(sets).astype(np.intp)
        total = 0.0
        n_slots = 0
        for lo in range(0, ids.shape[0], 65536):
            hi = min(lo + 65536, ids.shape[0])
            probs = self.predict_proba(ids[lo:hi])
            picked = np.take_along_axis(probs, sets[lo:hi], axis=1)
            total += float(-np.log2(picked).sum())
            n_slots += int(picked.size)
        return total / n_slots if n_slots else float("nan")

    # -- prediction -------------------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.counts is None or self.row_sums is None or self.train_marginal is None:
            raise RuntimeError("CountTablePredictor.fit has not been called")
        ids = np.asarray(X).ravel().astype(np.intp)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(
                f"token ids span [{ids.min()}, {ids.max()}], outside [0, {self.vocab_size})"
            )
        alpha = self.alpha if np.isfinite(self.alpha) else 1.0
        prior = alpha * self.top_k * self.train_marginal  # (n_experts,)
        num = self.counts[ids].astype(np.float64) + prior
        den = self.row_sums[ids].astype(np.float64) + alpha * self.top_k
        return num / den[:, None]

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """``log q(e | token_id)`` — the F6 feature block (plan T7.6).

        The plan's F6 concatenates "F1 embedding output". In the closed-form formulation the
        embedding row's role is played by the log of the smoothed conditional distribution, which
        is the same object up to the softmax's arbitrary additive constant per row.
        """
        return np.log(self.predict_proba(X)).astype(np.float32)


# -- frequency stratification ---------------------------------------------------------------------


def frequency_buckets(
    train_token_ids: np.ndarray,
    eval_token_ids: np.ndarray,
    *,
    n_buckets: int = 10,
    vocab_size: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign each eval row a train-frequency decile of its **token type** — plan T7.3.

    Returns ``(bucket_of_row, meta)``. Bucket 0 is the rarest decile of types, ``n_buckets - 1``
    the most frequent, and :data:`UNSEEN_BUCKET` (-1) collects types that never occur on train.

    Two decisions worth stating, because both change what the reported curve means:

    * Deciles are over **token types**, as the plan words it — rank types by train frequency and
      cut into ``n_buckets`` equal-sized groups of types. The top decile therefore holds a small
      minority of the vocabulary carrying most of the *tokens*, which is exactly the asymmetry the
      prediction is about; ``slots_per_bucket`` is returned so that is visible rather than
      implied. Equal-token-mass buckets would give a flatter, less interpretable picture.
    * Unseen types get their own bucket instead of joining decile 0. Lumping them in would mix
      "rare on train" with "absent from train", and F1's behaviour on those two is different by
      construction: absent types fall back exactly to F0.
    """
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")
    train = np.asarray(train_token_ids).ravel().astype(np.intp)
    ev = np.asarray(eval_token_ids).ravel().astype(np.intp)
    size = int(vocab_size) if vocab_size is not None else int(max(train.max(), ev.max()) + 1)

    freq = np.bincount(train, minlength=size).astype(np.int64)
    seen = np.flatnonzero(freq > 0)
    if not seen.size:
        raise ValueError("no token type occurs on train")

    # Rank seen types by frequency, then cut into equal-sized groups of types. array_split
    # handles a type count that is not divisible by n_buckets without dropping a group.
    order = seen[np.argsort(freq[seen], kind="stable")]
    bucket_of_type = np.full(size, UNSEEN_BUCKET, dtype=np.int32)
    for b, chunk in enumerate(np.array_split(order, n_buckets)):
        bucket_of_type[chunk] = b

    buckets = bucket_of_type[ev]
    labels = list(range(n_buckets)) + [UNSEEN_BUCKET]
    meta: dict[str, Any] = {
        "n_buckets": n_buckets,
        "unseen_bucket": UNSEEN_BUCKET,
        "types_per_bucket": {
            str(b): int((bucket_of_type == b).sum()) for b in labels
        },
        "rows_per_bucket": {str(b): int((buckets == b).sum()) for b in labels},
        "train_freq_range_per_bucket": {
            str(b): (
                [int(freq[bucket_of_type == b].min()), int(freq[bucket_of_type == b].max())]
                if (bucket_of_type == b).any() and b != UNSEEN_BUCKET
                else None
            )
            for b in labels
        },
    }
    return buckets, meta
