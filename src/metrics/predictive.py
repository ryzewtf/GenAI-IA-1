"""Family A — predictive metrics — plan T6.2.

The headline, systems-comparable family. All three metrics are computed on the top-k set:

* ``set_agreement@k`` = |S_pred ∩ S_true| / k
* ``recall@m`` for m ∈ {k, 2k, 4k} — fraction of true experts inside the predictor's top-m
* ``exact_match`` = 1 if S_pred == S_true else 0

Every function takes ``(pred_scores: (n, n_experts), true_sets: (n, k))`` where ``true_sets``
comes from ``topk.bin`` — the model's own emitted indices, never a recomputation (invariant I1).

Chunk-composability
-------------------
The plan requires these to compose over chunks, because Qwen3's logit trace is ~12 GB and no
metric may materialise a full model's array. Each metric therefore has a row-wise form
returning ``(n,)`` and :class:`PredictiveAccumulator` folds chunks into running totals. Row-wise
output is also what stratified reporting needs — T9.4 breaks every metric out by domain and
language, and T7.3 by token-frequency decile.

A note on two metrics that coincide
-----------------------------------
``set_agreement@k`` and ``recall@k`` are the same quantity: both are
|S_pred_top-k ∩ S_true| / k. The plan lists both because the two literatures name it
differently, and reporting both under one definition is part of the T9.5 metric audit. They are
computed once and reported twice rather than being allowed to drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "top_m_indices",
    "membership_mask",
    "set_agreement_rows",
    "recall_at_m_rows",
    "exact_match_rows",
    "set_agreement_at_k",
    "recall_at_m",
    "exact_match",
    "PredictiveAccumulator",
]


def _validate(pred_scores: np.ndarray, true_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_scores)
    true = np.asarray(true_sets)

    if pred.ndim != 2:
        raise ValueError(f"pred_scores must be (n, n_experts), got shape {pred.shape}")
    if true.ndim != 2:
        raise ValueError(f"true_sets must be (n, k), got shape {true.shape}")
    if pred.shape[0] != true.shape[0]:
        raise ValueError(
            f"row count mismatch: pred_scores has {pred.shape[0]}, true_sets has {true.shape[0]}"
        )

    n_experts = pred.shape[1]
    if true.size:
        lo, hi = int(true.min()), int(true.max())
        if lo < 0 or hi >= n_experts:
            raise ValueError(
                f"true_sets holds expert index range [{lo}, {hi}], outside [0, {n_experts}). "
                "This is the T5.3 label range check failing — the labels or n_experts are wrong."
            )
    return pred, true


def top_m_indices(pred_scores: np.ndarray, m: int) -> np.ndarray:
    """Indices of the ``m`` highest-scoring experts per row, ``(n, m)``.

    Order within the row is unspecified — only membership is used downstream. Ties are broken
    arbitrarily by ``argpartition``; with continuous predictor scores exact ties are measure-zero,
    but a constant predictor (F0) ties everywhere, which is why F0's reported agreement depends
    on ``m`` and not on any ordering claim.
    """
    m = int(m)
    n_experts = pred_scores.shape[1]
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if m >= n_experts:
        return np.broadcast_to(np.arange(n_experts), pred_scores.shape).copy()
    return np.argpartition(-pred_scores, m - 1, axis=1)[:, :m]


def membership_mask(true_sets: np.ndarray, n_experts: int) -> np.ndarray:
    """Boolean ``(n, n_experts)`` mask of the true expert sets.

    Turns set intersection into a gather, which is what keeps these metrics O(n·m) rather than
    O(n·k·m).
    """
    n = true_sets.shape[0]
    mask = np.zeros((n, n_experts), dtype=bool)
    if n:
        mask[np.arange(n)[:, None], true_sets] = True
    return mask


# -- row-wise forms ---------------------------------------------------------------------------


def recall_at_m_rows(pred_scores: np.ndarray, true_sets: np.ndarray, m: int) -> np.ndarray:
    """Per-row fraction of true experts inside the predictor's top-m, ``(n,)`` float64."""
    pred, true = _validate(pred_scores, true_sets)
    k = true.shape[1]
    if k == 0:
        return np.zeros(pred.shape[0])
    mask = membership_mask(true, pred.shape[1])
    picks = top_m_indices(pred, m)
    hits = np.take_along_axis(mask, picks, axis=1).sum(axis=1)
    return hits / k


def set_agreement_rows(pred_scores: np.ndarray, true_sets: np.ndarray) -> np.ndarray:
    """Per-row ``|S_pred ∩ S_true| / k``, ``(n,)`` float64. Identical to ``recall@k``."""
    return recall_at_m_rows(pred_scores, true_sets, m=np.asarray(true_sets).shape[1])


def exact_match_rows(pred_scores: np.ndarray, true_sets: np.ndarray) -> np.ndarray:
    """Per-row 1.0 if the predicted top-k set equals the true set exactly, ``(n,)`` float64.

    Set-level, so it is the one Family A metric sensitive to within-set structure — which the
    slot-level random variable in Family B deliberately discards (plan §1.2).
    """
    pred, true = _validate(pred_scores, true_sets)
    k = true.shape[1]
    if k == 0:
        return np.ones(pred.shape[0])
    return (set_agreement_rows(pred, true) == 1.0).astype(np.float64)


# -- scalar forms -------------------------------------------------------------------------------


def recall_at_m(pred_scores: np.ndarray, true_sets: np.ndarray, m: int) -> float:
    return float(recall_at_m_rows(pred_scores, true_sets, m).mean())


def set_agreement_at_k(pred_scores: np.ndarray, true_sets: np.ndarray) -> float:
    return float(set_agreement_rows(pred_scores, true_sets).mean())


def exact_match(pred_scores: np.ndarray, true_sets: np.ndarray) -> float:
    return float(exact_match_rows(pred_scores, true_sets).mean())


# -- chunk accumulation ------------------------------------------------------------------------


@dataclass
class PredictiveAccumulator:
    """Folds Family A metrics over chunks so no full-corpus array is ever materialised.

    >>> acc = PredictiveAccumulator(top_k=2, n_experts=4)
    >>> scores = np.array([[9.0, 8.0, 1.0, 0.0]])
    >>> acc.update(scores, np.array([[0, 1]]))
    >>> acc.result()["set_agreement@k"]
    1.0
    """

    top_k: int
    n_experts: int
    recall_multipliers: Sequence[int] = (1, 2, 4)
    n_rows: int = field(default=0, init=False)
    _agreement_sum: float = field(default=0.0, init=False)
    _exact_sum: float = field(default=0.0, init=False)
    _recall_sums: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        self._recall_sums = {m: 0.0 for m in self.recall_targets}

    @property
    def recall_targets(self) -> list[int]:
        """``m`` values for recall@m, capped at ``n_experts``.

        Capping matters for GPT-OSS: k=4 and n_experts=32 is fine, but a coarse model where
        4k > n_experts would otherwise report recall@m == 1.0 as if it were informative.
        """
        seen: list[int] = []
        for mult in self.recall_multipliers:
            m = min(self.top_k * int(mult), self.n_experts)
            if m not in seen:
                seen.append(m)
        return seen

    def update(self, pred_scores: np.ndarray, true_sets: np.ndarray) -> None:
        pred, true = _validate(pred_scores, true_sets)
        if true.shape[1] != self.top_k:
            raise ValueError(
                f"true_sets has k={true.shape[1]} but accumulator was built for "
                f"top_k={self.top_k}"
            )
        if pred.shape[1] != self.n_experts:
            raise ValueError(
                f"pred_scores has {pred.shape[1]} experts but accumulator was built for "
                f"{self.n_experts}"
            )
        if not pred.shape[0]:
            return

        mask = membership_mask(true, self.n_experts)
        agreement: np.ndarray | None = None

        for m in self.recall_targets:
            picks = top_m_indices(pred, m)
            hits = np.take_along_axis(mask, picks, axis=1).sum(axis=1) / self.top_k
            self._recall_sums[m] += float(hits.sum())
            if m == self.top_k:
                agreement = hits

        if agreement is None:  # top_k not among the recall targets
            agreement = recall_at_m_rows(pred, true, self.top_k)

        self._agreement_sum += float(agreement.sum())
        self._exact_sum += float((agreement == 1.0).sum())
        self.n_rows += int(pred.shape[0])

    def result(self) -> dict[str, float | int]:
        if not self.n_rows:
            raise ValueError("no rows accumulated")
        out: dict[str, float | int] = {
            "n": self.n_rows,
            "top_k": self.top_k,
            "n_experts": self.n_experts,
            "set_agreement@k": self._agreement_sum / self.n_rows,
            "exact_match": self._exact_sum / self.n_rows,
        }
        for m, total in self._recall_sums.items():
            out[f"recall@{m}"] = total / self.n_rows
        return out

    def extend(self, chunks: Iterable[tuple[np.ndarray, np.ndarray]]) -> "PredictiveAccumulator":
        for pred, true in chunks:
            self.update(pred, true)
        return self
