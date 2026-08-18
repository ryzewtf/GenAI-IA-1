"""Test-split evaluation: both metric families for one fitted predictor — plan §1.2.

The contract this module exists to hold:

* ``q_F`` was fit on **train** and early-stopped on **val**. Nothing here refits anything.
* ``H`` and ``CE`` are **both** estimated on **test**. Mixing splits between them is a silent
  bias, so ``mi_lower_bound`` is called with both split labels and raises if they differ
  (invariant I10).
* ``Î = H − CE`` is reported **as measured, including negative values.** Nothing clamps.

Why the epsilon-mix target is the *test* marginal
-------------------------------------------------
§1.2 defines exactly one marginal — ``p(e) = count(e) / (k · N_test)`` on test — and the mix
exists to guarantee a finite CE. The train marginal cannot serve: an expert that is dead on train
but selected on test would get zero mass and CE would be infinite, and T5.3 collects a dead-expert
histogram precisely because that case is expected. The leak this accepts is bounded and blunt —
the test *label marginal*, at weight 1e-6, with no per-token test information reaching the
predictor — and it is the plan's own construction rather than a choice made here.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..metrics.information import (
    CrossEntropyAccumulator,
    DEFAULT_EPSILON,
    entropy,
    expert_counts,
    marginal_distribution,
    mi_lower_bound,
)
from ..metrics.predictive import PredictiveAccumulator
from .base import Predictor

__all__ = ["evaluate_on_test", "TEST_SPLIT"]

TEST_SPLIT = "test"


def _proba(predictor: Predictor, X: Any, rows: np.ndarray) -> np.ndarray:
    """Row-subset probabilities, using a predictor's batched path where it has one."""
    if X is None:  # F0 — constant, so the row identity is irrelevant beyond the count
        probs = predictor.predict_proba(None)
        return np.broadcast_to(probs.reshape(1, -1)[0], (rows.size, probs.shape[-1]))
    if hasattr(X, "densify"):
        log_proba = getattr(predictor, "predict_log_proba", None)
        if log_proba is not None:
            return np.exp(log_proba(X, rows))
        raise TypeError(f"{type(predictor).__name__} cannot score a Design")
    return predictor.predict_proba(np.asarray(X)[rows])


def evaluate_on_test(
    predictor: Predictor,
    X_test: Any,
    y_test: np.ndarray,
    *,
    n_experts: int,
    top_k: int,
    estimator: str = "miller_madow",
    epsilon: float = DEFAULT_EPSILON,
    chunk_rows: int = 65536,
    buckets: np.ndarray | None = None,
    bucket_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Both metric families for one fitted predictor on the test split.

    ``buckets`` optionally assigns each test row a stratum (plan T7.3's frequency deciles, or
    T9.4's domain/language). Per-stratum **CE and Family A** are reported; a per-stratum ``Î`` is
    deliberately *not*, because it would need a per-stratum ``H`` and entropy estimated on a few
    thousand slots over 128 experts is badly enough biased that the resulting number would be
    dominated by stratum size rather than by predictability.
    """
    y = np.asarray(y_test)
    if y.ndim != 2 or not y.size:
        raise ValueError(f"y_test must be a non-empty (n, k) array, got shape {y.shape}")
    if y.shape[1] != top_k:
        raise ValueError(f"y_test has k={y.shape[1]} but top_k={top_k} was declared")
    n = int(y.shape[0])

    marginal = marginal_distribution(y, n_experts)
    entropy_bits = entropy(expert_counts(y, n_experts), estimator)

    predictive = PredictiveAccumulator(top_k=top_k, n_experts=n_experts)
    cross_entropy = CrossEntropyAccumulator(
        n_experts=n_experts, marginal=marginal, epsilon=epsilon, split=TEST_SPLIT
    )

    strata: dict[int, dict[str, Any]] = {}
    if buckets is not None:
        buckets = np.asarray(buckets).ravel()
        if buckets.shape[0] != n:
            raise ValueError(f"buckets has {buckets.shape[0]} rows but y_test has {n}")
        for label in np.unique(buckets):
            strata[int(label)] = {
                "predictive": PredictiveAccumulator(top_k=top_k, n_experts=n_experts),
                "cross_entropy": CrossEntropyAccumulator(
                    n_experts=n_experts, marginal=marginal, epsilon=epsilon, split=TEST_SPLIT
                ),
            }

    for lo in range(0, n, chunk_rows):
        rows = np.arange(lo, min(lo + chunk_rows, n))
        probs = _proba(predictor, X_test, rows)
        chunk_y = y[rows]
        predictive.update(probs, chunk_y)
        cross_entropy.update(probs, chunk_y)
        if buckets is not None:
            chunk_buckets = buckets[rows]
            for label, acc in strata.items():
                sel = chunk_buckets == label
                if sel.any():
                    acc["predictive"].update(probs[sel], chunk_y[sel])
                    acc["cross_entropy"].update(probs[sel], chunk_y[sel])

    ce_bits = cross_entropy.result()
    bound = mi_lower_bound(
        entropy_bits,
        ce_bits,
        n_experts,
        estimator=estimator,
        entropy_split=TEST_SPLIT,
        ce_split=TEST_SPLIT,
    )

    out: dict[str, Any] = {
        "split": TEST_SPLIT,
        "n_rows": n,
        "n_slots": int(y.size),
        "family_a": predictive.result(),
        "family_b": bound.to_json(),
        "epsilon_mix": epsilon,
        "entropy_estimator": estimator,
    }
    if buckets is not None:
        out["strata"] = {
            str(label): {
                "family_a": acc["predictive"].result(),
                "cross_entropy_bits": acc["cross_entropy"].result(),
                "n_slots": acc["cross_entropy"].n_slots,
            }
            for label, acc in sorted(strata.items())
        }
        out["stratification"] = dict(bucket_meta or {})
        out["strata_note"] = (
            "Per-stratum CE only. A per-stratum MI lower bound would need a per-stratum H, and "
            "entropy on a small stratum over many experts is too biased to compare."
        )
    return out
