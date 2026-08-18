"""F0 — the marginal baseline — plan T7.2.

A constant ``p(e)`` estimated on the **train** split. Every other feature must beat this.

The one thing this module exists to prevent
-------------------------------------------
Plan T7.2: *"Note this is the predictor; the H in §1.2 is estimated on test. They are different
numbers and the code must not conflate them."*

They are easy to conflate because they are the same formula applied to different rows, and if you
conflate them F0's ``Î`` comes out at exactly 0.000 for every model and layer — a number that
looks like a clean baseline and is actually a bug. :class:`MarginalPredictor` therefore records
which split it was fit on and refuses to be fit on test at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..metrics.information import marginal_distribution
from .base import FitReport

__all__ = ["MarginalPredictor", "reference_marginal_on_test"]


@dataclass
class MarginalPredictor:
    """Constant predictor: ``q(e | f) = p_train(e)`` for every row.

    ``X`` is ignored except for its row count, so ``fit`` accepts ``None`` designs.
    """

    n_experts: int
    top_k: int = 0
    layer: int = -1
    fit_split: str = "train"
    laplace: float = 1.0
    probs: np.ndarray | None = field(default=None, init=False)
    report: FitReport | None = field(default=None, init=False)

    name = "marginal"

    def __post_init__(self) -> None:
        if self.fit_split == "test":
            raise ValueError(
                "F0 is a predictor and must be fit on train. The marginal estimated on test is "
                "the H of plan §1.2, a different quantity — see this module's docstring."
            )
        if self.laplace < 0.0:
            raise ValueError(f"laplace must be >= 0, got {self.laplace}")

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        sets = np.asarray(y_train)
        if sets.ndim != 2 or not sets.size:
            raise ValueError(f"y_train must be a non-empty (n, k) array, got shape {sets.shape}")

        counts = np.bincount(
            sets.astype(np.intp).ravel(), minlength=self.n_experts
        ).astype(np.float64)
        # Add-one so a dead expert on train cannot give a selected expert zero mass on test. The
        # epsilon-mix in the metric would also cover it, but a predictor that emits a hard zero
        # for a reachable outcome is wrong independently of what the metric does about it.
        smoothed = counts + self.laplace
        self.probs = smoothed / smoothed.sum()
        self.top_k = int(sets.shape[1])

        val_ce = float("nan")
        if y_val is not None and np.asarray(y_val).size:
            val_sets = np.asarray(y_val).astype(np.intp)
            val_ce = float(-np.log2(self.probs[val_sets]).mean())

        self.report = FitReport(
            feature="F0",
            predictor=self.name,
            layer=self.layer,
            n_train_rows=int(sets.shape[0]),
            n_val_rows=int(np.asarray(y_val).shape[0]) if y_val is not None else 0,
            top_k=self.top_k,
            n_experts=self.n_experts,
            design_width=0,
            epochs_run=0,
            best_epoch=0,
            best_val_ce_bits=val_ce,
            hyperparams={"fit_split": self.fit_split, "laplace": self.laplace},
            blocks=[],
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.probs is None:
            raise RuntimeError("MarginalPredictor.fit has not been called")
        n = 1 if X is None else int(getattr(X, "n_rows", len(X)))
        return np.broadcast_to(self.probs, (n, self.n_experts))

    def train_marginal(self) -> np.ndarray:
        """The fitted train-split distribution. Not to be used as the §1.2 ``H`` reference."""
        if self.probs is None:
            raise RuntimeError("MarginalPredictor.fit has not been called")
        return self.probs.copy()


def reference_marginal_on_test(true_sets_test: np.ndarray, n_experts: int) -> np.ndarray:
    """The §1.2 reference distribution: ``p(e) = count(e) / (k * N_test)``, estimated on TEST.

    Deliberately a free function with ``test`` in its name rather than a method on the predictor,
    so that no code path can reach it by accident while holding a fitted F0.
    """
    return marginal_distribution(true_sets_test, n_experts)
