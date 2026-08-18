"""Predictor interface and feature-design primitives — plan T7.1.

The plan's interface is two methods::

    class Predictor(Protocol):
        def fit(self, X_train, y_train, X_val, y_val) -> None
        def predict_proba(self, X) -> np.ndarray   # (n, n_experts), rows sum to 1

``y`` throughout Phase 7 is the **true expert set**, ``(n, top_k)`` int, straight from
``topk.bin`` (invariant I1). It is never a single label: the slot-level random variable of plan
§1.2 treats the k selected experts as k draws from one distribution, so the natural training
target for a row is the uniform distribution over its selected set, ``multi_hot(S) / k``.

Why ``X`` is a :class:`Design` and not an array
-----------------------------------------------
The dense form of these features is large and mostly zeros. Qwen3's F6 design is
``3 * 128 = 384`` columns over 800k train rows: 1.2 GB in fp32, against a 4 GB RSS budget, and
that is *per layer* of 48. Every block here therefore stores its compact form (integer expert
indices, a token-id vector) and densifies only the rows of the current minibatch. The multi-hot
blocks in particular are exactly k ones per row, so densification is a scatter, not a matmul.

A ``Design`` **is** the ``X`` of the protocol — the plan's signature is satisfied as written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "UndefinedFeature",
    "Predictor",
    "FeatureBlock",
    "MultiHotBlock",
    "DenseBlock",
    "Design",
    "Standardizer",
    "FitReport",
    "soft_targets",
]


class UndefinedFeature(Exception):
    """The feature does not exist for this (layer, split) — not a failure.

    F2 conditions on layer ℓ−1 and so is undefined at ℓ = 0; F6 contains F2 and inherits that.
    The sweep records these as *skipped with a reason* rather than as errors or, worse, as a
    silently missing row in the results table (plan T7.4).
    """


@runtime_checkable
class Predictor(Protocol):
    """Fit on train, early-stop on val, report on test. **Never touch test during fitting.**"""

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any,
        y_val: np.ndarray,
    ) -> None: ...

    def predict_proba(self, X: Any) -> np.ndarray: ...


# -- feature blocks ----------------------------------------------------------------------------


class FeatureBlock(Protocol):
    """One contiguous column group of a design matrix."""

    @property
    def n_rows(self) -> int: ...

    @property
    def width(self) -> int: ...

    def densify(self, rows: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class MultiHotBlock:
    """Multi-hot indicator of an expert set, ``(b, n_experts)`` — plan T7.4.

    Exact-set lookup is what this replaces: C(128, 8) ≈ 1.43e12 possible sets, so the set is
    featurized rather than enumerated. Values are 1.0, not 1/k — scaling a linear layer's input
    by a constant is absorbed into the weights, and leaving it at 1.0 keeps the learned weights
    directly readable as "expert j at the conditioning site raises expert i's logit by w[j, i]".
    """

    sets: np.ndarray
    n_experts: int
    name: str = "multi_hot"

    def __post_init__(self) -> None:
        self.sets = np.asarray(self.sets)
        if self.sets.ndim != 2:
            raise ValueError(f"sets must be (n, k), got shape {self.sets.shape}")
        if self.sets.size:
            lo, hi = int(self.sets.min()), int(self.sets.max())
            if lo < 0 or hi >= self.n_experts:
                raise ValueError(
                    f"conditioning expert indices span [{lo}, {hi}], outside "
                    f"[0, {self.n_experts})"
                )

    @property
    def n_rows(self) -> int:
        return int(self.sets.shape[0])

    @property
    def width(self) -> int:
        return int(self.n_experts)

    def densify(self, rows: np.ndarray) -> np.ndarray:
        picks = self.sets[rows]
        out = np.zeros((picks.shape[0], self.n_experts), dtype=np.float32)
        out[np.arange(picks.shape[0])[:, None], picks.astype(np.intp)] = 1.0
        return out

    def describe(self) -> dict[str, Any]:
        return {"block": self.name, "kind": "multi_hot", "width": self.width}


@dataclass
class DenseBlock:
    """A dense float feature block, optionally standardized with **train-split** statistics.

    Used for the router-input probes (F4/F5/FV) and for F1's log-probability block inside F6.
    fp32 throughout: plan T7.5 keeps these probes in fp32 so that the one measurement defining
    the practical ceiling does not carry a precision question of its own.
    """

    values: np.ndarray
    standardizer: "Standardizer | None" = None
    name: str = "dense"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32)
        if self.values.ndim != 2:
            raise ValueError(f"values must be (n, d), got shape {self.values.shape}")
        if self.standardizer is not None and self.standardizer.width != self.values.shape[1]:
            raise ValueError(
                f"standardizer was fit on width {self.standardizer.width} but this block is "
                f"width {self.values.shape[1]}"
            )

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    def densify(self, rows: np.ndarray) -> np.ndarray:
        out = self.values[rows]
        if self.standardizer is not None:
            out = self.standardizer.transform(out)
        return np.ascontiguousarray(out, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "block": self.name,
            "kind": "dense",
            "width": self.width,
            "standardized": self.standardizer is not None,
        }


@dataclass
class Design:
    """Row-aligned feature blocks, densified per minibatch."""

    blocks: tuple[FeatureBlock, ...]

    def __post_init__(self) -> None:
        self.blocks = tuple(self.blocks)
        if not self.blocks:
            raise ValueError("a Design needs at least one block; F0 takes no design at all")
        counts = {b.n_rows for b in self.blocks}
        if len(counts) != 1:
            raise ValueError(f"blocks disagree on row count: {sorted(counts)}")

    @property
    def n_rows(self) -> int:
        return self.blocks[0].n_rows

    @property
    def width(self) -> int:
        return sum(b.width for b in self.blocks)

    def densify(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.intp)
        if len(self.blocks) == 1:
            return self.blocks[0].densify(rows)
        return np.hstack([b.densify(rows) for b in self.blocks])

    def describe(self) -> list[dict[str, Any]]:
        return [b.describe() for b in self.blocks]


# -- standardization ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Standardizer:
    """Zero-mean unit-variance transform. **Fit on train only** (plan T7.5)."""

    mean: np.ndarray
    scale: np.ndarray

    @property
    def width(self) -> int:
        return int(self.mean.shape[0])

    @classmethod
    def fit(cls, values: np.ndarray, *, floor: float = 1e-6) -> "Standardizer":
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"values must be (n, d), got shape {arr.shape}")
        if not arr.shape[0]:
            raise ValueError("cannot fit a standardizer on zero rows")
        mean = arr.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = arr.std(axis=0, dtype=np.float64).astype(np.float32)
        # A constant router-input dimension is a dead feature, not a divide-by-zero.
        return cls(mean=mean, scale=np.maximum(std, np.float32(floor)))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.scale


# -- targets -------------------------------------------------------------------------------------


def soft_targets(true_sets: np.ndarray, n_experts: int) -> np.ndarray:
    """``multi_hot(S) / k`` — the slot-level target of plan §1.2, ``(n, n_experts)`` float32.

    Minimising cross-entropy against this target is *exactly* minimising the per-slot CE the
    plan reports, because a row's k slots are k draws from one distribution over experts. The
    gradient of that loss w.r.t. the logits is ``softmax(z) - multi_hot(S)/k``, which is why the
    training loop below needs no special-casing for k > 1.
    """
    sets = np.asarray(true_sets)
    if sets.ndim != 2:
        raise ValueError(f"true_sets must be (n, k), got shape {sets.shape}")
    n, k = sets.shape
    if k == 0:
        raise ValueError("true_sets has k=0")
    out = np.zeros((n, n_experts), dtype=np.float32)
    if n:
        out[np.arange(n)[:, None], sets.astype(np.intp)] = np.float32(1.0 / k)
    return out


# -- reporting -----------------------------------------------------------------------------------


@dataclass
class FitReport:
    """What was fit, on how much, and where it stopped — written into every result JSON."""

    feature: str
    predictor: str
    layer: int
    n_train_rows: int = 0
    n_val_rows: int = 0
    top_k: int = 0
    n_experts: int = 0
    design_width: int = 0
    epochs_run: int = 0
    best_epoch: int = -1
    best_val_ce_bits: float = float("nan")
    val_ce_history: list[float] = field(default_factory=list)
    hyperparams: dict[str, Any] = field(default_factory=dict)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    n_excluded: int = 0
    exclusion_reason: str | None = None

    @property
    def n_train_slots(self) -> int:
        return self.n_train_rows * self.top_k

    @property
    def n_val_slots(self) -> int:
        return self.n_val_rows * self.top_k

    def to_json(self) -> dict[str, Any]:
        out = {
            "feature": self.feature,
            "predictor": self.predictor,
            "layer": self.layer,
            "n_train_rows": self.n_train_rows,
            "n_val_rows": self.n_val_rows,
            "n_train_slots": self.n_train_slots,
            "n_val_slots": self.n_val_slots,
            "top_k": self.top_k,
            "n_experts": self.n_experts,
            "design_width": self.design_width,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_ce_bits": self.best_val_ce_bits,
            "hyperparams": dict(self.hyperparams),
            "blocks": list(self.blocks),
            "n_excluded": self.n_excluded,
            "exclusion_reason": self.exclusion_reason,
        }
        return out
