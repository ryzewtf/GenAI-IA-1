"""Router-logit regression probes (F4R, FVR) with a set-aware decoder — audit L3 / M13.

Why a second router-input probe exists
--------------------------------------
F4/FV fit a multinomial-logistic probe to the *executed set* only. Two things limit that probe
before the information in the router input does:

* **The output family.** A softmax over experts cannot put its mass uniformly on a set, so even
  given the exact input the per-slot CE stays above the ``log2 k`` floor that only a set-uniform
  ``q`` reaches (:mod:`src.metrics.information`). FV, which sees the exact router input of layer
  ℓ, reached only 0.81–0.86 set agreement and ~0.63–0.70 of the attainable Î.
* **The supervision.** ~60k rows × k labels to fit a ``d × N`` map from scratch, and the L-BFGS
  fit stopped at its iteration cap on every layer.

The router itself is (up to a monotone softmax) a linear map of its input, and the trace stores
its scores. So this probe regresses the **router scores of layer ℓ** on the router input —
ridge, closed form, λ chosen on val — and decodes the predicted scores into a distribution that
*can* be set-uniform:

    τ_t     = midpoint of the k-th and (k+1)-th largest predicted score of row t
    q_t(e)  ∝ sigmoid((ẑ_t(e) − τ_t) / T)              T chosen on val

With exact scores (FVR) and T → 0 this tends to the uniform distribution on the executed set,
i.e. to the attainable ceiling. FVR is therefore both the alignment check (closed-form R² of
input → scores, plan T3.3) and the demonstration that the ceiling is reachable; F4R (input of
layer ℓ−1 → scores of layer ℓ) is the router-input predictability estimate that is not limited
by the probe's output family.

Targets
-------
For 6/7 models ``logits.bin`` holds softmax *probabilities*; only gpt-oss stores raw logits.
Softmax is shift-invariant, so centered log-probabilities equal centered logits and are the
linear target. Probabilities that underflowed to 0 in fp16 are floored before the log; only the
ranks near the k-th place matter to the decoder, and the floor is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["score_targets", "RidgeScoreProbe", "set_decode", "PROB_FLOOR"]

PROB_FLOOR = 6.0e-8
"""Smallest positive fp16 magnitude; probabilities below it are floored before the log."""

DEFAULT_LAMBDAS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
"""Ridge strengths relative to the mean eigenvalue of the train Gram matrix."""

DEFAULT_TEMPS: tuple[float, ...] = tuple(float(t) for t in np.geomspace(1e-3, 10.0, 41))
"""Decoder temperatures, in units of the row's own score spread (see :func:`set_decode`)."""


def score_targets(stored: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Centered linear router scores from the stored routing tensor, plus a description.

    Probabilities (non-negative rows summing to 1) become centered log-probabilities; anything
    else is treated as raw logits and only centered.
    """
    z = np.asarray(stored, dtype=np.float64)
    row_sums = z.sum(axis=1)
    is_probs = bool(np.all(z >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-2))
    info: dict[str, Any] = {"stored_kind": "probs" if is_probs else "logits"}
    if is_probs:
        floored = z < PROB_FLOOR
        info["frac_floored"] = float(floored.mean())
        z = np.log(np.maximum(z, PROB_FLOOR))
    z = z - z.mean(axis=1, keepdims=True)
    return z.astype(np.float32), info


def _row_scale(z: np.ndarray) -> np.ndarray:
    """Per-row spread used to make one temperature meaningful across rows and layers."""
    return np.maximum(z.std(axis=1, keepdims=True), 1e-6)


def set_decode(z: np.ndarray, k: int, temperature: float) -> np.ndarray:
    """Predicted scores → a distribution that approaches uniform-on-top-k as T → 0."""
    z = np.asarray(z, dtype=np.float64)
    n_experts = z.shape[1]
    if k >= n_experts:
        return np.full_like(z, 1.0 / n_experts)
    part = -np.partition(-z, kth=(k - 1, k), axis=1)
    tau = 0.5 * (part[:, k - 1] + part[:, k])[:, None]
    x = (z - tau) / (temperature * _row_scale(z))
    # log-sigmoid, stable, then normalise in log space
    log_s = -np.logaddexp(0.0, -x)
    log_s -= log_s.max(axis=1, keepdims=True)
    q = np.exp(log_s)
    q /= q.sum(axis=1, keepdims=True)
    return q


def _slot_ce_bits(q: np.ndarray, sets: np.ndarray) -> float:
    picked = np.take_along_axis(q, sets.astype(np.intp), axis=1)
    return float(-np.log2(np.maximum(picked, 1e-300)).mean())


def _set_agreement(z: np.ndarray, sets: np.ndarray, k: int) -> float:
    top = np.argpartition(-z, kth=k - 1, axis=1)[:, :k]
    hit = (top[:, :, None] == sets[:, None, :]).any(axis=2)
    return float(hit.mean())


@dataclass
class RidgeScoreProbe:
    """Ridge regression ``standardized router input → centered router scores``, set-decoded."""

    n_experts: int
    top_k: int
    feature: str = "F4R"
    layer: int = -1
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    temps: tuple[float, ...] = DEFAULT_TEMPS

    W: np.ndarray | None = field(default=None, init=False)
    b: np.ndarray | None = field(default=None, init=False)
    lam: float = field(default=float("nan"), init=False)
    temperature: float = field(default=float("nan"), init=False)
    report: dict[str, Any] = field(default_factory=dict, init=False)

    name = "ridge_score_probe"

    def fit(
        self,
        X_train: np.ndarray,
        Z_train: np.ndarray,
        X_val: np.ndarray,
        Z_val: np.ndarray,
        sets_val: np.ndarray,
    ) -> "RidgeScoreProbe":
        """Closed-form ridge for every λ; λ by val set agreement, then T by val per-slot CE."""
        # fp32 products, centred analytically and accumulated in fp64: no (n, d) fp64 copy, which
        # for gpt-oss (61k x 2880) would be 1.4 GB per copy.
        X = np.asarray(X_train, dtype=np.float32)
        Z = np.asarray(Z_train, dtype=np.float32)
        n, d = X.shape
        x_mean = X.mean(axis=0, dtype=np.float64)
        z_mean = Z.mean(axis=0, dtype=np.float64)
        gram = (X.T @ X).astype(np.float64) - n * np.outer(x_mean, x_mean)
        cross = (X.T @ Z).astype(np.float64) - n * np.outer(x_mean, z_mean)
        evals, evecs = np.linalg.eigh(gram)
        evals = np.maximum(evals, 0.0)
        proj = evecs.T @ cross  # (d, N)
        scale = float(evals.mean()) if evals.size else 1.0

        Xv = np.asarray(X_val, dtype=np.float64)
        sets_val = np.asarray(sets_val)
        per_lambda = []
        best = None
        for rel in self.lambdas:
            W = evecs @ (proj / (evals + rel * scale)[:, None])
            b = z_mean - x_mean @ W
            zv = Xv @ W + b
            agree = _set_agreement(zv, sets_val, self.top_k)
            r2 = _r2(zv, Z_val)
            per_lambda.append({"lambda_rel": rel, "val_set_agreement": agree, "val_r2": r2})
            key = (agree, r2)
            if best is None or key > best[0]:
                best = (key, rel, W, b, zv)
        assert best is not None
        _, self.lam, self.W, self.b, zv = best

        per_temp = []
        best_t = None
        for t in self.temps:
            ce = _slot_ce_bits(set_decode(zv, self.top_k, t), sets_val)
            per_temp.append(ce)
            if best_t is None or ce < best_t[0]:
                best_t = (ce, t)
        assert best_t is not None
        self.temperature = float(best_t[1])
        self.W = self.W.astype(np.float32)
        self.b = self.b.astype(np.float32)
        self.report = {
            "feature": self.feature,
            "predictor": self.name,
            "layer": self.layer,
            "n_train_rows": int(n),
            "n_val_rows": int(Xv.shape[0]),
            "design_width": int(d),
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "hyperparams": {
                "solver": "closed_form_ridge",
                "converged": True,
                "lambda_rel": self.lam,
                "lambda_grid": list(self.lambdas),
                "temperature": self.temperature,
                "temperature_grid": [min(self.temps), max(self.temps), len(self.temps)],
                "selection": "lambda by val set agreement (then val R2); T by val per-slot CE",
            },
            "per_lambda": per_lambda,
            "best_val_ce_bits": float(best_t[0]),
        }
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("fit() first")
        return np.asarray(X, dtype=np.float32) @ self.W + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return set_decode(self.predict_scores(X), self.top_k, self.temperature)


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    resid = float(((np.asarray(pred, dtype=np.float64) - target) ** 2).sum())
    total = float(((target - target.mean(axis=0)) ** 2).sum())
    return 1.0 - resid / total if total > 0 else math.nan


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Pooled R² over all (row, expert) entries, against the per-expert test mean."""
    return _r2(pred, target)
