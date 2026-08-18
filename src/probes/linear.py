"""Multinomial logistic probe — the shared predictor for F2, F3, F4, F6 and FV.

Plain NumPy, fp32, AdamW with a cosine schedule and early stopping on val cross-entropy. The
plan specifies that optimizer for F4/F5 (T7.5); using it for every gradient-fit feature keeps a
single code path, so a difference between two features is a difference in the *feature*, not in
how hard its predictor was trained.

Why NumPy rather than torch
---------------------------
These are one linear map per (model, layer, feature): at most ``hidden_dim × n_experts``, so
2816 × 128 for Gemma. The whole of F0–F3 and F6 is count tables and small linear maps and the
plan routes them to **CPU sessions** to keep them off the GPU quota (§Phase S). Depending on
torch here would put a 2 GB import and a CUDA-version question on the critical path of the phase
that needs neither. F5's 2-layer MLP is the one probe that genuinely wants a GPU and is the one
piece of Phase 7 left for a GPU session.

The loss is the per-slot cross-entropy of plan §1.2 — see :func:`~src.probes.base.soft_targets`
for why a k-expert row reduces to a single soft-target row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Design, FitReport, soft_targets

__all__ = ["SoftmaxProbe", "log_softmax"]

LOG2 = math.log(2.0)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``log softmax`` along the last axis."""
    z = np.asarray(z, dtype=np.float32)
    shift = z - z.max(axis=1, keepdims=True)
    return shift - np.log(np.exp(shift).sum(axis=1, keepdims=True))


@dataclass
class SoftmaxProbe:
    """Linear → softmax over experts, fit by AdamW on the per-slot cross-entropy.

    Parameters mirror the plan's F4/F5 recipe. ``feature`` and ``layer`` are carried only so the
    :class:`FitReport` is self-describing in the aggregated result JSON.
    """

    n_experts: int
    feature: str = "linear"
    layer: int = -1
    lr: float = 3e-2
    weight_decay: float = 1e-4
    batch_size: int = 4096
    max_epochs: int = 60
    patience: int = 5
    min_delta: float = 1e-5
    seed: int = 0
    warmup_frac: float = 0.05
    verbose: bool = False

    W: np.ndarray | None = field(default=None, init=False)
    b: np.ndarray | None = field(default=None, init=False)
    top_k: int = field(default=0, init=False)
    report: FitReport | None = field(default=None, init=False)

    name = "softmax_probe"

    # -- fitting ---------------------------------------------------------------------------------

    def fit(
        self,
        X_train: Design,
        y_train: np.ndarray,
        X_val: Design | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        sets = np.asarray(y_train)
        if sets.ndim != 2 or not sets.size:
            raise ValueError(f"y_train must be a non-empty (n, k) array, got {sets.shape}")
        if X_train.n_rows != sets.shape[0]:
            raise ValueError(
                f"row mismatch: design has {X_train.n_rows}, y_train has {sets.shape[0]}"
            )
        n, self.top_k = int(sets.shape[0]), int(sets.shape[1])
        d = X_train.width

        rng = np.random.default_rng(self.seed)
        # Zero init: for a convex softmax regression there is nothing to break symmetry between,
        # and it makes epoch 0 exactly the uniform predictor, so the val curve starts at log2(K).
        self.W = np.zeros((d, self.n_experts), dtype=np.float32)
        self.b = np.zeros(self.n_experts, dtype=np.float32)
        mW = np.zeros_like(self.W)
        vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b)
        vb = np.zeros_like(self.b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Targets are built per minibatch, not once: a full (n_train, n_experts) fp32 target
        # matrix is 410 MB for Qwen3's 800k train rows at 128 experts, against a 4 GB budget, and
        # it would be resident alongside the layer's feature blocks for no benefit.
        steps_per_epoch = max(1, math.ceil(n / self.batch_size))
        total_steps = steps_per_epoch * self.max_epochs
        warmup = max(1, int(self.warmup_frac * total_steps))
        step = 0

        best_ce = float("inf")
        best_epoch = -1
        best_state: tuple[np.ndarray, np.ndarray] | None = None
        history: list[float] = []
        stale = 0
        epochs_run = 0

        for epoch in range(self.max_epochs):
            perm = rng.permutation(n)
            for lo in range(0, n, self.batch_size):
                rows = perm[lo : lo + self.batch_size]
                Xb = X_train.densify(rows)
                Tb = soft_targets(sets[rows], self.n_experts)

                logq = log_softmax(Xb @ self.W + self.b)
                grad_z = (np.exp(logq) - Tb) / np.float32(rows.size)
                gW = Xb.T @ grad_z
                gb = grad_z.sum(axis=0)

                step += 1
                lr = self._lr_at(step, total_steps, warmup)
                # AdamW: decoupled weight decay, and no decay on the bias — the bias is the
                # model's copy of the marginal and shrinking it toward zero is shrinking toward
                # uniform, which is a different prior than "no feature effect".
                for p, g, m, v in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                    m *= beta1
                    m += (1.0 - beta1) * g
                    v *= beta2
                    v += (1.0 - beta2) * (g * g)
                    mhat = m / (1.0 - beta1**step)
                    vhat = v / (1.0 - beta2**step)
                    p -= lr * mhat / (np.sqrt(vhat) + eps)
                if self.weight_decay:
                    self.W -= np.float32(lr * self.weight_decay) * self.W

            epochs_run = epoch + 1
            if X_val is None or y_val is None or not np.asarray(y_val).size:
                continue

            ce = self.slot_ce(X_val, y_val)
            history.append(ce)
            if ce < best_ce - self.min_delta:
                best_ce, best_epoch = ce, epoch
                best_state = (self.W.copy(), self.b.copy())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            # Restore the early-stopping optimum. Without this the reported probe is whatever the
            # last epoch produced, which for F5-like capacity is the overfit one, and the negative
            # Î the plan expects to see becomes an artefact of not restoring rather than a finding.
            self.W, self.b = best_state

        self.report = FitReport(
            feature=self.feature,
            predictor=self.name,
            layer=self.layer,
            n_train_rows=n,
            n_val_rows=int(np.asarray(y_val).shape[0]) if y_val is not None else 0,
            top_k=self.top_k,
            n_experts=self.n_experts,
            design_width=int(d),
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            best_val_ce_bits=best_ce if math.isfinite(best_ce) else float("nan"),
            val_ce_history=history,
            hyperparams={
                "optimizer": "adamw",
                "schedule": "cosine",
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "seed": self.seed,
                "dtype": "float32",
            },
            blocks=X_train.describe(),
        )

    def _lr_at(self, step: int, total: int, warmup: int) -> float:
        if step <= warmup:
            return self.lr * step / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # -- prediction --------------------------------------------------------------------------------

    def predict_proba(self, X: Design) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X: Design, rows: np.ndarray | None = None) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("SoftmaxProbe.fit has not been called")
        idx = np.arange(X.n_rows) if rows is None else np.asarray(rows, dtype=np.intp)
        out = np.empty((idx.size, self.n_experts), dtype=np.float32)
        for lo in range(0, idx.size, self.batch_size):
            sl = slice(lo, min(lo + self.batch_size, idx.size))
            out[sl] = log_softmax(X.densify(idx[sl]) @ self.W + self.b)
        return out

    def slot_ce(self, X: Design, y: np.ndarray) -> float:
        """Per-slot cross-entropy in bits — the quantity early stopping watches.

        No epsilon-mix: a softmax output is strictly positive, so this is already finite. The
        epsilon-mix of §1.2 exists for the *test* report, where the mixing target is the test
        marginal; using that target here would be a val metric contaminated by test labels.
        """
        sets = np.asarray(y).astype(np.intp)
        logq = self.predict_log_proba(X)
        picked = np.take_along_axis(logq, sets, axis=1)
        return float(-picked.mean() / LOG2)
