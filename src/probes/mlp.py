"""Two-layer MLP probe — the predictor for F5 (plan §1.3, T7.5).

F5 conditions on the same variable as F4 — the captured **router input at layer ℓ−1** — and
differs only in the model class: F4 is linear, F5 is ``linear → GELU → linear → softmax``. The
pair is the whole point. F4 is the linear practical ceiling and F5 the nonlinear one, so the F5−F4
gap is the part of one-layer-lookahead predictability that a linear prefetcher cannot reach. Any
difference in *how hard the two were trained* would be read as that gap, so this module mirrors
:class:`~src.probes.linear.SoftmaxProbe` exactly — same optimizer, same schedule, same soft
targets, same early-stopping-with-restore, same ``slot_ce`` in bits — and reuses its
``log_softmax``/``LOG2`` rather than growing a second copy that could drift.

Plain NumPy, fp32. Plan T7.5 fixes fp32 for these probes so that the one measurement defining the
practical ceiling does not also carry a precision question; and staying in NumPy keeps torch off
the dependency list of a phase whose largest object is a ``hidden_dim × 256`` matrix.

Backprop is written out by hand below. There is no autograd here, so
:func:`tests/test_probes_mlp.py::test_analytic_gradients_match_central_differences` is the thing
standing between a sign error and a silently weak F5 — treat it as load-bearing.

Memory
------
The reason F5 was deferred is capacity, not code. Qwen3 is 48 layers × 128 experts with ~800k
train rows and ``hidden_dim`` 2048, against T6.1's 4 GB analysis budget on a 15.1 GiB machine.
Per split, in fp32:

* hidden activations ``800k × 256`` = **819 MB**
* output probabilities / targets ``800k × 128`` = **410 MB** each

so a whole-split forward pass is over 1.6 GB before the router-input block itself (6.5 GB at
``hidden_dim`` 2048 — which is why it stays memory-mapped behind
:class:`~src.probes.base.DenseBlock`). Nothing here is ever materialised for more than one
minibatch: at ``batch_size`` 4096 those same three arrays are 4 MB, 2 MB and 2 MB. The parameters
are tiny by comparison (``2048 × 256`` + ``256 × 128`` ≈ 2.2 MB), so the loop below is written
per-minibatch for both fitting *and* prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Design, FitReport, soft_targets
from .linear import LOG2, log_softmax

__all__ = ["MLPProbe", "gelu", "gelu_grad"]

_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximation GELU — the activation plan §1.3 specifies for F5.

    GELU rather than ReLU because the plan names it, and because the router input is a
    standardized, roughly zero-centred, roughly Gaussian vector: ReLU would hard-zero half of
    every such coordinate and half of the hidden units would receive no gradient at
    initialisation. The tanh approximation is used instead of the exact ``0.5 x (1 + erf(x/√2))``
    so that this module needs no ``scipy.special`` import and so that the derivative below is an
    exact derivative of the function actually evaluated — a gradient check against an
    approximation of a different function is not a gradient check.
    """
    x = np.asarray(x, dtype=np.float32)
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(np.float32(_GELU_C) * (x + np.float32(0.044715) * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """``d/dx`` of :func:`gelu`, evaluated at ``x`` (not at the activation)."""
    x = np.asarray(x, dtype=np.float32)
    inner = np.float32(_GELU_C) * (x + np.float32(0.044715) * x**3)
    t = np.tanh(inner)
    dinner = np.float32(_GELU_C) * (np.float32(1.0) + np.float32(3.0 * 0.044715) * x**2)
    return np.float32(0.5) * (np.float32(1.0) + t) + np.float32(0.5) * x * (np.float32(1.0) - t * t) * dinner


@dataclass
class MLPProbe:
    """``input → Linear(W1,b1) → GELU → Linear(W2,b2) → softmax`` over experts, fit by AdamW.

    ``hidden_width`` defaults to 256 rather than the plan's 512: 512 is the value budgeted for a
    GPU session at ``hidden_dim`` 2048, and it is recorded in ``hyperparams`` either way, so a run
    that uses the plan's width is self-describing in the result JSON.

    ``dropout`` is the plan's regularizer for F5 (T7.5 says 0.1) and is applied to the hidden
    activations on training minibatches only, inverted-scaling style, never at prediction time.
    It defaults to 0.0 so that the default probe is a deterministic function of its inputs and the
    F4/F5 comparison differs in model class alone; set it to 0.1 to follow the plan's recipe.
    """

    n_experts: int
    hidden_width: int = 256
    feature: str = "F5"
    layer: int = -1
    lr: float = 3e-3
    weight_decay: float = 1e-4
    dropout: float = 0.0
    batch_size: int = 4096
    max_epochs: int = 60
    patience: int = 5
    min_delta: float = 1e-5
    seed: int = 0
    warmup_frac: float = 0.05
    verbose: bool = False

    W1: np.ndarray | None = field(default=None, init=False)
    b1: np.ndarray | None = field(default=None, init=False)
    W2: np.ndarray | None = field(default=None, init=False)
    b2: np.ndarray | None = field(default=None, init=False)
    top_k: int = field(default=0, init=False)
    report: FitReport | None = field(default=None, init=False)

    name = "mlp_probe"

    # -- initialisation -----------------------------------------------------------------------

    def _init_params(self, d: int, rng: np.random.Generator) -> None:
        """He-normal for W1, Glorot-normal × 0.1 for W2, both biases zero.

        W1 uses **He (Kaiming) normal**, ``std = sqrt(2/d_in)`` — the fan-in scheme for a
        ReLU-family activation, which GELU is; it keeps the pre-activation variance ≈ 2 so the
        GELU is neither saturated nor operating only in its linear region.

        W2 uses **Glorot (Xavier) normal**, ``std = sqrt(2/(hidden_width + n_experts))``, scaled
        by 0.1. The extra factor is deliberate: unscaled Glorot on a 256-wide hidden layer gives
        output logits with a standard deviation of order 1, i.e. a confidently *wrong* predictor
        at epoch 0, and the shrink puts the initial logits near zero instead.

        ``b2`` is zero-initialised and — see the AdamW step — excluded from weight decay, for the
        same reason :class:`~src.probes.linear.SoftmaxProbe` does it: the output bias is the
        model's copy of the marginal, and decaying it toward zero shrinks the model toward
        *uniform*, which is a different prior from "this feature carries no information".

        Note the difference from SoftmaxProbe: because W2 is random rather than zero, epoch 0 here
        is **not** exactly the uniform predictor, so ``test_probes.py``'s
        ``ce == log2(n_experts)`` assertion does not transfer. What the F5 tests assert instead is
        that an unfitted probe sits *close to* uniform (within 0.25 bits of ``log2(n_experts)``)
        while not being exactly equal to it — the first half catches an init that has blown up,
        the second half catches a W2 that is silently zero and would make this probe linear.
        """
        self.W1 = (rng.normal(0.0, math.sqrt(2.0 / max(1, d)), size=(d, self.hidden_width))).astype(np.float32)
        self.b1 = np.zeros(self.hidden_width, dtype=np.float32)
        glorot = math.sqrt(2.0 / (self.hidden_width + self.n_experts))
        self.W2 = (0.1 * rng.normal(0.0, glorot, size=(self.hidden_width, self.n_experts))).astype(np.float32)
        self.b2 = np.zeros(self.n_experts, dtype=np.float32)

    # -- forward / backward -------------------------------------------------------------------

    def _forward(self, Xb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(pre_activation, hidden, log_softmax(logits))`` for one minibatch."""
        z1 = Xb @ self.W1 + self.b1
        h = gelu(z1)
        logq = log_softmax(h @ self.W2 + self.b2)
        return z1, h, logq

    def _loss_and_grads(
        self, Xb: np.ndarray, Tb: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Mean per-row cross-entropy in **nats** and its gradient w.r.t. every parameter.

        Because each soft-target row sums to 1 (``multi_hot(S)/k``), the per-row loss is the
        per-slot loss of plan §1.2 and no k-dependent bookkeeping appears anywhere below.

        The one line a reader will want to check: the CE-with-soft-targets gradient at the output
        layer is ``(softmax(z) - target) / batch``. Everything else is the chain rule on top of
        that — ``gW2 = hᵀ @ grad_z``, ``grad_h = grad_z @ W2ᵀ``, ``grad_z1 = grad_h * gelu'(z1)``,
        ``gW1 = Xbᵀ @ grad_z1``.
        """
        b = np.float32(Xb.shape[0])
        z1 = Xb @ self.W1 + self.b1
        h = gelu(z1)
        if mask is not None:
            h = h * mask
        logq = log_softmax(h @ self.W2 + self.b2)

        loss = float(-(Tb * logq).sum() / b)

        grad_z = (np.exp(logq) - Tb) / b
        gW2 = h.T @ grad_z
        gb2 = grad_z.sum(axis=0)
        grad_h = grad_z @ self.W2.T
        if mask is not None:
            grad_h = grad_h * mask
        grad_z1 = grad_h * gelu_grad(z1)
        gW1 = Xb.T @ grad_z1
        gb1 = grad_z1.sum(axis=0)
        return loss, (gW1, gb1, gW2, gb2)

    # -- fitting ------------------------------------------------------------------------------

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
        if self.hidden_width < 1:
            raise ValueError(f"hidden_width must be >= 1, got {self.hidden_width}")
        n, self.top_k = int(sets.shape[0]), int(sets.shape[1])
        d = X_train.width

        rng = np.random.default_rng(self.seed)
        self._init_params(d, rng)

        params = (self.W1, self.b1, self.W2, self.b2)
        mom = tuple(np.zeros_like(p) for p in params)
        vel = tuple(np.zeros_like(p) for p in params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Targets are built per minibatch, not once, exactly as in SoftmaxProbe: a full
        # (n_train, n_experts) fp32 target matrix is 410 MB for Qwen3's 800k train rows at 128
        # experts, and it would sit alongside the layer's router-input block for no benefit.
        steps_per_epoch = max(1, math.ceil(n / self.batch_size))
        total_steps = steps_per_epoch * self.max_epochs
        warmup = max(1, int(self.warmup_frac * total_steps))
        step = 0

        best_ce = float("inf")
        best_epoch = -1
        best_state: tuple[np.ndarray, ...] | None = None
        history: list[float] = []
        stale = 0
        epochs_run = 0
        keep = 1.0 - float(self.dropout)
        if not 0.0 < keep <= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        for epoch in range(self.max_epochs):
            perm = rng.permutation(n)
            for lo in range(0, n, self.batch_size):
                rows = perm[lo : lo + self.batch_size]
                Xb = X_train.densify(rows)
                Tb = soft_targets(sets[rows], self.n_experts)

                mask = None
                if keep < 1.0:
                    # Inverted dropout on the hidden activations, train-time only.
                    mask = (rng.random((rows.size, self.hidden_width)) < keep).astype(
                        np.float32
                    ) / np.float32(keep)

                _, grads = self._loss_and_grads(Xb, Tb, mask)

                step += 1
                lr = self._lr_at(step, total_steps, warmup)
                for p, g, m, v in zip(params, grads, mom, vel):
                    m *= beta1
                    m += (1.0 - beta1) * g
                    v *= beta2
                    v += (1.0 - beta2) * (g * g)
                    mhat = m / (1.0 - beta1**step)
                    vhat = v / (1.0 - beta2**step)
                    p -= lr * mhat / (np.sqrt(vhat) + eps)
                if self.weight_decay:
                    # Decoupled decay on the weight matrices only. b1 and b2 are excluded: b2 is
                    # the model's copy of the marginal (see _init_params) and b1 is a per-unit
                    # threshold whose zero has no privileged meaning.
                    decay = np.float32(lr * self.weight_decay)
                    self.W1 -= decay * self.W1
                    self.W2 -= decay * self.W2

            epochs_run = epoch + 1
            if X_val is None or y_val is None or not np.asarray(y_val).size:
                continue

            ce = self.slot_ce(X_val, y_val)
            history.append(ce)
            if ce < best_ce - self.min_delta:
                best_ce, best_epoch = ce, epoch
                best_state = tuple(p.copy() for p in params)
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            # Restore the early-stopping optimum. This matters more here than for the linear
            # probe: F5 is the feature plan §1.2 expects to overfit (50k hidden-state rows, 128
            # classes), so "the last epoch" and "the best epoch" are routinely different models,
            # and reporting the former would turn the plan's expected negative Î into an artefact
            # of not restoring rather than a finding about the probe.
            self.W1, self.b1, self.W2, self.b2 = best_state

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
                "architecture": "linear-gelu-linear",
                "hidden_width": int(self.hidden_width),
                "activation": "gelu_tanh",
                "init": "he_normal_w1/glorot_normal_x0.1_w2/zero_bias",
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "dropout": self.dropout,
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

    # -- prediction ---------------------------------------------------------------------------

    def predict_proba(self, X: Design) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X: Design, rows: np.ndarray | None = None) -> np.ndarray:
        if self.W1 is None or self.W2 is None:
            raise RuntimeError("MLPProbe.fit has not been called")
        idx = np.arange(X.n_rows) if rows is None else np.asarray(rows, dtype=np.intp)
        out = np.empty((idx.size, self.n_experts), dtype=np.float32)
        # Chunked so the (rows × hidden_width) activation never exists for the whole split; see
        # the module docstring's arithmetic. Dropout is not applied here by construction.
        for lo in range(0, idx.size, self.batch_size):
            sl = slice(lo, min(lo + self.batch_size, idx.size))
            _, _, logq = self._forward(X.densify(idx[sl]))
            out[sl] = logq
        return out

    def slot_ce(self, X: Design, y: np.ndarray) -> float:
        """Per-slot cross-entropy in bits — the quantity early stopping watches.

        No epsilon-mix, identical semantics to :meth:`SoftmaxProbe.slot_ce`: a softmax output is
        strictly positive so this is already finite, and §1.2's epsilon-mix belongs to the *test*
        report where the mixing target is the test marginal.
        """
        sets = np.asarray(y).astype(np.intp)
        logq = self.predict_log_proba(X)
        picked = np.take_along_axis(logq, sets, axis=1)
        return float(-picked.mean() / LOG2)
