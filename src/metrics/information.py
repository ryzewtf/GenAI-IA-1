"""Family B — information-theoretic metrics — plan T6.3 and T6.4.

The random variable, stated precisely (plan §1.2)
-------------------------------------------------
``E`` is the identity of a **uniformly-chosen member of the selected set** ``S_t``. This is a
*slot-level* view: the k selected experts are treated as k draws from one distribution over
experts. It is **not** the entropy of the selected set, and it deliberately discards within-set
dependence — which is exactly what makes ``log2(n_experts)`` the correct normalizer. Set-level
structure is reported separately, via ``exact_match`` in Family A.

::

    p(e)        = count(e) / (k * N_test)          # estimated on TEST, sums to 1
    H           = -sum_e p(e) log2 p(e)            # Miller-Madow corrected
    CE(F)       = -(1/(k*N_test)) sum_t sum_{e in S_t} log2 q_F(e | f_t)
    CE_norm(F)  = CE(F) / log2(n_experts)
    I_hat(E; F) = H - CE(F)                        # LOWER bound on mutual information
    ratio       = I_hat(E; F) / H

Two things this module refuses to do
------------------------------------
**It will not clamp a negative ``I_hat``.** The epsilon-mix guarantees *finite* CE (no ``log 0``);
it does **not** guarantee ``CE <= H`` — it bounds ``CE <= H + log2(1/epsilon)``. An overfitting
probe (most plausibly F5 on the 50k hidden-state subsample at 128 experts) can produce
``CE > H`` and therefore ``I_hat < 0``. A negative value is diagnostic information about the
probe; hiding it converts a visible failure into an invisible bias. See :class:`MILowerBound`.

**It will not silently mix splits.** ``H`` and ``CE`` must both be estimated on the same split.
Estimating ``H`` on train and ``CE`` on test is a silent bias in the numerator of every reported
ratio, so :func:`mi_lower_bound` cross-checks the split labels and raises on a mismatch.

DeepSeek's 2 shared experts are always active, therefore deterministic, therefore excluded —
``n_experts = 64`` for its normalizer. Gemma 4's parallel dense MLP is the same class of
always-on capacity but is not an indexed expert, so it does not enter these counts at all.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "expert_counts",
    "marginal_distribution",
    "entropy_plugin",
    "entropy_mm",
    "entropy_nsb",
    "epsilon_mix",
    "per_slot_cross_entropy",
    "CrossEntropyAccumulator",
    "MILowerBound",
    "mi_lower_bound",
    "load_balance_index",
]

LOG2 = math.log(2.0)
DEFAULT_EPSILON = 1e-6

EntropyEstimator = Literal["plugin", "miller_madow", "nsb"]


# -- counts and the marginal ----------------------------------------------------------------


def expert_counts(true_sets: np.ndarray, n_experts: int) -> np.ndarray:
    """Slot occupancy count per expert, ``(n_experts,)`` int64.

    Counts *slots*, not tokens: a token contributes k counts, one per selected expert. This is
    the slot-level view of §1.2 made concrete.
    """
    true = np.asarray(true_sets)
    if true.ndim != 2:
        raise ValueError(f"true_sets must be (n, k), got shape {true.shape}")
    if true.size:
        lo, hi = int(true.min()), int(true.max())
        if lo < 0 or hi >= n_experts:
            raise ValueError(
                f"expert indices span [{lo}, {hi}], outside [0, {n_experts}) — the T5.3 label "
                "range check is failing"
            )
    return np.bincount(true.ravel().astype(np.int64), minlength=n_experts).astype(np.int64)


def marginal_distribution(true_sets: np.ndarray, n_experts: int) -> np.ndarray:
    """``p(e)`` over slots, ``(n_experts,)`` float64, summing to 1.

    Estimate this on the **same split** as the cross-entropy it will be differenced against.
    Plan T7.2's F0 *predictor* is the train-split marginal; the ``H`` in §1.2 is the test-split
    marginal. They are different numbers and the code must not conflate them.
    """
    counts = expert_counts(true_sets, n_experts)
    total = counts.sum()
    if total == 0:
        raise ValueError("no slots — cannot estimate a marginal")
    return counts / float(total)


# -- entropy estimators --------------------------------------------------------------------


def entropy_plugin(counts: np.ndarray) -> float:
    """Naive plug-in entropy in bits. Biased downward; here for comparison only."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot estimate entropy from zero counts")
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def entropy_mm(counts: np.ndarray) -> float:
    """Miller-Madow corrected entropy in bits — the primary estimator (plan T6.3).

    Adds ``(m - 1) / (2N)`` nats to the plug-in estimate, where ``m`` is the number of
    *observed* (non-zero) bins. The plug-in estimator is biased downward by roughly that much,
    so the correction is a first-order debias, not a heuristic.

    Using observed rather than total bins is deliberate: it makes the correction adapt to a load
    imbalance that leaves experts unused. A dead expert is a real finding (plan T5.3), and it
    should not inflate the correction as if that bin were merely unsampled.

    .. warning::
       **Miller-Madow is not bounded above by ``log2(K)``.** When ``m`` approaches ``N`` the
       correction can push the estimate past the maximum entropy of the alphabet — e.g. counts
       ``[1,1,1,1]`` give ``2.0 + 0.54 = 2.54`` bits against a ``log2(4) = 2.0`` ceiling. This is
       a known property of a first-order bias correction, not a bug, and it is why
       :func:`load_balance_index` warns rather than silently reporting an index above 1.

       At the sample sizes this study collects the correction is negligible: 1M tokens at k=8 is
       8M slots, so for 64 experts the correction is ~5.7e-6 bits. Compute the load-balance index
       on full-corpus counts and the issue never arises.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot estimate entropy from zero counts")
    observed = int((counts > 0).sum())
    return entropy_plugin(counts) + (observed - 1) / (2.0 * total * LOG2)


def entropy_nsb(counts: np.ndarray, n_bins: int | None = None, n_grid: int = 256) -> float:
    """Nemenman-Shafee-Bialek entropy in bits — secondary, for the sensitivity appendix.

    Posterior-mean entropy under a Dirichlet prior of concentration ``beta``, mixed over a prior
    on ``beta`` chosen so the induced prior on entropy is close to uniform on ``[0, log K]``.
    That mixture is the whole point: a single fixed Dirichlet concentration encodes a strong and
    usually wrong prior belief about the entropy itself.

    Included because a reviewer attacking estimator bias is answered by showing the conclusion
    survives a different estimator. It should agree with Miller-Madow in the well-sampled regime
    and beat it when undersampled.
    """
    from scipy.special import gammaln, polygamma, psi  # local: scipy only needed here

    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("cannot estimate entropy from zero counts")

    n_bins = int(n_bins) if n_bins is not None else int(counts.size)
    nonzero = counts[counts > 0]
    n_observed = int(nonzero.size)
    if n_bins <= 1 or n_observed <= 1:
        return 0.0

    # Grid over beta, wide enough to cover the posterior mass at any sample size. The prior
    # density on beta is the Jacobian dxi/dbeta of the entropy reparametrisation, so integrating
    # in beta with that weight is equivalent to a uniform grid in xi without needing to invert
    # xi(beta) numerically.
    betas = np.logspace(-5, 4, n_grid) / n_bins
    kb = n_bins * betas

    # log evidence, up to a beta-independent constant
    log_like = (
        gammaln(kb)
        - gammaln(total + kb)
        + (gammaln(nonzero[None, :] + betas[:, None]) - gammaln(betas[:, None])).sum(axis=1)
    )

    # posterior mean entropy at each beta, in nats
    denom = total + kb
    weighted = (
        (nonzero[None, :] + betas[:, None])
        / denom[:, None]
        * psi(nonzero[None, :] + betas[:, None] + 1.0)
    ).sum(axis=1)
    unobserved = n_bins - n_observed
    if unobserved:
        weighted = weighted + unobserved * (betas / denom) * psi(betas + 1.0)
    mean_entropy = psi(denom + 1.0) - weighted

    # prior density on beta: dxi/dbeta where xi(beta) = psi(K beta + 1) - psi(beta + 1)
    jacobian = n_bins * polygamma(1, kb + 1.0) - polygamma(1, betas + 1.0)
    jacobian = np.clip(jacobian, 0.0, None)

    weights = jacobian * np.exp(log_like - log_like.max())
    normalizer = np.trapezoid(weights, betas)
    if not np.isfinite(normalizer) or normalizer <= 0:
        return entropy_mm(counts)  # degenerate posterior; fall back rather than emit garbage

    return float(np.trapezoid(weights * mean_entropy, betas) / normalizer / LOG2)


_ESTIMATORS = {
    "plugin": entropy_plugin,
    "miller_madow": entropy_mm,
    "nsb": lambda counts: entropy_nsb(counts),
}


def entropy(counts: np.ndarray, estimator: EntropyEstimator = "miller_madow") -> float:
    if estimator not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; choose from {sorted(_ESTIMATORS)}")
    return float(_ESTIMATORS[estimator](counts))


# -- cross-entropy --------------------------------------------------------------------------


def epsilon_mix(
    pred_probs: np.ndarray, marginal: np.ndarray, epsilon: float = DEFAULT_EPSILON
) -> np.ndarray:
    """``(1 - eps) * q + eps * p_marginal`` — guarantees finite CE, nothing more.

    Fix ``epsilon`` at 1e-6 and record it in ``run.yaml``. Do **not** tune it per predictor: a
    tuned mixture weight is an extra hyperparameter that buys a guarantee only on val, not on
    test (plan §1.2).
    """
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    marginal = np.asarray(marginal, dtype=np.float64)
    if marginal.ndim != 1:
        raise ValueError(f"marginal must be 1-D, got shape {marginal.shape}")
    return (1.0 - epsilon) * np.asarray(pred_probs, dtype=np.float64) + epsilon * marginal


@dataclass
class CrossEntropyAccumulator:
    """Per-slot held-out cross-entropy in bits, folded over chunks.

    ``marginal`` is the epsilon-mix target and must come from the same split being evaluated.
    ``split`` is carried purely so :func:`mi_lower_bound` can refuse to difference quantities
    estimated on different splits.
    """

    n_experts: int
    marginal: np.ndarray
    epsilon: float = DEFAULT_EPSILON
    split: str | None = None
    check_normalization: bool = True
    n_slots: int = field(default=0, init=False)
    _log_sum: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.marginal = np.asarray(self.marginal, dtype=np.float64)
        if self.marginal.shape != (self.n_experts,):
            raise ValueError(
                f"marginal must have shape ({self.n_experts},), got {self.marginal.shape}"
            )
        if not np.isclose(self.marginal.sum(), 1.0, atol=1e-6):
            raise ValueError(f"marginal must sum to 1, sums to {self.marginal.sum()}")

    def update(self, pred_probs: np.ndarray, true_sets: np.ndarray) -> None:
        pred = np.asarray(pred_probs, dtype=np.float64)
        true = np.asarray(true_sets)

        if pred.ndim != 2 or pred.shape[1] != self.n_experts:
            raise ValueError(
                f"pred_probs must be (n, {self.n_experts}) or (1, {self.n_experts}), got "
                f"{pred.shape}"
            )
        if true.ndim != 2:
            raise ValueError(f"true_sets must be (n, k), got {true.shape}")
        if pred.shape[0] == 1 and true.shape[0] != 1:
            pred = np.broadcast_to(pred, (true.shape[0], self.n_experts))
        elif pred.shape[0] != true.shape[0]:
            raise ValueError(
                f"row mismatch: pred_probs {pred.shape[0]}, true_sets {true.shape[0]}"
            )
        if not true.size:
            return

        if self.check_normalization:
            sums = pred.sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-4):
                worst = float(np.abs(sums - 1.0).max())
                raise ValueError(
                    f"pred_probs rows must sum to 1 (plan T7.1); worst deviation {worst:.2e}"
                )

        mixed = epsilon_mix(pred, self.marginal, self.epsilon)
        picked = np.take_along_axis(mixed, true.astype(np.intp), axis=1)
        if np.any(picked <= 0.0):
            raise ValueError(
                "epsilon-mixed probability of a selected expert is non-positive; the marginal "
                "assigns zero mass to an expert that the labels select. Estimate the marginal "
                "on the same split as the labels."
            )
        self._log_sum += float(-np.log2(picked).sum())
        self.n_slots += int(picked.size)

    def result(self) -> float:
        """Cross-entropy in bits per slot."""
        if not self.n_slots:
            raise ValueError("no slots accumulated")
        return self._log_sum / self.n_slots


def per_slot_cross_entropy(
    pred_probs: np.ndarray,
    true_sets: np.ndarray,
    marginal: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
    *,
    split: str | None = None,
) -> float:
    """One-shot per-slot cross-entropy in bits."""
    accumulator = CrossEntropyAccumulator(
        n_experts=np.asarray(marginal).shape[0],
        marginal=marginal,
        epsilon=epsilon,
        split=split,
    )
    accumulator.update(pred_probs, true_sets)
    return accumulator.result()


# -- the MI lower bound -------------------------------------------------------------------------


@dataclass(frozen=True)
class MILowerBound:
    """Result of ``I_hat = H - CE``, reported as measured.

    ``negative`` is set when ``CE > H``. Read that as *"no information detected beyond the
    marginal, and the probe is overfitting"* — and report it. Plan §1.2 requires flagging every
    (model, layer, feature) cell with ``I_hat < 0`` and reporting how many there are: a reviewer
    attacking estimator bias is disarmed by the honest count, not by a floor at zero.
    """

    entropy_bits: float
    cross_entropy_bits: float
    n_experts: int
    estimator: str = "miller_madow"
    split: str | None = None

    @property
    def mi_bits(self) -> float:
        """Signed. Never clamped."""
        return self.entropy_bits - self.cross_entropy_bits

    @property
    def ratio(self) -> float:
        """``I_hat / H``. Signed, and NaN if ``H`` is zero (a fully deterministic router)."""
        if self.entropy_bits == 0.0:
            return float("nan")
        return self.mi_bits / self.entropy_bits

    @property
    def ce_normalized(self) -> float:
        """``CE / log2(n_experts)`` — comparable across the panel."""
        if self.n_experts <= 1:
            return float("nan")
        return self.cross_entropy_bits / math.log2(self.n_experts)

    @property
    def entropy_normalized(self) -> float:
        """Also the load-balance index of plan T6.4."""
        if self.n_experts <= 1:
            return float("nan")
        return self.entropy_bits / math.log2(self.n_experts)

    @property
    def negative(self) -> bool:
        return self.mi_bits < 0.0

    def to_json(self) -> dict[str, float | int | bool | str | None]:
        return {
            "entropy_bits": self.entropy_bits,
            "cross_entropy_bits": self.cross_entropy_bits,
            "mi_bits": self.mi_bits,
            "ratio": self.ratio,
            "ce_normalized": self.ce_normalized,
            "entropy_normalized": self.entropy_normalized,
            "negative": self.negative,
            "n_experts": self.n_experts,
            "estimator": self.estimator,
            "split": self.split,
        }


def mi_lower_bound(
    entropy_bits: float,
    cross_entropy_bits: float,
    n_experts: int,
    *,
    estimator: str = "miller_madow",
    entropy_split: str | None = None,
    ce_split: str | None = None,
) -> MILowerBound:
    """Difference ``H - CE`` without clamping, refusing to mix splits.

    Held-out CE upper-bounds the true conditional entropy, so ``H - CE`` is a valid *lower*
    bound on mutual information — the standard variational framing, which the methods section
    must state explicitly.
    """
    if entropy_split is not None and ce_split is not None and entropy_split != ce_split:
        raise ValueError(
            f"refusing to difference H estimated on {entropy_split!r} against CE estimated on "
            f"{ce_split!r}: mixing splits between numerator and denominator is a silent bias "
            "(plan §1.2). Estimate both on test."
        )
    return MILowerBound(
        entropy_bits=float(entropy_bits),
        cross_entropy_bits=float(cross_entropy_bits),
        n_experts=int(n_experts),
        estimator=estimator,
        split=entropy_split or ce_split,
    )


# -- T6.4 load-balance index -------------------------------------------------------------------


def load_balance_index(
    counts_or_sets: np.ndarray,
    n_experts: int,
    *,
    estimator: EntropyEstimator = "miller_madow",
    is_counts: bool = False,
) -> float:
    """``H_marginal / log2(n_experts)`` per (model, layer) — plan T6.4.

    This is the aux-loss confound *instrument*, and the measured quantity that should carry the
    T9.4 argument. The config-declared ``aux_loss_coef`` is a field in a post-trained release
    whose training procedure was not published; HF config fields are not always the values used
    in training. GPT-OSS declares 0.9 against 0.001-0.01 elsewhere in the panel — a 900x spread
    — so if GPT-OSS shows the highest entropy that cannot be attributed to granularity.

    1.0 means perfectly uniform expert usage; lower means concentrated.

    The value is returned as measured — consistent with the no-clamping policy for ``I_hat`` —
    but a warning is raised when the Miller-Madow correction is worth more than ~1% of the
    plug-in entropy, which is the actual signal that the counts are too sparse to report. Note
    that perfectly balanced usage lands *marginally* above 1.0 by exactly the correction, since
    the plug-in estimate is already at the ceiling; that is expected and is not warned about.
    Always compute this on full-corpus counts.
    """
    counts = (
        np.asarray(counts_or_sets, dtype=np.float64)
        if is_counts
        else expert_counts(counts_or_sets, n_experts)
    )
    if n_experts <= 1:
        return float("nan")

    index = entropy(counts, estimator) / math.log2(n_experts)

    # Diagnose undersampling by the size of the bias correction relative to the entropy, not by
    # whether the index crosses 1.0. Perfectly balanced usage always lands marginally above 1.0
    # — the plug-in estimate is already at the ceiling and the correction adds to it — so a
    # crossing alone carries no information. A correction worth more than ~1% of the entropy
    # does.
    if estimator == "miller_madow":
        plugin_bits = entropy_plugin(counts)
        correction = entropy_mm(counts) - plugin_bits
        if plugin_bits > 0 and correction / plugin_bits > 0.01:
            total = float(np.sum(counts))
            observed = int((np.asarray(counts) > 0).sum())
            warnings.warn(
                f"load-balance index {index:.4f} rests on a Miller-Madow correction worth "
                f"{correction:.3f} bits ({correction / plugin_bits:.1%} of the plug-in entropy): "
                f"{observed} observed bins over only {total:.0f} slots. The estimate is "
                "undersampled — aggregate more slots before reporting it (plan T6.4/T9.4).",
                RuntimeWarning,
                stacklevel=2,
            )
    return index
