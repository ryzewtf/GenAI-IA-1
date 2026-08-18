"""Metric tests — plan T6.2 and T6.3 acceptance criteria.

The plan names two specific acceptance bars, and both are tested literally here:

* T6.2 — "verified against hand-computed values on a 10-row fixture"
* T6.3 — "on synthetic data with known ground-truth entropy, MM estimate within 1% at N = 10^6;
  a deliberately overfit predictor on a 500-sample fixture returns a negative I_hat with the
  flag set, rather than a clamped zero"
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics.information import (
    CrossEntropyAccumulator,
    entropy_mm,
    entropy_nsb,
    entropy_plugin,
    epsilon_mix,
    expert_counts,
    load_balance_index,
    marginal_distribution,
    mi_lower_bound,
    per_slot_cross_entropy,
)
from src.metrics.predictive import (
    PredictiveAccumulator,
    exact_match,
    recall_at_m,
    set_agreement_at_k,
    set_agreement_rows,
    top_m_indices,
)

# --------------------------------------------------------------------------------------------
# T6.2 — the hand-computed 10-row fixture
#
# Every row scores experts [6,5,4,3,2,1], so the predicted ranking is always 0>1>2>3>4>5:
#   top-2 = {0,1}   top-4 = {0,1,2,3}   top-6 = everything
# The true sets below were chosen to give a known mix of 2/2, 1/2 and 0/2 overlaps.
# --------------------------------------------------------------------------------------------

N_EXPERTS = 6
TOP_K = 2
_SCORES_ONE_ROW = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

FIXTURE_TRUE = np.array(
    [
        [0, 1],  # both in top-2   -> agree 1.0, exact 1, recall@4 1.0
        [0, 2],  # one in top-2    -> agree 0.5, exact 0, recall@4 1.0
        [2, 3],  # none in top-2   -> agree 0.0, exact 0, recall@4 1.0
        [4, 5],  # none anywhere   -> agree 0.0, exact 0, recall@4 0.0
        [1, 0],  # order irrelevant-> agree 1.0, exact 1, recall@4 1.0
        [0, 4],  # one in top-2    -> agree 0.5, exact 0, recall@4 0.5
        [3, 4],  # none in top-2   -> agree 0.0, exact 0, recall@4 0.5
        [1, 2],  # one in top-2    -> agree 0.5, exact 0, recall@4 1.0
        [5, 4],  # none anywhere   -> agree 0.0, exact 0, recall@4 0.0
        [0, 1],  # both in top-2   -> agree 1.0, exact 1, recall@4 1.0
    ]
)
FIXTURE_PRED = np.tile(np.array(_SCORES_ONE_ROW), (10, 1))

EXPECTED_AGREEMENT_ROWS = np.array([1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 1.0])
EXPECTED_RECALL4_ROWS = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 1.0])


def test_hand_computed_rows():
    np.testing.assert_allclose(
        set_agreement_rows(FIXTURE_PRED, FIXTURE_TRUE), EXPECTED_AGREEMENT_ROWS
    )


def test_hand_computed_scalars():
    assert set_agreement_at_k(FIXTURE_PRED, FIXTURE_TRUE) == pytest.approx(0.45)
    assert exact_match(FIXTURE_PRED, FIXTURE_TRUE) == pytest.approx(0.30)
    assert recall_at_m(FIXTURE_PRED, FIXTURE_TRUE, 2) == pytest.approx(0.45)
    assert recall_at_m(FIXTURE_PRED, FIXTURE_TRUE, 4) == pytest.approx(0.70)
    assert recall_at_m(FIXTURE_PRED, FIXTURE_TRUE, 6) == pytest.approx(1.00)


def test_recall_at_k_equals_set_agreement():
    """The two literatures name one quantity differently; it is computed once (see module doc)."""
    assert recall_at_m(FIXTURE_PRED, FIXTURE_TRUE, TOP_K) == pytest.approx(
        set_agreement_at_k(FIXTURE_PRED, FIXTURE_TRUE)
    )


def test_recall_is_monotone_in_m():
    values = [recall_at_m(FIXTURE_PRED, FIXTURE_TRUE, m) for m in range(1, N_EXPERTS + 1)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_top_m_saturates_at_n_experts():
    picks = top_m_indices(FIXTURE_PRED, 99)
    assert picks.shape == (10, N_EXPERTS)
    assert set(picks[0].tolist()) == set(range(N_EXPERTS))


# -- chunk composability ------------------------------------------------------------------------


def test_accumulator_matches_one_shot():
    accumulator = PredictiveAccumulator(top_k=TOP_K, n_experts=N_EXPERTS)
    accumulator.update(FIXTURE_PRED, FIXTURE_TRUE)
    result = accumulator.result()
    assert result["set_agreement@k"] == pytest.approx(0.45)
    assert result["exact_match"] == pytest.approx(0.30)
    assert result["recall@4"] == pytest.approx(0.70)
    assert result["n"] == 10


@pytest.mark.parametrize("chunks", [[10], [5, 5], [1, 2, 3, 4], [3, 3, 3, 1]])
def test_accumulation_is_independent_of_chunking(chunks):
    accumulator = PredictiveAccumulator(top_k=TOP_K, n_experts=N_EXPERTS)
    start = 0
    for size in chunks:
        accumulator.update(FIXTURE_PRED[start : start + size], FIXTURE_TRUE[start : start + size])
        start += size
    result = accumulator.result()
    assert result["set_agreement@k"] == pytest.approx(0.45)
    assert result["exact_match"] == pytest.approx(0.30)
    assert result["recall@4"] == pytest.approx(0.70)


def test_recall_targets_are_capped_at_n_experts():
    """A coarse model where 4k > n_experts must not report an uninformative recall@4k."""
    accumulator = PredictiveAccumulator(top_k=4, n_experts=8)
    assert accumulator.recall_targets == [4, 8]


def test_shape_and_range_errors_are_loud():
    with pytest.raises(ValueError, match="row count mismatch"):
        set_agreement_at_k(FIXTURE_PRED[:5], FIXTURE_TRUE)
    with pytest.raises(ValueError, match="outside"):
        set_agreement_at_k(FIXTURE_PRED[:1], np.array([[0, 99]]))
    with pytest.raises(ValueError, match="k=3"):
        PredictiveAccumulator(top_k=2, n_experts=6).update(
            FIXTURE_PRED, np.zeros((10, 3), dtype=int)
        )


# --------------------------------------------------------------------------------------------
# T6.3 — entropy estimators
# --------------------------------------------------------------------------------------------


def test_counts_are_slot_level_not_token_level():
    """A token contributes k counts, one per selected expert (plan §1.2)."""
    counts = expert_counts(np.array([[0, 1], [0, 2]]), n_experts=4)
    np.testing.assert_array_equal(counts, [2, 1, 1, 0])
    assert counts.sum() == 4  # 2 tokens * k=2 slots


def test_marginal_sums_to_one():
    p = marginal_distribution(np.array([[0, 1], [0, 2]]), n_experts=4)
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p, [0.5, 0.25, 0.25, 0.0])


def test_mm_within_one_percent_of_uniform_truth_at_1e6():
    """The literal T6.3 acceptance criterion."""
    n_experts, total = 64, 1_000_000
    counts = np.full(n_experts, total // n_experts, dtype=np.int64)
    truth = math.log2(n_experts)  # 6.0 bits
    assert abs(entropy_mm(counts) - truth) / truth < 0.01


def test_mm_within_one_percent_on_a_known_non_uniform_distribution():
    p = np.array([0.5, 0.25, 0.125, 0.125])
    truth = float(-(p * np.log2(p)).sum())  # 1.75 bits
    counts = (p * 1_000_000).astype(np.int64)
    assert abs(entropy_mm(counts) - truth) / truth < 0.01


def test_plugin_is_biased_downward_and_mm_corrects_upward():
    rng = np.random.default_rng(0)
    counts = np.bincount(rng.integers(0, 64, size=400), minlength=64)
    assert entropy_plugin(counts) < entropy_mm(counts)


def test_mm_can_exceed_log2_k_when_undersampled():
    """Documented property of a first-order bias correction, not a bug.

    Counts of all ones put the plug-in estimate at the ceiling already, and the correction adds
    to it. This is why load_balance_index warns instead of silently reporting an index above 1.
    """
    counts = np.ones(4, dtype=np.int64)
    assert entropy_plugin(counts) == pytest.approx(math.log2(4))
    assert entropy_mm(counts) > math.log2(4)


def test_mm_correction_is_negligible_at_collection_scale():
    """1M tokens at k=8 is 8M slots; the correction must not move the third decimal."""
    n_experts, slots = 64, 8_000_000
    counts = np.full(n_experts, slots // n_experts, dtype=np.int64)
    assert entropy_mm(counts) - entropy_plugin(counts) < 1e-4


def test_nsb_beats_plugin_when_undersampled():
    """NSB's whole justification is the undersampled regime; verify it there."""
    n_experts, truth = 64, math.log2(64)
    rng = np.random.default_rng(3)
    counts = np.bincount(rng.integers(0, n_experts, size=200), minlength=n_experts)
    assert abs(entropy_nsb(counts, n_bins=n_experts) - truth) <= abs(
        entropy_plugin(counts) - truth
    )


def test_nsb_agrees_with_mm_when_well_sampled():
    n_experts = 32
    counts = np.full(n_experts, 20_000, dtype=np.int64)
    assert entropy_nsb(counts, n_bins=n_experts) == pytest.approx(entropy_mm(counts), abs=0.05)


def test_entropy_is_zero_for_a_degenerate_router():
    counts = np.array([1000, 0, 0, 0])
    assert entropy_plugin(counts) == pytest.approx(0.0)
    assert entropy_nsb(counts, n_bins=4) == pytest.approx(0.0)


def test_entropy_rejects_empty_counts():
    with pytest.raises(ValueError, match="zero counts"):
        entropy_mm(np.zeros(8))


# --------------------------------------------------------------------------------------------
# T6.3 — cross-entropy
# --------------------------------------------------------------------------------------------


def test_cross_entropy_hand_computed():
    # q(0)=0.5 -> 1 bit, q(1)=0.25 -> 2 bits, mean over 2 slots = 1.5 bits
    pred = np.array([[0.5, 0.25, 0.125, 0.125]])
    true = np.array([[0, 1]])
    marginal = np.full(4, 0.25)
    assert per_slot_cross_entropy(pred, true, marginal, epsilon=0.0) == pytest.approx(1.5)


def test_cross_entropy_of_the_marginal_equals_its_entropy():
    """CE of the marginal predictor against its own distribution is H — the F0 sanity check."""
    true = np.array([[0, 1], [0, 2], [0, 1], [1, 2]])
    marginal = marginal_distribution(true, n_experts=4)
    ce = per_slot_cross_entropy(marginal[None, :], true, marginal, epsilon=0.0)
    assert ce == pytest.approx(entropy_plugin(expert_counts(true, 4)))


def test_cross_entropy_accumulates_independently_of_chunking():
    rng = np.random.default_rng(11)
    n, n_experts, k = 200, 8, 3
    true = np.stack([rng.choice(n_experts, size=k, replace=False) for _ in range(n)])
    pred = rng.dirichlet(np.ones(n_experts), size=n)
    marginal = marginal_distribution(true, n_experts)

    whole = CrossEntropyAccumulator(n_experts, marginal)
    whole.update(pred, true)

    chunked = CrossEntropyAccumulator(n_experts, marginal)
    for start in range(0, n, 37):
        chunked.update(pred[start : start + 37], true[start : start + 37])

    assert chunked.result() == pytest.approx(whole.result(), rel=1e-12)
    assert chunked.n_slots == whole.n_slots == n * k


def test_epsilon_mix_keeps_cross_entropy_finite_on_a_zero_probability_expert():
    pred = np.array([[1.0, 0.0, 0.0, 0.0]])
    true = np.array([[1]])  # predictor assigns exactly zero to the true expert
    marginal = np.full(4, 0.25)
    with np.errstate(divide="ignore"):
        assert not np.isfinite(-np.log2(pred[0, 1]))  # would be inf without the mix
    assert np.isfinite(per_slot_cross_entropy(pred, true, marginal))


def test_epsilon_mix_is_a_convex_combination():
    pred = np.array([[0.7, 0.3]])
    marginal = np.array([0.5, 0.5])
    mixed = epsilon_mix(pred, marginal, epsilon=0.1)
    np.testing.assert_allclose(mixed, [[0.68, 0.32]])
    assert mixed.sum() == pytest.approx(1.0)


def test_unnormalized_predictions_are_rejected():
    accumulator = CrossEntropyAccumulator(4, np.full(4, 0.25))
    with pytest.raises(ValueError, match="sum to 1"):
        accumulator.update(np.array([[0.9, 0.9, 0.9, 0.9]]), np.array([[0]]))


# --------------------------------------------------------------------------------------------
# T6.3 — the MI lower bound is signed and never clamped
# --------------------------------------------------------------------------------------------


def test_mi_is_signed_and_flags_negatives():
    result = mi_lower_bound(entropy_bits=1.0, cross_entropy_bits=2.5, n_experts=64)
    assert result.mi_bits == pytest.approx(-1.5)
    assert result.negative is True
    assert result.ratio == pytest.approx(-1.5)
    assert result.to_json()["mi_bits"] == pytest.approx(-1.5)


def test_overfit_probe_on_500_samples_yields_negative_mi():
    """The literal T6.3 acceptance criterion.

    A probe that has memorised training noise puts its mass on the wrong expert at evaluation
    time. CE then exceeds H and I_hat goes negative. The point of the test is that the value is
    *reported* as negative rather than floored at zero.
    """
    rng = np.random.default_rng(42)
    n, n_experts = 500, 32
    true = rng.integers(0, n_experts, size=(n, 1))

    # Confidently wrong: 0.99 on an expert that is never the true one for that row.
    pred = np.full((n, n_experts), 0.01 / (n_experts - 1))
    wrong = (true[:, 0] + 1 + rng.integers(0, n_experts - 1, size=n)) % n_experts
    pred[np.arange(n), wrong] = 0.99
    pred /= pred.sum(axis=1, keepdims=True)

    marginal = marginal_distribution(true, n_experts)
    entropy_bits = entropy_mm(expert_counts(true, n_experts))
    ce_bits = per_slot_cross_entropy(pred, true, marginal, split="test")

    result = mi_lower_bound(entropy_bits, ce_bits, n_experts, entropy_split="test", ce_split="test")
    assert result.negative is True
    assert result.mi_bits < 0.0
    assert result.to_json()["negative"] is True


def test_a_perfect_probe_recovers_almost_all_the_entropy():
    rng = np.random.default_rng(5)
    n, n_experts = 2000, 16
    true = rng.integers(0, n_experts, size=(n, 1))

    pred = np.full((n, n_experts), 1e-9)
    pred[np.arange(n), true[:, 0]] = 1.0
    pred /= pred.sum(axis=1, keepdims=True)

    marginal = marginal_distribution(true, n_experts)
    entropy_bits = entropy_mm(expert_counts(true, n_experts))
    ce_bits = per_slot_cross_entropy(pred, true, marginal)

    result = mi_lower_bound(entropy_bits, ce_bits, n_experts)
    assert result.negative is False
    assert result.ratio > 0.99


def test_mixing_splits_is_refused():
    with pytest.raises(ValueError, match="mixing splits"):
        mi_lower_bound(1.0, 0.5, 64, entropy_split="train", ce_split="test")


def test_normalizers_use_log2_n_experts():
    result = mi_lower_bound(entropy_bits=3.0, cross_entropy_bits=1.5, n_experts=64)
    assert result.ce_normalized == pytest.approx(1.5 / 6.0)
    assert result.entropy_normalized == pytest.approx(3.0 / 6.0)


def test_ratio_is_nan_for_a_deterministic_router():
    assert math.isnan(mi_lower_bound(0.0, 0.0, 64).ratio)


# --------------------------------------------------------------------------------------------
# T6.4 — load-balance index
# --------------------------------------------------------------------------------------------


def test_load_balance_index_bounds():
    uniform = np.full(64, 10_000, dtype=np.int64)
    assert load_balance_index(uniform, 64, is_counts=True) == pytest.approx(1.0, abs=1e-3)

    collapsed = np.zeros(64, dtype=np.int64)
    collapsed[0] = 640_000
    assert load_balance_index(collapsed, 64, is_counts=True) == pytest.approx(0.0, abs=1e-4)


def test_load_balance_index_from_expert_sets():
    """Balanced usage over enough slots that the bias correction is irrelevant."""
    true = np.tile(np.array([[0, 1], [2, 3]]), (5_000, 1))
    assert load_balance_index(true, 4) == pytest.approx(1.0, abs=1e-3)


def test_load_balance_index_warns_when_undersampled():
    """Badly sparse counts must be flagged, and still reported as measured rather than clamped."""
    with pytest.warns(RuntimeWarning, match="undersampled"):
        index = load_balance_index(np.ones(4, dtype=np.int64), 4, is_counts=True)
    assert index > 1.0, "reported as measured, not clamped"


def test_load_balance_index_is_quiet_on_well_sampled_counts(recwarn):
    """Balanced usage sits a hair above 1.0 by construction; that must not warn."""
    load_balance_index(np.full(64, 10_000, dtype=np.int64), 64, is_counts=True)
    load_balance_index(np.full(4, 5_000, dtype=np.int64), 4, is_counts=True)
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_load_balance_index_detects_imbalance_ordering():
    balanced = np.full(32, 1000, dtype=np.int64)
    skewed = np.array([20_000] + [1000] * 31, dtype=np.int64)
    assert load_balance_index(skewed, 32, is_counts=True) < load_balance_index(
        balanced, 32, is_counts=True
    )
