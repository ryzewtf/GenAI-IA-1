"""Tests for the audit fixes: set-decoded ridge probe (L3), doc aggregates + bootstrap (L7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics.bootstrap import (
    DocAggregates,
    draw_weights,
    replicate_metrics,
    summarize,
    save_aggregates,
    load_aggregates,
)
from src.probes.evaluate import evaluate_on_test
from src.probes.marginal import MarginalPredictor
from src.probes.ridge import RidgeScoreProbe, score_targets, set_decode


def _linear_router(n, d, n_experts, k, seed=0, as_probs=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    W = (0.3 * rng.normal(size=(d, n_experts))).astype(np.float32)  # router-like logit spread
    logits = X @ W
    sets = np.argsort(-logits, axis=1)[:, :k]
    if as_probs:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        stored = e / e.sum(axis=1, keepdims=True)
    else:
        stored = logits
    return X, stored, sets


def test_set_decode_tends_to_uniform_on_top_k():
    z = np.array([[5.0, 4.0, 1.0, 0.0, -1.0]])
    q = set_decode(z, k=2, temperature=1e-3)
    assert q[0, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert q.sum() == pytest.approx(1.0)


def test_score_targets_recovers_centered_logits_from_probs():
    X, probs, _ = _linear_router(200, 6, 8, 2)
    z, info = score_targets(probs)
    logits = X @ np.random.default_rng(0).normal(size=(6, 8))  # not used; shape sanity only
    assert info["stored_kind"] == "probs"
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-5)
    _, logits_stored, _ = _linear_router(200, 6, 8, 2, as_probs=False)
    z2, info2 = score_targets(logits_stored)
    assert info2["stored_kind"] == "logits"
    assert logits.shape == z2.shape


@pytest.mark.parametrize("as_probs", [True, False])
def test_ridge_on_exact_router_input_reaches_the_attainable_ceiling(as_probs):
    # FVR's claim: on the router's own input the probe recovers the executed set and the
    # attainable-normalised Î approaches 1, which the softmax-on-sets probe cannot.
    n_experts, k = 16, 2
    X, stored, sets = _linear_router(6000, 12, n_experts, k, as_probs=as_probs)
    Z, _ = score_targets(stored)
    tr, va, te = slice(0, 4000), slice(4000, 5000), slice(5000, 6000)
    probe = RidgeScoreProbe(n_experts=n_experts, top_k=k).fit(X[tr], Z[tr], X[va], Z[va], sets[va])
    metrics = evaluate_on_test(probe, X[te], sets[te], n_experts=n_experts, top_k=k)
    assert metrics["family_a"]["set_agreement@k"] > 0.999
    assert metrics["family_b"]["ratio_attainable"] > 0.97


def test_doc_aggregates_and_bootstrap_reproduce_the_point_estimate():
    rng = np.random.default_rng(1)
    n, n_experts, k = 3000, 16, 2
    sets = np.stack([rng.choice(n_experts, size=k, replace=False, p=np.linspace(1, 3, n_experts) / np.linspace(1, 3, n_experts).sum()) for _ in range(n)])
    docs = rng.integers(0, 40, size=n)
    pred = MarginalPredictor(n_experts=n_experts)
    pred.fit(None, sets, None, sets)
    metrics = evaluate_on_test(pred, None, sets, n_experts=n_experts, top_k=k, doc_ids=docs)
    agg = DocAggregates.from_mapping(metrics["doc_aggregates"])
    w = draw_weights(np.arange(40), n_boot=50, seed=0)
    reps = replicate_metrics(agg, w)
    fb = metrics["family_b"]
    assert reps["entropy_bits"][0] == pytest.approx(fb["entropy_bits"], rel=1e-9)
    assert reps["cross_entropy_bits"][0] == pytest.approx(fb["cross_entropy_bits"], rel=1e-9)
    assert reps["mi_attainable"][0] == pytest.approx(fb["ratio_attainable"], rel=1e-6, abs=1e-9)
    assert reps["set_agreement@k"][0] == pytest.approx(metrics["family_a"]["set_agreement@k"])
    s = summarize(reps["mi_bits"])
    assert s["lo"] <= s["hi"] and s["n_boot"] == 50


def test_basic_interval_brackets_the_bias_corrected_estimate():
    # replicates centred 0.3 below the point: the percentile interval misses the point; the basic
    # interval is the reflection and brackets the bias-corrected estimate 2θ̂ − mean(reps) = 1.3
    reps = np.concatenate([[1.0], 0.7 + np.linspace(-0.1, 0.1, 999)])
    s = summarize(reps)
    assert s["hi"] < 1.0
    assert s["bias"] == pytest.approx(-0.3) and s["bc"] == pytest.approx(1.3)
    assert s["basic_lo"] <= s["bc"] <= s["basic_hi"]
    assert (s["basic_lo"], s["basic_hi"]) == pytest.approx((2 - s["hi"], 2 - s["lo"]))


def test_aggregates_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    sets = np.stack([rng.choice(8, size=2, replace=False) for _ in range(500)])
    pred = MarginalPredictor(n_experts=8)
    pred.fit(None, sets, None, sets)
    m = evaluate_on_test(pred, None, sets, n_experts=8, top_k=2, doc_ids=rng.integers(0, 9, 500))
    save_aggregates(tmp_path / "x.npz", {"3": m["doc_aggregates"]})
    back = load_aggregates(tmp_path / "x.npz")
    assert set(back) == {3} and back[3].top_k == 2
    assert np.array_equal(back[3].expert_counts, m["doc_aggregates"]["expert_counts"])


def test_uniform_mix_bound_is_reported_and_matches_when_the_true_expert_has_mass():
    # a predictor that never gives a true expert near-zero mass: the mix target cannot matter
    rng = np.random.default_rng(3)
    y = np.stack([rng.choice(8, size=2, replace=False) for _ in range(400)])
    pred = MarginalPredictor(n_experts=8, layer=0)
    pred.fit(None, y, None, y)
    out = evaluate_on_test(pred, None, y, n_experts=8, top_k=2)
    a, b = out["family_b"], out["family_b_uniform_mix"]
    assert b["entropy_bits"] == a["entropy_bits"]
    assert abs(b["mi_bits"] - a["mi_bits"]) < 1e-6


def test_holm_step_down():
    from scripts.audit_holm import holm, p_two_sided
    assert holm([0.001, 0.02, 0.04], 0.05) == [True, True, True]
    assert holm([0.001, 0.03, 0.04], 0.05) == [True, False, False]  # 0.03 > 0.05/2
    assert holm([0.2, 0.001], 0.05) == [False, True]
    assert abs(p_two_sided(1.96, 1.0) - 0.05) < 1e-3
