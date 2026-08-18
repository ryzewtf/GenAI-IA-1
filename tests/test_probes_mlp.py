"""F5 — the two-layer MLP probe (plan T7.5).

Same organising principle as ``tests/test_probes.py``: every probe is tested against **two**
traces, one where the routing is a known function of the feature and one where it is random. F5
adds a third obligation the linear probes do not have. F5 exists only to measure what F4 cannot,
so the pair test here is not merely "does it learn" but "does it learn something a fitted
:class:`SoftmaxProbe` on the *same rows* provably cannot" — hence the XOR-structured trace, whose
Bayes-optimal linear predictor is the marginal.

And because backprop in :mod:`src.probes.mlp` is written by hand with no autograd behind it, the
gradient check below is load-bearing rather than decorative: a sign error in ``grad_z1`` would not
crash, it would quietly make F5 a weak probe and the F5−F4 gap a false negative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.probes.base import (
    Design,
    DenseBlock,
    Standardizer,
    UndefinedFeature,
    soft_targets,
)
from src.probes.evaluate import evaluate_on_test
from src.probes.features import CPU_FEATURES, CorpusIndex, build_features
from src.probes.linear import SoftmaxProbe
from src.probes.mlp import MLPProbe, gelu, gelu_grad
from src.traces.format import TraceSpec
from src.traces.reader import TraceReader
from src.traces.synth import make_synthetic_trace

N_EXPERTS = 8
TOP_K = 2


def _split(n, frac=(0.7, 0.15)):
    a = int(n * frac[0])
    b = a + int(n * frac[1])
    return slice(0, a), slice(a, b), slice(b, n)


def _designs(X: np.ndarray, tr: slice, va: slice, te: slice):
    """Train-fitted standardizer, as plan T7.5 requires — never refit on val or test."""
    st = Standardizer.fit(X[tr])
    return tuple(Design((DenseBlock(X[s], standardizer=st),)) for s in (tr, va, te))


# -- fixtures: the two traces --------------------------------------------------------------------


@pytest.fixture
def xor_router_input():
    """Expert set is an XOR of sign bits of the router input — genuinely nonlinear.

    Two independent XORs of coordinate signs select one of four expert *pairs*. Every marginal
    and every pairwise-linear projection of the label on ``X`` is flat by construction, so the
    best a linear softmax probe can do is the uniform marginal at ``log2(8) = 3`` bits, while the
    achievable floor is ``log2(2) = 1`` bit (the target is uniform over the pair). The remaining
    coordinates are pure noise, so a probe that "wins" by memorising them is visible as a val/test
    gap rather than as a good score.
    """
    rng = np.random.default_rng(3)
    n, d = 9000, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    bit0 = (X[:, 0] > 0) ^ (X[:, 1] > 0)
    bit1 = (X[:, 2] > 0) ^ (X[:, 3] > 0)
    group = (2 * bit0 + bit1).astype(np.int32)
    y = np.stack([2 * group, 2 * group + 1], axis=1).astype(np.int32)
    return X, y


@pytest.fixture
def independent_router_input():
    """Routing drawn independently of the router input — the null case."""
    rng = np.random.default_rng(29)
    n, d = 260, 160  # deliberately n < d: the regime plan §1.2 expects Î < 0 in
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.stack(
        [rng.choice(N_EXPERTS, size=TOP_K, replace=False) for _ in range(n)]
    ).astype(np.int32)
    return X, y


# -- activation and gradients ---------------------------------------------------------------------


def test_gelu_grad_matches_central_differences_on_the_activation():
    x = np.linspace(-4.0, 4.0, 41).astype(np.float32).reshape(1, -1)
    h = 1e-3
    numeric = (gelu(x + h) - gelu(x - h)) / (2 * h)
    assert np.allclose(gelu_grad(x), numeric, atol=2e-4)


def _reference_loss(params, Xb, Tb):
    """The F5 forward pass rewritten from scratch in **float64** — the gradient check's oracle.

    Central differences on :meth:`MLPProbe._loss_and_grads` itself do not work: that loss is fp32,
    so its ~1e-7 relative roundoff divided by a usable step size swamps gradient entries of order
    1e-3 and the "check" would only be able to detect gross sign errors. Differentiating an
    independent float64 transcription instead gives a numeric gradient good to ~1e-11, and because
    the analytic gradient is produced by the module's *own* forward pass, a discrepancy in either
    direction — forward or backward — shows up as a mismatch. This is a check on both halves.
    """
    W1, b1, W2, b2 = (np.asarray(p, dtype=np.float64) for p in params)
    X = np.asarray(Xb, dtype=np.float64)
    T = np.asarray(Tb, dtype=np.float64)
    z1 = X @ W1 + b1
    c = math.sqrt(2.0 / math.pi)
    h = 0.5 * z1 * (1.0 + np.tanh(c * (z1 + 0.044715 * z1**3)))
    z2 = h @ W2 + b2
    z2 = z2 - z2.max(axis=1, keepdims=True)
    logq = z2 - np.log(np.exp(z2).sum(axis=1, keepdims=True))
    return float(-(T * logq).sum() / X.shape[0])


def test_analytic_gradients_match_central_differences_for_every_parameter():
    """No autograd stands behind :meth:`MLPProbe._loss_and_grads`; this test does.

    Every parameter tensor is checked. A correct ``gW2`` with a wrong ``gW1`` is exactly the bug
    that would leave F5 behaving like a randomly-projected linear probe — it would still train, it
    would just never reach the nonlinear ceiling F5 exists to measure.
    """
    rng = np.random.default_rng(0)
    n_experts, d, hidden, batch = 5, 4, 6, 7
    probe = MLPProbe(n_experts=n_experts, hidden_width=hidden, dropout=0.0, seed=1)
    probe._init_params(d, np.random.default_rng(1))
    # Perturb the biases off zero so their gradients are checked at non-degenerate points, and
    # scale W2 up so the softmax is not sitting at the uniform point either.
    probe.b2 += rng.normal(0.0, 0.3, size=n_experts).astype(np.float32)
    probe.b1 += rng.normal(0.0, 0.3, size=hidden).astype(np.float32)
    probe.W2 = (probe.W2 * 8.0).astype(np.float32)

    Xb = rng.normal(size=(batch, d)).astype(np.float32)
    sets = np.stack([rng.choice(n_experts, 2, replace=False) for _ in range(batch)])
    Tb = soft_targets(sets, n_experts)

    loss, grads = probe._loss_and_grads(Xb, Tb)
    names = ("W1", "b1", "W2", "b2")
    params = [getattr(probe, n).astype(np.float64) for n in names]
    assert loss == pytest.approx(_reference_loss(params, Xb, Tb), rel=1e-5)

    step = 1e-6
    for slot, (name, analytic) in enumerate(zip(names, grads)):
        base = params[slot]
        numeric = np.zeros_like(base)
        for ix in np.ndindex(*base.shape):
            plus, minus = base.copy(), base.copy()
            plus[ix] += step
            minus[ix] -= step
            up = list(params)
            up[slot] = plus
            down = list(params)
            down[slot] = minus
            numeric[ix] = (_reference_loss(up, Xb, Tb) - _reference_loss(down, Xb, Tb)) / (2 * step)
        # The analytic side is fp32, so the achievable agreement is ~1e-6 relative on the largest
        # entries; the absolute floor covers entries that are themselves near zero.
        assert np.allclose(numeric, analytic, rtol=2e-4, atol=1e-7), (
            f"gradient mismatch for {name}: max abs err "
            f"{np.abs(numeric - np.asarray(analytic, np.float64)).max():.3e}"
        )


def test_output_layer_gradient_is_softmax_minus_target_over_batch():
    """The one line a reader will want to check, checked directly."""
    rng = np.random.default_rng(4)
    probe = MLPProbe(n_experts=6, hidden_width=5, dropout=0.0)
    probe._init_params(3, np.random.default_rng(2))
    Xb = rng.normal(size=(9, 3)).astype(np.float32)
    Tb = soft_targets(rng.integers(0, 6, size=(9, 2)), 6)
    _, (_, _, gW2, gb2) = probe._loss_and_grads(Xb, Tb)
    _, h, logq = probe._forward(Xb)
    grad_z = (np.exp(logq) - Tb) / np.float32(9)
    assert np.allclose(gb2, grad_z.sum(axis=0), atol=1e-6)
    assert np.allclose(gW2, h.T @ grad_z, atol=1e-6)


# -- trace 1: routing is a nonlinear function of the router input ---------------------------------


def test_mlp_beats_the_linear_probe_on_an_xor_structured_router_input(xor_router_input):
    """F5's whole reason to exist: nonlinearity F4 cannot represent.

    The XOR construction makes this a clean statement rather than a tuning contest — the linear
    probe's optimum *is* the marginal, so any gap is structural.
    """
    X, y = xor_router_input
    tr, va, te = _split(X.shape[0])
    dtr, dva, dte = _designs(X, tr, va, te)

    linear = SoftmaxProbe(n_experts=N_EXPERTS, feature="F4", lr=0.1, max_epochs=60)
    linear.fit(dtr, y[tr], dva, y[va])
    linear_ce = linear.slot_ce(dte, y[te])

    mlp = MLPProbe(
        n_experts=N_EXPERTS,
        hidden_width=64,
        feature="F5",
        lr=0.02,
        batch_size=256,
        max_epochs=120,
        patience=15,
        weight_decay=0.0,
    )
    mlp.fit(dtr, y[tr], dva, y[va])
    mlp_ce = mlp.slot_ce(dte, y[te])

    # The linear probe cannot do better than uniform here; the MLP's floor is 1 bit.
    assert linear_ce == pytest.approx(math.log2(N_EXPERTS), abs=0.1)
    assert mlp_ce < 1.5
    assert linear_ce - mlp_ce > 1.0

    out = evaluate_on_test(mlp, dte, y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    assert out["family_a"]["set_agreement@k"] > 0.9
    assert out["family_b"]["mi_bits"] > 1.0


def test_report_records_the_hidden_width_and_architecture(xor_router_input):
    X, y = xor_router_input
    tr, va, te = _split(X.shape[0])
    dtr, dva, _ = _designs(X, tr, va, te)
    mlp = MLPProbe(n_experts=N_EXPERTS, hidden_width=17, max_epochs=2, batch_size=512)
    mlp.fit(dtr, y[tr], dva, y[va])
    doc = mlp.report.to_json()
    assert doc["predictor"] == "mlp_probe"
    assert doc["feature"] == "F5"
    assert doc["hyperparams"]["hidden_width"] == 17
    assert doc["hyperparams"]["architecture"] == "linear-gelu-linear"
    assert doc["hyperparams"]["dtype"] == "float32"
    assert doc["design_width"] == X.shape[1]


def test_unfitted_probe_is_near_uniform_but_not_exactly_uniform(xor_router_input):
    """Unlike SoftmaxProbe's zero init, W2 is random — so log2(K) holds only approximately."""
    X, y = xor_router_input
    tr, va, te = _split(X.shape[0])
    dtr, _, _ = _designs(X, tr, va, te)
    probe = MLPProbe(n_experts=N_EXPERTS, hidden_width=64, max_epochs=0)
    probe.fit(dtr, y[tr], None, None)
    ce = probe.slot_ce(dtr, y[tr])
    assert abs(ce - math.log2(N_EXPERTS)) < 0.25
    assert ce != pytest.approx(math.log2(N_EXPERTS), abs=1e-6)
    # A silently-zero W2 would make this probe linear, and would pass the bound above.
    assert np.abs(probe.W2).max() > 0.0


# -- trace 2: routing is independent of the router input -----------------------------------------


def test_mlp_does_not_beat_the_marginal_when_routing_is_independent(independent_router_input):
    X, y = independent_router_input
    tr, va, te = _split(X.shape[0])
    dtr, dva, dte = _designs(X, tr, va, te)
    mlp = MLPProbe(
        n_experts=N_EXPERTS, hidden_width=64, lr=0.01, batch_size=64, max_epochs=60, patience=5
    )
    mlp.fit(dtr, y[tr], dva, y[va])
    out = evaluate_on_test(mlp, dte, y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    # No feature information to find, so Î must not be materially positive.
    assert out["family_b"]["mi_bits"] < 0.15
    assert out["family_b"]["cross_entropy_bits"] > math.log2(N_EXPERTS) - 0.3


def test_negative_mi_is_reported_not_clamped(independent_router_input):
    """Plan invariant I8 / §1.2: a negative Î is diagnostic and is never floored at zero."""
    X, y = independent_router_input
    tr, va, te = _split(X.shape[0])
    dtr, dva, dte = _designs(X, tr, va, te)
    mlp = MLPProbe(
        n_experts=N_EXPERTS,
        hidden_width=128,
        lr=0.05,
        batch_size=32,
        max_epochs=150,
        patience=150,
        min_delta=-1.0,  # defeat early stopping on purpose, as tests/test_probes.py does
        weight_decay=0.0,
    )
    mlp.fit(dtr, y[tr], dva, y[va])
    out = evaluate_on_test(mlp, dte, y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    fam = out["family_b"]
    assert fam["mi_bits"] < 0.0
    assert fam["negative"] is True
    assert fam["mi_bits"] == pytest.approx(fam["entropy_bits"] - fam["cross_entropy_bits"])


# -- optimisation contract -------------------------------------------------------------------------


def test_probe_restores_the_early_stopping_optimum(independent_router_input):
    """The reported F5 must be the best-val epoch, not the last (overfit) one."""
    X, y = independent_router_input
    tr, va, te = _split(X.shape[0])
    dtr, dva, _ = _designs(X, tr, va, te)
    probe = MLPProbe(
        n_experts=N_EXPERTS,
        hidden_width=64,
        lr=0.05,
        batch_size=32,
        max_epochs=40,
        patience=4,
        weight_decay=0.0,
    )
    probe.fit(dtr, y[tr], dva, y[va])
    history = probe.report.val_ce_history
    assert len(history) > 1
    assert probe.report.best_epoch == int(np.argmin(history))
    assert probe.report.best_val_ce_bits == pytest.approx(min(history), abs=1e-9)
    # The restored parameters reproduce the best epoch's val CE, not the last epoch's.
    assert probe.slot_ce(dva, y[va]) == pytest.approx(min(history), abs=1e-6)
    assert history[-1] > min(history)


def test_same_seed_is_identical_and_a_different_seed_is_not(independent_router_input):
    X, y = independent_router_input
    tr, va, te = _split(X.shape[0])
    dtr, dva, _ = _designs(X, tr, va, te)

    def run(seed):
        p = MLPProbe(
            n_experts=N_EXPERTS,
            hidden_width=32,
            lr=0.02,
            batch_size=64,
            max_epochs=12,
            patience=12,
            seed=seed,
        )
        p.fit(dtr, y[tr], dva, y[va])
        return p

    a, b, c = run(0), run(0), run(1)
    assert a.report.val_ce_history == b.report.val_ce_history
    assert np.array_equal(a.W1, b.W1) and np.array_equal(a.W2, b.W2)
    assert c.report.val_ce_history != a.report.val_ce_history


def test_dropout_is_train_only_and_prediction_is_deterministic(independent_router_input):
    X, y = independent_router_input
    tr, va, te = _split(X.shape[0])
    dtr, dva, _ = _designs(X, tr, va, te)
    probe = MLPProbe(
        n_experts=N_EXPERTS, hidden_width=32, dropout=0.1, max_epochs=3, batch_size=64
    )
    probe.fit(dtr, y[tr], dva, y[va])
    assert probe.report.hyperparams["dropout"] == 0.1
    first = probe.predict_proba(dva)
    assert np.allclose(first, probe.predict_proba(dva))
    assert np.allclose(first.sum(axis=1), 1.0, atol=1e-5)


def test_prediction_is_chunk_invariant(xor_router_input):
    """The per-minibatch prediction loop is a memory decision, not a numerical one."""
    X, y = xor_router_input
    tr, va, te = _split(X.shape[0])
    dtr, _, dte = _designs(X, tr, va, te)
    probe = MLPProbe(n_experts=N_EXPERTS, hidden_width=32, max_epochs=2, batch_size=512)
    probe.fit(dtr, y[tr], None, None)
    big = probe.slot_ce(dte, y[te])
    probe.batch_size = 37
    assert probe.slot_ce(dte, y[te]) == pytest.approx(big, abs=1e-5)


def test_fit_rejects_a_row_count_mismatch(xor_router_input):
    X, y = xor_router_input
    tr, va, te = _split(X.shape[0])
    dtr, _, _ = _designs(X, tr, va, te)
    probe = MLPProbe(n_experts=N_EXPERTS, max_epochs=1)
    with pytest.raises(ValueError, match="row mismatch"):
        probe.fit(dtr, y[tr][:-1], None, None)


def test_predict_before_fit_is_an_error(xor_router_input):
    X, y = xor_router_input
    tr, va, te = _split(X.shape[0])
    dtr, _, _ = _designs(X, tr, va, te)
    with pytest.raises(RuntimeError, match="fit has not been called"):
        MLPProbe(n_experts=N_EXPERTS).predict_proba(dtr)


# -- feature plumbing against a real trace --------------------------------------------------------

SPEC = TraceSpec(n_moe_layers=4, n_experts=N_EXPERTS, top_k=TOP_K, hidden_dim=6)


def _copycat_topk(rng, tokens, spec):
    n = tokens.shape[0]
    out = np.empty((n, spec.n_moe_layers, spec.top_k), dtype=np.int64)
    base = tokens["token_id"].astype(np.int64) % spec.n_experts
    out[:, 0, 0] = base
    out[:, 0, 1] = (base + 1) % spec.n_experts
    for layer in range(1, spec.n_moe_layers):
        out[:, layer] = out[:, layer - 1]
    return out


@pytest.fixture
def trace(tmp_path):
    truth = make_synthetic_trace(
        tmp_path,
        spec=SPEC,
        shard_sizes=(400, 250, 350),
        tokens_per_doc=10,
        hidden_every=2,
        topk_fn=_copycat_topk,
    )
    reader = TraceReader(tmp_path, truth.model, truth.corpus, doc_splits=truth.doc_splits)
    yield reader, truth
    reader.close()


def test_f5_is_undefined_at_layer_zero(trace):
    """F5 conditions on layer ℓ−1, exactly as F4 does — the rule lives in features.py, not here."""
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    with pytest.raises(UndefinedFeature, match="does not exist"):
        build_features("F5", reader, index, 0, "train")


def test_f5_conditions_on_the_previous_layer_and_the_hidden_subsample(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    f5 = build_features("F5", reader, index, 2, "train")
    assert f5.meta["hidden_layer"] == 1
    assert set(f5.row_index.tolist()) <= set(index.captured.tolist())
    assert "hidden subsample" in f5.exclusion_reason
    assert f5.standardizer is not None
    # Same rows and same design as F4 at the same layer: the F4/F5 comparison must differ only in
    # the model class (plan §1.3).
    f4 = build_features("F4", reader, index, 2, "train")
    assert np.array_equal(f4.row_index, f5.row_index)
    assert f4.X.width == f5.X.width


def test_mlp_probe_satisfies_the_predictor_protocol():
    from src.probes.base import Predictor

    assert isinstance(MLPProbe(n_experts=N_EXPERTS), Predictor)


def test_f5_is_not_a_cpu_budget_feature():
    """Plan §Phase S: F5 is the one probe budgeted a GPU session, so it is never a CPU default."""
    assert "F5" not in CPU_FEATURES
