"""Phase 7 tests — predictors, features, evaluation, sweep.

The organising principle: every probe is tested against **two** traces, one where the routing is a
known function of the feature and one where it is random. A probe that scores well on both is
wired up wrongly, and a probe that scores badly on both is broken. Only the pair distinguishes
them, and only the pair would have caught the class of bug the plan worries about most — a
feature silently conditioning on the label.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.metrics.information import marginal_distribution
from src.probes.base import (
    Design,
    DenseBlock,
    MultiHotBlock,
    Standardizer,
    UndefinedFeature,
    soft_targets,
)
from src.probes.counts import UNSEEN_BUCKET, CountTablePredictor, frequency_buckets
from src.probes.evaluate import evaluate_on_test
from src.probes.features import (
    CorpusIndex,
    build_features,
    consecutive_repetition_rate,
)
from src.probes.linear import SoftmaxProbe, log_softmax
from src.probes.marginal import MarginalPredictor, reference_marginal_on_test
from src.probes.train import output_path, sweep
from src.runtime.session import SessionBudget
from src.traces.format import TraceSpec
from src.traces.reader import TraceReader
from src.traces.synth import make_synthetic_trace

N_EXPERTS = 8
TOP_K = 2


# -- fixtures ------------------------------------------------------------------------------------


def _sets_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int32)


@pytest.fixture
def token_driven():
    """Routing determined entirely by ``token_id`` — F1's best case, F0's worst."""
    rng = np.random.default_rng(7)
    vocab = 40
    table = rng.normal(size=(vocab, N_EXPERTS))
    ids = rng.integers(0, vocab, size=6000)
    y = _sets_from_scores(table[ids], TOP_K)
    return ids, y, vocab


@pytest.fixture
def random_routing():
    rng = np.random.default_rng(11)
    ids = rng.integers(0, 40, size=6000)
    y = np.stack(
        [rng.choice(N_EXPERTS, size=TOP_K, replace=False) for _ in range(ids.size)]
    ).astype(np.int32)
    return ids, y


def _split(n, frac=(0.7, 0.15)):
    a = int(n * frac[0])
    b = a + int(n * frac[1])
    return slice(0, a), slice(a, b), slice(b, n)


# -- base primitives -------------------------------------------------------------------------------


def test_soft_targets_are_the_slot_level_distribution():
    sets = np.array([[0, 3], [1, 1]])
    t = soft_targets(sets, N_EXPERTS)
    assert np.allclose(t.sum(axis=1), [1.0, 0.5])  # row 1 is degenerate on purpose
    assert t[0, 0] == pytest.approx(0.5)
    assert t[0, 3] == pytest.approx(0.5)


def test_multi_hot_block_densifies_exactly_k_ones():
    sets = np.array([[0, 3], [2, 5], [1, 7]])
    block = MultiHotBlock(sets, N_EXPERTS)
    dense = block.densify(np.array([0, 2]))
    assert dense.shape == (2, N_EXPERTS)
    assert dense.sum(axis=1).tolist() == [2.0, 2.0]
    assert dense[1, 1] == 1.0 and dense[1, 7] == 1.0


def test_multi_hot_block_rejects_out_of_range_conditioning_experts():
    with pytest.raises(ValueError, match="outside"):
        MultiHotBlock(np.array([[0, N_EXPERTS]]), N_EXPERTS)


def test_design_refuses_row_count_disagreement():
    with pytest.raises(ValueError, match="row count"):
        Design(
            (
                MultiHotBlock(np.zeros((5, TOP_K), int), N_EXPERTS),
                DenseBlock(np.zeros((4, 3), np.float32)),
            )
        )


def test_design_concatenation_order_is_block_order():
    d = Design(
        (
            DenseBlock(np.full((3, 2), 9.0, np.float32), name="a"),
            MultiHotBlock(np.zeros((3, 1), int), N_EXPERTS, name="b"),
        )
    )
    assert d.width == 2 + N_EXPERTS
    dense = d.densify(np.arange(3))
    assert np.all(dense[:, :2] == 9.0)
    assert [b["block"] for b in d.describe()] == ["a", "b"]


def test_standardizer_survives_a_constant_dimension():
    values = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    s = Standardizer.fit(values)
    out = s.transform(values)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[:, 1], 0.0)  # dead feature, not a NaN


# -- F0 ---------------------------------------------------------------------------------------------


def test_f0_refuses_to_be_fit_on_test():
    """Plan T7.2: the train marginal is the predictor, the test marginal is H. Not the same."""
    with pytest.raises(ValueError, match="fit on train"):
        MarginalPredictor(n_experts=N_EXPERTS, fit_split="test")


def test_f0_predictor_and_the_section_1_2_reference_are_different_numbers(random_routing):
    ids, y = random_routing
    tr, _, te = _split(ids.size)
    f0 = MarginalPredictor(n_experts=N_EXPERTS)
    f0.fit(None, y[tr])
    assert not np.allclose(f0.train_marginal(), reference_marginal_on_test(y[te], N_EXPERTS))


def test_f0_gives_zero_information_on_random_routing(random_routing):
    ids, y = random_routing
    tr, va, te = _split(ids.size)
    f0 = MarginalPredictor(n_experts=N_EXPERTS)
    f0.fit(None, y[tr], None, y[va])
    out = evaluate_on_test(f0, None, y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    # H - CE for a marginal predictor is estimator noise around zero, not a positive result.
    assert abs(out["family_b"]["mi_bits"]) < 0.05


def test_f0_broadcasts_without_materialising_the_full_matrix(random_routing):
    ids, y = random_routing
    f0 = MarginalPredictor(n_experts=N_EXPERTS)
    f0.fit(None, y)
    probs = f0.predict_proba(None)
    assert probs.shape[-1] == N_EXPERTS
    assert probs.sum() == pytest.approx(1.0)


# -- F1 ---------------------------------------------------------------------------------------------


def test_f1_recovers_token_driven_routing(token_driven):
    ids, y, vocab = token_driven
    tr, va, te = _split(ids.size)
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=vocab)
    f1.fit(ids[tr], y[tr], ids[va], y[va])
    out = evaluate_on_test(f1, ids[te], y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    assert out["family_a"]["set_agreement@k"] > 0.9
    assert out["family_b"]["mi_bits"] > 1.0


def test_f1_finds_nothing_in_random_routing(random_routing):
    ids, y = random_routing
    tr, va, te = _split(ids.size)
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=40)
    f1.fit(ids[tr], y[tr], ids[va], y[va])
    out = evaluate_on_test(f1, ids[te], y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    # The paired assertion: same predictor, same code path, no signal in the trace.
    assert out["family_b"]["mi_bits"] < 0.15


def test_f1_is_the_closed_form_optimum_of_its_own_model(token_driven):
    """A gradient fit of ``softmax(W[token_id])`` cannot beat the smoothed count table."""
    ids, y, vocab = token_driven
    tr, va, te = _split(ids.size)

    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=vocab)
    f1.fit(ids[tr], y[tr], ids[va], y[va])
    table_ce = f1._slot_ce(ids[te], y[te])

    one_hot = np.zeros((ids.size, vocab), dtype=np.float32)
    one_hot[np.arange(ids.size), ids] = 1.0
    probe = SoftmaxProbe(
        n_experts=N_EXPERTS, feature="F1-sgd", lr=0.3, max_epochs=80, weight_decay=0.0
    )
    probe.fit(
        Design((DenseBlock(one_hot[tr]),)),
        y[tr],
        Design((DenseBlock(one_hot[va]),)),
        y[va],
    )
    sgd_ce = probe.slot_ce(Design((DenseBlock(one_hot[te]),)), y[te])
    assert table_ce <= sgd_ce + 0.05


def test_f1_falls_back_exactly_to_the_train_marginal_on_unseen_tokens():
    rng = np.random.default_rng(3)
    ids = rng.integers(0, 10, size=2000)
    y = np.stack([rng.choice(N_EXPERTS, TOP_K, replace=False) for _ in range(2000)]).astype(np.int32)
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=20)
    f1.fit(ids, y, ids, y)
    unseen = f1.predict_proba(np.array([15]))[0]
    assert np.allclose(unseen, f1.train_marginal, atol=1e-12)


def test_f1_alpha_is_selected_on_val_not_train(token_driven):
    ids, y, vocab = token_driven
    tr, va, _ = _split(ids.size)
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=vocab)
    f1.fit(ids[tr], y[tr], ids[va], y[va])
    history = f1.report.val_ce_history
    assert len(history) == len(f1.alpha_grid)
    assert f1.alpha == pytest.approx(f1.alpha_grid[int(np.argmin(history))])
    # Fit on train alone would drive alpha to the grid minimum every time.
    assert math.isfinite(f1.report.best_val_ce_bits)


def test_f1_rejects_token_ids_outside_the_declared_vocabulary():
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=5)
    with pytest.raises(ValueError, match="outside"):
        f1.fit(np.array([0, 9]), np.array([[0, 1], [2, 3]]))


# -- frequency deciles (T7.3) ------------------------------------------------------------------------


def test_frequency_buckets_separate_unseen_types_from_the_rarest_decile():
    train = np.repeat(np.arange(20), np.arange(1, 21))  # type v occurs v+1 times
    ev = np.array([0, 19, 99])  # rarest seen, most frequent seen, never seen
    buckets, meta = frequency_buckets(train, ev, n_buckets=5, vocab_size=100)
    assert buckets[0] == 0
    assert buckets[1] == 4
    assert buckets[2] == UNSEEN_BUCKET
    assert meta["rows_per_bucket"][str(UNSEEN_BUCKET)] == 1


def test_frequency_buckets_cover_every_seen_type_exactly_once():
    train = np.repeat(np.arange(23), 2)
    buckets, meta = frequency_buckets(train, np.arange(23), n_buckets=10, vocab_size=23)
    assert (buckets >= 0).all()
    assert sum(meta["types_per_bucket"][str(b)] for b in range(10)) == 23


def test_frequency_stratified_ce_is_reported_per_bucket(token_driven):
    ids, y, vocab = token_driven
    tr, va, te = _split(ids.size)
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=vocab)
    f1.fit(ids[tr], y[tr], ids[va], y[va])
    buckets, meta = frequency_buckets(ids[tr], ids[te], n_buckets=4, vocab_size=vocab)
    out = evaluate_on_test(
        f1, ids[te], y[te], n_experts=N_EXPERTS, top_k=TOP_K, buckets=buckets, bucket_meta=meta
    )
    assert set(out["strata"]) <= {str(b) for b in list(range(4)) + [UNSEEN_BUCKET]}
    total = sum(s["n_slots"] for s in out["strata"].values())
    assert total == out["n_slots"]
    # No per-stratum MI: it would need a per-stratum H, which is too biased to compare.
    assert all("mi_bits" not in s for s in out["strata"].values())


# -- SoftmaxProbe ------------------------------------------------------------------------------------


def test_log_softmax_is_normalized_and_stable():
    z = np.array([[1000.0, 1000.0, 999.0]], dtype=np.float32)
    lp = log_softmax(z)
    assert np.isfinite(lp).all()
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-5)


def test_probe_learns_a_set_to_set_map():
    """F2/F3's model class: multi-hot conditioning set -> expert distribution."""
    rng = np.random.default_rng(5)
    n = 8000
    cond = np.stack([rng.choice(N_EXPERTS, TOP_K, replace=False) for _ in range(n)]).astype(np.int32)
    # Target: shift the conditioning set by one expert, deterministically.
    y = ((cond + 1) % N_EXPERTS).astype(np.int32)
    tr, va, te = _split(n)
    probe = SoftmaxProbe(n_experts=N_EXPERTS, feature="F2", lr=0.2, max_epochs=60)
    probe.fit(
        Design((MultiHotBlock(cond[tr], N_EXPERTS),)),
        y[tr],
        Design((MultiHotBlock(cond[va], N_EXPERTS),)),
        y[va],
    )
    out = evaluate_on_test(
        probe,
        Design((MultiHotBlock(cond[te], N_EXPERTS),)),
        y[te],
        n_experts=N_EXPERTS,
        top_k=TOP_K,
    )
    assert out["family_a"]["set_agreement@k"] > 0.95
    assert out["family_a"]["exact_match"] > 0.9


def test_probe_restores_the_early_stopping_optimum():
    """Without restore, the reported probe is the last epoch — the overfit one for F5-like capacity."""
    rng = np.random.default_rng(13)
    n = 300
    X = rng.normal(size=(n, 60)).astype(np.float32)
    y = np.stack([rng.choice(N_EXPERTS, TOP_K, replace=False) for _ in range(n)]).astype(np.int32)
    tr, va, _ = _split(n)
    probe = SoftmaxProbe(
        n_experts=N_EXPERTS, feature="F4", lr=0.3, max_epochs=40, patience=3, weight_decay=0.0
    )
    dtr = Design((DenseBlock(X[tr]),))
    dva = Design((DenseBlock(X[va]),))
    probe.fit(dtr, y[tr], dva, y[va])
    history = probe.report.val_ce_history
    assert probe.report.best_epoch == int(np.argmin(history))
    assert probe.slot_ce(dva, y[va]) == pytest.approx(min(history), abs=1e-6)


def test_probe_starts_from_the_uniform_predictor():
    """Zero init means epoch 0 is exactly log2(K) — a val curve that starts elsewhere is a bug."""
    rng = np.random.default_rng(2)
    n = 400
    cond = np.stack([rng.choice(N_EXPERTS, TOP_K, replace=False) for _ in range(n)]).astype(np.int32)
    probe = SoftmaxProbe(n_experts=N_EXPERTS, max_epochs=0)
    probe.fit(Design((MultiHotBlock(cond, N_EXPERTS),)), cond, None, None)
    ce = probe.slot_ce(Design((MultiHotBlock(cond, N_EXPERTS),)), cond)
    assert ce == pytest.approx(math.log2(N_EXPERTS), abs=1e-5)


def test_overfitting_probe_reports_negative_mi_rather_than_zero():
    """Plan §1.2: report Î as measured. The floor-at-zero bug would hide exactly this case."""
    rng = np.random.default_rng(21)
    n = 240
    X = rng.normal(size=(n, 200)).astype(np.float32)
    y = np.stack([rng.choice(N_EXPERTS, TOP_K, replace=False) for _ in range(n)]).astype(np.int32)
    tr, va, te = _split(n)
    probe = SoftmaxProbe(
        n_experts=N_EXPERTS,
        feature="F5-like",
        lr=0.5,
        max_epochs=200,
        patience=200,
        min_delta=-1.0,  # defeat early stopping on purpose
        weight_decay=0.0,
    )
    probe.fit(Design((DenseBlock(X[tr]),)), y[tr], Design((DenseBlock(X[va]),)), y[va])
    out = evaluate_on_test(
        probe, Design((DenseBlock(X[te]),)), y[te], n_experts=N_EXPERTS, top_k=TOP_K
    )
    assert out["family_b"]["mi_bits"] < 0.0
    assert out["family_b"]["negative"] is True


# -- evaluation contract ------------------------------------------------------------------------------


def test_evaluation_reports_both_families_on_the_test_split(random_routing):
    ids, y = random_routing
    tr, va, te = _split(ids.size)
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=40)
    f1.fit(ids[tr], y[tr], ids[va], y[va])
    out = evaluate_on_test(f1, ids[te], y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    assert out["family_b"]["split"] == "test"
    assert out["family_a"]["n"] == y[te].shape[0]
    assert out["family_b"]["entropy_bits"] == pytest.approx(
        out["family_b"]["mi_bits"] + out["family_b"]["cross_entropy_bits"]
    )


def test_evaluation_is_chunk_invariant(token_driven):
    ids, y, vocab = token_driven
    tr, va, te = _split(ids.size)
    f1 = CountTablePredictor(n_experts=N_EXPERTS, vocab_size=vocab)
    f1.fit(ids[tr], y[tr], ids[va], y[va])
    a = evaluate_on_test(f1, ids[te], y[te], n_experts=N_EXPERTS, top_k=TOP_K, chunk_rows=10**9)
    b = evaluate_on_test(f1, ids[te], y[te], n_experts=N_EXPERTS, top_k=TOP_K, chunk_rows=37)
    assert a["family_b"]["cross_entropy_bits"] == pytest.approx(
        b["family_b"]["cross_entropy_bits"], abs=1e-9
    )
    assert a["family_a"]["set_agreement@k"] == pytest.approx(b["family_a"]["set_agreement@k"])


def test_epsilon_mix_target_is_the_test_marginal(random_routing):
    """A dead-on-train expert must not make CE infinite (plan T5.3 collects that histogram)."""
    ids, y = random_routing
    y = y.copy()
    tr, va, te = _split(ids.size)
    # Expert 7 appears only on test.
    y[tr] = np.where(y[tr] == 7, 0, y[tr])
    y[va] = np.where(y[va] == 7, 0, y[va])
    y[te][0] = [7, 1]
    f0 = MarginalPredictor(n_experts=N_EXPERTS, laplace=0.0)
    f0.fit(None, y[tr], None, y[va])
    assert f0.train_marginal()[7] == 0.0
    out = evaluate_on_test(f0, None, y[te], n_experts=N_EXPERTS, top_k=TOP_K)
    assert math.isfinite(out["family_b"]["cross_entropy_bits"])


def test_evaluation_rejects_a_k_that_disagrees_with_the_labels(random_routing):
    ids, y = random_routing
    f0 = MarginalPredictor(n_experts=N_EXPERTS)
    f0.fit(None, y)
    with pytest.raises(ValueError, match="top_k"):
        evaluate_on_test(f0, None, y, n_experts=N_EXPERTS, top_k=TOP_K + 1)


# -- features against a real trace ---------------------------------------------------------------------


SPEC = TraceSpec(n_moe_layers=4, n_experts=N_EXPERTS, top_k=TOP_K, hidden_dim=6)


def _copycat_topk(rng, tokens, spec):
    """Layer ℓ copies layer ℓ−1; layer 0 is a deterministic function of ``token_id``.

    So F2 should be near-perfect at ℓ ≥ 1, and F1 informative everywhere.
    """
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


def test_corpus_index_is_built_once_and_covers_every_split(trace):
    reader, truth = trace
    index = CorpusIndex.from_reader(reader)
    total = sum(index.rows(s).size for s in ("train", "val", "test"))
    assert total == reader.n_tokens
    assert index.n_tokens == truth.n_tokens


def test_f2_is_undefined_at_layer_zero(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    with pytest.raises(UndefinedFeature, match="does not exist"):
        build_features("F2", reader, index, 0, "train")


def test_f6_is_skipped_at_layer_zero_rather_than_degraded(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    with pytest.raises(UndefinedFeature, match="F2"):
        build_features("F6", reader, index, 0, "train")


def test_f3_excludes_document_initial_tokens_and_records_the_count(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    f3 = build_features("F3", reader, index, 1, "train")
    rows = index.rows("train")
    expected = int((index.pos_in_doc[rows] == 0).sum())
    assert f3.n_excluded == expected
    assert expected > 0
    assert "document-initial" in f3.exclusion_reason
    assert (index.pos_in_doc[f3.row_index] != 0).all()


def test_f3_conditioning_token_is_always_in_the_same_document(trace):
    """The no-leak guarantee: document-level splits + no doc-initial rows (plan T4.3)."""
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    for split in ("train", "val", "test"):
        f3 = build_features("F3", reader, index, 2, split)
        assert np.array_equal(
            index.doc_ids[f3.row_index], index.doc_ids[f3.row_index - 1]
        )
        in_split = set(index.rows(split).tolist())
        assert set((f3.row_index - 1).tolist()) <= in_split


def test_f2_labels_come_from_topk_and_match_the_ground_truth(trace):
    reader, truth = trace
    index = CorpusIndex.from_reader(reader)
    f2 = build_features("F2", reader, index, 2, "test")
    assert np.array_equal(f2.y, truth.topk[f2.row_index, 2, :].astype(np.int32))


def test_f4_and_fv_restrict_to_the_hidden_subsample(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    fv = build_features("FV", reader, index, 1, "train")
    assert set(fv.row_index.tolist()) <= set(index.captured.tolist())
    assert "hidden subsample" in fv.exclusion_reason
    assert fv.meta["hidden_layer"] == 1
    f4 = build_features("F4", reader, index, 1, "train")
    assert f4.meta["hidden_layer"] == 0


def test_router_input_standardizer_cannot_be_fit_on_non_train_rows(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    with pytest.raises(ValueError, match="train-fitted standardizer"):
        build_features("FV", reader, index, 1, "test")


def test_f2_recovers_the_copycat_structure(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    parts = {
        s: build_features("F2", reader, index, 3, s) for s in ("train", "val", "test")
    }
    probe = SoftmaxProbe(n_experts=N_EXPERTS, feature="F2", lr=0.2, max_epochs=60)
    probe.fit(parts["train"].X, parts["train"].y, parts["val"].X, parts["val"].y)
    out = evaluate_on_test(
        probe, parts["test"].X, parts["test"].y, n_experts=N_EXPERTS, top_k=TOP_K
    )
    # Layer 3 is a verbatim copy of layer 2, so F2 must find essentially all of it.
    assert out["family_a"]["set_agreement@k"] > 0.95
    assert out["family_b"]["mi_bits"] > 0.5


def test_repetition_rate_generalizes_to_the_mixtral_statistic(trace):
    reader, _ = trace
    index = CorpusIndex.from_reader(reader)
    stat = consecutive_repetition_rate(reader, index, 2, split="test")
    assert stat["random_baseline"] == pytest.approx(TOP_K / N_EXPERTS)
    assert stat["n_excluded_doc_initial"] > 0
    assert 0.0 <= stat["repetition_rate"] <= 1.0


def test_repetition_rate_is_one_when_routing_never_changes(tmp_path):
    def _frozen(rng, tokens, spec):
        out = np.zeros((tokens.shape[0], spec.n_moe_layers, spec.top_k), dtype=np.int64)
        out[..., 1] = 1
        return out

    truth = make_synthetic_trace(tmp_path, spec=SPEC, shard_sizes=(100,), topk_fn=_frozen)
    with TraceReader(tmp_path, truth.model, truth.corpus, doc_splits=truth.doc_splits) as reader:
        index = CorpusIndex.from_reader(reader)
        stat = consecutive_repetition_rate(reader, index, 1)
    assert stat["repetition_rate"] == pytest.approx(1.0)
    assert stat["exact_set_repeat_rate"] == pytest.approx(1.0)


# -- the sweep (T7.8) ------------------------------------------------------------------------------------


def test_sweep_writes_one_file_per_model_feature_not_per_layer(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    result = sweep(
        reader, model="synth", out_dir=out, features=("F0", "F1", "F3"), layers=(1, 2)
    )
    files = sorted(p.name for p in out.glob("*.json"))
    assert files == ["synth__F0.json", "synth__F1.json", "synth__F3.json"]
    doc = json.loads(output_path(out, "synth", "F1").read_text(encoding="utf-8"))
    assert sorted(doc["layers"]) == ["1", "2"]
    assert result.layers_done == [1, 2]


def test_sweep_is_idempotent_and_resumable(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    sweep(reader, model="synth", out_dir=out, features=("F0",), layers=(0, 1))
    first = json.loads(output_path(out, "synth", "F0").read_text(encoding="utf-8"))

    again = sweep(reader, model="synth", out_dir=out, features=("F0",), layers=(0, 1))
    assert again.layers_done == []
    assert again.layers_skipped == [0, 1]
    assert json.loads(output_path(out, "synth", "F0").read_text(encoding="utf-8"))["layers"] == first["layers"]

    extended = sweep(reader, model="synth", out_dir=out, features=("F0",), layers=(0, 1, 2))
    assert extended.layers_done == [2]
    assert sorted(
        json.loads(output_path(out, "synth", "F0").read_text(encoding="utf-8"))["layers"]
    ) == ["0", "1", "2"]


def test_sweep_force_refits_existing_cells(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    sweep(reader, model="synth", out_dir=out, features=("F0",), layers=(1,))
    forced = sweep(reader, model="synth", out_dir=out, features=("F0",), layers=(1,), force=True)
    assert forced.layers_done == [1]


def test_sweep_stops_on_the_session_budget_with_results_on_disk(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    spent = SessionBudget(wall_limit_s=1e-3, reserve_s=9e-4)
    result = sweep(
        reader, model="synth", out_dir=out, features=("F0",), layers=(0, 1, 2, 3), budget=spent
    )
    assert result.stopped_early
    assert "budget" in result.stop_reason
    # Nothing was attempted, but the file structure is still valid to resume from.
    assert result.layers_done == []


def test_sweep_records_skipped_cells_with_a_reason(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    result = sweep(reader, model="synth", out_dir=out, features=("F2",), layers=(0, 1))
    doc = json.loads(output_path(out, "synth", "F2").read_text(encoding="utf-8"))
    assert doc["layers"]["0"]["status"] == "skipped"
    assert doc["layers"]["1"]["status"] == "ok"
    assert result.skipped_cells and result.skipped_cells[0]["layer"] == 0


def test_sweep_pairs_f3_with_the_raw_repetition_rate(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    sweep(reader, model="synth", out_dir=out, features=("F3",), layers=(2,))
    doc = json.loads(output_path(out, "synth", "F3").read_text(encoding="utf-8"))
    stat = doc["layers"]["2"]["mixtral_table5_statistic"]
    assert stat["split"] == "test"
    assert "repetition_rate" in stat and "random_baseline" in stat


def test_sweep_records_the_fv_gate_as_appendix_only(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    sweep(reader, model="synth", out_dir=out, features=("FV",), layers=(1,))
    gate = json.loads(output_path(out, "synth", "FV").read_text(encoding="utf-8"))["layers"]["1"][
        "validation_gate"
    ]
    assert gate["appendix_only"] is True
    assert gate["threshold"] == 0.99
    assert isinstance(gate["pass"], bool)


def test_sweep_emits_f1_frequency_strata(trace, tmp_path):
    reader, _ = trace
    out = tmp_path / "results"
    sweep(reader, model="synth", out_dir=out, features=("F1",), layers=(1,), n_frequency_buckets=3)
    metrics = json.loads(output_path(out, "synth", "F1").read_text(encoding="utf-8"))["layers"]["1"][
        "metrics"
    ]
    assert "strata" in metrics
    assert metrics["stratification"]["n_buckets"] == 3


def test_sweep_carries_the_run_config_identity_into_every_result(trace, tmp_path):
    """Results that cannot be traced to a run config cannot be merged or defended (I2)."""
    reader, _ = trace
    out = tmp_path / "results"
    sweep(reader, model="synth", out_dir=out, features=("F0",), layers=(1,))
    doc = json.loads(output_path(out, "synth", "F0").read_text(encoding="utf-8"))
    assert doc["run_config_sha256"] == reader.run_config_sha256
    assert doc["logit_tensor_used"] == reader.logit_tensor_used


def test_sweep_offers_f5_on_request_but_never_by_default(trace, tmp_path):
    """F5 is the one GPU-budgeted probe (plan §Phase S): available explicitly, never a default.

    Full F5 behaviour is covered in ``tests/test_probes_mlp.py``; this asserts only the sweep's
    side of the contract — that asking for it works, and that not asking for it never gets it.
    """
    from src.probes.features import CPU_FEATURES

    assert "F5" not in CPU_FEATURES

    reader, _ = trace
    out = tmp_path / "results"
    result = sweep(
        reader,
        model="synth",
        out_dir=out,
        features=("F5",),
        layers=(1,),
        probe_kwargs={"max_epochs": 3, "hidden_width": 8},
    )
    assert result.features == ["F5"]
    doc = json.loads(output_path(out, "synth", "F5").read_text(encoding="utf-8"))
    cell = doc["layers"]["1"]
    assert cell["status"] == "ok"
    assert cell["fit"]["predictor"] == "mlp_probe"
    assert cell["fit"]["hyperparams"]["hidden_width"] == 8
    assert cell["meta"]["hidden_layer"] == 0

    # An unknown feature is still rejected — the guard was not loosened, only widened by one.
    with pytest.raises(ValueError, match="unknown feature"):
        sweep(reader, model="synth", out_dir=tmp_path, features=("F9",), layers=(1,))
