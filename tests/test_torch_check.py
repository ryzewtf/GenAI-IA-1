"""T3.2 tests — the PyTorch cross-validation comparison core.

torch is deliberately not a dependency of this file. The plan runs T3.2 on Kaggle (T0.1 measured
torch 2.10.0+cu128 and transformers 5.0.0 there); the workstation has neither. So everything that
can be settled without a 7 B model is settled here, on real arithmetic with real defects injected:
the rank statistics, the set/exact metrics, the document reconstruction, the layer mapping, and
the router-module discovery. What remains for Kaggle is the model load itself.

The rank statistics are checked against scipy rather than against hand-computed constants. A
hand-computed expectation for tie-averaged Spearman over 64 experts is exactly the kind of thing
that gets written to match the implementation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.traces.format import TOKEN_DTYPE, write_manifest
from src.traces.torch_check import (
    LayerComparison,
    TorchCheckError,
    average_ranks,
    compare_layer,
    compare_shard,
    documents_from_tokens,
    find_router_modules,
    main,
    spearman_rows,
)

scipy_stats = pytest.importorskip("scipy.stats")

N_LAYERS, N_EXPERTS, TOP_K = 4, 16, 4


# -- fixtures ----------------------------------------------------------------------------------


def build_shard(tmp_path, hf_logits, *, doc_lengths=(5, 7), layer_map=None, softmax=True):
    """Write a shard whose topk/logits are DERIVED from ``hf_logits``.

    Building the trace forwards from the same logits the comparison will be handed means a
    passing result is evidence about the comparison, not about the fixture: any defect the test
    injects afterwards is the only difference between the two sides.
    """
    n_tokens = sum(doc_lengths)
    n_layers = hf_logits.shape[1]
    shard = tmp_path / "shard_00000"
    shard.mkdir(parents=True, exist_ok=True)

    layer_map = list(range(n_layers)) if layer_map is None else list(layer_map)
    n_trace_layers = len(layer_map)

    tokens = np.zeros(n_tokens, dtype=TOKEN_DTYPE)
    row = 0
    for doc_id, length in enumerate(doc_lengths):
        for pos in range(length):
            tokens[row] = (1000 + row, doc_id, pos, 0)
            row += 1
    tokens.tofile(shard / "tokens.bin")

    selection = np.empty((n_tokens, n_trace_layers, hf_logits.shape[2]), dtype="<f2")
    topk = np.empty((n_tokens, n_trace_layers, TOP_K), dtype="<i4")
    for trace_layer, model_layer in enumerate(layer_map):
        block = hf_logits[:, model_layer, :].astype(np.float32)
        if softmax:
            shifted = block - block.max(axis=1, keepdims=True)
            probs = np.exp(shifted)
            block = probs / probs.sum(axis=1, keepdims=True)
        selection[:, trace_layer, :] = block.astype("<f2")
        topk[:, trace_layer, :] = np.argsort(
            -hf_logits[:, model_layer, :], axis=1, kind="stable"
        )[:, :TOP_K]
    selection.tofile(shard / "logits.bin")
    topk.tofile(shard / "topk.bin")

    write_manifest(shard, {
        "model": "fixture", "checkpoint_status": "base", "gguf_sha256": "0" * 64,
        "llama_cpp_commit": "0" * 40, "run_config_sha256": "0" * 64, "quant": "F16",
        "router_dtype": "F32", "logit_tensor_used": "ffn_moe_probs", "corpus": "c",
        "shard_id": 0, "shard_doc_range": [0, len(doc_lengths)], "n_tokens": n_tokens,
        "n_moe_layers": n_trace_layers, "n_experts": int(hf_logits.shape[2]), "top_k": TOP_K,
        "hidden_dim": 8, "hidden_subsample_n": 1, "n_captured": n_tokens, "hidden_stride": 1,
        "index_scheme": "doc_id*n_ctx+pos_in_doc", "index_doc_span": 64,
        "capture_flags": {"pre_norm": True}, "layer_index_map": layer_map,
        "device_plan": {"n_gpu": 0}, "file_sha256": {}, "collected_utc": "2026-01-01T00:00:00Z",
        "n_docs": len(doc_lengths),
    })
    return shard


@pytest.fixture
def logits():
    rng = np.random.default_rng(20260818)
    return rng.normal(size=(12, N_LAYERS, N_EXPERTS)).astype(np.float32)


# -- rank statistics ---------------------------------------------------------------------------


def test_average_ranks_matches_scipy_including_on_ties():
    rng = np.random.default_rng(1)
    values = rng.integers(0, 4, size=(20, 9)).astype(np.float64)  # forced heavy ties
    expected = np.stack([scipy_stats.rankdata(row, method="average") for row in values])
    np.testing.assert_allclose(average_ranks(values), expected)


def test_spearman_matches_scipy_on_tied_data():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 5, size=(8, 12)).astype(np.float64)
    b = rng.integers(0, 5, size=(8, 12)).astype(np.float64)
    got = spearman_rows(a, b)
    for i in range(a.shape[0]):
        expected = scipy_stats.spearmanr(a[i], b[i]).statistic
        assert got[i] == pytest.approx(expected, abs=1e-12) or (
            np.isnan(got[i]) and np.isnan(expected)
        )


def test_a_fully_tied_row_is_nan_not_zero():
    """A constant row has undefined rank correlation. Scoring it 0.0 would let a degenerate row
    masquerade as evidence of disagreement and drag a layer under the 0.999 gate."""
    rho = spearman_rows(np.ones((1, 6)), np.arange(6.0)[None, :])
    assert np.isnan(rho[0])


def test_softmax_does_not_change_the_spearman_score(logits):
    """I13: logits.bin holds ffn_moe_probs, not raw logits. The plan's gate is stated on raw
    logits, and it survives the substitution only because softmax is strictly monotone. If that
    ever stopped being true the whole metric would be measuring the storage format."""
    raw = logits[:, 0, :].astype(np.float64)
    shifted = raw - raw.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(spearman_rows(raw, probs), 1.0, atol=1e-12)


# -- compare_layer -----------------------------------------------------------------------------


def _layer_inputs(logits, layer=0):
    block = logits[:, layer, :]
    topk = np.argsort(-block, axis=1, kind="stable")[:, :TOP_K].astype(np.int32)
    shifted = block - block.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return block, topk, probs.astype(np.float16)


def test_an_aligned_layer_scores_perfectly(logits):
    block, topk, probs = _layer_inputs(logits)
    result = compare_layer(block, topk, probs, top_k=TOP_K)
    assert result.exact_match == 1.0
    assert result.set_agreement == 1.0
    assert result.spearman == pytest.approx(1.0, abs=1e-9)
    assert result.failures() == []


def test_a_one_row_shift_is_caught(logits):
    """The canonical off-by-one: the trace is fine, the HF replay is one token out of phase.
    Everything has the right shape and the right value range."""
    block, topk, probs = _layer_inputs(logits)
    result = compare_layer(block, np.roll(topk, 1, axis=0), np.roll(probs, 1, axis=0),
                           top_k=TOP_K)
    assert result.set_agreement < 0.9
    assert any("set_agreement" in f for f in result.failures())


def test_a_permuted_set_costs_exact_match_but_not_set_agreement(logits):
    """The two metrics exist to separate these. A reordering within the selected set is a
    floating-point near-tie; a different set is a different computation."""
    block, topk, probs = _layer_inputs(logits)
    swapped = topk.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    result = compare_layer(block, swapped, probs, top_k=TOP_K)
    assert result.set_agreement == 1.0
    assert result.exact_match == 0.0
    assert any("exact_match" in f for f in result.failures())
    assert not any("set_agreement" in f for f in result.failures())


def test_one_wrong_expert_in_every_row_lands_between_the_two_gates(logits):
    block, topk, probs = _layer_inputs(logits)
    broken = topk.copy()
    broken[:, -1] = (broken[:, -1] + 1) % N_EXPERTS
    result = compare_layer(block, broken, probs, top_k=TOP_K)
    assert result.set_agreement == pytest.approx((TOP_K - 1) / TOP_K, abs=0.05)
    assert result.exact_match == 0.0


def test_shape_disagreement_is_an_error_not_a_low_score(logits):
    block, topk, probs = _layer_inputs(logits)
    with pytest.raises(TorchCheckError, match="topk shape"):
        compare_layer(block, topk[:, :-1], probs, top_k=TOP_K)
    with pytest.raises(TorchCheckError, match="selection shape"):
        compare_layer(block, topk, probs[:, :-1], top_k=TOP_K)


def test_ties_are_reported_so_a_shortfall_can_be_attributed():
    """The measured OLMoE shard has ties in 42.5% of rows. A Spearman below the gate is either
    that or a real disagreement, and the report has to say which without a rerun."""
    block = np.tile(np.arange(N_EXPERTS, dtype=np.float32)[::-1], (5, 1))
    topk = np.argsort(-block, axis=1, kind="stable")[:, :TOP_K].astype(np.int32)
    flat = np.zeros_like(block, dtype=np.float16)
    flat[:, :TOP_K] = block[:, :TOP_K]
    result = compare_layer(block, topk, flat, top_k=TOP_K)
    assert result.frac_rows_with_ties == 1.0


# -- documents_from_tokens ---------------------------------------------------------------------


def test_documents_are_recovered_exactly_from_the_token_stream():
    tokens = np.zeros(9, dtype=TOKEN_DTYPE)
    for row, (doc, pos) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
                                      (2, 0), (2, 1), (2, 2), (2, 3)]):
        tokens[row] = (500 + row, doc, pos, 0)
    docs = documents_from_tokens(tokens)
    assert [len(d) for d in docs] == [3, 2, 4]
    assert docs[1].tolist() == [503, 504]


def test_a_position_counter_that_disagrees_with_the_boundary_is_refused():
    """If pos_in_doc does not restart at a doc_id change, the writer and the runner disagree
    about what a document is — and replaying that segmentation compares two different runs."""
    tokens = np.zeros(4, dtype=TOKEN_DTYPE)
    for row, (doc, pos) in enumerate([(0, 0), (0, 1), (1, 5), (1, 6)]):
        tokens[row] = (row, doc, pos, 0)
    with pytest.raises(TorchCheckError, match="pos_in_doc"):
        documents_from_tokens(tokens)


def test_an_interleaved_document_is_refused():
    tokens = np.zeros(4, dtype=TOKEN_DTYPE)
    for row, (doc, pos) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        tokens[row] = (row, doc, pos, 0)
    with pytest.raises(TorchCheckError):
        documents_from_tokens(tokens)


# -- compare_shard -----------------------------------------------------------------------------


def test_an_aligned_shard_passes_every_tier(tmp_path, logits):
    report = compare_shard(build_shard(tmp_path, logits), logits)
    assert report.ok, report.failures
    assert len(report.layers) == N_LAYERS
    assert report.worst.set_agreement == 1.0
    assert any("softmax" in n for n in report.notes)


def test_the_layer_index_map_is_applied_not_assumed(tmp_path, logits):
    """DeepSeek's layer 0 is dense, so trace layer 0 is model layer 1 (T3.5). A comparison that
    ignored the map would still line up on a model with an identity map and be silently wrong on
    the one model in the panel where it matters."""
    shard = build_shard(tmp_path, logits, layer_map=[1, 2, 3])
    assert compare_shard(shard, logits).ok


def test_an_off_by_one_layer_map_is_caught(tmp_path, logits):
    shard = build_shard(tmp_path, logits, layer_map=[0, 1, 2])
    manifest = json.loads((shard / "manifest.json").read_text())
    manifest["layer_index_map"] = [1, 2, 3]
    (shard / "manifest.json").write_text(json.dumps(manifest))
    report = compare_shard(shard, logits)
    assert not report.ok
    assert len(report.failures) >= 3, "every shifted layer should be implicated, not just one"


def test_a_map_pointing_past_the_captured_layers_is_an_error(tmp_path, logits):
    shard = build_shard(tmp_path, logits, layer_map=[0, 1, 2])
    manifest = json.loads((shard / "manifest.json").read_text())
    manifest["layer_index_map"] = [0, 1, 99]
    (shard / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TorchCheckError, match="captured only"):
        compare_shard(shard, logits)


def test_a_token_count_mismatch_is_refused_rather_than_truncated(tmp_path, logits):
    """Reconciling by truncation is the tempting fix and it is always wrong: the two sides
    disagree about which tokens exist, so any alignment after truncation is a coincidence."""
    shard = build_shard(tmp_path, logits)
    with pytest.raises(TorchCheckError, match="different segmentation"):
        compare_shard(shard, logits[:-1])


def test_a_router_width_mismatch_names_the_checkpoint(tmp_path, logits):
    shard = build_shard(tmp_path, logits)
    with pytest.raises(TorchCheckError, match="n_experts"):
        compare_shard(shard, logits[:, :, :-1])


def test_a_single_bad_layer_is_reported_by_name(tmp_path, logits):
    """The plan's interpretation rule needs to know WHICH layers failed: a shortfall in one or two
    layers is a name-filter or layer-index bug, spread evenly it is floating point."""
    corrupted = logits.copy()
    shard = build_shard(tmp_path, corrupted)
    corrupted[:, 2, :] = corrupted[:, 2, ::-1]
    report = compare_shard(shard, corrupted)
    assert not report.ok
    assert all("layer 2" in f for f in report.failures)


def test_the_report_round_trips_as_json(tmp_path, logits):
    report = compare_shard(build_shard(tmp_path, logits), logits)
    payload = json.loads(json.dumps(report.to_json()))
    assert payload["task"] == "T3.2"
    assert payload["ok"] is True
    assert len(payload["layers"]) == N_LAYERS
    assert set(payload["thresholds"]) == {"set_agreement", "exact_match", "spearman"}


# -- router module discovery -------------------------------------------------------------------


class FakeLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.zeros((out_features, in_features))


class FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return list(self._modules)


def _fake_moe_model(n_layers, n_experts, hidden=8):
    modules = []
    for i in range(n_layers):
        modules.append((f"model.layers.{i}.mlp.gate", FakeLinear(hidden, n_experts)))
        modules.append((f"model.layers.{i}.self_attn.q_proj", FakeLinear(hidden, hidden)))
    modules.append(("lm_head", FakeLinear(hidden, 32000)))
    return FakeModel(modules)


def test_routers_are_found_by_shape_not_by_name():
    model = _fake_moe_model(4, N_EXPERTS)
    found = find_router_modules(model, N_EXPERTS, 4)
    assert [name for name, _ in found] == [f"model.layers.{i}.mlp.gate" for i in range(4)]


def test_a_missing_router_is_refused_rather_than_partially_hooked():
    """Hooking 15 routers on a 16-layer model produces a full-looking report over the wrong
    layers. This is the exact failure T3.2 exists to catch, so it must not be survivable."""
    model = _fake_moe_model(4, N_EXPERTS)
    model._modules = [m for m in model._modules if not m[0].startswith("model.layers.2.mlp")]
    with pytest.raises(TorchCheckError, match="found 3 candidate router modules"):
        find_router_modules(model, N_EXPERTS, 4)


def test_discovery_is_ordered_by_layer_index_not_by_registration_order():
    """named_modules() walks in registration order, which is layer order today. Sorting makes
    that an assertion instead of an assumption, because a wrong order maps every layer to the
    wrong one and still produces a complete, plausible report."""
    model = _fake_moe_model(4, N_EXPERTS)
    model._modules = list(reversed(model._modules))
    found = find_router_modules(model, N_EXPERTS, 4)
    assert [name for name, _ in found] == [f"model.layers.{i}.mlp.gate" for i in range(4)]


def test_a_head_outside_the_decoder_stack_is_not_mistaken_for_a_router():
    model = _fake_moe_model(4, N_EXPERTS)
    model._modules.append(("some_aux_head", FakeLinear(8, N_EXPERTS)))
    assert len(find_router_modules(model, N_EXPERTS, 4)) == 4


# -- CLI ---------------------------------------------------------------------------------------


def test_cli_passes_and_writes_its_report(tmp_path, logits, capsys):
    shard = build_shard(tmp_path, logits)
    saved = tmp_path / "hf.npy"
    np.save(saved, logits)
    report = tmp_path / "t3.2.json"
    rc = main([str(shard), "--model-id", "fixture", "--hf-logits", str(saved),
               "--report", str(report)])
    assert rc == 0
    assert "T3.2 PASSED" in capsys.readouterr().out
    assert json.loads(report.read_text())["ok"] is True


def test_cli_separates_could_not_run_from_disagreement(tmp_path, logits, capsys):
    """Exit 2 (could not run) and exit 1 (ran and disagreed) are different outcomes, and for
    GPT-OSS the plan gives the first one a documented fallback. Collapsing them would make the
    limitations section unwritable."""
    shard = build_shard(tmp_path, logits)
    saved = tmp_path / "hf.npy"

    np.save(saved, logits[:-1])
    attempt = tmp_path / "attempt.json"
    assert main([str(shard), "--model-id", "fixture", "--hf-logits", str(saved),
                 "--attempt-log", str(attempt)]) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert json.loads(attempt.read_text())["ran"] is False

    np.save(saved, np.roll(logits, 1, axis=0))
    assert main([str(shard), "--model-id", "fixture", "--hf-logits", str(saved)]) == 1
    assert "T3.2 FAILED" in capsys.readouterr().out


def test_thresholds_are_overridable_from_the_cli(tmp_path, logits):
    shard = build_shard(tmp_path, logits)
    saved = tmp_path / "hf.npy"
    np.save(saved, logits)
    assert main([str(shard), "--model-id", "fixture", "--hf-logits", str(saved),
                 "--min-exact-match", "1.1"]) == 1


def test_failures_carry_the_measured_value_not_just_a_verdict():
    comparison = LayerComparison(
        layer=3, model_layer=4, n_tokens=100, exact_match=0.5, set_agreement=0.8,
        spearman=0.5, spearman_min=0.1, frac_rows_with_ties=0.0,
    )
    text = " ".join(comparison.failures())
    assert "layer 3" in text and "model layer 4" in text
    assert "0.800000" in text and "0.500000" in text
