"""Post-collection shard validation tests — plan T5.3, plus the Python half of T3.1.

``src/traces/validate.py`` is the last thing that looks at a shard before hours of Kaggle GPU
time turn into a published dataset, so the only failure that matters here is a check that does
not fire. Every test below therefore comes in a pair: a **corrupt** shard, constructed to carry
exactly the defect one check exists to catch, asserted to be caught; and a **healthy** shard,
asserted not to be flagged by that same check. A validator with no false-negative test is
decoration; one with no false-positive test fails good data and gets switched off.

Fixtures come from :mod:`src.traces.synth` rather than hand-rolled bytes, so the "healthy" case
is the same generator the reader tests use and a corruption is always a *diff* against it.

The invariants that get the most attention are the ones that fail silently:

* **I12** — ``ffn_moe_topk`` is a strided view. The corrupt fixture here is built by literally
  performing the contiguous read (``argsort(-logits).ravel()[:n*k]``), which is what a harness
  that ignores ``nb[1]`` produces, and both the exact reconstruction and the adjacency proxy
  are asserted to fire on it.
* **I15** — truncation lives only in ``capture_stats.json``; a truncated shard is otherwise
  perfectly well formed.
* **I2** — shards whose ``run_config_sha256`` differs are a different experiment and merging
  them is a hard error, not a warning.
* **I8** — nothing in the load-balance path may clamp; a concentrated router must report the
  measured index, not a floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.traces.format import (
    FLAG_HIDDEN_CAPTURED,
    HIDDEN_INDEX_DTYPE,
    LOGIT_DTYPE,
    MANIFEST_NAME,
    TOKEN_DTYPE,
    TOPK_DTYPE,
    TraceSpec,
    read_manifest,
)
from src.traces.reader import ShardHandle
from src.traces.synth import make_synthetic_trace
from src.traces.validate import (
    CHECK_NAMES,
    DEFAULT_SAMPLE_TOKENS,
    STATS_NAME,
    Finding,
    ValidationReport,
    check_hidden_stride,
    check_sizes,
    check_topk_strided_view,
    discover_shards,
    main,
    read_stats,
    validate_shard,
    validate_shards,
)

SPEC = TraceSpec(n_moe_layers=4, n_experts=16, top_k=3, hidden_dim=8)

# 40/25/35 is deliberately unequal AND deliberately mid-document: at 10 tokens per document,
# shard 1 ends inside document 6 and shard 2 picks it up at position 5. A shard is not a whole
# number of documents and does not begin on a subsample boundary, which is exactly the point.
SHARD_SIZES = (40, 25, 35)

#: Synthetic capture parameters, mirrored from `make_synthetic_trace`'s defaults so the tests can
#: state expected row counts instead of discovering them.
TOKENS_PER_DOC = 10
HIDDEN_EVERY = 4
DOC_INDEX_SPAN = 64

#: A document's block base is a multiple of DOC_INDEX_SPAN, itself a multiple of HIDDEN_EVERY, so
#: every document contributes ceil(TOKENS_PER_DOC / HIDDEN_EVERY) = 3 rows when whole.
#: Shard 0 is 4 whole documents.
SHARD0_CAPTURED = 12


# -- fixtures and helpers ---------------------------------------------------------------------


@pytest.fixture
def trace(tmp_path):
    return make_synthetic_trace(tmp_path, spec=SPEC, shard_sizes=SHARD_SIZES, seed=7)


def shard(trace, index: int = 0) -> Path:
    return trace.root / trace.model / trace.corpus / f"shard_{index:05d}"


def trace_dir(trace) -> Path:
    return trace.root / trace.model / trace.corpus


def one_shard(tmp_path, *, spec: TraceSpec = SPEC, n_tokens: int = 300, **kwargs) -> Path:
    """A single-shard trace, for the checks that need a lot of rows in one place."""
    tr = make_synthetic_trace(tmp_path, spec=spec, shard_sizes=(n_tokens,), **kwargs)
    return tr.root / tr.model / tr.corpus / "shard_00000"


def load(path: Path, dtype, shape=None) -> np.ndarray:
    array = np.fromfile(path, dtype=dtype)
    return array if shape is None else array.reshape(shape)


def store(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array).tobytes())


def truncate(path: Path, n_bytes: int) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - n_bytes])


def patch_manifest(shard_dir: Path, **updates) -> None:
    path = shard_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run(shard_dir: Path, check: str, **kwargs) -> ValidationReport:
    return validate_shard(shard_dir, checks=[check], **kwargs)


def messages(report: ValidationReport, severity: str) -> str:
    return " || ".join(f.message for f in report.findings if f.severity == severity)


def errors(report: ValidationReport) -> str:
    return messages(report, "error")


def warnings(report: ValidationReport) -> str:
    return messages(report, "warning")


def collect(iterator) -> list[Finding]:
    return list(iterator)


def _handle(shard_dir: Path) -> ShardHandle:
    """The handle ``validate_shard`` would build, for the checks exercised directly."""
    manifest = read_manifest(shard_dir)
    return ShardHandle(shard_dir, manifest, TraceSpec.from_manifest(manifest))


# ==============================================================================================
# Finding / ValidationReport plumbing
# ==============================================================================================


def test_a_finding_rejects_a_severity_the_gate_cannot_rank():
    with pytest.raises(ValueError, match="unknown severity"):
        Finding("size_arithmetic", "critical", "boom")


def test_only_error_severity_decides_the_verdict_warnings_do_not():
    report = ValidationReport("d", 0, "m")
    report.add([Finding("expert_usage", "warning", "dead expert")])
    report.add([Finding("truncation", "info", "fine")])
    assert report.ok
    assert report.n_warnings == 1

    report.add([Finding("truncation", "error", "truncated")])
    assert not report.ok
    assert report.n_errors == 1
    assert [f.message for f in report.errors()] == ["truncated"]


def test_a_report_serialises_numpy_scalars_that_json_would_refuse():
    report = ValidationReport("d", 0, "m")
    report.add(
        [
            Finding(
                "expert_usage",
                "info",
                "hist",
                {"count": np.int64(7), "rate": np.float32(0.5), "row": np.arange(3)},
            )
        ]
    )
    payload = json.loads(report.to_json())
    assert payload["findings"][0]["detail"] == {"count": 7, "rate": 0.5, "row": [0, 1, 2]}
    assert payload["ok"] is True


def test_a_non_finite_float_in_a_detail_survives_json_as_a_string():
    report = ValidationReport("d", 0, "m")
    report.add([Finding("logits_sanity", "info", "m", {"min_margin": float("inf")})])
    assert json.loads(report.to_json())["findings"][0]["detail"]["min_margin"] == "inf"


# ==============================================================================================
# check_sizes — T5.3 size arithmetic
# ==============================================================================================


def test_a_healthy_shard_has_no_size_arithmetic_findings(trace):
    for i in range(len(SHARD_SIZES)):
        assert collect(check_sizes(_handle(shard(trace, i)))) == []


def test_a_missing_stream_is_an_error_not_a_crash(trace):
    (shard(trace, 0) / "hidden.bin").unlink()
    report = run(shard(trace, 0), "size_arithmetic")
    assert "hidden.bin: missing" in errors(report)


def test_a_shard_short_by_whole_token_rows_reads_as_a_killed_session(trace):
    """The last document never got flushed — re-run the shard."""
    truncate(shard(trace, 0) / "tokens.bin", 2 * SPEC.token_stride)
    report = run(shard(trace, 0), "size_arithmetic")
    assert "2 whole token row(s) short" in errors(report)


def test_a_shard_short_by_a_layer_row_points_at_the_node_spec_not_the_session(trace):
    """One layer's tensor was never matched, so the writer emitted short rows.

    The two remedies are opposite — re-run the shard vs. fix the node spec — so the finding has
    to say which. ``topk_stride`` is 48 B and a layer row is 12 B, so 12 is not a token row.
    """
    truncate(shard(trace, 0) / "topk.bin", SPEC.topk_stride // SPEC.n_moe_layers)
    report = run(shard(trace, 0), "size_arithmetic")
    assert "layer-row(s) short" in errors(report)
    assert "check the node spec" in errors(report)


def test_a_stream_that_is_not_a_whole_number_of_rows_is_called_structurally_broken(trace):
    truncate(shard(trace, 0) / "topk.bin", 5)
    report = run(shard(trace, 0), "size_arithmetic")
    assert "structurally broken" in errors(report)


def test_hidden_index_on_disk_outranks_the_manifests_captured_count(trace):
    """The manifest number is written before the last flush; the file is the authority."""
    patch_manifest(shard(trace, 0), n_captured=999)
    report = run(shard(trace, 0), "size_arithmetic")
    assert "n_captured=999" in errors(report)
    assert f"holds {SHARD0_CAPTURED} rows" in errors(report)


def test_the_collection_wide_subsample_budget_is_not_a_per_shard_row_count(trace):
    """Regression: `hidden_subsample_n` is a budget for the whole collection, not this shard.

    T4.4 converts it to an integer stride and the harness applies that stride per shard, so any
    one shard's row count equals the budget only by coincidence. Comparing the two failed a
    perfectly healthy shard — the first real capture off OLMoE tripped it.
    """
    patch_manifest(shard(trace, 0), hidden_subsample_n=50_000)
    report = run(shard(trace, 0), "size_arithmetic")
    assert not errors(report)


def test_a_subsample_larger_than_the_token_count_is_impossible(trace):
    """The subsample is drawn from the tokens; it cannot outnumber them."""
    patch_manifest(shard(trace, 0), n_tokens=5)
    report = run(shard(trace, 0), "size_arithmetic")
    assert "cannot be larger than what it subsamples" in errors(report)


# ==============================================================================================
# check_lockstep — T3.1
# ==============================================================================================


def test_a_healthy_shard_is_in_three_stream_lockstep(trace):
    for i in range(len(SHARD_SIZES)):
        report = run(shard(trace, i), "lockstep")
        assert report.ok, errors(report)


def test_a_stream_missing_token_rows_breaks_lockstep(trace):
    """A filter that matched the wrong node for one stream lands here, not in a crash."""
    truncate(shard(trace, 1) / "topk.bin", SPEC.topk_stride)
    report = run(shard(trace, 1), "lockstep")
    assert "three-stream lockstep failure" in errors(report)
    assert "'topk': 24" in errors(report)


def test_hidden_and_hidden_index_must_agree_on_the_captured_count(trace):
    truncate(shard(trace, 0) / "hidden.bin", SPEC.hidden_stride)
    report = run(shard(trace, 0), "lockstep")
    assert "subsample streams are not in lockstep" in errors(report)


def test_a_non_increasing_hidden_index_is_caught(trace):
    """Global indices must ascend or ``hidden()`` resolves the wrong rows."""
    path = shard(trace, 0) / "hidden_index.bin"
    index = load(path, HIDDEN_INDEX_DTYPE)
    index[2], index[3] = index[3], index[2]
    store(path, index)
    report = run(shard(trace, 0), "lockstep")
    assert "not strictly increasing" in errors(report)


def test_the_capture_flags_and_the_index_stream_must_name_the_same_tokens(trace):
    """Two independently written streams; agreement between them is the only real evidence.

    A shifted index with the flags untouched means every F4/F5 lookup silently takes a
    neighbour's router input — no crash, no NaN, just a relabelled feature. `lockstep` owns the
    count half of that agreement; `hidden_stride` owns the values, because comparing them needs
    to know how an index is formed.
    """
    path = shard(trace, 0) / "hidden_index.bin"
    index = load(path, HIDDEN_INDEX_DTYPE)
    index[3] += 1  # still strictly increasing, still in range
    store(path, index)
    report = run(shard(trace, 0), "hidden_stride", stats=healthy_stats())
    assert "are not doc_id *" in errors(report)
    assert run(shard(trace, 0), "lockstep").ok, "the counts still agree; only a value moved"


def test_a_dropped_capture_flag_is_caught_by_the_row_count(trace):
    path = shard(trace, 0) / "tokens.bin"
    tokens = load(path, TOKEN_DTYPE)
    tokens["flags"][8] &= ~np.uint32(FLAG_HIDDEN_CAPTURED)
    store(path, tokens)
    report = run(shard(trace, 0), "lockstep")
    assert "carry FLAG_HIDDEN_CAPTURED but hidden_index.bin has" in errors(report)


def test_lockstep_reports_a_skip_rather_than_raising_on_an_unmappable_stream(trace):
    """A truncated upload must come back as a finding, not a memmap traceback."""
    truncate(shard(trace, 0) / "tokens.bin", 5)
    report = run(shard(trace, 0), "lockstep")
    assert "skipped: tokens.bin failed the size check" in messages(report, "info")


def test_a_hidden_index_that_is_not_a_whole_number_of_uint32_fails_at_the_manifest(trace):
    """``n_captured`` is derived from this file's length, so a ragged one is unreadable."""
    truncate(shard(trace, 0) / "hidden_index.bin", 2)
    report = validate_shard(shard(trace, 0))
    assert not report.ok
    assert "not a whole number of uint32" in errors(report)


# ==============================================================================================
# check_topk_range_and_distinctness — the necessary-but-not-sufficient label check
# ==============================================================================================


def test_healthy_labels_are_in_range_and_distinct_and_say_so_without_erroring(trace):
    report = run(shard(trace, 0), "topk_labels")
    assert report.ok
    assert "NOT sufficient" in messages(report, "info")


@pytest.mark.parametrize("bad_value", [-1, SPEC.n_experts, 10_000])
def test_an_expert_index_outside_the_routed_range_is_an_error(trace, bad_value):
    """An I32 stream read as float, or a wrong n_experts, both land here (I5)."""
    path = shard(trace, 0) / "topk.bin"
    topk = load(path, TOPK_DTYPE, (SHARD_SIZES[0], SPEC.n_moe_layers, SPEC.top_k))
    topk[5, 2, 1] = bad_value
    store(path, topk)
    report = run(shard(trace, 0), "topk_labels")
    assert f"outside [0, {SPEC.n_experts})" in errors(report)


def test_a_repeated_expert_within_one_token_layer_row_is_an_error(trace):
    path = shard(trace, 0) / "topk.bin"
    topk = load(path, TOPK_DTYPE, (SHARD_SIZES[0], SPEC.n_moe_layers, SPEC.top_k))
    topk[7, 1, 2] = topk[7, 1, 0]
    store(path, topk)
    report = run(shard(trace, 0), "topk_labels")
    assert f"do not hold {SPEC.top_k} distinct experts" in errors(report)


def test_the_label_check_skips_rather_than_raises_when_topk_is_unmappable(trace):
    truncate(shard(trace, 0) / "topk.bin", 5)
    report = run(shard(trace, 0), "topk_labels")
    assert report.ok
    assert "skipped" in messages(report, "info")


# ==============================================================================================
# check_topk_strided_view — I12, the worst silent-corruption path in the plan
# ==============================================================================================


def _rewrite_topk_from_logits(shard_dir: Path, spec: TraceSpec, n_tokens: int, *, strided: bool):
    """Rebuild ``topk.bin`` as either the correct de-strided read or the I12 contiguous one.

    The corrupt variant is not a hand-picked pattern: it is literally what a harness that
    ignores ``nb[1]`` produces — flat positions ``[t*k, t*k+k)`` of the full
    ``[n_experts, n_tokens]`` argsort, instead of ``[t*n_experts, t*n_experts+k)``. Token 0 is
    identical under both readings, which is why eyeballing the first row proves nothing.
    """
    logits = load(
        shard_dir / "logits.bin", LOGIT_DTYPE, spec.logit_shape(n_tokens)
    ).astype(np.float32)
    topk = np.empty(spec.topk_shape(n_tokens), dtype=TOPK_DTYPE)
    for layer in range(spec.n_moe_layers):
        order = np.argsort(-logits[:, layer, :], axis=1, kind="stable")
        if strided:
            topk[:, layer, :] = order.ravel()[: n_tokens * spec.top_k].reshape(
                n_tokens, spec.top_k
            )
        else:
            topk[:, layer, :] = order[:, : spec.top_k]
    store(shard_dir / "topk.bin", topk)


def test_a_contiguous_read_of_the_strided_topk_view_is_identified_exactly(tmp_path):
    """I12: in-range, distinct, wrong indices for every token after the first."""
    sd = one_shard(tmp_path, n_tokens=300, seed=3)
    _rewrite_topk_from_logits(sd, SPEC, 300, strided=True)
    report = run(sd, "topk_strided_view")
    assert "matches a CONTIGUOUS read of the strided ffn_moe_topk view" in errors(report)
    detail = [f for f in report.findings if f.is_error][0].detail
    assert detail["strided_hypothesis_agreement"] == pytest.approx(1.0)
    assert detail["correct_hypothesis_agreement"] < 0.5


def test_the_i12_corruption_also_trips_the_adjacency_proxy_that_needs_no_logits(tmp_path):
    """Consecutive segments of one permutation are disjoint far more often than routing is.

    The proxy is one-sided in the right direction: real routers have temporal locality (that is
    feature F3), which pushes adjacent-token overlap *up*, so an excess of disjointness cannot
    be explained by routing behaviour.
    """
    sd = one_shard(tmp_path, n_tokens=300, seed=3)
    _rewrite_topk_from_logits(sd, SPEC, 300, strided=True)
    findings = collect(check_topk_strided_view(_handle(sd)))
    proxy = [f for f in findings if "disjoint far more often" in f.message]
    assert proxy, [f.message for f in findings]
    assert proxy[0].detail["flagged_layers"]


def test_a_correctly_de_strided_shard_is_not_flagged_as_i12(tmp_path):
    sd = one_shard(tmp_path, n_tokens=300, seed=3)
    _rewrite_topk_from_logits(sd, SPEC, 300, strided=False)
    report = run(sd, "topk_strided_view")
    assert report.ok, errors(report)
    assert "no I12 signature" in messages(report, "info")
    assert warnings(report) == ""


def test_routing_unrelated_to_the_logits_is_not_mistaken_for_i12(trace):
    """The synthetic router is independent of ``logits.bin``; neither hypothesis wins."""
    for i in range(len(SHARD_SIZES)):
        report = run(shard(trace, i), "topk_strided_view")
        assert report.ok, errors(report)


def test_the_i12_corruption_is_not_expressible_when_every_expert_is_selected(tmp_path):
    spec = TraceSpec(n_moe_layers=2, n_experts=4, top_k=4, hidden_dim=4)
    sd = one_shard(tmp_path, spec=spec, n_tokens=60)
    report = run(sd, "topk_strided_view")
    assert "not expressible" in messages(report, "info")
    assert report.ok


# ==============================================================================================
# check_expert_usage — dead experts, saturation, load balance (T6.4)
# ==============================================================================================


def _constant_router(experts):
    def topk_fn(rng, tokens, spec):
        out = np.empty(spec.topk_shape(tokens.shape[0]), dtype=TOPK_DTYPE)
        out[:, :, :] = np.asarray(experts, dtype=TOPK_DTYPE)
        return out

    return topk_fn


def test_balanced_routing_reports_a_load_balance_index_near_one_and_no_warning(tmp_path):
    sd = one_shard(tmp_path, n_tokens=400, seed=11)
    report = run(sd, "expert_usage")
    assert report.ok
    assert warnings(report) == ""
    detail = [f for f in report.findings if f.check == "expert_usage"][0].detail
    assert detail["load_balance_index_mean"] == pytest.approx(1.0, abs=0.05)


def test_an_expert_never_selected_in_the_shard_is_reported_as_dead(tmp_path):
    """q(e) is exactly zero for it, which is what §1.2's epsilon-mix covers."""
    sd = one_shard(tmp_path, n_tokens=200, topk_fn=_constant_router([0, 1, 2]))
    report = run(sd, "expert_usage")
    assert "dead expert slot(s)" in warnings(report)
    dead = [f for f in report.findings if "dead expert" in f.message][0].detail
    assert dead["n_dead"] == SPEC.n_moe_layers * (SPEC.n_experts - 3)


def test_an_expert_taking_far_more_than_its_uniform_share_is_reported_as_saturated(tmp_path):
    sd = one_shard(tmp_path, n_tokens=200, topk_fn=_constant_router([0, 1, 2]))
    report = run(sd, "expert_usage")
    assert "x uniform in" in warnings(report)
    hot = [f for f in report.findings if "x uniform" in f.message][0].detail
    assert {e["expert"] for e in hot["saturated_per_layer"]["0"]} == {0, 1, 2}


def test_a_collapsed_router_warns_and_reports_the_measured_index_unclamped(tmp_path):
    """I8's spirit: the number reported is the one measured, not a floor.

    Three experts out of sixteen, perfectly evenly used, is exactly log2(3)/log2(16).
    """
    sd = one_shard(tmp_path, n_tokens=200, topk_fn=_constant_router([0, 1, 2]))
    report = run(sd, "expert_usage")
    lbi = [f for f in report.findings if "load-balance index" in f.message][0]
    assert lbi.severity == "warning"
    expected = np.log2(3) / np.log2(SPEC.n_experts)
    assert lbi.detail["load_balance_index_mean"] == pytest.approx(expected, abs=0.02)


def test_expert_usage_counts_are_folded_into_the_report_counters(tmp_path):
    sd = one_shard(tmp_path, n_tokens=200, seed=5)
    report = run(sd, "expert_usage")
    per_layer = report.counters["load_balance_index_per_layer"]
    assert set(per_layer) == {str(i) for i in range(SPEC.n_moe_layers)}


def test_expert_usage_skips_rather_than_raising_when_topk_is_unmappable(trace):
    truncate(shard(trace, 0) / "topk.bin", 5)
    report = run(shard(trace, 0), "expert_usage")
    assert report.ok
    assert "skipped" in messages(report, "info")


# ==============================================================================================
# check_logits_sanity
# ==============================================================================================


def test_finite_logits_report_a_computable_margin(trace):
    report = run(shard(trace, 0), "logits_sanity")
    assert report.ok
    detail = [f for f in report.findings][0].detail
    assert detail["min_margin"] is not None and detail["min_margin"] >= 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_single_non_finite_logit_is_an_error(trace, bad):
    """np.partition sorts NaN last, so one NaN silently moves which pair the margin spans."""
    path = shard(trace, 0) / "logits.bin"
    logits = load(path, LOGIT_DTYPE, SPEC.logit_shape(SHARD_SIZES[0]))
    logits[3, 1, 4] = np.float16(bad)
    store(path, logits)
    report = run(shard(trace, 0), "logits_sanity")
    assert "NaN and" in errors(report)
    first = [f for f in report.findings if f.is_error][0].detail["first"]
    assert (first["token"], first["layer"], first["expert"]) == (3, 1, 4)


def test_a_model_that_selects_every_expert_has_no_margin_and_says_so(tmp_path):
    spec = TraceSpec(n_moe_layers=2, n_experts=4, top_k=4, hidden_dim=4)
    sd = one_shard(tmp_path, spec=spec, n_tokens=40)
    report = run(sd, "logits_sanity")
    assert "flip analysis is not available" in warnings(report)
    assert report.ok


def test_logits_sanity_skips_rather_than_raising_when_logits_are_unmappable(trace):
    truncate(shard(trace, 0) / "logits.bin", 3)
    report = run(shard(trace, 0), "logits_sanity")
    assert report.ok
    assert "skipped" in messages(report, "info")


# ==============================================================================================
# check_selection_argsort_agreement — I13
# ==============================================================================================


def test_topk_reproduced_by_the_logit_ranking_is_reported_as_consistent(tmp_path):
    sd = one_shard(tmp_path, n_tokens=300, seed=3)
    _rewrite_topk_from_logits(sd, SPEC, 300, strided=False)
    report = run(sd, "selection_argsort")
    assert report.ok
    assert warnings(report) == ""
    detail = report.findings[0].detail
    assert detail["set_agreement"] == pytest.approx(1.0)
    assert detail["exact_match"] == pytest.approx(1.0)


def test_a_logit_stream_that_does_not_explain_the_labels_is_reported_never_silent(trace):
    """I13: if the manifest names an earlier node, every T8.2 margin is the wrong quantity.

    Warning, not error — GPT-OSS's router bias and fp16 near-ties both make <1.0 legitimate, so
    a gate here would fail correct traces. What is forbidden is passing silently.
    """
    report = run(shard(trace, 0), "selection_argsort")
    assert report.ok  # never an error
    assert "logit_tensor_used=" in warnings(report)
    assert "(I13)" in warnings(report)
    finding = [f for f in report.findings if f.severity == "warning"][0]
    assert finding.detail["logit_tensor_used"] == "ffn_moe_logits"
    assert finding.detail["set_agreement"] < 0.5


def test_the_selection_check_samples_across_the_shard_not_only_a_prefix(tmp_path):
    """A chain that is only wrong for later documents would hide in a prefix (I13's LLaMA-4
    branch re-resolves ``selection_probs`` after the chain is already named)."""
    sd = one_shard(tmp_path, n_tokens=400, seed=3)
    _rewrite_topk_from_logits(sd, SPEC, 400, strided=False)
    topk = load(sd / "topk.bin", TOPK_DTYPE, SPEC.topk_shape(400))
    # Corrupt only the tail: roll the labels for the last quarter of the shard.
    topk[300:] = np.roll(topk[300:], 1, axis=0)
    store(sd / "topk.bin", topk)
    report = run(sd, "selection_argsort", sample_tokens=64)
    assert "reproduces topk.bin at only" in warnings(report)


def test_the_selection_check_skips_when_either_stream_is_unmappable(trace):
    truncate(shard(trace, 0) / "logits.bin", 3)
    report = run(shard(trace, 0), "selection_argsort")
    assert "skipped" in messages(report, "info")


# ==============================================================================================
# check_truncation / capture_stats — I15
# ==============================================================================================


def healthy_stats(**overrides) -> dict:
    stats = {
        "n_docs_truncated": 0,
        "n_tokens_dropped": 0,
        "first_truncated_doc": None,
        "exit_code": 0,
        "topk_layout": "strided",
        "nodes_captured": 3 * SPEC.n_moe_layers * 5,
        "n_tokens": SHARD_SIZES[0],
        "n_moe_layers": SPEC.n_moe_layers,
        "n_experts": SPEC.n_experts,
        "top_k": SPEC.top_k,
        "hidden_dim": SPEC.hidden_dim,
        "n_captured": SHARD0_CAPTURED,
        "hidden_stride": HIDDEN_EVERY,
        "index_scheme": "doc_id*n_ctx+pos_in_doc",
        "index_doc_span": DOC_INDEX_SPAN,
    }
    stats.update(overrides)
    return stats


def test_a_shard_with_a_clean_capture_stats_file_raises_nothing(trace):
    report = run(shard(trace, 0), "capture_stats", stats=healthy_stats())
    assert report.ok, errors(report)
    assert warnings(report) == ""
    assert "no documents truncated (I15)" in messages(report, "info")


def test_a_truncated_document_invalidates_the_cross_model_comparison(trace):
    """I15 is unrecoverable from the streams: a truncated shard is otherwise well formed."""
    stats = healthy_stats(n_docs_truncated=2, n_tokens_dropped=511, first_truncated_doc=17)
    report = run(shard(trace, 0), "capture_stats", stats=stats)
    assert "exceeded n_ctx and were truncated" in errors(report)
    assert "(I15)" in errors(report)
    detail = [f for f in report.findings if f.is_error][0].detail
    assert detail == {
        "n_docs_truncated": 2,
        "first_truncated_doc": 17,
        "n_tokens_dropped": 511,
    }


def test_a_missing_capture_stats_file_is_a_warning_because_i15_becomes_unverifiable(trace):
    report = run(shard(trace, 0), "capture_stats")
    assert "I15 truncation" in warnings(report)
    assert read_stats(shard(trace, 0)) is None


def test_capture_stats_is_read_from_the_shard_directory_when_present(trace):
    (shard(trace, 0) / STATS_NAME).write_text(json.dumps(healthy_stats()), encoding="utf-8")
    report = run(shard(trace, 0), "capture_stats")
    assert report.ok, errors(report)
    assert warnings(report) == ""


def test_a_malformed_capture_stats_file_reads_as_absent_rather_than_raising(trace):
    (shard(trace, 0) / STATS_NAME).write_text("{not json", encoding="utf-8")
    assert read_stats(shard(trace, 0)) is None
    report = run(shard(trace, 0), "capture_stats")
    assert "not found" in warnings(report)


def test_a_nonzero_capture_exit_code_forbids_marking_the_shard_collected(trace):
    report = run(shard(trace, 0), "capture_stats", stats=healthy_stats(exit_code=137))
    assert "exited 137" in errors(report)


def test_a_mixed_topk_layout_means_the_de_striding_misidentified_a_layout(trace):
    """The third independent angle on I12; nothing in ggml should produce both in one shard."""
    report = run(shard(trace, 0), "capture_stats", stats=healthy_stats(topk_layout="mixed"))
    assert "topk_layout='mixed'" in errors(report)


@pytest.mark.parametrize("layout", [None, "none"])
def test_an_unrecorded_topk_layout_warns_that_the_de_striding_path_never_ran(trace, layout):
    report = run(shard(trace, 0), "capture_stats", stats=healthy_stats(topk_layout=layout))
    assert "de-striding path was never exercised" in warnings(report)


def test_a_node_count_that_is_not_three_per_moe_layer_suggests_an_over_broad_filter(trace):
    stats = healthy_stats(nodes_captured=3 * SPEC.n_moe_layers * 5 + 1)
    report = run(shard(trace, 0), "capture_stats", stats=stats)
    assert "is not a multiple of 3 x n_moe_layers" in warnings(report)


@pytest.mark.parametrize("key", ["n_tokens", "n_moe_layers", "n_experts", "top_k", "hidden_dim"])
def test_capture_stats_disagreeing_with_the_manifest_about_shape_is_an_error(trace, key):
    stats = healthy_stats(**{key: healthy_stats()[key] + 1})
    report = run(shard(trace, 0), "capture_stats", stats=stats)
    assert f"says {key}=" in errors(report)


def test_capture_stats_disagreeing_about_the_captured_count_is_an_error(trace):
    report = run(shard(trace, 0), "capture_stats", stats=healthy_stats(n_captured=99))
    assert "n_captured=99" in errors(report)
    assert f"holds {SHARD0_CAPTURED} rows" in errors(report)


# ==============================================================================================
# check_split_coverage — T4.3
# ==============================================================================================


def test_every_document_in_a_healthy_shard_has_a_split(trace):
    report = run(shard(trace, 2), "split_coverage", doc_splits=trace.doc_splits)
    assert report.ok, errors(report)
    counts = report.counters["tokens_per_split"]
    assert sum(counts.values()) == SHARD_SIZES[2]


def test_a_document_with_no_split_assignment_is_an_error_not_a_silent_drop(trace):
    splits = dict(trace.doc_splits)
    del splits[0]  # doc 0 covers tokens 0..9 of shard 0
    report = run(shard(trace, 0), "split_coverage", doc_splits=splits)
    assert "no split assignment" in errors(report)
    detail = [f for f in report.findings if f.is_error][0].detail
    assert detail["unknown_tokens"] == 10
    assert detail["example_doc_ids"] == [0]


def test_split_coverage_says_it_was_not_checked_when_no_mapping_is_supplied(trace):
    report = run(shard(trace, 0), "split_coverage")
    assert report.ok
    assert "no doc_splits supplied" in messages(report, "info")


def test_an_empty_required_split_is_only_a_verdict_the_whole_shard_set_can_reach(trace):
    """Per-shard emptiness is normal (a shard covers one ``shard_doc_range``); trace-level is not."""
    per_shard = run(shard(trace, 0), "split_coverage", doc_splits=trace.doc_splits)
    assert per_shard.counters["tokens_per_split"]["test"] == 0
    assert per_shard.ok

    all_train = {d: "train" for d in trace.doc_splits}
    reports = validate_shards(
        trace_dir(trace), doc_splits=all_train, checks=["split_coverage"]
    )
    assert "contain zero tokens across the whole shard set" in errors(reports[-1])
    assert reports[-1].errors()[0].detail["empty_splits"] == ["val", "test"]


def test_a_shard_set_whose_splits_are_all_populated_is_accepted(trace):
    reports = validate_shards(
        trace_dir(trace), doc_splits=trace.doc_splits, checks=["split_coverage"]
    )
    assert reports[-1].ok, errors(reports[-1])
    totals = reports[-1].counters["tokens_per_split"]
    assert sum(totals.values()) == sum(SHARD_SIZES)
    assert all(v > 0 for v in totals.values())


# ==============================================================================================
# check_hidden_stride — the global-stride rule
# ==============================================================================================


def _stride_stats(**overrides) -> dict:
    stats = {"hidden_stride": HIDDEN_EVERY, "index_doc_span": DOC_INDEX_SPAN}
    stats.update(overrides)
    return stats


def test_every_healthy_shard_matches_the_global_stride_rule(trace):
    """Regression: shard 2 begins at document 6 position 5, mid-document and mid-subsample.

    The old rule was a running token count from a per-shard base, and a shard that did not begin
    on a stride boundary made the base ambiguous by up to one stride — a validator that guessed
    it failed this healthy fixture. Under `doc_id * n_ctx + pos_in_doc` there is nothing to
    guess: `tokens.bin` names the document and position of every token.
    """
    for i in range(len(SHARD_SIZES)):
        report = run(shard(trace, i), "hidden_stride", stats=_stride_stats())
        assert report.ok, f"shard {i}: {errors(report)}"
        assert "tokens.flags agrees" in messages(report, "info")


def test_the_index_of_every_captured_row_is_checked_not_just_the_count(trace):
    """The count can be right while the rows belong to different tokens."""
    report = run(shard(trace, 0), "hidden_stride", stats=_stride_stats())
    detail = report.findings[0].detail
    assert detail["n_captured"] == SHARD0_CAPTURED
    assert detail["index_doc_span"] == DOC_INDEX_SPAN


def test_without_capture_stats_the_check_degrades_to_ascent_and_says_so(trace):
    """An honest partial check beats a confident wrong one: the span is not inferable."""
    report = run(shard(trace, 0), "hidden_stride")
    assert report.ok
    assert "only checked for strict ascent" in warnings(report)


def test_a_wrong_declared_doc_span_is_caught(trace):
    """The span is `n_ctx`. A shard captured under a different one does not concatenate."""
    report = run(shard(trace, 2), "hidden_stride", stats=_stride_stats(index_doc_span=32))
    assert errors(report)


def test_captured_indices_that_are_not_on_the_stride_are_caught(trace):
    """A re-shard moved which tokens carry hidden states; F4/F5 would read a neighbour's input."""
    path = shard(trace, 0) / "hidden_index.bin"
    index = load(path, HIDDEN_INDEX_DTYPE)
    index[2] += 1  # still ascending, no longer a stride multiple of its document block
    store(path, index)
    report = run(shard(trace, 0), "hidden_stride", stats=_stride_stats())
    assert "are not doc_id *" in errors(report)
    example = [f for f in report.findings if f.is_error][0].detail["examples"][0]
    assert example["row"] == 2 and example["got"] == example["want"] + 1


def test_captured_rows_missing_from_the_end_are_caught_by_the_implied_count(trace):
    truncate(shard(trace, 0) / "hidden_index.bin", 3 * HIDDEN_INDEX_DTYPE.itemsize)
    report = run(shard(trace, 0), "hidden_stride", stats=_stride_stats())
    assert f"selects {SHARD0_CAPTURED} of this shard's" in errors(report)
    assert "holds 9 rows" in errors(report)


def test_a_flag_that_disagrees_with_the_stride_rule_is_caught(trace):
    """`tokens.flags` and the index stream are written by different code; they must agree."""
    path = shard(trace, 0) / "tokens.bin"
    tokens = load(path, TOKEN_DTYPE)
    tokens["flags"][1] |= 1  # token 1 is not on the stride
    store(path, tokens)
    report = run(shard(trace, 0), "hidden_stride", stats=_stride_stats())
    assert "tokens.flags disagrees" in errors(report) or "selects" in errors(report)


def test_a_non_ascending_index_is_refused_before_anything_else(trace):
    """The reader concatenates shards unrewritten, so ascent is the load-bearing property."""
    path = shard(trace, 0) / "hidden_index.bin"
    index = load(path, HIDDEN_INDEX_DTYPE)
    index[5], index[6] = index[6], index[5]
    store(path, index)
    report = run(shard(trace, 0), "hidden_stride", stats=_stride_stats())
    assert "not strictly ascending" in errors(report)


def test_a_declared_stride_with_no_captured_rows_at_all_is_an_error(trace):
    sd = shard(trace, 0)
    (sd / "hidden_index.bin").write_bytes(b"")
    (sd / "hidden.bin").write_bytes(b"")
    report = run(sd, "hidden_stride", stats={"hidden_stride": 4})
    assert "should have captured at least one row" in errors(report)


def test_no_captured_rows_and_no_declared_stride_is_merely_reported(trace):
    sd = shard(trace, 0)
    (sd / "hidden_index.bin").write_bytes(b"")
    (sd / "hidden.bin").write_bytes(b"")
    report = run(sd, "hidden_stride")
    assert report.ok
    assert "no hidden states captured" in messages(report, "info")


def test_an_empty_subsample_never_reaches_the_memmap_that_would_refuse_it(trace):
    """A zero-row ``hidden_index.bin`` is legal (T4.4 may drop the subsample) and numpy will
    not map a zero-length file, so the empty case has to be answered before mapping."""
    sd = shard(trace, 0)
    (sd / "hidden_index.bin").write_bytes(b"")
    (sd / "hidden.bin").write_bytes(b"")
    assert collect(check_hidden_stride(_handle(sd))) == [
        Finding("hidden_stride", "info", "no hidden states captured", {"n_captured": 0})
    ]


# ==============================================================================================
# validate_shard orchestration
# ==============================================================================================


def test_a_healthy_shard_passes_every_check_at_once(trace):
    for i in range(len(SHARD_SIZES)):
        report = validate_shard(shard(trace, i), doc_splits=trace.doc_splits)
        assert report.ok, f"shard {i}: {errors(report)}"


def test_the_report_carries_the_shape_counters_downstream_aggregation_needs(trace):
    report = validate_shard(shard(trace, 1), doc_splits=trace.doc_splits)
    assert report.counters["n_tokens"] == SHARD_SIZES[1]
    assert report.counters["n_experts"] == SPEC.n_experts
    assert report.counters["top_k"] == SPEC.top_k
    assert report.counters["n_moe_layers"] == SPEC.n_moe_layers
    assert report.shard_id == 1
    assert report.model == "synth-moe"
    assert report.corpus == "synth-v1"


def test_a_broken_manifest_comes_back_as_a_finding_not_an_exception(trace):
    """The caller is a collection loop that must record a verdict for every shard."""
    (shard(trace, 0) / MANIFEST_NAME).write_text("{ not json", encoding="utf-8")
    report = validate_shard(shard(trace, 0))
    assert not report.ok
    assert report.by_check("manifest")
    assert "malformed JSON" in errors(report)


def test_a_manifest_missing_a_required_key_comes_back_as_a_finding(trace):
    path = shard(trace, 0) / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    del manifest["run_config_sha256"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_shard(shard(trace, 0))
    assert "run_config_sha256" in errors(report)


def test_an_unknown_check_name_is_refused_loudly(trace):
    with pytest.raises(ValueError, match="unknown check"):
        validate_shard(shard(trace, 0), checks=["no_such_check"])


def test_every_registered_check_runs_and_is_named_in_check_names(trace):
    assert set(CHECK_NAMES) == {
        "size_arithmetic",
        "lockstep",
        "topk_labels",
        "topk_strided_view",
        "expert_usage",
        "logits_sanity",
        "selection_argsort",
        "capture_stats",
        "split_coverage",
        "hidden_stride",
    }
    for name in CHECK_NAMES:
        report = validate_shard(shard(trace, 0), checks=[name], doc_splits=trace.doc_splits)
        assert report.ok, f"{name}: {errors(report)}"


# ==============================================================================================
# validate_shards — the verdicts only the shard SET can reach (S.3 / I2)
# ==============================================================================================


def test_discover_shards_accepts_either_a_shard_or_a_trace_directory(trace):
    assert discover_shards(shard(trace, 1)) == [shard(trace, 1)]
    assert discover_shards(trace_dir(trace)) == [shard(trace, i) for i in range(3)]


def test_a_shard_set_yields_one_report_per_shard_plus_a_trace_level_report(trace):
    reports = validate_shards(trace_dir(trace), doc_splits=trace.doc_splits)
    assert len(reports) == len(SHARD_SIZES) + 1
    assert [r.shard_id for r in reports] == [0, 1, 2, -1]
    assert reports[-1].counters["n_shards"] == 3
    assert reports[-1].counters["n_tokens"] == sum(SHARD_SIZES)
    assert all(r.ok for r in reports), [errors(r) for r in reports]


@pytest.mark.parametrize(
    "key,value",
    [
        ("run_config_sha256", "1" * 64),
        ("logit_tensor_used", "ffn_moe_probs_biased"),
        ("gguf_sha256", "a" * 64),
        ("llama_cpp_commit", "deadbee"),
        ("quant", "Q8_0"),
        ("n_experts", 32),
        ("checkpoint_status", "instruct"),
    ],
)
def test_shards_collected_under_different_conditions_are_a_hard_error_to_merge(
    trace, key, value
):
    """I2 / S.3. Platform is part of the run config (I3), so this is the whole guard."""
    patch_manifest(shard(trace, 1), **{key: value})
    reports = validate_shards(trace_dir(trace), checks=["size_arithmetic"])
    trace_report = reports[-1]
    assert not trace_report.ok
    message = errors(trace_report)
    assert f"manifest key {key!r} differs across shards" in message
    assert "different experiment" in message
    assert "(S.3, I2)" in message


def test_identical_run_config_across_shards_is_not_flagged(trace):
    reports = validate_shards(trace_dir(trace), checks=["size_arithmetic"])
    assert reports[-1].by_check("shard_set") == []


def test_two_shards_claiming_the_same_id_would_shadow_each_other(trace):
    patch_manifest(shard(trace, 1), shard_id=0)
    reports = validate_shards(trace_dir(trace), checks=["size_arithmetic"])
    assert "duplicate shard id(s) [0]" in errors(reports[-1])


def test_a_directory_with_no_shards_is_an_error_rather_than_an_empty_pass(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    reports = validate_shards(empty)
    assert len(reports) == 1
    assert "no shard directories found" in errors(reports[0])


def test_a_corrupt_shard_in_a_set_does_not_stop_the_others_from_being_validated(trace):
    truncate(shard(trace, 1) / "topk.bin", SPEC.topk_stride)
    reports = validate_shards(trace_dir(trace), checks=["size_arithmetic", "lockstep"])
    assert [r.ok for r in reports[:3]] == [True, False, True]


# ==============================================================================================
# CLI — T5.2 keys on the exit code, not on the log
# ==============================================================================================


def test_the_cli_exits_zero_on_a_healthy_trace(trace, capsys):
    assert main([str(trace_dir(trace))]) == 0
    out = capsys.readouterr().out
    assert "PASS shard_00000" in out
    assert "0 error(s)" in out


def test_the_cli_exits_one_so_a_bad_shard_is_never_marked_complete(trace, capsys):
    truncate(shard(trace, 2) / "logits.bin", SPEC.logit_stride)
    assert main([str(trace_dir(trace))]) == 1
    assert "FAIL shard_00002" in capsys.readouterr().out


def test_the_cli_writes_a_machine_readable_report(trace, tmp_path, capsys):
    out_path = tmp_path / "reports" / "validation.json"
    assert main([str(trace_dir(trace)), "--json", str(out_path)]) == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload) == len(SHARD_SIZES) + 1
    assert payload[0]["ok"] is True
    assert {f["check"] for f in payload[0]["findings"]} <= set(CHECK_NAMES) | {"truncation"}


def test_the_cli_runs_only_the_requested_checks(trace, capsys):
    assert main([str(trace_dir(trace)), "--checks", "size_arithmetic", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "topk_strided_view" not in out


def test_verbose_shows_info_findings_that_the_default_hides(trace, capsys):
    main([str(trace_dir(trace)), "--checks", "topk_labels"])
    quiet = capsys.readouterr().out
    main([str(trace_dir(trace)), "--checks", "topk_labels", "--verbose"])
    loud = capsys.readouterr().out
    assert "NOT sufficient" not in quiet
    assert "NOT sufficient" in loud


def test_the_default_sample_budget_is_bounded_so_a_4m_token_shard_stays_cheap():
    assert DEFAULT_SAMPLE_TOKENS <= 8192


def test_a_mismatch_at_an_exact_fp16_tie_is_attributed_to_storage_not_the_chain(tmp_path):
    """Real OLMoE capture: 56 of 32432 rows disagreed, every one an exact fp16 tie.

    llama.cpp breaks the k/(k+1) tie in fp32 before logits.bin is down-cast, so the two sets are
    indistinguishable on disk and topk.bin is authoritative. Reporting those rows as evidence
    about `logit_tensor_used` would put a floor of benign noise under the one check that catches
    a wrong node of the selection chain (I13).
    """
    sd = one_shard(tmp_path, n_tokens=300, seed=11)
    lg = load(sd / "logits.bin", LOGIT_DTYPE, SPEC.logit_shape(300))
    # Flatten every expert in one row to one value: every ordering of it is then a tie, so a
    # disagreement is guaranteed and is entirely explained by storage precision.
    lg[0, 0, :] = np.float16(0.25)
    store(sd / "logits.bin", lg)
    _rewrite_topk_from_logits(sd, SPEC, 300, strided=False)
    # ...and then pick the tied row's experts in an order argsort would not produce.
    topk = load(sd / "topk.bin", TOPK_DTYPE, SPEC.topk_shape(300))
    topk[0, 0, :] = np.arange(SPEC.n_experts - SPEC.top_k, SPEC.n_experts)
    store(sd / "topk.bin", topk)

    report = run(sd, "selection_argsort")
    assert not errors(report)
    text = " ".join(f.message for f in report.findings)
    assert "storage precision" in text
