"""Phase 8 tests — precision-ladder drift, the T8.3 bands, and T8.5.

The traces here are synthetic and the drift is *injected by hand*, so every metric has an exact
expected value rather than a plausible one. That matters more than usual for this phase: T8.3 can
change what the paper claims, so a drift number that is off by a factor is not a cosmetic bug.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis.precision import (
    DEVICE_SPLIT_FLOOR,
    PrecisionError,
    check_device_split,
    compare_traces,
    interpret,
    main,
)
from src.traces.format import LOGIT_DTYPE, TOPK_DTYPE, TraceSpec
from src.traces.reader import TraceReader
from src.traces.synth import make_synthetic_trace

SPEC = TraceSpec(n_moe_layers=4, n_experts=16, top_k=3, hidden_dim=8)
SHARDS = (40, 25, 35)
N_TOKENS = sum(SHARDS)


# -- fixtures ----------------------------------------------------------------------------------


def _shard_bounds():
    out, start = [], 0
    for size in SHARDS:
        out.append((start, start + size))
        start += size
    return out


def _write_streams(root, model, corpus, topk, logits):
    """Overwrite topk.bin/logits.bin across the shards of an existing synthetic trace."""
    trace_dir = root / model / corpus
    for shard_id, (lo, hi) in enumerate(_shard_bounds()):
        shard = trace_dir / f"shard_{shard_id:05d}"
        np.ascontiguousarray(topk[lo:hi], dtype=TOPK_DTYPE).tofile(shard / "topk.bin")
        np.ascontiguousarray(logits[lo:hi], dtype=LOGIT_DTYPE).tofile(shard / "logits.bin")


def _base_topk():
    """A fixed, distinct selection per (token, layer). Deterministic and easy to perturb."""
    topk = np.empty((N_TOKENS, SPEC.n_moe_layers, SPEC.top_k), dtype=np.int32)
    for t in range(N_TOKENS):
        for l in range(SPEC.n_moe_layers):
            topk[t, l] = ((np.arange(SPEC.top_k) + t + 5 * l) % SPEC.n_experts)
    return topk


def _logits_with_margins(margins):
    """Build a probability block whose k-th minus (k+1)-th gap is exactly ``margins[t, l]``.

    The reader derives margins from ``logits.bin``, so controlling the gap directly is the only
    way to test the margin conditioning without also testing the model.
    """
    logits = np.zeros((N_TOKENS, SPEC.n_moe_layers, SPEC.n_experts), dtype=np.float32)
    k = SPEC.top_k
    for t in range(N_TOKENS):
        for l in range(SPEC.n_moe_layers):
            values = np.linspace(1.0, 0.5, SPEC.n_experts)
            values[k:] -= margins[t, l]
            logits[t, l] = values
    return logits.astype(LOGIT_DTYPE)


def make_pair(tmp_path, *, flips=None, margins=None, cand_quant="Q4_K_M", ref_quant="F16",
              cand_overrides=None):
    """Two traces of the same corpus differing only where ``flips`` says so.

    ``flips`` is a boolean ``(n_tokens, n_layers)`` array; a True entry replaces one selected
    expert in the candidate, which costs exactly one of ``top_k`` overlap for that row.
    """
    if margins is None:
        margins = np.full((N_TOKENS, SPEC.n_moe_layers), 0.05, dtype=np.float32)
    topk = _base_topk()
    logits = _logits_with_margins(margins)

    ref_root = tmp_path / "ref"
    cand_root = tmp_path / "cand"
    for root, quant, extra in ((ref_root, ref_quant, {}),
                               (cand_root, cand_quant, cand_overrides or {})):
        make_synthetic_trace(
            root, spec=SPEC, shard_sizes=SHARDS, tokens_per_doc=10, hidden_every=4, seed=7,
            manifest_overrides={"quant": quant, **extra},
        )

    _write_streams(ref_root, "synth-moe", "synth-v1", topk, logits)

    cand_topk = topk.copy()
    if flips is not None:
        rows, layers = np.nonzero(flips)
        # Replace the last selected expert with one that is definitely not already in the set.
        cand_topk[rows, layers, -1] = (cand_topk[rows, layers, -1] + SPEC.top_k) % SPEC.n_experts
    _write_streams(cand_root, "synth-moe", "synth-v1", cand_topk, logits)

    return (TraceReader(ref_root, "synth-moe", "synth-v1"),
            TraceReader(cand_root, "synth-moe", "synth-v1"))


# -- identity and exact arithmetic ---------------------------------------------------------------


def test_two_identical_traces_show_no_drift(tmp_path):
    reference, candidate = make_pair(tmp_path)
    result = compare_traces(reference, candidate)
    assert result.mean_set_agreement == 1.0
    assert all(l.exact_match == 1.0 and l.flip_rate == 0.0 for l in result.layers)
    assert all(l.spearman == pytest.approx(1.0) for l in result.layers)


def test_a_known_flip_count_produces_the_exact_expected_metrics(tmp_path):
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[:20, 1] = True  # 20 of 100 tokens flip one expert, in layer 1 only
    reference, candidate = make_pair(tmp_path, flips=flips)
    result = compare_traces(reference, candidate)

    layer1 = result.layers[1]
    assert layer1.flip_rate == pytest.approx(0.20)
    assert layer1.exact_match == pytest.approx(0.80)
    # 20 rows lose exactly 1 of top_k overlap.
    assert layer1.set_agreement == pytest.approx(1.0 - 20 / (N_TOKENS * SPEC.top_k))
    assert all(result.layers[l].flip_rate == 0.0 for l in (0, 2, 3))


def test_set_agreement_reads_topk_not_a_recomputation(tmp_path):
    """Both traces here carry IDENTICAL logits.bin and differing topk.bin. A comparison that
    recomputed selections from the stored logits would report perfect agreement, which is exactly
    the failure the plan warns about: it would measure the fp16 storage, not the quantization."""
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[:50, :] = True
    reference, candidate = make_pair(tmp_path, flips=flips)
    result = compare_traces(reference, candidate)
    assert result.mean_set_agreement < 1.0
    assert all(l.spearman == pytest.approx(1.0) for l in result.layers), (
        "logits are identical, so the Spearman leg must be clean while the set leg is not"
    )


# -- margin conditioning -------------------------------------------------------------------------


def test_flips_at_small_margins_land_in_the_low_bins(tmp_path):
    """The expected, benign shape: drift concentrated where the k-th and (k+1)-th logits are
    nearly tied is arithmetic, and the plan says so."""
    rng = np.random.default_rng(3)
    margins = rng.uniform(0.01, 1.0, size=(N_TOKENS, SPEC.n_moe_layers)).astype(np.float32)
    small = margins[:, 0] <= np.quantile(margins[:, 0], 0.20)
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[small, 0] = True

    reference, candidate = make_pair(tmp_path, flips=flips, margins=margins)
    layer0 = compare_traces(reference, candidate).layers[0]

    by_margin = np.array(layer0.flip_rate_by_margin)
    assert by_margin[0] > 0, "the smallest-margin bin must carry flips"
    assert by_margin[-1] == 0.0, "the largest-margin bin must be clean"


def test_flips_at_large_margins_are_visible_as_the_bug_signature(tmp_path):
    """The whole point of conditioning. A uniform flip rate is indistinguishable from arithmetic
    until you see that the widest-margin tokens are flipping too, which arithmetic cannot do."""
    rng = np.random.default_rng(4)
    margins = rng.uniform(0.01, 1.0, size=(N_TOKENS, SPEC.n_moe_layers)).astype(np.float32)
    large = margins[:, 0] >= np.quantile(margins[:, 0], 0.80)
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[large, 0] = True

    reference, candidate = make_pair(tmp_path, flips=flips, margins=margins)
    layer0 = compare_traces(reference, candidate).layers[0]
    assert layer0.flip_rate_by_margin[-1] > 0
    assert layer0.flip_rate_by_margin[0] == 0.0


def test_degenerate_margins_do_not_manufacture_empty_clean_bins(tmp_path):
    """When every margin is identical the quantile edges collapse. Reporting five bins, four of
    them empty with a flip rate of 0.0, would read as 'clean at every margin but the smallest' —
    a conclusion drawn from no data at all."""
    margins = np.full((N_TOKENS, SPEC.n_moe_layers), 0.25, dtype=np.float32)
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[:30, 0] = True
    reference, candidate = make_pair(tmp_path, flips=flips, margins=margins)
    layer0 = compare_traces(reference, candidate).layers[0]

    assert len(layer0.flip_rate_by_margin) == 1
    assert layer0.n_by_margin == (N_TOKENS,)
    assert layer0.flip_rate_by_margin[0] == pytest.approx(0.30)


def test_every_margin_bin_reports_its_population(tmp_path):
    reference, candidate = make_pair(tmp_path)
    layer0 = compare_traces(reference, candidate).layers[0]
    assert sum(layer0.n_by_margin) == N_TOKENS
    assert len(layer0.n_by_margin) == len(layer0.flip_rate_by_margin)


# -- depth trend -----------------------------------------------------------------------------


def test_flip_rate_rising_with_depth_is_detected(tmp_path):
    """T8.2's testable prediction. Error accumulates through the quantized layers beneath the
    router, so deeper layers should flip more; measuring it turns a limitation into a result."""
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    for layer in range(SPEC.n_moe_layers):
        flips[: 10 * (layer + 1), layer] = True
    reference, candidate = make_pair(tmp_path, flips=flips)
    result = compare_traces(reference, candidate)

    assert result.depth_trend() == pytest.approx(1.0)
    assert [l.depth for l in result.layers] == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]
    assert "rises with depth" in interpret(result).action


def test_a_flat_profile_does_not_claim_a_depth_trend(tmp_path):
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[:10, :] = True
    result = compare_traces(*make_pair(tmp_path, flips=flips))
    assert "rises with depth" not in interpret(result).action


# -- comparability guards ----------------------------------------------------------------------


def test_traces_over_different_text_are_refused(tmp_path):
    """Quantization does not change the tokenizer. Differing token ids mean the two runs saw
    different text, and every drift number would be measuring the corpus."""
    reference, _ = make_pair(tmp_path)
    other = tmp_path / "other"
    make_synthetic_trace(other, spec=SPEC, shard_sizes=SHARDS, tokens_per_doc=10,
                         hidden_every=4, seed=99, manifest_overrides={"quant": "Q4_K_M"})
    candidate = TraceReader(other, "synth-moe", "synth-v1")
    with pytest.raises(PrecisionError, match="token ids diverge"):
        compare_traces(reference, candidate)


def test_a_different_expert_count_is_refused(tmp_path):
    reference, _ = make_pair(tmp_path)
    other = tmp_path / "wide"
    make_synthetic_trace(other, spec=TraceSpec(4, 32, 3, 8), shard_sizes=SHARDS,
                         tokens_per_doc=10, hidden_every=4, seed=7,
                         manifest_overrides={"quant": "Q4_K_M"})
    with pytest.raises(PrecisionError, match="n_experts"):
        compare_traces(reference, TraceReader(other, "synth-moe", "synth-v1"))


def test_a_different_token_count_is_refused(tmp_path):
    reference, _ = make_pair(tmp_path)
    other = tmp_path / "short"
    make_synthetic_trace(other, spec=SPEC, shard_sizes=(40, 25), tokens_per_doc=10,
                         hidden_every=4, seed=7, manifest_overrides={"quant": "Q4_K_M"})
    with pytest.raises(PrecisionError, match="different amounts of text"):
        compare_traces(reference, TraceReader(other, "synth-moe", "synth-v1"))


def test_comparing_a_quant_against_itself_is_labelled_not_refused(tmp_path):
    """Legitimate as a nondeterminism floor — it is how you learn what 'zero drift' looks like on
    this hardware. Dangerous as an unlabelled ladder rung, because it will look excellent."""
    reference, candidate = make_pair(tmp_path, ref_quant="Q4_K_M", cand_quant="Q4_K_M")
    result = compare_traces(reference, candidate)
    assert any("BOTH SIDES ARE Q4_K_M" in n for n in result.notes)
    assert any("nondeterminism" in n for n in result.notes)


def test_max_tokens_is_recorded_not_silent(tmp_path):
    reference, candidate = make_pair(tmp_path)
    result = compare_traces(reference, candidate, max_tokens=50)
    assert result.n_tokens == 50
    assert any("first 50 of 100" in n for n in result.notes)


def test_chunking_does_not_change_the_answer(tmp_path):
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[::3, 2] = True
    reference, candidate = make_pair(tmp_path, flips=flips)
    whole = compare_traces(reference, candidate, chunk_tokens=10_000)
    split = compare_traces(reference, candidate, chunk_tokens=7)
    assert whole.mean_set_agreement == pytest.approx(split.mean_set_agreement)
    assert [l.flip_rate for l in whole.layers] == [
        pytest.approx(l.flip_rate) for l in split.layers
    ]


# -- T8.3 bands ------------------------------------------------------------------------------


def _gate_for(tmp_path, flip_fraction):
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    # Each flipped row costs 1/top_k of that row's agreement, so aim the fraction at the metric.
    n_flip = int(round(flip_fraction * N_TOKENS * SPEC.top_k))
    rows = np.arange(N_TOKENS)
    for layer in range(SPEC.n_moe_layers):
        flips[rows[:n_flip], layer] = True
    return interpret(compare_traces(*make_pair(tmp_path, flips=flips)))


def test_high_agreement_is_a_bounded_limitation(tmp_path):
    gate = _gate_for(tmp_path, 0.01)
    assert gate.band == "bounded"
    assert gate.proceed_as_planned
    assert "Proceed as planned" in gate.action


def test_the_middle_band_demands_a_rerun_on_f16(tmp_path):
    gate = _gate_for(tmp_path, 0.05)
    assert gate.band == "prominent"
    assert not gate.proceed_as_planned
    assert "F16" in gate.action and "PROMINENTLY" in gate.action


def test_severe_drift_says_the_paper_changes(tmp_path):
    gate = _gate_for(tmp_path, 0.20)
    assert gate.band == "severe"
    assert "THE PAPER CHANGES" in gate.action
    assert "Do not bury" in gate.action


def test_one_bad_layer_hidden_behind_a_good_mean_is_called_out(tmp_path):
    """A uniform 0.98 and a 0.85 layer under a 0.98 mean are different situations, and only the
    second needs a depth-resolved caveat. The mean alone cannot tell them apart."""
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[:, 3] = True  # layer 3 flips every token: set_agreement 1 - 1/3
    reference, candidate = make_pair(tmp_path, flips=flips)
    gate = interpret(compare_traces(reference, candidate))
    assert gate.worst_layer == 3
    assert gate.mean_set_agreement > gate.worst_set_agreement
    assert "depth-resolved caveat" in gate.action


# -- T8.5 ------------------------------------------------------------------------------------


def test_device_split_passes_on_identical_traces(tmp_path):
    result = check_device_split(compare_traces(*make_pair(tmp_path)))
    assert result.passed
    assert result.floor == DEVICE_SPLIT_FLOOR
    assert "appendix alongside T3.7" in result.verdict


def test_device_split_is_judged_on_the_worst_layer_not_the_mean(tmp_path):
    """One layer landing on a different device and drifting is the exact failure mode. A 16-layer
    mean dilutes it by sixteen, and T8.5's job is to notice it, not to average it away."""
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[:5, 2] = True  # layer 2 only: 5% of rows, worst-layer agreement ~0.983
    comparison = compare_traces(*make_pair(tmp_path, flips=flips))
    assert comparison.mean_set_agreement > 0.99

    result = check_device_split(comparison)
    assert not result.passed
    assert result.worst_layer == 2
    assert "split-dependent" in result.verdict
    assert "Qwen3 and Gemma 4" in result.verdict


# -- report and CLI ---------------------------------------------------------------------------


def test_the_comparison_round_trips_as_json(tmp_path):
    result = compare_traces(*make_pair(tmp_path))
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["task"] == "T8.2"
    assert payload["reference_quant"] == "F16"
    assert payload["candidate_quant"] == "Q4_K_M"
    assert len(payload["layers"]) == SPEC.n_moe_layers
    assert "flip_rate_by_margin" in payload["layers"][0]


def test_cli_ladder_mode(tmp_path, capsys):
    make_pair(tmp_path)
    report = tmp_path / "t8.json"
    rc = main(["--reference-root", str(tmp_path / "ref"), "--candidate-root", str(tmp_path / "cand"),
               "--model", "synth-moe", "--corpus", "synth-v1", "--report", str(report)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "T8.2 mean set agreement 1.000000" in out
    assert "[bounded]" in out
    assert json.loads(report.read_text())["gate"]["task"] == "T8.3"


def test_cli_device_split_mode_uses_the_t8_5_floor(tmp_path, capsys):
    flips = np.zeros((N_TOKENS, SPEC.n_moe_layers), dtype=bool)
    flips[:5, 2] = True
    make_pair(tmp_path, flips=flips)
    rc = main(["--reference-root", str(tmp_path / "ref"), "--candidate-root", str(tmp_path / "cand"),
               "--model", "synth-moe", "--corpus", "synth-v1", "--mode", "device-split"])
    assert rc == 1
    assert "T8.5 FAIL" in capsys.readouterr().out


def test_cli_reports_could_not_run_separately_from_drift(tmp_path, capsys):
    make_pair(tmp_path)
    rc = main(["--reference-root", str(tmp_path / "ref"), "--candidate-root", str(tmp_path / "cand"),
               "--model", "synth-moe", "--corpus", "nonexistent"])
    assert rc == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
