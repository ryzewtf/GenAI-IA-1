"""T0.5 tests — llama-bench parsing, capture overhead, the Phase 5 projection, and Gate Q1.

The measurement itself is `KGPU`. Everything that *consumes* the measurement is arithmetic, and
Gate Q1 is a decision about whether the study collects 1M or 500k tokens per model — so the
arithmetic is worth more test coverage than the measurement.

The inconclusive branch is exercised with the real numbers a workstation run actually produced,
because that branch existing is the whole reason the tool is trustworthy: a first local run
reported full capture as *faster* than no capture at all.
"""

from __future__ import annotations

import csv
import json

import pytest

from src.runtime.throughput import (
    DROP_ORDER,
    GATE_Q1_FRACTION,
    WEEKLY_GPU_HOURS,
    CaptureOverhead,
    ThroughputError,
    append_quota_log,
    evaluate_gate_q1,
    overhead_from_stats,
    parse_llama_bench_json,
    project_phase5,
    write_calibration_csv,
)

PINNED = "7077abbe14c510cb829c93a1328c2815b5805ebd"


def bench_entry(**overrides):
    entry = {
        "build_commit": PINNED[:8],
        "model_filename": "olmoe-Q4_K_M.gguf",
        "n_prompt": 2048,
        "n_gen": 0,
        "avg_ts": 1200.5,
        "stddev_ts": 8.25,
        "n_gpu_layers": 99,
        "n_threads": 4,
        "n_batch": 2048,
        "n_ubatch": 512,
        "split_mode": "layer",
        "tensor_split": "0.50/0.50",
        "backends": "CUDA",
        "gpu_info": "Tesla T4",
    }
    entry.update(overrides)
    return entry


def stats(mode, rate, tokens=20_000, exit_code=0):
    return {"capture_mode": mode, "prefill_tok_per_s": rate, "n_tokens_decoded": tokens,
            "exit_code": exit_code}


# -- llama-bench parsing ------------------------------------------------------------------------


def test_a_normal_run_parses():
    rows = parse_llama_bench_json(json.dumps([bench_entry(), bench_entry(n_prompt=512)]))
    assert len(rows) == 2
    assert rows[0].avg_ts == 1200.5
    assert rows[0].rel_stddev == pytest.approx(8.25 / 1200.5)


def test_numeric_fields_are_accepted_as_strings():
    """llama-bench declares avg_ts as FLOAT but its JSON writer is string-oriented and the exact
    quoting has changed upstream before. Depending on it would make the parser break on a
    llama.cpp bump for no reason."""
    rows = parse_llama_bench_json(json.dumps([bench_entry(avg_ts="1200.5", n_prompt="2048")]))
    assert rows[0].avg_ts == 1200.5
    assert rows[0].n_prompt == 2048


def test_generation_rows_are_dropped_not_averaged_in():
    """The plan pins -n 0 because the study is prefill-only. Token-generation throughput on an MoE
    model is a different number by an order of magnitude — one token per forward, memory-bound —
    and folding one into the projection produces a plausible, badly wrong budget."""
    rows = parse_llama_bench_json(json.dumps([bench_entry(), bench_entry(n_gen=128, avg_ts=45.0)]))
    assert len(rows) == 1
    assert rows[0].avg_ts == 1200.5


def test_a_run_with_only_generation_rows_is_an_error_with_the_fix_in_it():
    with pytest.raises(ThroughputError, match="Re-run llama-bench with -n 0"):
        parse_llama_bench_json(json.dumps([bench_entry(n_gen=128)]))


def test_a_different_build_commit_is_refused():
    """Throughput measured on a different build does not describe the binary that will collect,
    and build.llama_cpp_commit is inside run_config_sha256 precisely because it is load-bearing."""
    payload = json.dumps([bench_entry(build_commit="deadbee")])
    with pytest.raises(ThroughputError, match="different binary"):
        parse_llama_bench_json(payload, expect_commit=PINNED)


def test_an_abbreviated_commit_still_matches_the_pinned_one():
    rows = parse_llama_bench_json(json.dumps([bench_entry()]), expect_commit=PINNED)
    assert rows[0].build_commit == PINNED[:8]


def test_malformed_output_says_what_it_got():
    with pytest.raises(ThroughputError, match="not JSON"):
        parse_llama_bench_json("<html>403</html>")
    with pytest.raises(ThroughputError, match="got dict"):
        parse_llama_bench_json(json.dumps({"avg_ts": 1}))
    with pytest.raises(ThroughputError, match="no rows"):
        parse_llama_bench_json("[]")


# -- capture overhead ---------------------------------------------------------------------------


def test_a_clean_measurement_reports_all_three_ratios():
    overhead = overhead_from_stats({
        "no-callback": [stats("no-callback", 1000.0), stats("no-callback", 1010.0)],
        "filter-off": [stats("filter-off", 900.0), stats("filter-off", 910.0)],
        "full": [stats("full", 800.0), stats("full", 805.0)],
    })
    assert overhead.observability_ratio == pytest.approx(1005.0 / 905.0)
    assert overhead.readback_ratio == pytest.approx(905.0 / 802.5)
    assert overhead.total_ratio == pytest.approx(1005.0 / 802.5)
    assert overhead.conclusive
    assert overhead.total_ratio > 1.0, "a ratio of TIME, so capture costing more must exceed 1"


def test_noise_larger_than_the_effect_is_reported_as_inconclusive():
    """These are the real numbers from a workstation run: full capture came out FASTER than no
    capture at all, which cannot be true. A tool that reported 0.968 here would have fed a
    fabricated speedup straight into Gate Q1."""
    overhead = overhead_from_stats({
        "no-callback": [stats("no-callback", r) for r in (116.404, 169.426, 161.517, 202.193)],
        "filter-off": [stats("filter-off", r) for r in (152.798, 143.025, 150.447, 179.648)],
        "full": [stats("full", r) for r in (177.213, 164.678, 163.253, 195.087)],
    })
    assert overhead.total_ratio < 1.0
    assert not overhead.conclusive
    assert any("INCONCLUSIVE" in n for n in overhead.notes)
    assert overhead.worst_spread > 0.2


def test_the_median_is_used_not_the_mean():
    """One stalled run — a background compile, a page-cache miss — should not move the answer, and
    with three repeats a mean is one outlier away from useless."""
    overhead = overhead_from_stats({
        "no-callback": [stats("no-callback", r) for r in (1000.0, 1000.0, 10.0)],
        "full": [stats("full", 500.0)],
    })
    assert overhead.timings["no-callback"].median == 1000.0


def test_a_failed_run_is_not_timed():
    """A failed capture's wall time measures how long it took to fail."""
    with pytest.raises(ThroughputError, match="exit_code"):
        overhead_from_stats({"full": [stats("full", 800.0, exit_code=6)]})


def test_a_mislabelled_stats_file_is_caught():
    with pytest.raises(ThroughputError, match="filed under"):
        overhead_from_stats({"full": [stats("filter-off", 800.0)]})


def test_modes_that_decoded_different_amounts_of_work_are_refused():
    """A throughput ratio between different token counts is not an overhead. This is the shape of
    a corpus that was edited between legs."""
    with pytest.raises(ThroughputError, match="different token counts"):
        overhead_from_stats({
            "no-callback": [stats("no-callback", 1000.0, tokens=20_000)],
            "full": [stats("full", 800.0, tokens=19_000)],
        })


def test_an_unknown_mode_is_refused():
    with pytest.raises(ThroughputError, match="unknown capture mode"):
        overhead_from_stats({"turbo": [stats("turbo", 1.0)]})


def test_missing_modes_are_noted_and_the_ratio_is_nan_not_one():
    overhead = overhead_from_stats({"full": [stats("full", 800.0)]})
    assert overhead.total_ratio != overhead.total_ratio  # NaN
    assert not overhead.conclusive
    assert any("modes not measured" in n for n in overhead.notes)


# -- projection ---------------------------------------------------------------------------------

RATES = {"a": 1000.0, "b": 500.0, "c": 250.0, "d": 200.0, "e": 2000.0, "fast": 4000.0}


def test_projection_is_hours_of_tokens_over_rate():
    projection = project_phase5(RATES, primary_models=["a"], tokens_per_model=3_600_000)
    assert projection.primary_hours == pytest.approx(1.0)
    assert projection.total_hours == pytest.approx(1.0)


def test_capture_overhead_multiplies_the_projection():
    plain = project_phase5(RATES, primary_models=["a"], tokens_per_model=3_600_000)
    slowed = project_phase5(RATES, primary_models=["a"], tokens_per_model=3_600_000,
                            overhead_ratio=1.5)
    assert slowed.total_hours == pytest.approx(plain.total_hours * 1.5)


def test_additions_and_the_scale_run_are_separable():
    projection = project_phase5(
        RATES, primary_models=["a", "b"], addition_models=["fast"], scale_model="fast",
        tokens_per_model=1_000_000, scale_tokens=4_000_000,
    )
    assert projection.scale_hours == pytest.approx(4 * projection.per_model_hours["fast"])
    assert projection.total_hours == pytest.approx(
        projection.primary_hours + projection.additions_hours + projection.scale_hours
    )


def test_a_model_with_no_measurement_is_refused_rather_than_defaulted():
    """The plan's exact words are 'do not guess it, measure it'. A default here would produce a
    complete-looking budget with an invented term in it."""
    with pytest.raises(ThroughputError, match="measured, not guessed"):
        project_phase5(RATES, primary_models=["a", "missing"])


def test_nonsense_inputs_are_refused():
    with pytest.raises(ThroughputError, match="overhead_ratio"):
        project_phase5(RATES, primary_models=["a"], overhead_ratio=0.0)
    with pytest.raises(ThroughputError, match="tokens_per_model"):
        project_phase5(RATES, primary_models=["a"], tokens_per_model=0)
    with pytest.raises(ThroughputError, match="non-positive"):
        project_phase5({"a": 0.0}, primary_models=["a"])


# -- Gate Q1 ------------------------------------------------------------------------------------


def test_a_comfortable_budget_passes_and_says_so():
    gate = evaluate_gate_q1({"a": 100_000.0}, primary_models=["a"], tokens_per_model=1_000_000)
    assert gate.passed
    assert gate.reduced is None
    assert "PASS" in gate.verdict and "1,000,000 tokens per model" in gate.verdict
    assert gate.budget_hours == pytest.approx(WEEKLY_GPU_HOURS * GATE_Q1_FRACTION)


def test_an_over_budget_projection_costs_the_prescribed_cut():
    """Gate Q1 does not invent a remedy — the plan already fixed it at 1M -> 500k. What the tool
    adds is whether that cut is actually sufficient, because if it is not, the drop list is the
    next lever and someone has to choose deliberately."""
    gate = evaluate_gate_q1({"a": 10.0}, primary_models=["a"], tokens_per_model=1_000_000)
    assert not gate.passed
    assert gate.reduced is not None
    assert gate.reduced.total_hours == pytest.approx(gate.projection.total_hours / 2)
    assert "FAIL" in gate.verdict
    assert "500,000 tokens per model" in gate.verdict
    assert "T9.4 sample-size sensitivity curve" in gate.verdict


def test_a_cut_that_is_still_not_enough_surfaces_the_drop_order():
    gate = evaluate_gate_q1({"a": 1.0}, primary_models=["a"], tokens_per_model=1_000_000)
    assert not gate.passed
    assert "still exceeds the budget" in gate.verdict
    assert gate.drop_order == DROP_ORDER
    assert DROP_ORDER[0] in gate.verdict
    assert "Do not cut the panel's five primary models" in gate.verdict


def test_a_cut_that_is_enough_says_it_fits():
    # Just over budget: halving comfortably clears it.
    hours_budget = WEEKLY_GPU_HOURS * GATE_Q1_FRACTION
    rate = 1_000_000 / ((hours_budget * 1.2) * 3600)
    gate = evaluate_gate_q1({"a": rate}, primary_models=["a"], tokens_per_model=1_000_000)
    assert not gate.passed
    assert "which fits." in gate.verdict


def test_the_gate_round_trips_as_json():
    gate = evaluate_gate_q1({"a": 10.0}, primary_models=["a"])
    payload = json.loads(json.dumps(gate.to_json()))
    assert payload["gate"] == "Q1"
    assert payload["passed"] is False
    assert payload["reduced_projection"]["tokens_per_model"] == 500_000
    assert payload["projection"]["total_hours"] > payload["budget_hours"]


# -- artifacts ----------------------------------------------------------------------------------


def test_calibration_csv_has_a_row_per_bench_row_and_overhead_as_comments(tmp_path):
    rows = parse_llama_bench_json(json.dumps([bench_entry(), bench_entry(n_prompt=512)]))
    overhead = overhead_from_stats({
        "no-callback": [stats("no-callback", 1000.0)],
        "filter-off": [stats("filter-off", 900.0)],
        "full": [stats("full", 800.0)],
    })
    path = write_calibration_csv(tmp_path / "throughput_calibration.csv", rows, overhead)
    text = path.read_text(encoding="utf-8")

    assert text.startswith("# capture overhead")
    body = [line for line in text.splitlines() if not line.startswith("#")]
    parsed = list(csv.DictReader(body))
    assert len(parsed) == 2
    assert parsed[0]["model"] == "olmoe-Q4_K_M.gguf"
    assert "n_gen" not in parsed[0], "n_gen is always 0 here; a column of zeros invites confusion"


def test_an_inconclusive_overhead_says_so_in_the_csv(tmp_path):
    overhead = overhead_from_stats({
        "no-callback": [stats("no-callback", r) for r in (116.4, 202.2)],
        "filter-off": [stats("filter-off", r) for r in (152.8, 179.6)],
        "full": [stats("full", r) for r in (177.2, 195.1)],
    })
    path = write_calibration_csv(tmp_path / "c.csv",
                                parse_llama_bench_json(json.dumps([bench_entry()])), overhead)
    text = path.read_text(encoding="utf-8")
    assert "conclusive=False" in text
    assert "INCONCLUSIVE" in text


def test_quota_log_appends_and_keeps_one_header(tmp_path):
    """The weekly allowance is a rolling budget, so a rewritten log is a lost budget."""
    path = tmp_path / "quota_log.csv"
    append_quota_log(path, session_id="s1", notebook="setup", gpu_hours=0.5,
                     tokens_processed=0, purpose="build", date="2026-08-18")
    append_quota_log(path, session_id="s2", notebook="collect", gpu_hours=3.25,
                     tokens_processed=1_000_000, purpose="T5.2 olmoe", date="2026-08-19")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert [r["session_id"] for r in rows] == ["s1", "s2"]
    assert rows[1]["gpu_hours"] == "3.25"
    assert path.read_text(encoding="utf-8").count("session_id,date") == 1


def test_an_empty_overhead_is_not_conclusive():
    assert not CaptureOverhead().conclusive
