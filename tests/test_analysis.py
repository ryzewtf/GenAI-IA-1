"""Phase 9 tests — result documents in, tables and the reconciliation out (plan T9.1–T9.6).

Two things are being defended here, and they are different in kind.

**The tables (T9.1–T9.4, T9.6) are a reporting boundary**, so the tests are about what survives the
crossing: a checkpoint that never produced a document must not quietly become a one-model table, a
layer a killed session never reached must not read as a floor of zeros, and a negative ``Î`` must
render as a negative number. Invariant I8 is the sharp one — the signed lower bound is *never*
clamped — so it is asserted at every renderer, not once at the source.

**The reconciliation (T9.5) is the study's central claim**, so its tests are hand-computed. The
field's "28% vs 99%" is a metric mismatch: a consecutive-token repetition rate against a cache-hit
rate at capacity. Every number this module puts on a shared axis is derived on paper in a comment
above the assertion, so a wrong formula fails the test rather than re-pinning itself.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.analysis.reconcile import (
    DEFINITIONAL_AXES,
    LITERATURE_FIGURES,
    UNAVAILABLE_ANCHORS,
    LiteratureFigure,
    reconcile,
)
from src.analysis.tables import (
    MISSING_MARKER,
    METRICS,
    MissingCell,
    Table,
    cell_metric,
    confound_table,
    expected_shape_check,
    load_results,
    negative_mi_cells,
    normalized_depth,
    pair_table,
    primary_table,
)
from src.probes.features import CorpusIndex, consecutive_repetition_rate
from src.traces.format import TraceSpec
from src.traces.reader import TraceReader
from src.traces.synth import make_synthetic_trace

SHA_A = "a" * 64
SHA_B = "b" * 64

# ==================================================================================================
# Fixture construction — result documents in exactly the shape src.probes.train.sweep writes.
# ==================================================================================================


def family_b(entropy_bits: float, ce_bits: float, n_experts: int) -> dict:
    """A family_b payload that is arithmetically consistent: mi_bits IS H − CE (invariant I8).

    ``load_results`` re-derives the difference and refuses a document where it disagrees, so a
    fixture that fudged this would be rejected before any table saw it.
    """
    mi = entropy_bits - ce_bits
    return {
        "entropy_bits": entropy_bits,
        "cross_entropy_bits": ce_bits,
        "mi_bits": mi,
        "ratio": mi / entropy_bits if entropy_bits else None,
        "ce_normalized": ce_bits / math.log2(n_experts),
        "entropy_normalized": entropy_bits / math.log2(n_experts),
        "negative": mi < 0.0,
        "n_experts": n_experts,
        "estimator": "miller_madow",
        "split": "test",
    }


def family_a(top_k: int, n_experts: int, *, agreement: float, exact: float, recall: dict) -> dict:
    out = {
        "n": 1000,
        "top_k": top_k,
        "n_experts": n_experts,
        "set_agreement@k": agreement,
        "exact_match": exact,
    }
    out.update({f"recall@{m}": v for m, v in recall.items()})
    return out


def ok_cell(
    layer: int,
    *,
    top_k: int,
    n_experts: int,
    entropy_bits: float = 6.0,
    ce_bits: float = 4.5,
    agreement: float = 0.5,
    exact: float = 0.2,
    recall: dict | None = None,
    model_layer: int | None = None,
    extra: dict | None = None,
) -> dict:
    if recall is None:
        recall = {top_k: agreement, 2 * top_k: min(1.0, agreement + 0.2), 4 * top_k: 0.99}
    record = {
        "layer": layer,
        "model_layer": layer if model_layer is None else model_layer,
        "status": "ok",
        "fit": None,
        "metrics": {
            "split": "test",
            "n_rows": 1000,
            "n_slots": 1000 * top_k,
            "family_a": family_a(
                top_k, n_experts, agreement=agreement, exact=exact, recall=recall
            ),
            "family_b": family_b(entropy_bits, ce_bits, n_experts),
        },
        "rows": {"train": 8000, "val": 1000, "test": 1000},
        "excluded": {"train": 0, "val": 0, "test": 0, "reason": None},
        "meta": {},
    }
    record.update(extra or {})
    return record


def skipped_cell(layer: int, reason: str = "F2 is undefined at layer 0") -> dict:
    return {"layer": layer, "model_layer": layer, "status": "skipped", "reason": reason}


def doc(
    model: str,
    feature: str,
    records: list,
    *,
    n_experts: int = 64,
    top_k: int = 8,
    n_moe_layers: int = 4,
    sha: str = SHA_A,
    logit_tensor: str | None = "ffn_moe_probs",
    shard_ids: tuple = (0, 1, 2),
) -> dict:
    return {
        "model": model,
        "feature": feature,
        "run_config_sha256": sha,
        "logit_tensor_used": logit_tensor,
        "n_moe_layers": n_moe_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "shard_ids": list(shard_ids),
        "entropy_estimator": "miller_madow",
        "layers": {str(r["layer"]): r for r in records},
    }


def write_docs(out_dir: Path, docs: list) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in docs:
        path = out_dir / f"{d['model']}__{d['feature']}.json"
        path.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return out_dir


# --------------------------------------------------------------------------------------------
# The panel used by most table tests.
#
# olmoe : 64 experts, top-8, 4 MoE layers. F0 is the marginal; F1 beats it at layer 0 and is
#         NEGATIVE at layer 1 (an overfit probe — a required output, §1.2/I8). Layers 2 and 3
#         were never reached: a killed session, not a shorter model.
# gptoss: 32 experts, top-4, 2 MoE layers. Matched 12.5% activated density, different granularity.
#
# Hand-computed mi_bits, from H − CE:
#   olmoe/F0 layer 0: 6.0 − 6.0 = 0.0      olmoe/F1 layer 0: 6.0 − 4.5 = +1.5
#   olmoe/F0 layer 1: 6.0 − 6.0 = 0.0      olmoe/F1 layer 1: 6.0 − 6.5 = −0.5
#   => mean over the two layers that have a value: (1.5 + (−0.5)) / 2 = 0.5
# --------------------------------------------------------------------------------------------

OLMOE = {"n_experts": 64, "top_k": 8, "n_moe_layers": 4}
GPTOSS = {"n_experts": 32, "top_k": 4, "n_moe_layers": 2}
# The same geometry without n_moe_layers, for cell records (a cell has no layer count).
OLMOE_CELL = {"n_experts": 64, "top_k": 8}
GPTOSS_CELL = {"n_experts": 32, "top_k": 4}


@pytest.fixture
def panel(tmp_path: Path) -> Path:
    return write_docs(
        tmp_path / "results",
        [
            doc(
                "olmoe",
                "F0",
                [
                    ok_cell(0, **OLMOE_CELL, entropy_bits=6.0, ce_bits=6.0, agreement=0.125),
                    ok_cell(1, **OLMOE_CELL, entropy_bits=6.0, ce_bits=6.0, agreement=0.125),
                ],
                **OLMOE,
            ),
            doc(
                "olmoe",
                "F1",
                [
                    ok_cell(0, **OLMOE_CELL, entropy_bits=6.0, ce_bits=4.5, agreement=0.5),
                    ok_cell(1, **OLMOE_CELL, entropy_bits=6.0, ce_bits=6.5, agreement=0.3),
                ],
                **OLMOE,
            ),
            doc(
                "gptoss",
                "F0",
                [
                    ok_cell(0, **GPTOSS_CELL, entropy_bits=5.0, ce_bits=5.0, agreement=0.125),
                    ok_cell(1, **GPTOSS_CELL, entropy_bits=5.0, ce_bits=5.0, agreement=0.125),
                ],
                **GPTOSS,
            ),
            doc(
                "gptoss",
                "F1",
                [
                    ok_cell(0, **GPTOSS_CELL, entropy_bits=5.0, ce_bits=4.0, agreement=0.6),
                    ok_cell(1, **GPTOSS_CELL, entropy_bits=5.0, ce_bits=3.0, agreement=0.7),
                ],
                **GPTOSS,
            ),
        ],
    )


MODELS_CONFIG = {
    "models": {
        "olmoe": {
            "checkpoint_status": "base",
            "shared_experts": 0,
            "parallel_dense_mlp": False,
            "aux_loss": {"key": "router_aux_loss_coef", "value": 0.01},
        },
        "gptoss": {
            "checkpoint_status": "instruct",
            "shared_experts": 0,
            "parallel_dense_mlp": False,
            "aux_loss": {},
        },
    },
    "pairs": {
        "pair_g": {
            "members": ["gptoss", "olmoe"],
            "holds_fixed": ["activated_density_12.5pct"],
            "not_matched": ["post_training", "depth", "vocabulary"],
            "framing": "matched density, with these differences documented",
            "controls": ["olmoe-instruct"],
        },
        "pair_broken": {"members": ["olmoe"], "framing": "one member"},
        "pair_absent": {"members": ["olmoe", "gemma"], "framing": "a checkpoint that never ran"},
    },
}


# ==================================================================================================
# load_results — the merge gate (I2, I3, I13) and the anti-clamp check (I8)
# ==================================================================================================


def test_a_results_directory_loads_into_one_model_per_name(panel):
    resultset = load_results(panel)
    assert resultset.model_names == ["gptoss", "olmoe"]
    assert resultset.features() == ["F0", "F1"]
    assert resultset.run_config_sha256 == SHA_A
    assert resultset.models["olmoe"].features == ["F0", "F1"]


def test_a_single_shard_and_a_merged_shard_document_load_the_same_way(tmp_path):
    """A one-shard checkpoint and a three-shard merge differ only in shard_ids.

    The shard count is provenance, not a metric: nothing downstream may branch on it, so the two
    must produce identical tables. This is the case a resumed collection actually produces —
    OLMoE finished in one shard, the model that was killed and restarted has three.
    """
    single = write_docs(
        tmp_path / "single",
        [doc("olmoe", "F1", [ok_cell(0, **OLMOE_CELL)], **OLMOE, shard_ids=(0,))],
    )
    merged = write_docs(
        tmp_path / "merged",
        [doc("olmoe", "F1", [ok_cell(0, **OLMOE_CELL)], **OLMOE, shard_ids=(0, 1, 2))],
    )
    a, b = load_results(single), load_results(merged)
    assert a.models["olmoe"].shard_ids == [0]
    assert b.models["olmoe"].shard_ids == [0, 1, 2]
    assert primary_table(a, metric="mi_bits").to_csv() == primary_table(b, metric="mi_bits").to_csv()


def test_two_run_configs_in_one_directory_are_refused(tmp_path):
    """I2/I3 — merging two run configs silently averages two experiments."""
    out = write_docs(
        tmp_path / "r",
        [
            doc("olmoe", "F0", [ok_cell(0, **OLMOE_CELL)], **OLMOE, sha=SHA_A),
            doc("gptoss", "F0", [ok_cell(0, **GPTOSS_CELL)], **GPTOSS, sha=SHA_B),
        ],
    )
    with pytest.raises(ValueError, match="two run configs"):
        load_results(out)


def test_two_run_configs_for_one_model_are_refused_even_when_the_check_is_relaxed(tmp_path):
    """Within one model the documents describe shards of ONE trace; a mismatch is always fatal."""
    out = write_docs(
        tmp_path / "r",
        [
            doc("olmoe", "F0", [ok_cell(0, **OLMOE_CELL)], **OLMOE, sha=SHA_A),
            doc("olmoe", "F1", [ok_cell(0, **OLMOE_CELL)], **OLMOE, sha=SHA_B),
        ],
    )
    with pytest.raises(ValueError, match="two run configs"):
        load_results(out, require_uniform_run_config=False)


def test_relaxing_the_check_permits_two_models_from_two_run_configs(tmp_path):
    out = write_docs(
        tmp_path / "r",
        [
            doc("olmoe", "F0", [ok_cell(0, **OLMOE_CELL)], **OLMOE, sha=SHA_A),
            doc("gptoss", "F0", [ok_cell(0, **GPTOSS_CELL)], **GPTOSS, sha=SHA_B),
        ],
    )
    resultset = load_results(out, require_uniform_run_config=False)
    assert resultset.model_names == ["gptoss", "olmoe"]
    assert resultset.run_config_sha256 is None  # no single config describes this set


def test_disagreeing_trace_geometry_for_one_model_is_refused(tmp_path):
    out = write_docs(
        tmp_path / "r",
        [
            doc("olmoe", "F0", [ok_cell(0, **OLMOE_CELL)], **OLMOE),
            doc(
                "olmoe",
                "F1",
                [ok_cell(0, top_k=8, n_experts=64)],
                n_experts=64,
                top_k=8,
                n_moe_layers=16,
            ),
        ],
    )
    with pytest.raises(ValueError, match="[Tt]wo traces have been swept"):
        load_results(out)


def test_a_different_selection_tensor_is_a_different_experiment(tmp_path):
    """I13 — logit_tensor_used names the label stream; two values are two experiments."""
    out = write_docs(
        tmp_path / "r",
        [
            doc("olmoe", "F0", [ok_cell(0, **OLMOE_CELL)], **OLMOE, logit_tensor="ffn_moe_probs"),
            doc("olmoe", "F1", [ok_cell(0, **OLMOE_CELL)], **OLMOE, logit_tensor="ffn_moe_logits"),
        ],
    )
    with pytest.raises(ValueError, match="logit_tensor_used differs"):
        load_results(out)


def test_a_clamped_mi_bits_is_refused_at_the_reporting_boundary(tmp_path):
    """I8 — an upstream clamp shows up here as arithmetic that does not add up, not as a table."""
    cell = ok_cell(0, **OLMOE_CELL, entropy_bits=6.0, ce_bits=6.5)
    assert cell["metrics"]["family_b"]["mi_bits"] == pytest.approx(-0.5)
    cell["metrics"]["family_b"]["mi_bits"] = 0.0  # the clamp a well-meaning script would apply
    out = write_docs(tmp_path / "r", [doc("olmoe", "F1", [cell], **OLMOE)])
    with pytest.raises(ValueError, match="never\n?\\s*clamped|is not\\s"):
        load_results(out)


def test_a_nan_mi_bits_survives_the_consistency_check(tmp_path):
    """A deterministic router gives H = 0 and a NaN ratio; NaN == NaN must not read as a clamp."""
    cell = ok_cell(0, **OLMOE_CELL, entropy_bits=float("nan"), ce_bits=float("nan"))
    out = write_docs(tmp_path / "r", [doc("olmoe", "F1", [cell], **OLMOE)])
    resultset = load_results(out)
    assert math.isnan(resultset.models["olmoe"].cells["F1"][0]["metrics"]["family_b"]["mi_bits"])


# ==================================================================================================
# Missing cells — a blank is not a zero
# ==================================================================================================


def test_a_layer_a_killed_session_never_reached_is_missing_not_zero(panel):
    """olmoe declares 4 MoE layers and the documents carry 2. Layers 2-3 are holes."""
    model = load_results(panel).models["olmoe"]
    assert model.expected_layers() == [0, 1, 2, 3]
    hole = model.cell("F1", 3)
    assert isinstance(hole, MissingCell) and hole.kind == "not_reached"
    assert str(hole) == MISSING_MARKER != "0"


def test_a_feature_with_no_document_is_missing_not_zero(panel):
    model = load_results(panel).models["olmoe"]
    hole = model.cell("F4", 0)
    assert isinstance(hole, MissingCell) and hole.kind == "feature_absent"


def test_an_undefined_feature_skip_is_reported_as_skipped_with_its_reason(tmp_path):
    out = write_docs(
        tmp_path / "r",
        [doc("olmoe", "F2", [skipped_cell(0), ok_cell(1, **OLMOE_CELL)], **OLMOE)],
    )
    model = load_results(out).models["olmoe"]
    hole = model.cell("F2", 0)
    assert isinstance(hole, MissingCell) and hole.kind == "skipped"
    assert "undefined at layer 0" in hole.reason


def test_a_metric_absent_from_a_cell_that_ran_is_missing_not_zero(panel):
    model = load_results(panel).models["olmoe"]
    value = cell_metric(model, "F1", 0, "recall@4k")
    assert value == pytest.approx(0.99)
    stripped = load_results(panel).models["gptoss"]
    del stripped.cells["F1"][0]["metrics"]["family_a"]["exact_match"]
    hole = cell_metric(stripped, "F1", 0, "exact_match")
    assert isinstance(hole, MissingCell) and hole.kind == "metric_absent"


def test_every_renderer_agrees_that_a_missing_cell_is_not_a_number():
    table = Table(
        name="t",
        columns=["model", "value"],
        rows=[["olmoe", MissingCell("not_reached", "killed session")], ["gptoss", 0.0]],
    )
    assert table.n_missing == 1
    assert table.missing_cells() == [
        {"row": 0, "column": "value", "kind": "not_reached", "reason": "killed session"}
    ]
    assert f"| olmoe | {MISSING_MARKER} |" in table.to_markdown()
    assert "olmoe,--\r\n" not in table.to_csv()  # csv writer must not add its own line ending
    assert table.to_csv().splitlines()[1] == "olmoe,--"
    assert table.to_csv().splitlines()[2] == "gptoss,0.0"
    assert table.to_json()["rows"][0][1] == {"missing": "not_reached", "reason": "killed session"}
    assert table.to_json()["rows"][1][1] == 0.0


def test_a_row_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="has 1 cells against 2 columns"):
        Table(name="t", columns=["a", "b"], rows=[["only-one"]])


def test_a_pipe_in_a_cell_cannot_break_the_markdown_grid():
    table = Table(name="t", columns=["framing"], rows=[["matched | documented"]])
    body = table.to_markdown().splitlines()[4]
    assert body == "| matched \\| documented |"


def test_floats_render_without_rounding():
    """The module never rounds a value it renders — repr round-trips exactly."""
    value = 0.1 + 0.2
    table = Table(name="t", columns=["v"], rows=[[value]])
    assert repr(value) in table.to_markdown()
    assert float(table.to_csv().splitlines()[1]) == value


# ==================================================================================================
# T9.1 primary table
# ==================================================================================================


def test_the_mean_reduction_averages_only_the_layers_that_have_a_value(panel):
    # olmoe/F1 mi_bits: layer 0 = 6.0 − 4.5 = +1.5, layer 1 = 6.0 − 6.5 = −0.5.
    # Layers 2 and 3 were never reached, so the mean is over 2 values: (1.5 − 0.5)/2 = 0.5.
    # Dividing by the 4 expected layers instead would give 0.25 — the imputation this forbids.
    table = primary_table(load_results(panel), metric="mi_bits", layer_reduction="mean")
    row = next(r for r in table.rows if r[0] == "olmoe")
    assert row[table.columns.index("F1")] == pytest.approx(0.5)
    assert table.meta["coverage"]["olmoe::F1"] == {"n_used": 2, "n_expected": 4}


def test_a_partially_covered_cell_is_listed_so_the_mean_is_not_read_as_complete(panel):
    table = primary_table(load_results(panel), metric="mi_bits")
    partial = {(c["model"], c["feature"]) for c in table.meta["partial_cells"]}
    assert ("olmoe", "F1") in partial
    assert ("gptoss", "F1") not in partial  # gptoss declares 2 layers and has both


def test_a_cell_with_no_usable_layer_renders_as_missing_not_as_zero(tmp_path):
    out = write_docs(
        tmp_path / "r", [doc("olmoe", "F2", [skipped_cell(0), skipped_cell(1)], **OLMOE)]
    )
    table = primary_table(load_results(out), metric="mi_bits")
    assert isinstance(table.rows[0][table.columns.index("F2")], MissingCell)
    assert table.n_missing == 1
    assert MISSING_MARKER in table.to_markdown()


def test_a_model_that_never_produced_results_is_a_hard_error_not_a_shorter_table(panel):
    """A checkpoint that failed collection must not silently drop out of a cross-model table."""
    resultset = load_results(panel)
    with pytest.raises(KeyError, match="gemma"):
        primary_table(resultset, models=["olmoe", "gemma"])
    with pytest.raises(KeyError, match="gemma"):
        resultset.require(["gemma"])


def test_an_unknown_metric_or_reduction_is_refused(panel):
    resultset = load_results(panel)
    with pytest.raises(ValueError, match="unknown metric"):
        primary_table(resultset, metric="accuracy")
    with pytest.raises(ValueError, match="unknown layer_reduction"):
        primary_table(resultset, layer_reduction="median")


def test_recall_at_k_resolves_against_each_model_s_own_top_k(panel):
    """recall@k is recall@8 on OLMoE and recall@4 on GPT-OSS; one column, two budgets."""
    resultset = load_results(panel)
    olmoe, gptoss = resultset.models["olmoe"], resultset.models["gptoss"]
    olmoe.cells["F1"][0]["metrics"]["family_a"] = family_a(
        8, 64, agreement=0.5, exact=0.2, recall={8: 0.5, 16: 0.7, 32: 0.99}
    )
    gptoss.cells["F1"][0]["metrics"]["family_a"] = family_a(
        4, 32, agreement=0.6, exact=0.3, recall={4: 0.6, 8: 0.8, 16: 0.95}
    )
    assert cell_metric(olmoe, "F1", 0, "recall@k") == pytest.approx(0.5)
    assert cell_metric(olmoe, "F1", 0, "recall@4k") == pytest.approx(0.99)
    assert cell_metric(gptoss, "F1", 0, "recall@k") == pytest.approx(0.6)
    assert cell_metric(gptoss, "F1", 0, "recall@4k") == pytest.approx(0.95)


def test_per_layer_rows_carry_the_model_layer_alongside_the_trace_layer(tmp_path):
    """DeepSeek's trace layer i is model layer i+1; the trace index is the axis (moe_layer_offset)."""
    out = write_docs(
        tmp_path / "r",
        [
            doc(
                "deepseek",
                "F1",
                [ok_cell(0, top_k=6, n_experts=64, model_layer=1)],
                n_experts=64,
                top_k=6,
                n_moe_layers=2,
            )
        ],
    )
    table = primary_table(load_results(out), metric="mi_bits", layer_reduction="per_layer")
    assert table.columns[:3] == ["model", "layer", "model_layer"]
    assert table.rows[0][:3] == ["deepseek", 0, 1]
    # Layer 1 was never reached, so even its model_layer is unknown — not guessed as 2.
    assert isinstance(table.rows[1][2], MissingCell)


def test_normalized_depth_is_the_only_cross_model_depth_axis():
    # Mixtral layer 15 of 32 sits at 15/31; OLMoE layer 15 of 16 sits at 15/15 = the last layer.
    assert normalized_depth(15, 32) == pytest.approx(15 / 31)
    assert normalized_depth(15, 16) == 1.0
    assert normalized_depth(0, 16) == 0.0
    # A single-MoE-layer model has no interior depth: 0.0, not a ZeroDivisionError.
    assert normalized_depth(0, 1) == 0.0


def test_depth_bins_put_a_shallow_and_a_deep_model_on_the_same_rows(tmp_path):
    """A 4-layer and a 16-layer model share bin labels; n_layers_in_bin differs by construction."""
    shallow = doc("shallow", "F1", [ok_cell(i, **OLMOE_CELL) for i in range(4)], **OLMOE)
    deep_cells = [ok_cell(i, top_k=8, n_experts=64) for i in range(16)]
    deep = doc("deep", "F1", deep_cells, n_experts=64, top_k=8, n_moe_layers=16)
    out = write_docs(tmp_path / "r", [shallow, deep])
    table = primary_table(
        load_results(out), metric="mi_bits", layer_reduction="per_normalized_depth", depth_bins=4
    )
    labels = {r[0]: [] for r in table.rows}
    for r in table.rows:
        labels[r[0]].append(r[1])
    assert labels["shallow"] == labels["deep"]
    counts = {(r[0], r[1]): r[3] for r in table.rows}
    # shallow: depths 0, 1/3, 2/3, 1 -> bins 0, 1, 2, 3 -> one layer each.
    assert [counts[("shallow", lab)] for lab in labels["shallow"]] == [1, 1, 1, 1]
    # deep: 16 layers over 4 bins, and the last bin is closed so it takes the l=15 endpoint.
    assert sum(counts[("deep", lab)] for lab in labels["deep"]) == 16


def test_an_empty_depth_bin_is_missing_rather_than_absent(tmp_path):
    """With more bins than layers some rows have nothing to average; they must still appear."""
    out = write_docs(
        tmp_path / "r",
        [doc("olmoe", "F1", [ok_cell(0, **OLMOE_CELL), ok_cell(1, **OLMOE_CELL)], **OLMOE)],
    )
    table = primary_table(
        load_results(out), metric="mi_bits", layer_reduction="per_normalized_depth", depth_bins=8
    )
    assert len(table.rows) == 8
    assert table.n_missing > 0


def test_the_primary_table_surfaces_the_selection_tensor_per_model(panel):
    notes = " ".join(primary_table(load_results(panel)).notes)
    assert "logit_tensor_used" in notes and "ffn_moe_probs" in notes


# ==================================================================================================
# T9.1 negative Î — invariant I8, at every renderer
# ==================================================================================================


def test_a_negative_mi_bits_reaches_the_rendered_table_as_a_negative_number(panel):
    """I8: the signed lower bound is NEVER clamped, so the table shows −0.5, not 0.0 and not '--'."""
    table = primary_table(
        load_results(panel), metric="mi_bits", layer_reduction="per_layer", models=["olmoe"]
    )
    cell = table.rows[1][table.columns.index("F1")]
    assert cell == pytest.approx(-0.5)
    assert repr(-0.5) in table.to_markdown()
    assert "-0.5" in table.to_csv()
    assert table.to_json()["rows"][1][table.columns.index("F1")] == pytest.approx(-0.5)


def test_a_negative_ratio_survives_the_mean_reduction_without_a_floor(panel):
    # olmoe/F1 ratio: layer 0 = 1.5/6 = 0.25, layer 1 = −0.5/6 = −0.0833...
    # mean = (0.25 − 1/12)/2 = 1/12 = 0.08333...; clamping layer 1 at zero would give 0.125.
    table = primary_table(load_results(panel), metric="mi_ratio", layer_reduction="mean")
    row = next(r for r in table.rows if r[0] == "olmoe")
    assert row[table.columns.index("F1")] == pytest.approx(1 / 12)


def test_negative_mi_cells_reports_the_count_with_its_denominator(panel):
    """T9.1 asks for the COUNT of negative Î cells; a count without a denominator is unreadable."""
    negatives = negative_mi_cells(load_results(panel))
    assert negatives.count == 1
    assert negatives.by_model == {"olmoe": 1}
    assert negatives.by_feature == {"F1": 1}
    # 4 documents; olmoe declares 4 layers with 2 present, gptoss declares 2 with 2 present.
    assert negatives.n_cells_examined == 2 * 2 + 2 * 2
    assert negatives.n_missing_cells == 2 * 2  # olmoe layers 2-3, for each of F0 and F1
    assert negatives.cells[0]["mi_bits"] == pytest.approx(-0.5)


def test_the_negative_mi_listing_renders_the_measured_value_not_an_absolute(panel):
    table = negative_mi_cells(load_results(panel)).to_table()
    assert table.rows[0][table.columns.index("mi_bits")] == pytest.approx(-0.5)
    assert repr(-0.5) in table.to_markdown()
    assert "1 of 8 evaluated cells" in table.to_markdown()


def test_the_negative_mi_object_offers_no_clamped_view(panel):
    """There is deliberately no clamp/floor/abs on this path, and the object is frozen."""
    negatives = negative_mi_cells(load_results(panel))
    with pytest.raises(dataclasses.FrozenInstanceError):
        negatives.cells = ()
    assert not [n for n in dir(negatives) if "clamp" in n or "clip" in n or "floor" in n]
    assert not [m for m in METRICS if "clamp" in m]


def test_a_nan_mi_bits_is_not_counted_as_negative(tmp_path):
    out = write_docs(
        tmp_path / "r",
        [
            doc(
                "olmoe",
                "F1",
                [ok_cell(0, **OLMOE_CELL, entropy_bits=float("nan"), ce_bits=float("nan"))],
                **OLMOE,
            )
        ],
    )
    assert negative_mi_cells(load_results(out)).count == 0


# ==================================================================================================
# T9.2 / T9.3 pair tables
# ==================================================================================================


def test_the_pair_framing_travels_in_every_row_not_in_a_caption(panel):
    """§1.5: no pair but A is controlled, so a reader copying a delta copies the qualification."""
    table = pair_table(load_results(panel), "pair_g", MODELS_CONFIG, metric="mi_bits")
    assert {"holds_fixed", "not_matched", "framing"} <= set(table.columns)
    for row in table.rows:
        assert row[table.columns.index("framing")] == (
            "matched density, with these differences documented"
        )
        assert "post_training" in row[table.columns.index("not_matched")]
    assert "controlled" not in table.meta["framing"]


def test_the_pair_delta_is_the_second_member_minus_the_first(panel):
    # pair_g members are [gptoss, olmoe] in that order.
    # mi_bits mean: gptoss/F1 = ((5.0−4.0) + (5.0−3.0))/2 = 1.5 ; olmoe/F1 = 0.5 (above).
    # delta_b_minus_a = olmoe − gptoss = 0.5 − 1.5 = −1.0.
    table = pair_table(load_results(panel), "pair_g", MODELS_CONFIG, metric="mi_bits")
    row = next(r for r in table.rows if r[table.columns.index("feature")] == "F1")
    assert row[table.columns.index("gptoss (mi_bits)")] == pytest.approx(1.5)
    assert row[table.columns.index("olmoe (mi_bits)")] == pytest.approx(0.5)
    assert row[table.columns.index("delta_b_minus_a")] == pytest.approx(-1.0)


def test_a_pair_with_an_absent_member_is_refused(panel):
    with pytest.raises(KeyError, match="gemma"):
        pair_table(load_results(panel), "pair_absent", MODELS_CONFIG)


def test_a_pair_that_is_not_two_models_is_refused(panel):
    with pytest.raises(ValueError, match="has 1 members"):
        pair_table(load_results(panel), "pair_broken", MODELS_CONFIG)
    with pytest.raises(KeyError, match="no pair"):
        pair_table(load_results(panel), "pair_z", MODELS_CONFIG)


def test_a_delta_needs_both_members_and_is_missing_otherwise(tmp_path):
    """One member reached a feature the other never did — the delta is a hole, not a value."""
    out = write_docs(
        tmp_path / "r",
        [
            doc("gptoss", "F1", [ok_cell(0, **GPTOSS_CELL)], **GPTOSS),
            doc("olmoe", "F1", [skipped_cell(0), skipped_cell(1)], **OLMOE),
        ],
    )
    table = pair_table(load_results(out), "pair_g", MODELS_CONFIG, metric="mi_bits")
    row = table.rows[0]
    assert isinstance(row[table.columns.index("olmoe (mi_bits)")], MissingCell)
    assert isinstance(row[table.columns.index("delta_b_minus_a")], MissingCell)
    assert table.n_missing >= 2


def test_the_tokenizer_caveat_is_attached_to_pair_g_and_pair_t(panel):
    notes = " ".join(pair_table(load_results(panel), "pair_g", MODELS_CONFIG).notes)
    assert "tokenizer" in notes.lower()
    assert "word-level control" in notes
    assert "olmoe-instruct" in notes  # the declared post-training control (T9.2)


def test_the_shipped_models_yaml_pair_definitions_are_usable(tmp_path):
    """Guards the column names this table reads out of the real configs/models.yaml."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "models.yaml").read_text(
            encoding="utf-8"
        )
    )
    members = config["pairs"]["pair_g"]["members"]
    out = write_docs(
        tmp_path / "r",
        [doc(m, "F1", [ok_cell(0, **OLMOE_CELL), ok_cell(1, **OLMOE_CELL)], **OLMOE) for m in members],
    )
    table = pair_table(load_results(out), "pair_g", config, metric="mi_ratio")
    assert table.meta["members"] == members
    assert table.meta["framing"] == "matched density, with these differences documented"
    assert "controlled" not in table.meta["framing"]


def test_per_layer_pair_rows_warn_that_they_are_not_aligned_across_members(panel):
    table = pair_table(
        load_results(panel), "pair_g", MODELS_CONFIG, metric="mi_bits", layer_reduction="per_layer"
    )
    assert "layer" in table.columns
    assert any("NOT aligned across members" in n for n in table.notes)


def test_pair_rows_can_be_aligned_across_members_on_normalized_depth(panel):
    """The only reduction whose rows line up across members of different depth (§1.4).

    olmoe has 4 MoE layers and gptoss 2, so a per-layer join would compare unrelated positions;
    binned normalized depth gives both members the same row keys.
    """
    table = pair_table(
        load_results(panel),
        "pair_g",
        MODELS_CONFIG,
        metric="mi_bits",
        layer_reduction="per_normalized_depth",
        depth_bins=2,
    )
    labels = [r[table.columns.index("depth_bin")] for r in table.rows]
    assert set(labels) == {"[0.00,0.50)", "[0.50,1.00]"}
    # gptoss layers 0 and 1 sit at depths 0.0 and 1.0, so each bin holds exactly one of them:
    # bin 0 -> mi 5.0-4.0 = 1.0, bin 1 -> mi 5.0-3.0 = 2.0.
    first = next(
        r
        for r in table.rows
        if r[table.columns.index("depth_bin")] == "[0.00,0.50)"
        and r[table.columns.index("feature")] == "F1"
    )
    assert first[table.columns.index("gptoss (mi_bits)")] == pytest.approx(1.0)


# ==================================================================================================
# T9.4 confounds
# ==================================================================================================


def test_measured_load_balance_and_declared_aux_loss_stay_in_separate_columns(panel):
    """They are different quantities and are free to disagree; that disagreement is the result."""
    table = confound_table(
        load_results(panel), MODELS_CONFIG, load_balance={"olmoe": {0: 0.9, 1: 0.7}}
    )
    row = next(r for r in table.rows if r[0] == "olmoe")
    assert row[table.columns.index("load_balance_measured")] == pytest.approx(0.8)
    assert row[table.columns.index("load_balance_n_layers")] == 2
    assert row[table.columns.index("aux_loss_coef_declared")] == pytest.approx(0.01)
    assert "load_balance" not in table.columns  # never collapsed into one column


def test_a_model_with_no_measured_load_balance_gets_a_blank_not_a_default(panel):
    table = confound_table(load_results(panel), MODELS_CONFIG, load_balance={"olmoe": 0.9})
    row = next(r for r in table.rows if r[0] == "gptoss")
    cell = row[table.columns.index("load_balance_measured")]
    assert isinstance(cell, MissingCell) and cell.kind == "metric_absent"
    assert MISSING_MARKER in table.to_markdown()


def test_an_absent_aux_loss_coefficient_is_blank_rather_than_zero(panel):
    table = confound_table(load_results(panel), MODELS_CONFIG)
    row = next(r for r in table.rows if r[0] == "gptoss")
    assert isinstance(row[table.columns.index("aux_loss_coef_declared")], MissingCell)
    assert isinstance(row[table.columns.index("aux_loss_key")], MissingCell)


def test_activated_density_is_derived_from_the_trace_manifest(panel):
    # OLMoE 8/64 and GPT-OSS 4/32 are both 12.5% — the matched density Pair G rests on (§1.5).
    table = confound_table(load_results(panel), MODELS_CONFIG)
    density = {r[0]: r[table.columns.index("activated_density")] for r in table.rows}
    assert density["olmoe"] == pytest.approx(0.125)
    assert density["gptoss"] == pytest.approx(0.125)


def test_a_model_absent_from_models_yaml_still_gets_a_row_with_blanks(panel):
    table = confound_table(load_results(panel), {"models": {}})
    assert {r[0] for r in table.rows} == {"olmoe", "gptoss"}
    row = next(r for r in table.rows if r[0] == "olmoe")
    assert isinstance(row[table.columns.index("checkpoint_status")], MissingCell)


# ==================================================================================================
# T9.6 expected-shape check — written for a nuance result
# ==================================================================================================


def _ladder_docs(values_per_layer: dict) -> list:
    """values_per_layer: {feature: {layer: mi_ratio}} -> documents with matching H/CE."""
    docs = []
    for feature, per_layer in values_per_layer.items():
        records = []
        for layer, ratio in per_layer.items():
            # ratio = (H − CE)/H with H = 6.0 -> CE = 6.0 * (1 − ratio)
            records.append(ok_cell(layer, **OLMOE_CELL, entropy_bits=6.0, ce_bits=6.0 * (1.0 - ratio)))
        docs.append(doc("olmoe", feature, records, **OLMOE))
    return docs


def test_a_ladder_that_holds_on_every_layer_agrees(tmp_path):
    out = write_docs(
        tmp_path / "r",
        _ladder_docs(
            {
                "F0": {0: 0.0, 1: 0.0},
                "F1": {0: 0.2, 1: 0.3},
                "F2": {0: 0.3, 1: 0.4},
                "F4": {0: 0.5, 1: 0.6},
            }
        ),
    )
    report = expected_shape_check(load_results(out), metric="mi_ratio")
    ladder = next(e for e in report.expectations if e.name == "feature_ladder_ordering")
    assert ladder.verdict == "agrees"
    assert ladder.per_model["olmoe"]["fraction_ordered"] == 1.0
    assert ladder.per_model["olmoe"]["n_layers_evaluated"] == 2


def test_a_ladder_that_holds_on_some_layers_is_mixed_not_a_refutation(tmp_path):
    """T9.6 is written for a nuance result: partial agreement is a first-class verdict."""
    out = write_docs(
        tmp_path / "r",
        _ladder_docs(
            {
                "F0": {0: 0.0, 1: 0.0},
                "F1": {0: 0.2, 1: 0.5},
                "F2": {0: 0.3, 1: 0.1},  # layer 1 breaks the ordering
                "F4": {0: 0.5, 1: 0.6},
            }
        ),
    )
    ladder = next(
        e
        for e in expected_shape_check(load_results(out), metric="mi_ratio").expectations
        if e.name == "feature_ladder_ordering"
    )
    assert ladder.verdict == "mixed"
    assert ladder.per_model["olmoe"]["layers_out_of_order"] == [1]
    assert ladder.per_model["olmoe"]["fraction_ordered"] == pytest.approx(0.5)


def test_a_ladder_with_fewer_than_two_features_is_not_evaluable(tmp_path):
    out = write_docs(tmp_path / "r", _ladder_docs({"F1": {0: 0.2, 1: 0.3}}))
    report = expected_shape_check(load_results(out), metric="mi_ratio")
    ladder = next(e for e in report.expectations if e.name == "feature_ladder_ordering")
    assert ladder.verdict == "not_evaluable"
    assert "fewer than two ladder features" in ladder.per_model["olmoe"]["reason"]


def test_an_interior_peak_in_depth_agrees_and_a_flat_profile_does_not(tmp_path):
    peaked = write_docs(
        tmp_path / "peaked",
        _ladder_docs({"F1": {0: 0.1, 1: 0.6, 2: 0.5, 3: 0.2}}),
    )
    flat = write_docs(tmp_path / "flat", _ladder_docs({"F1": {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3}}))
    peaked_detail = next(
        e for e in expected_shape_check(load_results(peaked)).expectations if e.name == "depth_variation"
    ).per_model["olmoe"]
    flat_detail = next(
        e for e in expected_shape_check(load_results(flat)).expectations if e.name == "depth_variation"
    ).per_model["olmoe"]
    assert peaked_detail["verdict"] == "agrees"
    assert peaked_detail["peak_layer"] == 1
    assert peaked_detail["peak_normalized_depth"] == pytest.approx(1 / 3)
    assert peaked_detail["spread"] == pytest.approx(0.5)
    assert flat_detail["verdict"] == "disagrees"
    assert flat_detail["spread"] == pytest.approx(0.0, abs=1e-12)


def test_a_depth_profile_needs_at_least_three_layers(tmp_path):
    out = write_docs(tmp_path / "r", _ladder_docs({"F1": {0: 0.1, 1: 0.6}}))
    detail = next(
        e for e in expected_shape_check(load_results(out)).expectations if e.name == "depth_variation"
    ).per_model["olmoe"]
    assert detail["verdict"] == "not_evaluable"
    assert "at least 3" in detail["reason"]


def test_the_shape_check_emits_a_histogram_and_never_a_boolean(tmp_path):
    out = write_docs(
        tmp_path / "r",
        _ladder_docs(
            {"F0": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, "F1": {0: 0.1, 1: 0.6, 2: 0.5, 3: 0.2}}
        ),
    )
    report = expected_shape_check(load_results(out))
    assert sum(report.verdict_counts.values()) == len(report.expectations)
    payload = report.to_json()
    assert not any(isinstance(v, bool) for v in payload["verdict_counts"].values())
    assert "pass" not in payload and "passed" not in payload
    assert set(payload["verdict_counts"]) == {"agrees", "mixed", "disagrees", "not_evaluable"}
    assert report.to_table().n_missing == 0


def test_a_results_directory_where_nothing_was_collected_produces_empty_but_valid_tables(tmp_path):
    """The whole-session failure case: no documents at all. Every renderer must still work, and
    nothing may report a zero it did not measure."""
    empty = tmp_path / "empty"
    empty.mkdir()
    resultset = load_results(empty)
    assert resultset.model_names == [] and resultset.run_config_sha256 is None
    assert primary_table(resultset).rows == []
    assert negative_mi_cells(resultset).count == 0
    assert negative_mi_cells(resultset).n_cells_examined == 0
    report = expected_shape_check(resultset)
    assert report.verdict_counts["not_evaluable"] == len(report.expectations)
    table = report.to_table()
    assert table.n_missing == len(report.expectations)
    assert MISSING_MARKER in table.to_markdown()


def test_the_shape_check_refuses_an_unknown_metric(panel):
    with pytest.raises(ValueError, match="unknown metric"):
        expected_shape_check(load_results(panel), metric="accuracy")


# ==================================================================================================
# T9.5 reconciliation — the study's central claim, hand-computed
# ==================================================================================================


def test_the_mixtral_figure_is_stored_as_the_range_it_was_measured_as():
    """The quoted '28%' is the TOP of a 23.6-28.4% range at ONE layer. A bare 0.28 is the artefact
    this module exists to stop propagating (§1.7 / correction C5)."""
    figure = next(f for f in LITERATURE_FIGURES if f.name == "mixtral_first_choice_layer15")
    assert figure.value is None
    assert figure.span() == (0.236, 0.284)
    assert figure.random_baseline == 0.125
    assert figure.layer == 15 and figure.n_layers == 32
    assert "consecutive" in figure.definition


def test_excess_over_chance_is_the_rate_minus_the_source_s_own_chance_level():
    # Mixtral-8x7B, layer 15: 23.6-28.4% against a 1/8 = 12.5% first-choice chance level.
    #   0.236 − 0.125 = 0.111
    #   0.284 − 0.125 = 0.159
    # The headline "28%" is therefore 15.9 points of signal, not 28.
    figure = next(f for f in LITERATURE_FIGURES if f.name == "mixtral_first_choice_layer15")
    low, high = figure.excess_over_random()
    assert low == pytest.approx(0.111)
    assert high == pytest.approx(0.159)


def test_a_figure_whose_chance_level_is_not_pinned_yields_no_excess():
    """§1.7 gives the first-or-second-choice chance level only as '~46%'. A tilde is not a baseline,
    so plugging 0.46 in here would manufacture a number nobody measured."""
    figure = next(f for f in LITERATURE_FIGURES if f.name == "mixtral_first_or_second_choice_layer15")
    assert figure.span() == (0.616, 0.670)
    assert figure.random_baseline is None
    assert figure.excess_over_random() is None
    assert "~46%" in figure.random_baseline_gap


def test_the_ninety_nine_percent_cache_hit_rate_is_a_gap_and_not_a_number():
    """The prefetching side of the '28% vs 99%' contrast is not pinned down by any parameter §1.7
    states, so it must never be encoded as 0.99."""
    figure = next(f for f in LITERATURE_FIGURES if f.name == "prefetch_cache_hit_rate")
    assert not figure.is_pinned
    assert figure.span() is None
    assert "cache capacity" in figure.gap_reason
    assert 0.99 not in {f.value for f in LITERATURE_FIGURES}


def test_a_literature_figure_must_be_either_pinned_or_explained():
    with pytest.raises(ValueError, match="must say why"):
        LiteratureFigure(
            name="bare", quantity="q", definition="d", source="s", plan_reference="§1.7"
        )
    with pytest.raises(ValueError, match="cannot be both pinned and a gap"):
        LiteratureFigure(
            name="both",
            quantity="q",
            definition="d",
            source="s",
            plan_reference="§1.7",
            value=0.28,
            gap_reason="unpinned",
        )
    with pytest.raises(ValueError, match="needs a definition and a source"):
        LiteratureFigure(
            name="undefined", quantity="q", definition="", source="", plan_reference="§1.7", value=0.28
        )


def test_no_mixtral_anchor_is_claimed_for_this_panel():
    """O3 is closed: the trace dataset that was to provide one has no Mixtral in it."""
    assert any("Mixtral" in a["anchor"] for a in UNAVAILABLE_ANCHORS)
    assert all("unavailable" in a["status"] for a in UNAVAILABLE_ANCHORS)


# --------------------------------------------------------------------------------------------
# The reconciliation over a real result set.
#
# Two models, both with 16 MoE layers so the depth alignment is hand-checkable:
#   olmoe  : 64 experts, top-8 -> chance = 8/64  = 0.125 (Mixtral's own 12.5%, by coincidence)
#   qwen   : 128 experts, top-8 -> chance = 8/128 = 0.0625
# --------------------------------------------------------------------------------------------

RECON_OLMOE = {"n_experts": 64, "top_k": 8, "n_moe_layers": 16}
RECON_QWEN = {"n_experts": 128, "top_k": 8, "n_moe_layers": 16}


def repetition_stat(layer: int, rate: float, *, top_k: int, n_experts: int, exact: float = 0.0):
    return {
        "layer": layer,
        "model_layer": layer,
        "split": "test",
        "n_rows": 5000,
        "n_excluded_doc_initial": 500,
        "repetition_rate": rate,
        "random_baseline": top_k / n_experts,
        "exact_set_repeat_rate": exact,
    }


@pytest.fixture
def recon_panel(tmp_path: Path):
    """F1/F3/F4 for both models, with the F3 cells carrying the repetition statistic sweep stores.

    Layer 7 is given the interesting rate because that is the layer normalized-depth alignment
    picks for Mixtral's layer 15 (see the derivation in the alignment test).
    """
    docs = []
    for name, geometry in (("olmoe", RECON_OLMOE), ("qwen", RECON_QWEN)):
        rate = 0.30 if name == "olmoe" else 0.23
        for feature in ("F1", "F3", "F4"):
            records = []
            for layer in range(geometry["n_moe_layers"]):
                extra = None
                if feature == "F3":
                    extra = {
                        "mixtral_table5_statistic": repetition_stat(
                            layer,
                            rate if layer == 7 else 0.10,
                            top_k=geometry["top_k"],
                            n_experts=geometry["n_experts"],
                            exact=0.04,
                        )
                    }
                records.append(
                    ok_cell(
                        layer,
                        top_k=geometry["top_k"],
                        n_experts=geometry["n_experts"],
                        entropy_bits=6.0,
                        ce_bits=4.0,
                        agreement=0.5,
                        recall={
                            geometry["top_k"]: 0.55,
                            2 * geometry["top_k"]: 0.80,
                            4 * geometry["top_k"]: 0.99,
                        },
                        extra=extra,
                    )
                )
            docs.append(doc(name, feature, records, **geometry))
    return write_docs(tmp_path / "recon", docs)


def test_the_study_is_aligned_to_a_literature_figure_by_normalized_depth(recon_panel):
    """Absolute layer indices would compare unrelated positions in two stacks (§1.4).

    Mixtral's layer 15 of 32 sits at 15/31 = 0.4838709677...
    A 16-MoE-layer model's layers sit at l/15, so the two candidates are
        l = 7 -> 7/15  = 0.4666666..., |error| = |7/15 − 15/31| = 8/465 = 0.01720430107526882
        l = 8 -> 8/15  = 0.5333333..., |error| = |8/15 − 15/31| = 23/465 = 0.04946236559139785
    so layer 7 is the alignment, with the residual reported rather than hidden.
    """
    report = reconcile(load_results(recon_panel))
    comparison = next(
        c for c in report.comparisons if c.figure.name == "mixtral_first_choice_layer15"
    )
    assert comparison.figure.normalized_depth == pytest.approx(15 / 31)
    point = next(p for p in comparison.points if p.model == "olmoe")
    assert point.layer == 7
    assert point.normalized_depth == pytest.approx(7 / 15)
    assert point.depth_alignment_error == pytest.approx(8 / 465)


def test_the_repetition_statistic_is_read_from_the_f3_cells_the_sweep_wrote(recon_panel):
    """T7.4 stores the raw statistic next to F3 so the reconciliation reads the same number the
    results directory reports, rather than a second copy that can drift."""
    report = reconcile(load_results(recon_panel))
    comparison = next(
        c for c in report.comparisons if c.figure.name == "mixtral_first_choice_layer15"
    )
    point = next(p for p in comparison.points if p.model == "olmoe")
    assert point.repetition_rate == pytest.approx(0.30)
    assert point.exact_set_repeat_rate == pytest.approx(0.04)
    assert point.n_rows == 5000
    assert "|S_t ∩ S_{t-1}| / k" in point.definition


def test_the_chance_level_moves_with_granularity_and_reverses_the_raw_ordering(recon_panel):
    """The axis on which raw rates are comparable across the panel is excess over chance.

    Hand derivation, both models aligned to Mixtral's layer 15 (trace layer 7 above):

        olmoe : k/n = 8/64  = 0.1250 ; rate 0.30 -> excess 0.30 − 0.1250 = 0.1750
        qwen  : k/n = 8/128 = 0.0625 ; rate 0.23 -> excess 0.23 − 0.0625 = 0.1675
        mixtral layer 15    = 0.1250 ; rate 0.236-0.284 -> excess 0.111-0.159

    qwen's RAW rate (0.23) is below the bottom of Mixtral's raw range (0.236), yet its excess over
    chance (0.1675) is above the TOP of Mixtral's excess range (0.159). The ordering flips between
    the two axes, so a formula that forgot the chance term — or that used a fixed 12.5% for every
    model — would rank these three the other way round and fail here.
    """
    report = reconcile(load_results(recon_panel))
    comparison = next(
        c for c in report.comparisons if c.figure.name == "mixtral_first_choice_layer15"
    )
    points = {p.model: p for p in comparison.points}

    assert points["olmoe"].random_baseline == pytest.approx(0.125)
    assert points["qwen"].random_baseline == pytest.approx(0.0625)
    assert points["olmoe"].excess_over_random == pytest.approx(0.175)
    assert points["qwen"].excess_over_random == pytest.approx(0.1675)

    lit_low, lit_high = comparison.figure.excess_over_random()
    assert points["qwen"].repetition_rate < comparison.figure.span()[0]  # raw: below Mixtral
    assert points["qwen"].excess_over_random > lit_high  # excess: above Mixtral
    assert points["olmoe"].excess_over_random > lit_high


def test_the_chance_level_falls_back_to_k_over_n_experts_when_the_stat_omits_it(recon_panel):
    """A statistic written by an older sweep carries no random_baseline; k/n_experts is the
    definition, so it is re-derived rather than left blank."""
    stat = repetition_stat(7, 0.30, top_k=8, n_experts=64)
    del stat["random_baseline"]
    report = reconcile(
        load_results(recon_panel), repetition_stats={"olmoe": {7: stat}}, models=["olmoe"]
    )
    point = next(
        c for c in report.comparisons if c.figure.name == "mixtral_first_choice_layer15"
    ).points[0]
    assert point.random_baseline == pytest.approx(8 / 64)
    assert point.excess_over_random == pytest.approx(0.175)


def test_the_unpinned_cache_hit_rate_is_never_compared_against(recon_panel):
    """The 99% side of the contrast produces a GAP, not a comparison row. Closing it with an
    approximation would reproduce exactly the error §1.7 documents."""
    report = reconcile(load_results(recon_panel))
    compared = {c.figure.name for c in report.comparisons}
    assert "prefetch_cache_hit_rate" not in compared
    assert "prefetch_cache_hit_rate" in {f.name for f in report.unpinned}
    gaps = {(g["figure"], g["kind"]) for g in report.gaps}
    assert ("prefetch_cache_hit_rate", "figure") in gaps
    assert ("mixtral_first_or_second_choice_layer15", "random_baseline") in gaps
    assert all(g["reason"] for g in report.gaps)
    assert any("GAP — prefetch_cache_hit_rate" in n for n in report.to_table().notes)


def test_the_cache_style_axis_is_reported_under_this_study_s_own_definition(recon_panel):
    """recall@m at budget m IS a per-layer, single-step cache-hit rate at capacity m. Putting the
    28%-like and the 99%-like numbers in one report is the reconciliation: they never contradicted
    each other, they were measured on different axes."""
    report = reconcile(load_results(recon_panel))
    table = report.cache_budget_axis
    row = next(r for r in table.rows if r[0] == "olmoe" and r[3] == "F3" and r[4] == 7)
    assert row[table.columns.index("recall@k")] == pytest.approx(0.55)
    assert row[table.columns.index("recall@2k")] == pytest.approx(0.80)
    assert row[table.columns.index("recall@4k")] == pytest.approx(0.99)
    # The same (model, layer) whose budget-free repetition rate is 0.30 reaches 0.99 at budget 4k.
    point = next(
        p
        for c in report.comparisons
        if c.figure.name == "mixtral_first_choice_layer15"
        for p in c.points
        if p.model == "olmoe"
    )
    assert point.repetition_rate == pytest.approx(0.30)
    assert any("no literature figure is attached" in n.lower() for n in table.notes)


def test_the_cache_budget_row_reports_absolute_capacity_per_model(recon_panel):
    """recall@k is 8 experts on OLMoE and 8 on Qwen but out of 64 vs 128 — the capacity column is
    what stops the two being read as one budget."""
    table = reconcile(load_results(recon_panel)).cache_budget_axis
    for name, n_experts in (("olmoe", 64), ("qwen", 128)):
        row = next(r for r in table.rows if r[0] == name)
        assert row[table.columns.index("top_k")] == 8
        assert row[table.columns.index("n_experts")] == n_experts


def test_every_comparison_names_the_definitional_axes_that_separate_the_numbers(recon_panel):
    report = reconcile(load_results(recon_panel))
    for comparison in report.comparisons:
        axis_names = {a.axis for a in comparison.axes}
        assert any("top-1" in a for a in axis_names)
        assert any("chance level" in a for a in axis_names)
        assert any("per-layer" in a for a in axis_names)
        assert comparison.caveats
    first_choice = next(
        c for c in report.comparisons if c.figure.name == "mixtral_first_choice_layer15"
    )
    assert any("top-1 identity repetition" in c for c in first_choice.caveats)
    assert "capacity_vs_single_step" in DEFINITIONAL_AXES


def test_a_model_with_no_repetition_statistic_is_reported_as_unavailable(recon_panel):
    """A missing statistic is a hole with a reason, never a zero rate."""
    report = reconcile(load_results(recon_panel), repetition_stats={"olmoe": {7: repetition_stat(7, 0.3, top_k=8, n_experts=64)}})
    comparison = next(
        c for c in report.comparisons if c.figure.name == "mixtral_first_choice_layer15"
    )
    qwen = next(p for p in comparison.points if p.model == "qwen")
    assert qwen.repetition_rate is None
    assert qwen.excess_over_random is None
    assert "no consecutive-repetition statistic" in qwen.unavailable_reason
    table = report.to_table()
    assert table.n_missing > 0
    assert MISSING_MARKER in table.to_markdown()


def test_a_model_absent_from_the_result_set_is_refused(recon_panel):
    with pytest.raises(KeyError, match="mixtral"):
        reconcile(load_results(recon_panel), models=["olmoe", "mixtral"])


def test_the_report_serializes_the_definition_it_measured_everything_under(recon_panel):
    payload = reconcile(load_results(recon_panel)).to_json()
    assert "|S_t ∩ S_{t-1}| / k" in payload["this_study_definition"]
    assert "k / n_experts" in payload["this_study_definition"]
    assert payload["gaps"] and payload["unavailable_anchors"]
    assert json.dumps(payload)  # the report has to survive a round trip to disk
    assert any("metric mismatch" in n for n in payload["notes"])


def test_the_rendered_reconciliation_keeps_the_literature_and_study_columns_apart(recon_panel):
    table = reconcile(load_results(recon_panel)).to_table()
    assert "lit_low" in table.columns and "our_repetition_rate" in table.columns
    row = next(
        r
        for r in table.rows
        if r[0] == "mixtral_first_choice_layer15" and r[table.columns.index("model")] == "olmoe"
    )
    assert row[table.columns.index("lit_low")] == pytest.approx(0.236)
    assert row[table.columns.index("lit_excess_high")] == pytest.approx(0.159)
    assert row[table.columns.index("our_repetition_rate")] == pytest.approx(0.30)
    assert row[table.columns.index("our_excess")] == pytest.approx(0.175)
    # The unpinned baseline renders as a hole on the literature side, not as 0.46.
    unpinned_row = next(
        r for r in table.rows if r[0] == "mixtral_first_or_second_choice_layer15"
    )
    assert isinstance(unpinned_row[table.columns.index("lit_random")], MissingCell)
    assert isinstance(unpinned_row[table.columns.index("lit_excess_low")], MissingCell)


# ==================================================================================================
# The definition itself — the reconciliation is only as good as the statistic it reads.
# ==================================================================================================


def _fixed_topk(sets_per_layer):
    """topk_fn making layer L's set at token t equal to sets_per_layer[L](t)."""

    def build(rng, tokens, spec):
        n = tokens.shape[0]
        out = np.zeros((n, spec.n_moe_layers, spec.top_k), dtype=np.int32)
        for layer, fn in enumerate(sets_per_layer):
            for t in range(n):
                out[t, layer] = fn(t)
        return out

    return build


def test_the_repetition_rate_is_the_mean_overlap_fraction_over_consecutive_tokens(tmp_path):
    """|S_t ∩ S_{t-1}| / k, hand-computed on three constructed layers (16 experts, top-4).

    layer 0 — every token routes to {0,1,2,3}. Every consecutive pair overlaps in all 4 slots,
              so the rate is 4/4 = 1.0 and every pair is an exact set repeat.
    layer 1 — token t routes to {4t, 4t+1, 4t+2, 4t+3} mod 16. Consecutive sets are disjoint
              blocks, so the overlap is 0/4 = 0.0 on every pair.
    layer 2 — tokens alternate between {0,1,2,3} and {2,3,4,5}. Every consecutive pair shares
              exactly {2,3}, so the overlap is 2/4 = 0.5 on every pair and never an exact repeat.

    The chance level is k/n_experts = 4/16 = 0.25 at every layer, independent of the routing.
    """
    spec = TraceSpec(n_moe_layers=3, n_experts=16, top_k=4, hidden_dim=8)
    truth = make_synthetic_trace(
        tmp_path / "traces",
        spec=spec,
        topk_fn=_fixed_topk(
            [
                lambda t: [0, 1, 2, 3],
                lambda t: [(4 * t + i) % 16 for i in range(4)],
                lambda t: [0, 1, 2, 3] if t % 2 == 0 else [2, 3, 4, 5],
            ]
        ),
    )
    with TraceReader(
        tmp_path / "traces", truth.model, truth.corpus, doc_splits=truth.doc_splits
    ) as reader:
        index = CorpusIndex.from_reader(reader)
        constant = consecutive_repetition_rate(reader, index, 0, split="test")
        disjoint = consecutive_repetition_rate(reader, index, 1, split="test")
        half = consecutive_repetition_rate(reader, index, 2, split="test")

    assert constant["repetition_rate"] == pytest.approx(1.0)
    assert constant["exact_set_repeat_rate"] == pytest.approx(1.0)
    assert disjoint["repetition_rate"] == pytest.approx(0.0)
    assert half["repetition_rate"] == pytest.approx(0.5)
    assert half["exact_set_repeat_rate"] == pytest.approx(0.0)
    for stat in (constant, disjoint, half):
        assert stat["random_baseline"] == pytest.approx(0.25)
        assert stat["n_rows"] > 0
        assert stat["n_excluded_doc_initial"] > 0  # doc-initial tokens have no previous token (I11)


def test_that_statistic_feeds_the_reconciliation_unchanged(tmp_path):
    """End to end: the number the trace produces is the number the reconciliation reports.

    The half-overlap layer gives 0.5 at a 4/16 = 0.25 chance level, so the excess over chance is
    0.5 − 0.25 = 0.25 — above the top of Mixtral's 0.111-0.159 excess range.
    """
    spec = TraceSpec(n_moe_layers=3, n_experts=16, top_k=4, hidden_dim=8)
    truth = make_synthetic_trace(
        tmp_path / "traces",
        spec=spec,
        topk_fn=_fixed_topk(
            [
                lambda t: [0, 1, 2, 3] if t % 2 == 0 else [2, 3, 4, 5],
                lambda t: [0, 1, 2, 3] if t % 2 == 0 else [2, 3, 4, 5],
                lambda t: [0, 1, 2, 3] if t % 2 == 0 else [2, 3, 4, 5],
            ]
        ),
    )
    with TraceReader(
        tmp_path / "traces", truth.model, truth.corpus, doc_splits=truth.doc_splits
    ) as reader:
        index = CorpusIndex.from_reader(reader)
        stats = {
            layer: consecutive_repetition_rate(reader, index, layer, split="test")
            for layer in range(spec.n_moe_layers)
        }

    geometry = {"n_experts": 16, "top_k": 4, "n_moe_layers": 3}
    out = write_docs(
        tmp_path / "results",
        [doc("synth", "F3", [ok_cell(i, top_k=4, n_experts=16) for i in range(3)], **geometry)],
    )
    report = reconcile(load_results(out), repetition_stats={"synth": stats})
    point = next(
        c for c in report.comparisons if c.figure.name == "mixtral_first_choice_layer15"
    ).points[0]
    assert point.repetition_rate == pytest.approx(0.5)
    assert point.random_baseline == pytest.approx(0.25)
    assert point.excess_over_random == pytest.approx(0.25)
    assert point.excess_over_random > next(
        f for f in LITERATURE_FIGURES if f.name == "mixtral_first_choice_layer15"
    ).excess_over_random()[1]
