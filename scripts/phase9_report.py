#!/usr/bin/env python3
"""Phase 9 driver: results/probes + Phase 6 load balance -> results/phase9/ (T9.1-T9.4, T9.6).

Thin glue over src.analysis.tables; it computes nothing itself. The panel is split-engine (one
run_config per engine), so load_results runs with require_uniform_run_config=False: that still
refuses a mismatch *within* a model, and the engine difference is carried as a named not_matched
axis on Pair G / Pair T (bounded by the Phase 8 engine rung).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.analysis import tables as T

PROBES = Path("results/probes")
OUT = Path("results/phase9")
FEATURES = ["F0", "F1", "F2", "F3", "F4", "F4R", "F5", "F6"]  # FV/FVR are validation-only, never reported
# mi_attainable = Î/(H − log2 k) is the primary Family B metric (audit L1/L2); mi_ratio = Î/H is
# kept for continuity but is capped at (H − log2 k)/H, which differs by model, so it must not be
# compared across models with different k.
METRICS = ["mi_attainable", "mi_ratio", "ce_normalized", "set_agreement@k", "recall@k", "recall@2k",
           "recall@4k", "exact_match"]


def write(name: str, table: T.Table) -> None:
    (OUT / f"{name}.md").write_text(table.to_markdown(), encoding="utf-8")
    (OUT / f"{name}.csv").write_text(table.to_csv(), encoding="utf-8")


def main(argv=None) -> int:
    # v0.2 tables read results/probes_v2 (converged F4/FV, re-swept ladder, F4R) into results/phase9_v2;
    # the defaults keep rebuilding the v0.1 tables unchanged.
    global PROBES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default=str(PROBES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    PROBES, OUT = Path(args.probes), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    rs = T.load_results(PROBES, require_uniform_run_config=False)
    lb = {r["model"]: r["lbi_mean"] for r in
          json.loads(Path("results/phase6_load_balance.json").read_text(encoding="utf-8"))}

    # T9.1 primary table, one per metric, plus the depth view
    for m in METRICS:
        write(f"t9_1_{m.replace('@', '_at_')}", T.primary_table(rs, metric=m, features=FEATURES))
    for m in ("mi_ratio", "mi_attainable"):
        write(f"t9_4_depth_{m}", T.primary_table(rs, metric=m, features=FEATURES,
                                                 layer_reduction="per_normalized_depth"))
    neg = T.negative_mi_cells(rs)
    (OUT / "negative_mi_cells.json").write_text(json.dumps(neg.to_json(), indent=2), encoding="utf-8")

    # T9.2 / T9.3 / Pair A, mean and per-depth; post-training control (base vs instruct) via pair
    cfg.setdefault("pairs", {})["post_training"] = {
        "members": ["olmoe-0125", "olmoe-0125-instruct"],
        "holds_fixed": ["architecture", "tokenizer", "vocabulary", "engine", "aux_loss_coef"],
        "not_matched": ["post_training"],
        "framing": "post-training control for Pair G (T9.2)",
    }
    # Pairs on every metric that a cross-model claim may rest on. "{pair}_{red}" keeps its old
    # meaning (Î/H) so earlier readers do not silently change; the k-invariant versions carry a
    # metric suffix. Family A at k is matched-budget only where the pair also matches k/N.
    pair_metrics = {"": "mi_ratio", "_attainable": "mi_attainable",
                    "_recall_at_k": "recall@k", "_recall_at_2k": "recall@2k"}
    for pair in ("pair_g", "pair_t", "pair_a", "post_training"):
        for red in ("mean", "per_normalized_depth"):
            for suffix, metric in pair_metrics.items():
                t = T.pair_table(rs, pair, cfg, metric=metric, layer_reduction=red)
                fcol = t.columns.index("feature")
                t.rows = [r for r in t.rows if r[fcol] in FEATURES]  # FV: validation only (§1.3)
                write(f"{pair}_{red}{suffix}", t)

    # T9.4 confounds (measured load balance vs declared aux-loss), T9.6 shape check
    write("t9_4_confounds", T.confound_table(rs, cfg, load_balance=lb))
    shape = T.expected_shape_check(rs, metric="mi_attainable")
    (OUT / "t9_6_shape.json").write_text(json.dumps(shape.to_json(), indent=2), encoding="utf-8")

    print(f"wrote {OUT}/")
    print("logit tensors:", rs.logit_tensors())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
