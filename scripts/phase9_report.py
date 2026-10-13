#!/usr/bin/env python3
"""Phase 9 driver: results/probes + Phase 6 load balance -> results/phase9/ (T9.1-T9.4, T9.6).

Thin glue over src.analysis.tables; it computes nothing itself. The panel is split-engine (one
run_config per engine), so load_results runs with require_uniform_run_config=False: that still
refuses a mismatch *within* a model, and the engine difference is carried as a named not_matched
axis on Pair G / Pair T (bounded by the Phase 8 engine rung).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.analysis import tables as T

PROBES = Path("results/probes")
OUT = Path("results/phase9")
FEATURES = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]  # FV is validation-only, never reported
METRICS = ["mi_ratio", "ce_normalized", "set_agreement@k", "recall@k", "exact_match"]


def write(name: str, table: T.Table) -> None:
    (OUT / f"{name}.md").write_text(table.to_markdown(), encoding="utf-8")
    (OUT / f"{name}.csv").write_text(table.to_csv(), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    rs = T.load_results(PROBES, require_uniform_run_config=False)
    lb = {r["model"]: r["lbi_mean"] for r in
          json.loads(Path("results/phase6_load_balance.json").read_text(encoding="utf-8"))}

    # T9.1 primary table, one per metric, plus the depth view
    for m in METRICS:
        write(f"t9_1_{m.replace('@', '_at_')}", T.primary_table(rs, metric=m, features=FEATURES))
    write("t9_4_depth_mi_ratio", T.primary_table(rs, metric="mi_ratio", features=FEATURES,
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
    for pair in ("pair_g", "pair_t", "pair_a", "post_training"):
        for red in ("mean", "per_normalized_depth"):
            t = T.pair_table(rs, pair, cfg, metric="mi_ratio", layer_reduction=red)
            fcol = t.columns.index("feature")
            t.rows = [r for r in t.rows if r[fcol] in FEATURES]  # FV: validation only (§1.3)
            write(f"{pair}_{red}", t)

    # T9.4 confounds (measured load balance vs declared aux-loss), T9.6 shape check
    write("t9_4_confounds", T.confound_table(rs, cfg, load_balance=lb))
    shape = T.expected_shape_check(rs, metric="mi_ratio")
    (OUT / "t9_6_shape.json").write_text(json.dumps(shape.to_json(), indent=2), encoding="utf-8")

    print(f"wrote {OUT}/")
    print("logit tensors:", rs.logit_tensors())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
