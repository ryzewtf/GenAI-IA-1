#!/usr/bin/env python3
"""Seed spread of the MLP probe F5 — audit L7 (single seed everywhere).

F5 is the only stochastic predictor whose differences the paper interprets (F6 is stochastic too and
moved by ≤ 0.014 between the v0.1 sweep and the re-sweep). Seeds 1 and 2 are refit on the
convergence-study layers (≈ 0.25, 0.5, 0.75 relative depth) and compared with seed 0 from the
canonical v0.2 results.

    python -m scripts.audit_seeds run    [--models ...] [--seeds 1,2] [--out E:/temp2/seeds]
    python -m scripts.audit_seeds report [--out E:/temp2/seeds]     # -> results/audit/seeds.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scripts import phase7_sweep
from scripts.audit_convergence import N_LAYERS, study_layers
from scripts.phase6_metrics import ENGINE_BY_MODEL

SEED0 = Path("results/probes_v2")


def attainable(rec) -> float:
    b = rec["metrics"]["family_b"]
    return b["mi_bits"] / (b["entropy_bits"] - math.log2(rec["fit"]["top_k"]))


def run(args) -> int:
    rc = 0
    for seed in map(int, args.seeds.split(",")):
        for model in args.models.split(","):
            rc |= phase7_sweep.main([
                "--root", args.root, "--no-validate-sizes", "--models", model, "--features", "F5",
                "--layers", *map(str, study_layers(N_LAYERS[model])), "--seed", str(seed),
                "--out", str(Path(args.out) / f"s{seed}"),
            ])
    return rc


def report(args) -> int:
    cells, per_model = [], {}
    for model in ENGINE_BY_MODEL:
        base = SEED0 / f"{model}__F5.json"
        runs = {0: json.loads(base.read_text(encoding="utf-8"))["layers"]} if base.exists() else {}
        for d in sorted(Path(args.out).glob("s*")):
            p = d / f"{model}__F5.json"
            if p.exists():
                runs[int(d.name[1:])] = json.loads(p.read_text(encoding="utf-8"))["layers"]
        if len(runs) < 2:
            continue
        for layer in study_layers(N_LAYERS[model]):
            vals = {s: attainable(r[str(layer)]) for s, r in runs.items()
                    if r.get(str(layer), {}).get("status") == "ok"}
            if len(vals) < 2:
                continue
            v = list(vals.values())
            cells.append({"model": model, "layer": layer, "per_seed": vals,
                          "range": max(v) - min(v), "sd": st.stdev(v)})
        mine = [c for c in cells if c["model"] == model]
        if mine:
            per_model[model] = {"n_cells": len(mine), "seeds": sorted(runs),
                                "mean_range": st.mean(c["range"] for c in mine),
                                "max_range": max(c["range"] for c in mine),
                                "mean_sd": st.mean(c["sd"] for c in mine)}
    out = Path("results/audit")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"feature": "F5", "metric": "Î/(H − log2 k)", "per_model": per_model, "cells": cells,
               "max_range_overall": max((c["range"] for c in cells), default=None)}
    (out / "seeds.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# F5 seed spread, Î/(H − log2 k)", "",
          "| model | layer | " + " | ".join(f"seed {s}" for s in (0, 1, 2)) + " | range |", "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {c['model']} | {c['layer']} | " + " | ".join(f"{c['per_seed'].get(s, float('nan')):.4f}" for s in (0, 1, 2))
           + f" | {c['range']:.4f} |" for c in cells]
    (out / "seeds.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(cells)} cells; max range {summary['max_range_overall']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", default=",".join(ENGINE_BY_MODEL))
    r.add_argument("--seeds", default="1,2")
    r.add_argument("--root", default="E:/temp/traces")
    r.add_argument("--out", default="E:/temp2/seeds")
    rep = sub.add_parser("report")
    rep.add_argument("--out", default="E:/temp2/seeds")
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
