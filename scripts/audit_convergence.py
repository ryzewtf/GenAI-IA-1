#!/usr/bin/env python3
"""Convergence study for the softmax router-input probes F4 / FV — audit L6 (and M10's artifact).

Every F4 and FV fit in the main sweep stopped at the 150-iteration L-BFGS cap. This driver refits
both at three relative depths per model (≈0.25, 0.5, 0.75) for an L2 grid, with a much larger
iteration budget, then picks the L2 per layer **on val** and reports how far the converged,
val-selected probe moves from the capped one. The output is the saved stability-band artifact the
draft cited without one.

    python -m scripts.audit_convergence run   --models ... [--maxiter 2000] [--out DIR] [--l2 0.03,0.1]   # fits (slow)
    python -m scripts.audit_convergence report [--out DIR]                              # summary only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts import phase7_sweep
from scripts.phase6_metrics import ENGINE_BY_MODEL

L2_GRID = (1e-4, 1e-3, 1e-2)
N_LAYERS = {"gpt-oss-20b": 24, "olmoe-0125": 16, "olmoe-0924": 16, "olmoe-0125-instruct": 16,
            "deepseek-v2-lite": 26, "qwen3-30b-a3b": 48, "gemma-4-26b-a4b": 30}
ROOT = Path("E:/temp2/conv")
BASE = Path("results/probes")


def study_layers(n: int) -> list[int]:
    return sorted({max(1, round(f * (n - 1))) for f in (0.25, 0.5, 0.75)})


def run(args) -> int:
    rc = 0
    for model in args.models.split(","):
        layers = [str(x) for x in study_layers(N_LAYERS[model])]
        for l2 in (map(float, args.l2.split(",")) if args.l2 else L2_GRID):
            rc |= phase7_sweep.main([
                "--root", args.root, "--no-validate-sizes", "--models", model,
                "--features", "F4", "FV", "--layers", *layers,
                "--out", str(Path(args.out) / f"l2_{l2:g}"),
                "--probe-kwargs", json.dumps({"lbfgs_maxiter": args.maxiter, "l2": l2}),
            ])
    return rc


def attainable(rec) -> float:
    b = rec["metrics"]["family_b"]
    return b["mi_bits"] / (b["entropy_bits"] - math.log2(rec["fit"]["top_k"]))


def report(args) -> int:
    rows = []
    for model in ENGINE_BY_MODEL:
        for feat in ("F4", "FV"):
            base_p = BASE / f"{model}__{feat}.json"
            if not base_p.exists():
                continue
            base = json.loads(base_p.read_text(encoding="utf-8"))["layers"]
            fits = {}
            # Every l2_* directory present counts: a model whose val optimum sat on the grid edge gets
            # extra L2 values via `run --l2`, and selection then sees them too.
            for d in Path(args.out).glob("l2_*"):
                p = d / f"{model}__{feat}.json"
                if p.exists():
                    fits[float(d.name[3:])] = json.loads(p.read_text(encoding="utf-8"))["layers"]
            if not fits:
                continue
            for layer in study_layers(N_LAYERS[model]):
                key = str(layer)
                cand = {l2: f[key] for l2, f in fits.items() if key in f and f[key].get("status") == "ok"}
                if not cand or key not in base:
                    continue
                best = min(cand, key=lambda l2: cand[l2]["fit"]["best_val_ce_bits"])
                b = cand[best]
                rows.append({
                    "model": model, "feature": feat, "layer": layer,
                    "capped_150": {"attainable": attainable(base[key]),
                                   "set_agreement@k": base[key]["metrics"]["family_a"]["set_agreement@k"],
                                   "val_ce": base[key]["fit"]["best_val_ce_bits"]},
                    "per_l2": {f"{l2:g}": {"attainable": attainable(r), "val_ce": r["fit"]["best_val_ce_bits"],
                                          "converged": r["fit"]["hyperparams"]["converged"],
                                          "n_iter": r["fit"]["hyperparams"]["n_iter"]}
                               for l2, r in cand.items()},
                    "selected_l2": best,
                    "selected": {"attainable": attainable(b),
                                 "set_agreement@k": b["metrics"]["family_a"]["set_agreement@k"],
                                 "converged": b["fit"]["hyperparams"]["converged"],
                                 "n_iter": b["fit"]["hyperparams"]["n_iter"]},
                })
    for r in rows:
        r["delta_attainable"] = r["selected"]["attainable"] - r["capped_150"]["attainable"]
        tried = sorted(float(k) for k in r["per_l2"])
        r["selected_l2_on_edge"] = r["selected_l2"] in (tried[0], tried[-1])
    out = Path("results/audit")
    out.mkdir(parents=True, exist_ok=True)
    deltas = [r["delta_attainable"] for r in rows]
    summary = {"n_cells": len(rows), "l2_grid": sorted({float(d.name[3:]) for d in Path(args.out).glob("l2_*")}),
               "n_selected_on_grid_edge": sum(r["selected_l2_on_edge"] for r in rows),
               "max_abs_delta_attainable": max(map(abs, deltas)) if deltas else None,
               "mean_delta_attainable": sum(deltas) / len(deltas) if deltas else None,
               "n_converged": sum(r["selected"]["converged"] for r in rows), "cells": rows}
    (out / "convergence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# F4 / FV convergence study (val-selected L2, large iteration budget)", "",
          f"{len(rows)} cells; converged {summary['n_converged']}; max |Δ attainable| "
          f"{summary['max_abs_delta_attainable']}", "",
          "| model | feature | layer | capped (150 it) | selected L2 | converged fit | Δ | converged? | iters |",
          "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    md += [f"| {r['model']} | {r['feature']} | {r['layer']} | {r['capped_150']['attainable']:.3f} | "
           f"{r['selected_l2']:g} | {r['selected']['attainable']:.3f} | {r['delta_attainable']:+.3f} | "
           f"{r['selected']['converged']} | {r['selected']['n_iter']} |" for r in rows]
    (out / "convergence.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:3]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", required=True)
    r.add_argument("--root", default="E:/temp/traces")
    r.add_argument("--maxiter", type=int, default=2000)
    r.add_argument("--out", default=str(ROOT))
    r.add_argument("--l2", default=None, help="comma-separated L2 values to fit instead of L2_GRID")
    rep = sub.add_parser("report")
    rep.add_argument("--out", default=str(ROOT))
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
