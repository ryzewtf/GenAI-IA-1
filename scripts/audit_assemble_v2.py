#!/usr/bin/env python3
"""Assemble the v0.2 canonical probe results from the audit re-runs.

    results/probes_v2/<model>__<feature>.json        per-layer results
    results/probes_v2_docagg/<model>__<feature>.npz  per-document aggregates (bootstrap input)

Sources, one per feature, so no feature is ever taken from two runs:

* F0 F1 F2 F3 F5 F6 - the re-sweep with aggregates and the M10 F6 refit rule (``--src-ladder``);
* F4 FV             - the converged L-BFGS refit, val-selected l2 per model (``--src-conv``, audit L6);
* F4R FVR           - the ridge router-score probes (``results/probes`` + ``results/probes_docagg``, L3).

``results/probes`` itself is left as it is: it holds the v0.1 run, and the convergence report uses
its 150-iteration F4/FV as the capped baseline. The script refuses to run unless every
(model, feature) has both a result file and an aggregate file, and every layer that is ``ok`` in the
v0.1 file of the same name is ``ok`` in the source too (a sweep still running writes partial files).

Usage: python -m scripts.audit_assemble_v2 [--src-ladder E:/temp2/probes_v2 --src-ladder-agg E:/temp2/agg_v2
                                             --src-conv E:/temp2/probes_conv --src-conv-agg E:/temp2/agg_conv]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.phase6_metrics import ENGINE_BY_MODEL

LADDER = ("F0", "F1", "F2", "F3", "F5", "F6")
CONVERGED = ("F4", "FV")
RIDGE = ("F4R", "FVR")
REFERENCE = Path("results/probes")


def ok_layers(path: Path) -> set[str]:
    layers = json.loads(path.read_text(encoding="utf-8"))["layers"]
    return {k for k, v in layers.items() if v.get("status") == "ok"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ladder", default="E:/temp2/probes_v2")
    ap.add_argument("--src-ladder-agg", default="E:/temp2/agg_v2")
    ap.add_argument("--src-conv", default="E:/temp2/probes_conv")
    ap.add_argument("--src-conv-agg", default="E:/temp2/agg_conv")
    ap.add_argument("--src-ridge", default="results/probes")
    ap.add_argument("--src-ridge-agg", default="results/probes_docagg")
    ap.add_argument("--out", default="results/probes_v2")
    ap.add_argument("--out-agg", default="results/probes_v2_docagg")
    args = ap.parse_args(argv)

    sources = ([(f, args.src_ladder, args.src_ladder_agg) for f in LADDER]
               + [(f, args.src_conv, args.src_conv_agg) for f in CONVERGED]
               + [(f, args.src_ridge, args.src_ridge_agg) for f in RIDGE])
    plan, missing = [], []
    for model in ENGINE_BY_MODEL:
        for feature, src, src_agg in sources:
            name = f"{model}__{feature}"
            res, agg = Path(src) / f"{name}.json", Path(src_agg) / f"{name}.npz"
            for p in (res, agg):
                if not p.exists():
                    missing.append(str(p))
            ref = REFERENCE / f"{name}.json"
            if res.exists() and ref.exists():
                lacking = ok_layers(ref) - ok_layers(res)
                if lacking:
                    missing.append(f"{res}: layers {sorted(lacking, key=int)} not ok yet")
            plan.append((res, agg, name))
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))

    out, out_agg = Path(args.out), Path(args.out_agg)
    out.mkdir(parents=True, exist_ok=True)
    out_agg.mkdir(parents=True, exist_ok=True)
    for res, agg, name in plan:
        shutil.copy2(res, out / f"{name}.json")
        shutil.copy2(agg, out_agg / f"{name}.npz")
    print(f"assembled {len(plan)} (model, feature) cells into {out} and {out_agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
