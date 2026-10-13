#!/usr/bin/env python3
"""T8.3 'prominent'-band obligation: re-run the primary analysis at F16 and show conclusions hold.

Q4_K_M vs F16 OLMoE-0125 landed at 0.956 set agreement (0.90–0.97 band). The plan's action is to
re-run the primary analysis on F16 and show the conclusions are unchanged. Inputs are two full probe
sweeps (F0–F6, FV) on the Phase 8 ladder traces — same 151k tokens, same doc splits — one per rung.

Reports, per feature: mean Î/H at each rung and the delta; then the two checks the panel's
conclusions actually rest on:
  1. feature ordering (rank of F0..F6 by mean Î/H) is identical across rungs;
  2. |Δ| per feature is below the smallest cross-model gap the write-up interprets
     (``--min-gap``, default 0.02 = the F4 L-BFGS stability band; a smaller gap is never read).

Usage: python -m scripts.phase8_conclusions [--ref E:/temp/p8probe/f16] [--cand E:/temp/p8probe/q4_k_m]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURES = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]


def means(d: Path, model: str) -> dict[str, float]:
    out = {}
    for f in FEATURES:
        p = d / f"{model}__{f}.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        xs = [L["metrics"]["family_b"]["ratio"] for L in doc["layers"].values()
              if L.get("status") == "ok"]
        if xs:
            out[f] = sum(xs) / len(xs)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="E:/temp/p8probe/f16")
    ap.add_argument("--cand", default="E:/temp/p8probe/q4_k_m")
    ap.add_argument("--model", default="olmoe-0125")
    ap.add_argument("--min-gap", type=float, default=0.02)
    ap.add_argument("--out", default="results/phase9/t8_3_f16_rerun.json")
    args = ap.parse_args(argv)

    ref, cand = means(Path(args.ref), args.model), means(Path(args.cand), args.model)
    feats = [f for f in FEATURES if f in ref and f in cand]
    rows = [{"feature": f, "f16": ref[f], "q4_k_m": cand[f], "delta": cand[f] - ref[f]}
            for f in feats]
    rank = lambda m: sorted(feats, key=lambda f: m[f])  # noqa: E731
    order_same = rank(ref) == rank(cand)
    max_abs = max(abs(r["delta"]) for r in rows) if rows else float("nan")
    verdict = "unchanged" if order_same and max_abs < args.min_gap else "CHANGED"
    report = {
        "task": "T8.3 F16 re-run", "model": args.model, "rows": rows,
        "feature_order_f16": rank(ref), "feature_order_q4_k_m": rank(cand),
        "feature_order_identical": order_same, "max_abs_delta": max_abs,
        "min_interpreted_gap": args.min_gap, "conclusions": verdict,
        "missing_features": [f for f in FEATURES if f not in feats],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['feature']}  F16 {r['f16']:.4f}  Q4_K_M {r['q4_k_m']:.4f}  Δ {r['delta']:+.4f}")
    print(f"order identical={order_same}  max|Δ|={max_abs:.4f}  -> conclusions {verdict}")
    return 0 if verdict == "unchanged" else 1


if __name__ == "__main__":
    raise SystemExit(main())
