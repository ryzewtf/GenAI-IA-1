#!/usr/bin/env python3
"""Aggregate results/probes/<model>__<feature>.json into results/phase7_summary.json (T9.1 input).

One row per (model, feature): layer-mean Family A (set_agreement@k, recall@k, exact_match) and
Family B (CE_norm, signed I/H), the count of layers with negative I (never clamped, §1.2), and the
count of layers refit after a val-flagged divergence. FV is tagged validation-only (§1.3): it is a
data-alignment check and must never be reported as a finding.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROBES = Path("results/probes")
OUT = Path("results/phase7_summary.json")
FEATURES = ("F0", "F1", "F2", "F3", "F4", "F4R", "F5", "F6", "FV", "FVR")


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default=str(PROBES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    rows = []
    for path in sorted(Path(args.probes).glob("*__*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("feature") not in FEATURES:
            continue
        fit = [L for L in d["layers"].values() if "metrics" in L]
        a = [L["metrics"]["family_a"] for L in fit]
        b = [L["metrics"]["family_b"] for L in fit]
        k = a[0]["top_k"] if a else None
        rows.append({
            "model": d["model"],
            "feature": d["feature"],
            "role": "validation_only" if d["feature"] in ("FV", "FVR") else "result",
            "mean_set_agreement_at_k": mean([x["set_agreement@k"] for x in a]),
            "mean_recall_at_k": mean([x[f"recall@{k}"] for x in a]) if k else None,
            "mean_exact_match": mean([x["exact_match"] for x in a]),
            "mean_ce_normalized": mean([x["ce_normalized"] for x in b]),
            "mean_mi_ratio": mean([x["ratio"] for x in b]),
            # Î/(H − log2 k): the k-invariant primary (audit L1); derived for older files.
            "mean_mi_attainable": mean([x["mi_bits"] / (x["entropy_bits"] - math.log2(k))
                                        for x in b]) if k else None,
            "n_negative_mi_layers": sum(1 for x in b if x["negative"]),
            "n_layers_ok": len(fit),
            "n_layers_total": len(d["layers"]),
            "n_refit_layers": sum(1 for L in fit if "refit_note" in (L.get("fit") or {})),
        })
    rows.sort(key=lambda r: (r["model"], FEATURES.index(r["feature"])))
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    neg = {}
    for r in rows:
        neg[r["feature"]] = neg.get(r["feature"], 0) + r["n_negative_mi_layers"]
    print(f"wrote {out}: {len(rows)} rows")
    print("negative-I layers by feature:", neg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
