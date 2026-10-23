#!/usr/bin/env python3
"""Holm correction over every interpreted difference in ``results/audit/bootstrap.json`` — CHECKLISTS [H].

The bootstrap marks a difference "resolved" when its per-comparison 95% interval excludes 0. The
draft interprets many such differences at once (four pairs × ten predictors, plus the within-model
ladder), so this applies Holm's step-down procedure to the whole family at α = 0.05.

p-values come from the normal approximation with the bootstrap standard error,
``p = 2 Φ(−|θ̂| / se)``. A percentile p-value cannot go below 1/(B+1) ≈ 0.001 with B = 1000, which
is above Holm's smallest threshold α/m for a family of this size, so it could never reject.

    python -m scripts.audit_holm [--bootstrap results/audit/bootstrap.json] [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def p_two_sided(point: float, se: float) -> float:
    if se <= 0:
        return 0.0 if point != 0 else 1.0
    return math.erfc(abs(point) / se / math.sqrt(2.0))


def holm(pvals: list[float], alpha: float) -> list[bool]:
    order = sorted(range(len(pvals)), key=pvals.__getitem__)
    reject = [False] * len(pvals)
    m = len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] > alpha / (m - rank):
            break
        reject[i] = True
    return reject


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="results/audit/bootstrap.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="results/audit")
    args = ap.parse_args(argv)
    boot = json.loads(Path(args.bootstrap).read_text(encoding="utf-8"))

    family = []
    for p in boot["pairs"]:
        s = p["mi_attainable"]
        family.append({"kind": "pair", "name": p["pair"], "feature": p["feature"], **_fields(s)})
    for r in boot["ladders"]:
        key = f"mi_attainable ({r['a']}-{r['b']})"
        family.append({"kind": "ladder", "name": r["model"], "feature": f"{r['a']}-{r['b']}", **_fields(r[key])})

    pvals = [c["p"] for c in family]
    for c, rej in zip(family, holm(pvals, args.alpha)):
        c["holm_reject"] = rej
    changed = [c for c in family if c["resolved"] and not c["holm_reject"]]
    out = {"alpha": args.alpha, "m": len(family), "method": "Holm step-down; p = 2Φ(−|θ̂|/se_boot)",
           "n_resolved_uncorrected": sum(c["resolved"] for c in family),
           "n_reject_holm": sum(c["holm_reject"] for c in family),
           "resolved_but_not_holm": changed, "family": family}
    dst = Path(args.out)
    (dst / "holm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"# Holm correction over {out['m']} interpreted differences (α = {args.alpha})", "",
          f"Resolved by the per-comparison 95% interval: {out['n_resolved_uncorrected']}. "
          f"Rejected by Holm: {out['n_reject_holm']}.", "",
          "## Resolved per comparison but not after Holm", "",
          "| kind | pair / model | feature | Δ | se | p |", "|---|---|---|---:|---:|---:|"]
    md += [f"| {c['kind']} | {c['name']} | {c['feature']} | {c['point']:+.4f} | {c['se']:.4f} | {c['p']:.3g} |"
           for c in changed] or ["| — | — | — | — | — | — |"]
    md += ["", "## Pairs", "", "| pair | feature | Δ | se | p | Holm |", "|---|---|---:|---:|---:|---|"]
    md += [f"| {c['name']} | {c['feature']} | {c['point']:+.4f} | {c['se']:.4f} | {c['p']:.2g} | "
           f"{'✓' if c['holm_reject'] else 'n.s.'} |" for c in family if c["kind"] == "pair"]
    (dst / "holm.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:8 + len(changed)]))
    return 0


def _fields(s: dict) -> dict:
    return {"point": s["point"], "se": s["se"], "basic_lo": s["basic_lo"], "basic_hi": s["basic_hi"],
            "resolved": bool(s["resolved"]), "p": p_two_sided(s["point"], s["se"])}


if __name__ == "__main__":
    raise SystemExit(main())
