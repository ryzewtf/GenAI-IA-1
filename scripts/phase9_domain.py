#!/usr/bin/env python3
"""T9.4 domain/language breakdown — the plan's PRIMARY cross-model comparison.

Reads the --doc-strata sweep outputs (one dir per model, or per model-layer shard), whose test
metrics carry ``metrics.doc_strata.{domain,lang}.<label> = {family_a, cross_entropy_bits, n_slots}``.

Per stratum the evaluator deliberately reports CE and Family A only, never a per-stratum Î (a
per-stratum H on a small stratum over 128 experts is too biased). The comparable quantity here is
therefore the **CE gain over the marginal within the stratum**:

    gain_bits  = CE(F0 | stratum) − CE(feature | stratum)
    gain_ratio = gain_bits / CE(F0 | stratum)

F0 is the train-split marginal, so both terms are cross-entropies against held-out routing in that
stratum and neither needs an entropy estimate. Values are means over the swept layers (5 normalized
depths per model). Rows with no F0 at that (layer, stratum) are dropped, not imputed.

Usage: python -m scripts.phase9_domain [--src E:/temp/dom] [--out results/phase9]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FEATURES = ["F1", "F2", "F3", "F4", "F5", "F6"]  # F0 is the baseline; FV never reported


def load(src: Path) -> dict:
    """model -> feature -> layer -> doc_strata"""
    acc: dict = defaultdict(lambda: defaultdict(dict))
    for path in sorted(src.glob("*/*__*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for L, rec in (d.get("layers") or {}).items():
            ds = (rec.get("metrics") or {}).get("doc_strata")
            if rec.get("status") == "ok" and ds:
                acc[d["model"]][d["feature"]][int(L)] = ds
    return acc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="E:/temp/dom")
    ap.add_argument("--out", default="results/phase9")
    args = ap.parse_args(argv)
    acc = load(Path(args.src))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in sorted(acc):
        f0 = acc[model].get("F0", {})
        for feat in FEATURES:
            per = defaultdict(list)  # (kind, label) -> [(gain_bits, gain_ratio, agree, slots)]
            for L, ds in acc[model].get(feat, {}).items():
                if L not in f0:
                    continue
                for kind, labels in ds.items():
                    for label, m in labels.items():
                        base = f0[L].get(kind, {}).get(label)
                        if not base:
                            continue
                        ce0, ce = base["cross_entropy_bits"], m["cross_entropy_bits"]
                        per[(kind, label or "(none)")].append(
                            (ce0 - ce, (ce0 - ce) / ce0, m["family_a"]["set_agreement@k"],
                             m["n_slots"]))
            for (kind, label), xs in sorted(per.items()):
                n = len(xs)
                rows.append({
                    "model": model, "kind": kind, "stratum": label, "feature": feat,
                    "n_layers": n,
                    "gain_bits": sum(x[0] for x in xs) / n,
                    "gain_ratio": sum(x[1] for x in xs) / n,
                    "set_agreement@k": sum(x[2] for x in xs) / n,
                    "mean_test_slots": sum(x[3] for x in xs) / n,
                })

    (out / "t9_4_domain.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(out / "t9_4_domain.csv", "w", newline="", encoding="utf-8") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # Markdown pivots: model x domain, one block per feature, for gain_ratio and agreement.
    md = ["**T9.4 domain breakdown — CE gain over marginal within stratum (ratio), mean of 5 "
          "normalized depths**", "",
          "gain_ratio = (CE_F0 − CE_feature)/CE_F0 on test rows of that domain. Not Î/H (no "
          "per-stratum entropy is estimated); comparable across models and domains.", ""]
    for kind in ("domain", "lang"):
        labels = sorted({r["stratum"] for r in rows if r["kind"] == kind})
        models = sorted({r["model"] for r in rows})
        for metric in ("gain_ratio", "set_agreement@k"):
            for feat in FEATURES:
                cell = {(r["model"], r["stratum"]): r[metric] for r in rows
                        if r["kind"] == kind and r["feature"] == feat}
                if not cell:
                    continue
                md.append(f"### {kind} · {feat} · {metric}")
                md.append("| model | " + " | ".join(labels) + " |")
                md.append("|---|" + "---:|" * len(labels))
                for m in models:
                    md.append(f"| {m} | " + " | ".join(
                        f"{cell[(m, l)]:.3f}" if (m, l) in cell else "--" for l in labels) + " |")
                md.append("")
    (out / "t9_4_domain.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out}/t9_4_domain.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
