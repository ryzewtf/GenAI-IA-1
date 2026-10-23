#!/usr/bin/env python3
"""Document-bootstrap intervals for every reported number and every interpreted difference — audit L7.

Reads the per-document aggregates written by the probe sweeps (``--agg-dir``) and the ridge probe
driver, and writes ``results/audit/bootstrap.{json,md}``:

* **cells**: per (model, feature), the layer-mean of each metric with a 95% percentile interval;
* **pairs**: per pair and feature, the paired difference (second − first member) of layer means,
  with its interval, and whether the interval excludes zero ("resolved"). This replaces the fixed
  ±0.02 band, which the draft applied inconsistently;
* **ladders**: within-model paired differences between predictors (e.g. F4R − F4, FVR − F4R);
* **strata**: per domain and per language, each stratum scored against **its own** marginal
  entropy on **the same rows** as the feature (audit M3), as an attainable-normalised Î with an
  interval. Strata are groups of whole documents, so they come straight out of the aggregates.

One draw of document multiplicities is shared by every cell (keyed by doc_id), so all differences
are paired over the same resampled text.

Usage: python -m scripts.audit_bootstrap [--agg results/probes_docagg] [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.metrics.bootstrap import (
    DocAggregates,
    draw_weights,
    layer_mean,
    load_aggregates,
    replicate_metrics,
    summarize,
)

PAIRS = {
    "pair_g": ("gpt-oss-20b", "olmoe-0125"),
    "pair_t": ("qwen3-30b-a3b", "gemma-4-26b-a4b"),
    "pair_a": ("olmoe-0924", "olmoe-0125"),
    "post_training": ("olmoe-0125", "olmoe-0125-instruct"),
}
LADDER_DIFFS = [("F4R", "F4"), ("F4R", "F5"), ("F5", "F4"), ("FVR", "F4R"), ("FV", "F4"),
                ("F6", "F2"), ("F2", "F1"), ("F4R", "F2")]
METRICS = ("mi_attainable", "mi_ratio", "recall@k", "recall@2k", "recall@4k", "exact_match")


def metric_key(metric: str, k: int) -> str:
    return {"recall@k": f"recall@{k}", "recall@2k": f"recall@{2 * k}",
            "recall@4k": f"recall@{4 * k}"}.get(metric, metric)


def subset(agg: DocAggregates, keep: np.ndarray) -> DocAggregates:
    return DocAggregates(
        doc_ids=agg.doc_ids[keep], rows=agg.rows[keep], ce_bits_sum=agg.ce_bits_sum[keep],
        exact_sum=agg.exact_sum[keep], recall_m=agg.recall_m,
        recall_hits_sum=agg.recall_hits_sum[:, keep], expert_counts=agg.expert_counts[keep],
        top_k=agg.top_k, n_experts=agg.n_experts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", nargs="+", default=["results/probes_docagg"])
    ap.add_argument("--corpus", default="corpora/mixed-v2.jsonl")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/audit")
    args = ap.parse_args(argv)

    cells: dict[tuple[str, str], dict[int, DocAggregates]] = {}
    for d in args.agg:
        for p in sorted(Path(d).glob("*__*.npz")):
            model, feature = p.stem.split("__")
            # the same probe name in two dirs (e.g. capped vs converged F4) must not shadow silently
            if (model, feature) in cells:
                raise SystemExit(f"duplicate aggregates for {model} {feature}: {p}")
            cells[(model, feature)] = load_aggregates(p)
    if not cells:
        raise SystemExit("no aggregates found")

    # "lang_no_flores" drops the FLORES-200 parallel documents, whose translations are split
    # independently across train/val/test (audit M12): if the language contrast survives without
    # them, cross-split parallel text is not what drives it.
    strata = {"domain": {}, "lang": {}, "lang_no_flores": {}}
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                d = int(r["doc_id"])
                strata["domain"][d] = str(r["domain"])
                strata["lang"][d] = str(r.get("lang") or "")
                if r.get("source") != "flores200":
                    strata["lang_no_flores"][d] = str(r.get("lang") or "")

    universe = np.unique(np.concatenate([a.doc_ids for c in cells.values() for a in c.values()]))
    weights = draw_weights(universe, args.n_boot, seed=args.seed)

    reps: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (model, feature), per_layer in sorted(cells.items()):
        k = next(iter(per_layer.values())).top_k
        layer_reps = [replicate_metrics(a, weights) for _, a in sorted(per_layer.items())]
        reps[(model, feature)] = {m: layer_mean([r[metric_key(m, k)] for r in layer_reps])
                                  for m in METRICS}

    out: dict = {"n_boot": args.n_boot, "n_docs_universe": int(universe.size), "cells": [],
                 "pairs": [], "ladders": [], "strata": []}
    for (model, feature), r in sorted(reps.items()):
        out["cells"].append({"model": model, "feature": feature, "n_layers": len(cells[(model, feature)]),
                             **{m: summarize(v) for m, v in r.items()}})

    def diff(a, b, metric):
        s = summarize(reps[b][metric] - reps[a][metric])
        s["resolved"] = bool(s["basic_lo"] > 0 or s["basic_hi"] < 0)
        return s

    features = sorted({f for _, f in reps})
    for name, (ma, mb) in PAIRS.items():
        for f in features:
            if (ma, f) in reps and (mb, f) in reps:
                out["pairs"].append({"pair": name, "first": ma, "second": mb, "feature": f,
                                     **{m: diff((ma, f), (mb, f), m) for m in METRICS}})
    for model in sorted({m for m, _ in reps}):
        for fa, fb in LADDER_DIFFS:
            if (model, fa) in reps and (model, fb) in reps:
                # common layers only, so a difference never mixes layer sets
                common = sorted(set(cells[(model, fa)]) & set(cells[(model, fb)]))
                k = cells[(model, fa)][common[0]].top_k
                ra = [replicate_metrics(cells[(model, fa)][L], weights) for L in common]
                rb = [replicate_metrics(cells[(model, fb)][L], weights) for L in common]
                rec = {"model": model, "a": fa, "b": fb, "n_layers": len(common)}
                for m in ("mi_attainable", "recall@k"):
                    key = metric_key(m, k)
                    s = summarize(layer_mean([x[key] for x in ra]) - layer_mean([x[key] for x in rb]))
                    s["resolved"] = bool(s["basic_lo"] > 0 or s["basic_hi"] < 0)
                    rec[f"{m} ({fa}-{fb})"] = s
                out["ladders"].append(rec)

    for (model, feature), per_layer in sorted(cells.items()):
        if feature not in ("F1", "F2", "F4", "F4R", "F5"):
            continue
        for kind, mapping in strata.items():
            labels = defaultdict(list)
            n_docs_by: dict[str, int] = {}
            for L, a in sorted(per_layer.items()):
                lab = np.array([mapping.get(int(d), "") for d in a.doc_ids])
                for value in np.unique(lab):
                    if kind.startswith("lang") and value == "":
                        continue
                    keep = lab == value
                    if keep.sum() < 3:
                        continue
                    labels[value].append(replicate_metrics(subset(a, keep), weights))
                    n_docs_by[value] = max(n_docs_by.get(value, 0), int(keep.sum()))
            for value, layer_reps in sorted(labels.items()):
                k = next(iter(per_layer.values())).top_k
                out["strata"].append({
                    "model": model, "feature": feature, "kind": kind, "stratum": value,
                    "n_layers": len(layer_reps), "n_test_docs": n_docs_by.get(value, 0),
                    "mi_attainable": summarize(layer_mean([x["mi_attainable"] for x in layer_reps])),
                    "recall@k": summarize(layer_mean([x[f"recall@{k}"] for x in layer_reps])),
                })

    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "bootstrap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def fmt(s):
        return f"{s['point']:.3f} [{s['basic_lo']:.3f}, {s['basic_hi']:.3f}]"

    def fmt_bc(s):
        return f"{s['bc']:.3f} [{s['basic_lo']:.3f}, {s['basic_hi']:.3f}]"

    md = [f"# Document-bootstrap intervals ({args.n_boot} replicates over {universe.size} test documents)", "",
          "Layer-mean values, 95% basic (reflected-percentile) intervals, which correct the downward "
          "resampling bias of plug-in entropy terms (visible in F0 and in strata of 11–20 documents; "
          "`bootstrap.json` also keeps the percentile interval and the bias). Differences are paired "
          "over the same resampled documents; **resolved** = interval excludes 0.", "", "## Cells", "",
          "| model | feature | Î/(H−log2k) | Î/H | recall@k | recall@2k | recall@4k |", "|---|---|---|---|---|---|---|"]
    md += [f"| {c['model']} | {c['feature']} | {fmt(c['mi_attainable'])} | {fmt(c['mi_ratio'])} | "
           f"{fmt(c['recall@k'])} | {fmt(c['recall@2k'])} | {fmt(c['recall@4k'])} |" for c in out["cells"]]
    md += ["", "## Pairs (second − first)", "",
           "| pair | feature | Δ Î/(H−log2k) | Δ recall@k | Δ recall@2k |", "|---|---|---|---|---|"]
    for p in out["pairs"]:
        md.append(f"| {p['pair']} | {p['feature']} | {fmt(p['mi_attainable'])}{' ✓' if p['mi_attainable']['resolved'] else ''} | "
                  f"{fmt(p['recall@k'])}{' ✓' if p['recall@k']['resolved'] else ''} | {fmt(p['recall@2k'])} |")
    md += ["", "## Within-model predictor differences", "", "| model | a − b | Δ Î/(H−log2k) | Δ recall@k |", "|---|---|---|---|"]
    for r in out["ladders"]:
        a, b = r["a"], r["b"]
        md.append(f"| {r['model']} | {a} − {b} | {fmt(r[f'mi_attainable ({a}-{b})'])} | {fmt(r[f'recall@k ({a}-{b})'])} |")
    md += ["", "## Strata (own-marginal, same rows; bias-corrected points)", "", "| model | feature | kind | stratum | test docs | Î/(H−log2k) | recall@k |", "|---|---|---|---|---:|---|---|"]
    # small strata carry a plug-in bias of up to ~0.04, so they are shown bias-corrected (raw in the json)
    md += [f"| {s['model']} | {s['feature']} | {s['kind']} | {s['stratum']} | {s['n_test_docs']} | "
           f"{fmt_bc(s['mi_attainable'])} | {fmt_bc(s['recall@k'])} |" for s in out["strata"]]
    (dst / "bootstrap.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {dst}/bootstrap.{{json,md}}: {len(out['cells'])} cells, {len(out['pairs'])} pair rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
