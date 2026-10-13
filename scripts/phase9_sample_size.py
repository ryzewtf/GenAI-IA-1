#!/usr/bin/env python3
"""T9.4 sample-size convergence: OLMoE metrics vs training-token budget N.

Val and test are held fixed (the corpus doc splits); only the TRAIN rows are subsampled, by whole
documents in a fixed seeded order, until the cumulative train-token count reaches N. Each N is a
nested superset of the smaller ones, so the curve is monotone in data, not in random draws.

The plan's grid is N ∈ {100k, 250k, 500k, 1M, 4M}. The collected OLMoE trace has ~1.22M train
tokens, so 4M is not reachable: the grid stops at the full train split and that is stated in the
output (``n_train_available``). Features: F1 (closed-form) and F2 (sparse AdamW) at early/mid/late
depth — the token-ID and prev-layer claims are the ones whose estimates depend on table coverage.

Usage: python -m scripts.phase9_sample_size [--model olmoe-0125] [--root E:/temp/traces]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.phase6_metrics import ENGINE_BY_MODEL
from src.probes.evaluate import evaluate_on_test
from src.probes.features import CorpusIndex, build_features
from src.probes.train import _make_predictor
from src.traces.reader import TraceReader

GRID = (100_000, 250_000, 500_000, 1_000_000, 4_000_000)


def train_subsets(index: CorpusIndex, grid, seed: int) -> dict[int, np.ndarray]:
    rows = index.rows("train")
    docs = np.unique(index.doc_ids[rows])
    np.random.default_rng(seed).shuffle(docs)
    by_doc = {int(d): rows[index.doc_ids[rows] == d] for d in docs}
    out, taken, total = {}, [], 0
    it = iter(docs)
    for n in sorted(grid):
        while total < n:
            d = next(it, None)
            if d is None:
                break
            taken.append(by_doc[int(d)])
            total += len(by_doc[int(d)])
        out[total] = np.sort(np.concatenate(taken))  # keyed by the actual (whole-doc) count
        if total < n:  # grid point beyond the data: the full split is the last point
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmoe-0125")
    ap.add_argument("--root", default="E:/temp/traces")
    ap.add_argument("--splits", default="corpora/mixed-v2.splits.json")
    ap.add_argument("--features", nargs="+", default=["F1", "F2"])
    ap.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/phase9/t9_4_sample_size.json")
    args = ap.parse_args(argv)

    splits = {int(k): v for k, v in json.loads(Path(args.splits).read_text("utf-8")).items()}
    root = Path(args.root) / ENGINE_BY_MODEL[args.model] / "traces"
    res = {"model": args.model, "grid_requested": list(GRID), "points": []}
    with TraceReader(root, args.model, "mixed-v2", doc_splits=splits, validate_sizes=False) as r:
        full = CorpusIndex.from_reader(r)
        res["n_train_available"] = int(len(full.rows("train")))
        subsets = train_subsets(full, GRID, args.seed)
        for L in sorted({round(f * (r.n_moe_layers - 1)) for f in args.depths}):
            for n, tr in subsets.items():
                index = replace(full, split_rows={**full.split_rows, "train": tr})
                for feat in args.features:
                    tr_f = build_features(feat, r, index, L, "train")
                    va = build_features(feat, r, index, L, "val", standardizer=tr_f.standardizer)
                    te = build_features(feat, r, index, L, "test", standardizer=tr_f.standardizer)
                    p = _make_predictor(feat, n_experts=r.n_experts, vocab_size=index.vocab_size,
                                        layer=L, seed=args.seed)
                    p.fit(tr_f.X, tr_f.y, va.X, va.y)
                    m = evaluate_on_test(p, te.X, te.y, n_experts=r.n_experts, top_k=r.top_k)
                    pt = {"layer": L, "n_train_tokens": int(n), "feature": feat,
                          "mi_ratio": m["family_b"]["ratio"],
                          "set_agreement@k": m["family_a"]["set_agreement@k"]}
                    res["points"].append(pt)
                    print(f"L{L:2} N={n:>9,} {feat} Î/H {pt['mi_ratio']:.4f} "
                          f"agree {pt['set_agreement@k']:.4f}", flush=True)
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
