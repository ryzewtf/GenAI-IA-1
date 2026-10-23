#!/usr/bin/env python3
"""T10.3 extension — lookahead decay: router input h_{ℓ−j} → expert set S_ℓ, j ∈ {1,2,4,8}.

The ProMoE regime: how much of layer ℓ's routing is linearly decodable from the residual stream j
layers earlier (what a prefetcher actually has in hand). Inputs are built with the FV design, which
reads the router input of exactly the source layer (F4 reads ℓ−1), so j=1 reproduces F4 — a built-in
cross-check against results/probes. j=0 would be FV (validation only) and is not run. Same convex L-BFGS
softmax probe as F4, same captured-row subsample, same doc splits, standardizer fitted on train of
the *source* layer; only the label layer is shifted. Pure CPU on existing traces.

Targets: every model, ℓ at normalized depths {0.3, 0.5, 0.7, 0.9} (ℓ−j < 0 is skipped).

Usage: python -m scripts.phase10_lookahead [--models a,b] [--root E:/temp/traces]
       [--probe-kwargs '{"lbfgs_maxiter": 2000, "l2": 0.001}'] [--out ...]

The v0.1 run used the F4 defaults (150 L-BFGS iterations), which the convergence study (audit L6)
showed stop short of the optimum; pass the converged settings and a new --out to refit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.phase6_metrics import ENGINE_BY_MODEL
from src.probes.evaluate import evaluate_on_test
from src.probes.features import CorpusIndex, build_features
from src.probes.train import _make_predictor
from src.traces.reader import TraceReader

LAGS = (1, 2, 4, 8)
DEPTHS = (0.3, 0.5, 0.7, 0.9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/temp/traces")
    ap.add_argument("--splits", default="corpora/mixed-v2.splits.json")
    ap.add_argument("--models", default=",".join(ENGINE_BY_MODEL))
    ap.add_argument("--out", default="results/phase10/lookahead.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-kwargs", default=None, help="JSON of probe fields, as in phase7_sweep")
    args = ap.parse_args(argv)
    pkw = json.loads(args.probe_kwargs) if args.probe_kwargs else {}

    splits = {int(k): v for k, v in json.loads(Path(args.splits).read_text("utf-8")).items()}
    out_path = Path(args.out)
    res = json.loads(out_path.read_text("utf-8")) if out_path.exists() else {}
    for model in args.models.split(","):
        root = Path(args.root) / ENGINE_BY_MODEL[model] / "traces"
        with TraceReader(root, model, "mixed-v2", doc_splits=splits, validate_sizes=False) as r:
            index = CorpusIndex.from_reader(r)
            mres = res.setdefault(model, {})
            for f in DEPTHS:
                target = round(f * (r.n_moe_layers - 1))
                labels = r.topk_sets(target)
                for j in LAGS:
                    src = target - j
                    key = f"{target}:{j}"
                    if src < 0 or key in mres:
                        continue
                    tr = build_features("FV", r, index, src, "train")
                    va = build_features("FV", r, index, src, "val", standardizer=tr.standardizer)
                    te = build_features("FV", r, index, src, "test", standardizer=tr.standardizer)
                    ys = [labels[x.row_index].astype("int32") for x in (tr, va, te)]
                    p = _make_predictor("F4", n_experts=r.n_experts, vocab_size=index.vocab_size,
                                        layer=target, seed=args.seed)
                    for name, value in pkw.items():
                        if not hasattr(p, name):
                            raise ValueError(f"probe_kwargs[{name!r}] is not a field of {type(p).__name__}")
                        setattr(p, name, value)
                    p.fit(tr.X, ys[0], va.X, ys[1])
                    m = evaluate_on_test(p, te.X, ys[2], n_experts=r.n_experts, top_k=r.top_k)
                    mres[key] = {"target_layer": target, "source_layer": src, "lag": j,
                                 "depth": f, "mi_ratio": m["family_b"]["ratio"],
                                 "mi_attainable": m["family_b"]["ratio_attainable"],
                                 "converged": p.report.hyperparams.get("converged"),
                                 "n_iter": p.report.hyperparams.get("n_iter"),
                                 "set_agreement@k": m["family_a"]["set_agreement@k"],
                                 "recall@k": m["family_a"][f"recall@{r.top_k}"],
                                 "n_test_rows": m["n_rows"]}
                    print(f"{model:22} ℓ={target:2} j={j} Î/H {mres[key]['mi_ratio']:.4f} "
                          f"agree {mres[key]['set_agreement@k']:.4f}", flush=True)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
