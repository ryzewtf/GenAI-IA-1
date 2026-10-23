#!/usr/bin/env python3
"""Router-score regression probes F4R / FVR on every model and layer — audit L3, M13, L7.

For each MoE layer ℓ, on the hidden-state subsample (the same rows as F4/F5/FV):

* **FVR**: router input of ℓ → router scores of ℓ. Its test R² is the closed-form alignment
  gate (plan T3.3) that the paper cites and that previously had no saved artifact (M13); its
  attainable-normalised Î shows how close a probe can get to the ``H − log2 k`` ceiling when it
  is given complete information (L3).
* **F4R**: router input of ℓ−1 → router scores of ℓ. The one-layer-ahead estimate that is not
  limited by a softmax output family (L3).
* **Train-size curve** for F4R (fractions of the train documents): whether F4R is still rising
  with data, i.e. whether "data-limited" is a supported reading (L3).

Writes ``<out>/<model>__F4R.json`` / ``__FVR.json`` in the probe-result schema (so
``src.analysis.tables`` reads them like any other feature), and per-document aggregates for the
bootstrap into ``<agg>/<model>__<feature>.npz``.

Usage:
    python -m scripts.audit_ridge_probe --models olmoe-0125 [--layers 0 1 2] [--root "FULL RESULTS"]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.phase6_metrics import ENGINE_BY_MODEL
from src.metrics.bootstrap import save_aggregates
from src.probes.base import Standardizer
from src.probes.evaluate import evaluate_on_test
from src.probes.features import CorpusIndex
from src.probes.ridge import RidgeScoreProbe, r2_score, score_targets
from src.probes.train import _atomic_write_json
from src.traces.reader import TraceReader

FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2)


def captured_logits(reader: TraceReader, rows: np.ndarray) -> np.ndarray:
    """Stored routing tensor for the given (captured) rows, all layers: ``(n, L, N)`` float16."""
    out = np.empty((rows.size, reader.n_moe_layers, reader.n_experts), dtype=np.float16)
    for handle, ls, le, os_, oe in reader._spans(slice(0, int(rows.max()) + 1)):
        base = os_ - ls  # global row = local row + base
        sel = (rows >= os_) & (rows < oe)
        if sel.any():
            out[sel] = handle.logits[rows[sel] - base]
    return out


def fit_eval(feature, layer, model_layer, Xs, Z, sets, docs, n_experts, top_k, meta, train_frac_rows=None):
    tr, va, te = "train", "val", "test"
    probe = RidgeScoreProbe(n_experts=n_experts, top_k=top_k, feature=feature, layer=layer)
    probe.fit(Xs[tr], Z[tr], Xs[va], Z[va], sets[va])
    metrics = evaluate_on_test(probe, Xs[te], sets[te], n_experts=n_experts, top_k=top_k,
                               doc_ids=docs[te])
    agg = metrics.pop("doc_aggregates")
    metrics["r2_test"] = r2_score(probe.predict_scores(Xs[te]), Z[te])
    record = {
        "layer": layer, "model_layer": model_layer, "status": "ok", "fit": probe.report,
        "metrics": metrics, "meta": meta,
        "rows": {s: int(sets[s].shape[0]) for s in (tr, va, te)},
    }
    curve = []
    for frac, sub in (train_frac_rows or {}).items():
        p = RidgeScoreProbe(n_experts=n_experts, top_k=top_k, feature=feature, layer=layer)
        p.fit(Xs[tr][sub], Z[tr][sub], Xs[va], Z[va], sets[va])
        m = evaluate_on_test(p, Xs[te], sets[te], n_experts=n_experts, top_k=top_k)
        curve.append({"train_fraction": frac, "n_train_rows": int(sub.size),
                      "ratio_attainable": m["family_b"]["ratio_attainable"],
                      "set_agreement@k": m["family_a"]["set_agreement@k"]})
    if curve:
        curve.append({"train_fraction": 1.0, "n_train_rows": int(sets[tr].shape[0]),
                      "ratio_attainable": metrics["family_b"]["ratio_attainable"],
                      "set_agreement@k": metrics["family_a"]["set_agreement@k"]})
        record["train_size_curve"] = curve
    return record, agg


def run_model(model, args, doc_splits):
    engine = ENGINE_BY_MODEL[model]
    root = Path(args.root) / engine / "traces"
    fast = Path(args.fast_root) / engine / "traces" if args.fast_root else root
    out_dir, agg_dir = Path(args.out), Path(args.agg)
    # Router inputs are stored token-major ([row, layer, dim]), so one layer is a strided read of
    # the whole file: fine on NVMe, minutes of seeking per layer on a hard disk. Hidden states,
    # sets and tokens come from the fast copy; the routing scores (absent from that copy) are read
    # once, for the captured rows only, from the full trace.
    with TraceReader(root, model, args.corpus, doc_splits=doc_splits,
                     validate_sizes=False) as full,          TraceReader(fast, model, args.corpus, doc_splits=doc_splits,
                     validate_sizes=False) as reader:
        if full.shard_ids != reader.shard_ids or full.run_config_sha256 != reader.run_config_sha256:
            raise RuntimeError(f"{model}: fast copy at {fast} does not match the full trace")
        index = CorpusIndex.from_reader(reader)
        cap = reader.captured_rows()
        if not np.array_equal(cap, full.captured_rows()):
            raise RuntimeError(f"{model}: captured rows differ between the two copies")
        rows = {s: np.intersect1d(index.rows(s), cap) for s in ("train", "val", "test")}
        allrows = np.concatenate([rows[s] for s in ("train", "val", "test")])
        order = np.argsort(allrows)
        t0 = time.perf_counter()
        logits_all = np.empty((allrows.size, reader.n_moe_layers, reader.n_experts), np.float16)
        logits_all[order] = captured_logits(full, allrows[order])
        pos = {}
        lo = 0
        for s in ("train", "val", "test"):
            pos[s] = np.arange(lo, lo + rows[s].size)
            lo += rows[s].size
        print(f"{model}: logits for {allrows.size} captured rows in {time.perf_counter()-t0:.0f}s",
              flush=True)

        docs = {s: index.doc_ids[rows[s]] for s in rows}
        rng = np.random.default_rng(args.seed)
        train_docs = np.unique(docs["train"])
        frac_rows = {}
        for frac in FRACTIONS:
            keep = rng.choice(train_docs, size=max(1, int(round(frac * train_docs.size))), replace=False)
            frac_rows[frac] = np.flatnonzero(np.isin(docs["train"], keep))

        n_experts, top_k = reader.n_experts, reader.top_k
        layers = args.layers if args.layers else list(range(reader.n_moe_layers))
        docs_out = {}
        for feat in ("F4R", "FVR"):
            path = out_dir / f"{model}__{feat}.json"
            docs_out[feat] = (json.loads(path.read_text(encoding="utf-8")) if path.exists() and not args.force
                              else {"model": model, "feature": feat, "n_moe_layers": reader.n_moe_layers,
                                    "n_experts": n_experts, "top_k": top_k,
                                    "run_config_sha256": reader.run_config_sha256,
                                    "logit_tensor_used": reader.logit_tensor_used,
                                    "shard_ids": reader.shard_ids,
                                    "entropy_estimator": "miller_madow", "layers": {}})
        aggs = {"F4R": {}, "FVR": {}}
        for feat in aggs:
            p = agg_dir / f"{model}__{feat}.npz"
            if p.exists() and not args.force:
                with np.load(p) as z:
                    for key in z.files:
                        L, name = key.split("/", 1)
                        aggs[feat].setdefault(L, {})[name] = z[key]

        def hidden_std(layer):
            vals = {s: reader.hidden(layer, rows[s]) for s in rows}
            st = Standardizer.fit(vals["train"])
            return {s: st.transform(v) for s, v in vals.items()}

        prev = None
        for layer in layers:
            todo = [f for f in ("F4R", "FVR") if str(layer) not in docs_out[f]["layers"]
                    and not (f == "F4R" and layer == 0)]
            if not todo:
                prev = None
                continue
            t1 = time.perf_counter()
            cur = hidden_std(layer)
            Z, sets = {}, {}
            info = None
            for s in rows:
                Z[s], info = score_targets(logits_all[pos[s], layer, :].astype(np.float32))
                sets[s] = reader.topk_sets(layer)[rows[s]].astype(np.int32)
            ml = reader.model_layer(layer)
            if "FVR" in todo:
                rec, agg = fit_eval("FVR", layer, ml, cur, Z, sets, docs, n_experts, top_k,
                                    {"hidden_layer": layer, "targets": info})
                docs_out["FVR"]["layers"][str(layer)] = rec
                aggs["FVR"][str(layer)] = agg
            if "F4R" in todo:
                if prev is None or prev[0] != layer - 1:
                    prev = (layer - 1, hidden_std(layer - 1))
                rec, agg = fit_eval("F4R", layer, ml, prev[1], Z, sets, docs, n_experts, top_k,
                                    {"hidden_layer": layer - 1, "targets": info},
                                    train_frac_rows=frac_rows if not args.no_curve else None)
                docs_out["F4R"]["layers"][str(layer)] = rec
                aggs["F4R"][str(layer)] = agg
            prev = (layer, cur)
            for feat in todo:
                docs_out[feat]["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                _atomic_write_json(out_dir / f"{model}__{feat}.json", docs_out[feat])
                save_aggregates(agg_dir / f"{model}__{feat}.npz", aggs[feat])
            fv = docs_out["FVR"]["layers"].get(str(layer), {}).get("metrics", {})
            f4 = docs_out["F4R"]["layers"].get(str(layer), {}).get("metrics", {})
            print(f"{model} L{layer}: FVR R2={fv.get('r2_test', float('nan')):.4f} "
                  f"agree={fv.get('family_a', {}).get('set_agreement@k', float('nan')):.4f} "
                  f"att={fv.get('family_b', {}).get('ratio_attainable', float('nan')):.3f} | "
                  f"F4R agree={f4.get('family_a', {}).get('set_agreement@k', float('nan')):.4f} "
                  f"att={f4.get('family_b', {}).get('ratio_attainable', float('nan')):.3f} "
                  f"({time.perf_counter()-t1:.0f}s)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="FULL RESULTS")
    ap.add_argument("--fast-root", default="E:/temp/traces",
                    help="NVMe copy with hidden/topk/tokens (no logits); '' to read all from --root")
    ap.add_argument("--corpus", default="mixed-v2")
    ap.add_argument("--splits", default="corpora/mixed-v2.splits.json")
    ap.add_argument("--out", default="results/probes")
    ap.add_argument("--agg", default="results/probes_docagg")
    ap.add_argument("--models", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-curve", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    raw = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    doc_splits = {int(k): str(v) for k, v in raw.items()}
    rc = 0
    for model in args.models.split(","):
        try:
            run_model(model, args, doc_splits)
        except Exception as exc:  # report and continue with the next model
            import traceback
            traceback.print_exc()
            print(f"!! {model}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
