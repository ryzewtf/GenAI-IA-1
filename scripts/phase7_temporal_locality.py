"""Phase 7 (split-free) — consecutive-token expert repetition across the panel (T7.4 / §1.7 / T9.5).

This is the descriptive statistic the Mixtral paper reports (first-choice repeats 23.6-28.4% at its
layer 15 vs a 12.5% random baseline) computed for every model and MoE layer in our panel. F3 is its
*learned* generalization (a linear map on the same conditioning variable), so this raw rate is F3's
floor and the gap between them is the interesting result — reporting both side by side is what makes
the "28%" vs systems-paper "99%" numbers comparable (plan T9.5).

Split-free on purpose: ``|S_t cap S_{t-1}| / k`` averaged over non-document-initial tokens needs only
``pos_in_doc`` (which lives in ``tokens.bin``), never the train/val/test split, so it runs today
without the corpus file. The split-conditioned probe fits (F0-F6/FV) are the separate, split-gated
step that still needs the corpus.

Usage:
    python -m scripts.phase7_temporal_locality [--root "FULL RESULTS"] [--corpus mixed-v2] \
        [--out results/phase7_temporal_locality] [--models m1,m2,...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.probes.features import CorpusIndex, consecutive_repetition_rate
from src.traces.reader import TraceReader

# Reuse the engine map from the Phase 6 driver.
from scripts.phase6_metrics import ENGINE_BY_MODEL


def index_without_splits(reader: TraceReader) -> CorpusIndex:
    """CorpusIndex built from the trace alone (empty split_rows).

    ``consecutive_repetition_rate(split=None)`` uses only ``n_tokens`` and ``pos_in_doc``, so the
    split row-sets — the one thing that needs the corpus file — are left empty here."""
    records = reader.tokens()
    token_ids = np.asarray(records["token_id"], dtype=np.int64)
    return CorpusIndex(
        token_ids=token_ids,
        doc_ids=np.asarray(records["doc_id"], dtype=np.int64),
        pos_in_doc=np.asarray(records["pos_in_doc"], dtype=np.int64),
        split_rows={},
        captured=reader.captured_token_ids(),
        vocab_size=int(token_ids.max()) + 1,
    )


def run_model(root: Path, engine: str, model: str, corpus: str) -> dict:
    trace_root = root / engine / "traces"
    t0 = time.perf_counter()
    with TraceReader(trace_root, model, corpus, validate_sizes=False) as reader:
        index = index_without_splits(reader)
        layers = [consecutive_repetition_rate(reader, index, L) for L in range(reader.n_moe_layers)]
        rates = np.array([x["repetition_rate"] for x in layers])
        peak = int(np.argmax(rates))
        return {
            "model": model,
            "engine": engine,
            "n_experts": reader.n_experts,
            "top_k": reader.top_k,
            "n_moe_layers": reader.n_moe_layers,
            "random_baseline": float(reader.top_k / reader.n_experts),
            "rep_rate_mean": float(rates.mean()),
            "rep_rate_peak": float(rates.max()),
            "rep_rate_peak_model_layer": layers[peak]["model_layer"],
            # mid-depth is the regime Mixtral's layer-15 figure sits in
            "rep_rate_mid": float(rates[len(rates) // 2]),
            "seconds": round(time.perf_counter() - t0, 1),
            "layers": layers,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 temporal-locality driver (T7.4/§1.7/T9.5)")
    ap.add_argument("--root", default="FULL RESULTS")
    ap.add_argument("--corpus", default="mixed-v2")
    ap.add_argument("--out", default="results/phase7_temporal_locality")
    ap.add_argument("--models", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    models = args.models.split(",") if args.models else list(ENGINE_BY_MODEL)
    results = []
    for model in models:
        engine = ENGINE_BY_MODEL.get(model)
        if engine is None:
            print(f"!! {model}: unknown model; skipped", file=sys.stderr)
            continue
        try:
            s = run_model(root, engine, model, args.corpus)
        except Exception as exc:
            print(f"!! {model}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(s)
        lift = s["rep_rate_mean"] / s["random_baseline"]
        print(
            f"{model:22} rep mean={s['rep_rate_mean']:.4f} peak={s['rep_rate_peak']:.4f}"
            f"@L{s['rep_rate_peak_model_layer']:<2} random={s['random_baseline']:.4f} "
            f"lift={lift:4.1f}x ({s['seconds']}s)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_experts", "top_k", "model_layer", "trace_layer",
                    "repetition_rate", "exact_set_repeat_rate", "random_baseline", "n_rows"])
        for s in results:
            for L in s["layers"]:
                w.writerow([s["model"], s["n_experts"], s["top_k"], L["model_layer"], L["layer"],
                            f"{L['repetition_rate']:.6f}", f"{L['exact_set_repeat_rate']:.6f}",
                            f"{L['random_baseline']:.6f}", L["n_rows"]])
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.csv')} ({len(results)} model(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
