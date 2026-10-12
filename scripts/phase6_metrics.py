"""Phase 6 driver — corpus-wide marginal + load-balance index across the panel (T6.3/T6.4).

Runs on CPU, memory-bounded (memmap + chunked accumulation; never materialises a model's
logit array). For each panel model it opens the shard-assembled trace with ``TraceReader``,
accumulates slot-occupancy counts per MoE layer over the whole corpus, and reports the
Miller-Madow marginal entropy and the load-balance index ``H / log2(n_experts)`` (T6.4) per
layer. The load-balance index is the aux-loss confound *instrument* that carries the T9.4
argument, so these are real Phase 9 inputs, not just a reader smoke test.

Usage:
    python -m scripts.phase6_metrics [--root "FULL RESULTS"] [--corpus mixed-v2] \
        [--out results/phase6_load_balance] [--chunk-tokens N] [--models m1,m2,...]

Writes ``<out>.json`` (full per-layer tables) and ``<out>.csv`` (one row per model+layer),
and prints a per-model summary (mean / worst-layer load-balance index).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from src.metrics.information import entropy_mm, load_balance_index
from src.traces.reader import TraceReader

# Which engine panel each model's traces live under (FULL RESULTS/<engine>/traces/...).
ENGINE_BY_MODEL = {
    "gpt-oss-20b": "llamacpp",
    "deepseek-v2-lite": "llamacpp",
    "gemma-4-26b-a4b": "llamacpp",
    "olmoe-0125": "vllm",
    "olmoe-0125-instruct": "vllm",
    "olmoe-0924": "vllm",
    "qwen3-30b-a3b": "vllm",
}


def model_counts_per_layer(reader: TraceReader, chunk_tokens: int) -> np.ndarray:
    """Accumulate slot-occupancy counts ``(n_moe_layers, n_experts)`` over the whole corpus.

    Chunked so peak memory is O(chunk_tokens x n_moe_layers x k), not O(corpus)."""
    n_layers = reader.n_moe_layers
    n_experts = reader.n_experts
    counts = np.zeros((n_layers, n_experts), dtype=np.int64)
    for window in reader.iter_chunks(chunk_tokens=chunk_tokens):
        for layer in range(n_layers):
            sets = reader.topk_sets(layer, window)  # (chunk_n, k) int16, model's own indices
            flat = sets.reshape(-1).astype(np.int64, copy=False)
            counts[layer] += np.bincount(flat, minlength=n_experts)[:n_experts]
    return counts


def run_model(root: Path, engine: str, model: str, corpus: str, chunk_tokens: int) -> dict:
    trace_root = root / engine / "traces"
    t0 = time.perf_counter()
    # Sizes already T5.3-validated for this dataset; skip the re-check for speed.
    with TraceReader(trace_root, model, corpus, validate_sizes=False) as reader:
        n_experts = reader.n_experts
        uniform_bits = math.log2(n_experts)
        counts = model_counts_per_layer(reader, chunk_tokens)
        layers = []
        for trace_layer in range(reader.n_moe_layers):
            c = counts[trace_layer]
            h = entropy_mm(c)
            lbi = load_balance_index(c, n_experts, is_counts=True)
            layers.append(
                {
                    "trace_layer": trace_layer,
                    "model_layer": reader.model_layer(trace_layer),
                    "entropy_bits": h,
                    "load_balance_index": lbi,
                    "n_slots": int(c.sum()),
                    "experts_unused": int((c == 0).sum()),
                }
            )
        lbis = np.array([x["load_balance_index"] for x in layers])
        worst = int(np.argmin(lbis))
        summary = {
            "model": model,
            "engine": engine,
            "run_config_sha256": reader.run_config_sha256[:8],
            "n_moe_layers": reader.n_moe_layers,
            "n_experts": n_experts,
            "top_k": reader.top_k,
            "n_tokens": int(reader.n_tokens),
            "uniform_entropy_bits": uniform_bits,
            "lbi_mean": float(lbis.mean()),
            "lbi_min": float(lbis.min()),
            "lbi_worst_layer": layers[worst]["model_layer"],
            "seconds": round(time.perf_counter() - t0, 1),
            "layers": layers,
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 marginal + load-balance driver (T6.3/T6.4)")
    ap.add_argument("--root", default="FULL RESULTS", help="dataset root holding <engine>/traces/")
    ap.add_argument("--corpus", default="mixed-v2")
    ap.add_argument("--out", default="results/phase6_load_balance", help="output path stem")
    ap.add_argument("--chunk-tokens", type=int, default=100_000)
    ap.add_argument("--models", default=None, help="comma-separated subset; default = all 7")
    args = ap.parse_args(argv)

    root = Path(args.root)
    models = args.models.split(",") if args.models else list(ENGINE_BY_MODEL)

    results = []
    for model in models:
        engine = ENGINE_BY_MODEL.get(model)
        if engine is None:
            print(f"!! {model}: unknown model (no engine mapping); skipped", file=sys.stderr)
            continue
        try:
            summary = run_model(root, engine, model, args.corpus, args.chunk_tokens)
        except Exception as exc:  # surface which model broke, keep going
            print(f"!! {model}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(summary)
        print(
            f"{model:22} exp={summary['n_experts']:<3} layers={summary['n_moe_layers']:<2} "
            f"LBI mean={summary['lbi_mean']:.4f} min={summary['lbi_min']:.4f} "
            f"@L{summary['lbi_worst_layer']:<2} ({summary['seconds']}s)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "engine", "n_experts", "top_k", "model_layer", "trace_layer",
                    "entropy_bits", "load_balance_index", "n_slots", "experts_unused"])
        for s in results:
            for L in s["layers"]:
                w.writerow([s["model"], s["engine"], s["n_experts"], s["top_k"],
                            L["model_layer"], L["trace_layer"], f"{L['entropy_bits']:.6f}",
                            f"{L['load_balance_index']:.6f}", L["n_slots"], L["experts_unused"]])
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.csv')} "
          f"({len(results)} model(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
