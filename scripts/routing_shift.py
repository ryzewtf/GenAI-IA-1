#!/usr/bin/env python3
"""How much routing itself moves between OLMoE checkpoints on identical tokens — audit L4.

Predictability (Table 5) barely changes across 0924 → 0125 → 0125-Instruct, but that says nothing
about whether the *same* tokens are sent to the *same* experts. This measures it directly: per
layer, mean top-k set overlap |S_a ∩ S_b| / k and exact-set agreement between two checkpoints,
over every token of the corpus (the three traces are token-identical; checked per shard), plus the
overlap broken out by the reference checkpoint's routing margin (quintiles, as in Phase 8).

Writes results/phase9/routing_shift.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.traces.reader import TraceReader

ROOT = Path("FULL RESULTS/vllm/traces")
PAIRS = {
    "continued_pretraining (0924 -> 0125)": ("olmoe-0924", "olmoe-0125"),
    "post_training (0125 -> 0125-instruct)": ("olmoe-0125", "olmoe-0125-instruct"),
}
CHUNK = 262_144


def overlap_rows(a: np.ndarray, b: np.ndarray, n_experts: int) -> np.ndarray:
    mask = np.zeros((a.shape[0], n_experts), dtype=bool)
    mask[np.arange(a.shape[0])[:, None], a.astype(np.intp)] = True
    return np.take_along_axis(mask, b.astype(np.intp), axis=1).sum(axis=1)


def main() -> int:
    readers = {m: TraceReader(ROOT, m, "mixed-v2", validate_sizes=False)
               for m in {x for p in PAIRS.values() for x in p}}
    toks = None
    for m, r in readers.items():
        t = np.asarray(r.tokens()["token_id"])
        if toks is None:
            toks = t
        elif t.shape != toks.shape or not np.array_equal(t, toks):
            raise SystemExit(f"{m}: token stream differs; pairs are not on identical tokens")
    n_tokens = int(toks.shape[0])
    out: dict = {"n_tokens": n_tokens, "pairs": {}}
    md = ["# Routing shift between OLMoE checkpoints (identical tokens)", "",
          f"All {n_tokens:,} tokens. Overlap = |S_a ∩ S_b| / k; chance = k/N = 0.125. Margin "
          "quintiles use the first checkpoint's k-th minus (k+1)-th routing probability.", ""]
    for name, (ma, mb) in PAIRS.items():
        ra, rb = readers[ma], readers[mb]
        k, n_exp, L = ra.top_k, ra.n_experts, ra.n_moe_layers
        per_layer = []
        for layer in range(L):
            hits = np.empty(n_tokens, dtype=np.int16)
            margin = np.empty(n_tokens, dtype=np.float32)
            for lo in range(0, n_tokens, CHUNK):
                sl = slice(lo, min(lo + CHUNK, n_tokens))
                hits[sl] = overlap_rows(ra.topk_sets(layer, sl), rb.topk_sets(layer, sl), n_exp)
                margin[sl] = ra.margins(layer, sl)
            ov = hits / k
            edges = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
            q = np.searchsorted(edges, margin, side="right")
            per_layer.append({
                "layer": layer,
                "overlap": float(ov.mean()),
                "exact": float((hits == k).mean()),
                "overlap_by_margin_quintile": [float(ov[q == i].mean()) for i in range(5)],
            })
        o = np.array([p["overlap"] for p in per_layer])
        e = np.array([p["exact"] for p in per_layer])
        mq = np.array([p["overlap_by_margin_quintile"] for p in per_layer]).mean(axis=0)
        out["pairs"][name] = {
            "members": [ma, mb], "mean_overlap": float(o.mean()), "min_overlap": float(o.min()),
            "min_layer": int(o.argmin()), "mean_exact": float(e.mean()),
            "mean_overlap_by_margin_quintile": mq.tolist(), "per_layer": per_layer,
        }
        md += [f"## {name}", "",
               f"mean overlap **{o.mean():.4f}** (min {o.min():.4f} at layer {o.argmin()}), "
               f"exact-set agreement {e.mean():.4f}; slots moved {1 - o.mean():.3f}.",
               "overlap by margin quintile (smallest → largest): "
               + ", ".join(f"{x:.3f}" for x in mq), "",
               "| layer | overlap | exact |", "|---:|---:|---:|"]
        md += [f"| {p['layer']} | {p['overlap']:.4f} | {p['exact']:.4f} |" for p in per_layer]
        md.append("")
        print(f"{name}: overlap {o.mean():.4f} min {o.min():.4f} exact {e.mean():.4f} by-margin {np.round(mq, 3)}",
              flush=True)
    for r in readers.values():
        r.close()
    dst = Path("results/phase9")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "routing_shift.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (dst / "routing_shift.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote results/phase9/routing_shift.{json,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
