"""Document-level bootstrap for every reported metric — audit L7.

Splits are document-level (plan T4.3), so the sampling unit is the test **document**, not the
token: tokens inside a document are strongly dependent, and a token bootstrap would make
intervals look several times tighter than they are.

The bootstrap works on per-document sufficient statistics written by
:func:`src.probes.evaluate.evaluate_on_test` (``doc_aggregates``): slot-CE sums, slot and row
counts, recall@m hits, exact matches and per-document expert counts. A replicate is a vector of
document multiplicities ``w``; every metric of a replicate is then a ratio of ``w``-weighted sums,
and ``H`` is re-estimated from the ``w``-weighted expert counts with the same Miller–Madow
correction, so the interval covers the entropy estimate as well as the CE.

**Paired comparisons.** All models read the same corpus, so test documents share ``doc_id``
across models, features and layers. Drawing the multiplicities once per replicate, keyed by
``doc_id``, and applying them to both sides of a difference gives a paired interval: the
document-to-document variation common to both sides cancels, as it should when two predictors or
two models are scored on the same text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "DocAggregates",
    "draw_weights",
    "replicate_metrics",
    "summarize",
    "save_aggregates",
    "load_aggregates",
]

LN2 = math.log(2.0)


@dataclass
class DocAggregates:
    """Per-document sufficient statistics for one (model, feature, layer) cell."""

    doc_ids: np.ndarray            # (D,)
    rows: np.ndarray               # (D,)
    ce_bits_sum: np.ndarray        # (D,)  sum over the document's slots of -log2 q_mix
    exact_sum: np.ndarray          # (D,)
    recall_m: np.ndarray           # (R,)
    recall_hits_sum: np.ndarray    # (R, D) sum over rows of |top-m ∩ S| / k
    expert_counts: np.ndarray      # (D, N)
    top_k: int
    n_experts: int

    @classmethod
    def from_mapping(cls, m: Mapping[str, np.ndarray]) -> "DocAggregates":
        return cls(
            doc_ids=np.asarray(m["doc_ids"], dtype=np.int64),
            rows=np.asarray(m["rows"], dtype=np.float64),
            ce_bits_sum=np.asarray(m["ce_bits_sum"], dtype=np.float64),
            exact_sum=np.asarray(m["exact_sum"], dtype=np.float64),
            recall_m=np.asarray(m["recall_m"], dtype=np.int64),
            recall_hits_sum=np.asarray(m["recall_hits_sum"], dtype=np.float64),
            expert_counts=np.asarray(m["expert_counts"], dtype=np.float64),
            top_k=int(np.asarray(m["top_k"])),
            n_experts=int(np.asarray(m["n_experts"])),
        )


def draw_weights(doc_universe: np.ndarray, n_boot: int, seed: int = 0) -> dict[int, np.ndarray]:
    """Multiplicities ``(n_boot,)`` per doc_id: a multinomial resample of the document universe.

    Returned as a mapping so every cell, whatever subset of documents it covers, looks its own
    documents up in the *same* draw — that is what makes model-vs-model differences paired.
    """
    docs = np.unique(np.asarray(doc_universe, dtype=np.int64))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(docs.size, np.full(docs.size, 1.0 / docs.size), size=n_boot)
    return {int(d): counts[:, i].astype(np.float64) for i, d in enumerate(docs)}


def _weights_for(agg: DocAggregates, weights: Mapping[int, np.ndarray], n_boot: int) -> np.ndarray:
    out = np.empty((n_boot, agg.doc_ids.size))
    for j, d in enumerate(agg.doc_ids):
        w = weights.get(int(d))
        if w is None:
            raise KeyError(f"doc_id {int(d)} is not in the bootstrap universe")
        out[:, j] = w
    return out


def _entropy_mm_rows(counts: np.ndarray) -> np.ndarray:
    """Miller–Madow entropy in bits for each row of a ``(B, N)`` count matrix."""
    total = counts.sum(axis=1, keepdims=True)
    p = np.where(total > 0, counts / np.maximum(total, 1e-300), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plug = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    observed = (counts > 0).sum(axis=1)
    return plug + (observed - 1) / (2.0 * total[:, 0] * LN2)


def replicate_metrics(
    agg: DocAggregates, weights: Mapping[int, np.ndarray], *, include_point: bool = True
) -> dict[str, np.ndarray]:
    """Every metric for every replicate, ``(B,)`` each (``(B+1,)`` with the point estimate first)."""
    n_boot = len(next(iter(weights.values())))
    W = _weights_for(agg, weights, n_boot)
    if include_point:
        W = np.vstack([np.ones((1, W.shape[1])), W])
    rows = W @ agg.rows
    slots = rows * agg.top_k
    ce = (W @ agg.ce_bits_sum) / slots
    H = _entropy_mm_rows(W @ agg.expert_counts)
    mi = H - ce
    out = {
        "entropy_bits": H,
        "cross_entropy_bits": ce,
        "mi_bits": mi,
        "mi_ratio": mi / H,
        "mi_attainable": mi / (H - math.log2(agg.top_k)),
        "exact_match": (W @ agg.exact_sum) / rows,
    }
    for i, m in enumerate(agg.recall_m):
        out[f"recall@{int(m)}"] = (W @ agg.recall_hits_sum[i]) / rows
    out["set_agreement@k"] = out[f"recall@{agg.top_k}"]
    return out


def summarize(values: np.ndarray, *, level: float = 0.95, has_point: bool = True) -> dict[str, float]:
    """Point estimate (replicate 0) plus percentile and basic intervals over the remaining replicates.

    ``lo``/``hi`` are the percentile interval. ``basic_lo``/``basic_hi`` reflect it around the point
    (2θ̂ − q_hi, 2θ̂ − q_lo) and ``bc`` = θ̂ − bias is the bias-corrected estimate they bracket. Plug-in
    entropy/KL terms are biased low when documents are internally correlated (the effective sample is
    smaller than the row count Miller–Madow assumes); the bootstrap sees the same bias again, so the
    replicates sit below θ̂. It is negligible for whole-model cells and paired differences (< 0.003)
    but reaches ~0.04 in strata of 11–20 documents, and it is the whole of F0's "negative" Î.
    """
    values = np.asarray(values, dtype=np.float64)
    point = float(values[0]) if has_point else float(values.mean())
    reps = values[1:] if has_point else values
    lo, hi = np.quantile(reps, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "basic_lo": float(2 * point - hi), "basic_hi": float(2 * point - lo),
            "bias": float(reps.mean() - point), "bc": float(2 * point - reps.mean()),
            "se": float(reps.std(ddof=1)),
            "n_boot": int(reps.size)}


def save_aggregates(path: Path | str, cells: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    """One ``.npz`` per (model, feature); keys are ``"<layer>/<field>"``."""
    flat = {f"{layer}/{name}": np.asarray(v) for layer, agg in cells.items() for name, v in agg.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **flat)
    tmp.replace(path)


def load_aggregates(path: Path | str) -> dict[int, DocAggregates]:
    with np.load(path) as z:
        grouped: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            layer, name = key.split("/", 1)
            grouped.setdefault(layer, {})[name] = z[key]
    return {int(layer): DocAggregates.from_mapping(m) for layer, m in grouped.items()}


def layer_mean(per_layer: Sequence[np.ndarray]) -> np.ndarray:
    """Replicate-wise mean over layers (each ``(B+1,)``) — the bootstrap of a layer-mean table cell."""
    return np.mean(np.vstack(per_layer), axis=0)
