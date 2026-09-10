"""Synthetic multi-shard traces for testing — supports plan T6.1's acceptance criterion.

The reader's correctness cannot be validated against real traces until Phase 5, and by then a
bug in it would already have been used to draw conclusions. These fixtures give a trace whose
exact contents are known in advance, so every accessor can be checked for equality rather than
plausibility.

The generated data deliberately satisfies the T5.3 label invariants — every ``topk`` value in
``[0, n_experts)``, ``top_k`` distinct entries per row — so a test that *breaks* one of them is
unambiguous about what it is testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .format import (
    FLAG_HIDDEN_CAPTURED,
    HIDDEN_DTYPE,
    HIDDEN_INDEX_DTYPE,
    LOGIT_DTYPE,
    STREAM_FILES,
    TOKEN_DTYPE,
    TOPK_DTYPE,
    TraceSpec,
    write_manifest,
)

__all__ = ["SyntheticTrace", "make_synthetic_trace"]


@dataclass
class SyntheticTrace:
    """Ground truth for a generated trace, concatenated across shards in shard order."""

    root: Path
    model: str
    corpus: str
    spec: TraceSpec
    tokens: np.ndarray  # structured, (n_tokens,)
    topk: np.ndarray  # (n_tokens, n_layers, top_k) int32
    logits: np.ndarray  # (n_tokens, n_layers, n_experts) float16
    hidden: np.ndarray  # (n_captured, n_layers, hidden_dim) float16
    hidden_index: np.ndarray  # (n_captured,) uint32, global token indices
    shard_sizes: list[int]
    doc_splits: dict[int, str]

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_captured(self) -> int:
        return int(self.hidden_index.shape[0])


def make_synthetic_trace(
    root: Path | str,
    *,
    model: str = "synth-moe",
    corpus: str = "synth-v1",
    spec: TraceSpec | None = None,
    shard_sizes: Sequence[int] = (40, 25, 35),
    tokens_per_doc: int = 10,
    hidden_every: int = 4,
    seed: int = 0,
    run_config_sha256: str = "0" * 64,
    manifest_overrides: dict[str, Any] | None = None,
    topk_fn: Callable[[np.random.Generator, np.ndarray, TraceSpec], np.ndarray] | None = None,
    n_docs_per_split_cycle: int = 10,
    doc_index_span: int = 64,
) -> SyntheticTrace:
    """Write a complete multi-shard trace under ``root/model/corpus/`` and return its truth.

    Parameters
    ----------
    shard_sizes:
        Token count per shard. Deliberately unequal by default — equal shards would hide
        off-by-one errors in the reader's global-index arithmetic.
    hidden_every:
        The subsample stride (plan T4.4). A token is captured when its **global index** is a
        multiple of it, and the global index is ``doc_id * doc_index_span + pos_in_doc`` — the
        same per-document reserved block `moe_trace` uses, not a running token count. Which
        tokens are captured is therefore a function of the corpus alone, and the index stream
        ascends strictly across shards so they concatenate unrewritten (T2.3).
    doc_index_span:
        Index values reserved per document; ``n_ctx`` in a real capture. Must be at least
        ``tokens_per_doc`` or the blocks would collide.
    """
    if doc_index_span < tokens_per_doc:
        raise ValueError(
            f"doc_index_span={doc_index_span} is below tokens_per_doc={tokens_per_doc}; the "
            "per-document index blocks would overlap"
        )
    spec = spec or TraceSpec(n_moe_layers=4, n_experts=16, top_k=3, hidden_dim=8)
    rng = np.random.default_rng(seed)
    root = Path(root)
    trace_dir = root / model / corpus

    n_tokens = int(sum(shard_sizes))

    # -- tokens ---------------------------------------------------------------------------
    tokens = np.zeros(n_tokens, dtype=TOKEN_DTYPE)
    tokens["token_id"] = rng.integers(0, 5000, size=n_tokens, dtype=np.uint32)
    tokens["doc_id"] = np.arange(n_tokens, dtype=np.uint32) // tokens_per_doc
    tokens["pos_in_doc"] = np.arange(n_tokens, dtype=np.uint32) % tokens_per_doc

    global_index = (
        tokens["doc_id"].astype(np.int64) * doc_index_span + tokens["pos_in_doc"].astype(np.int64)
    )
    captured = np.flatnonzero(global_index % hidden_every == 0).astype(np.int64)
    tokens["flags"][captured] |= FLAG_HIDDEN_CAPTURED

    # -- topk: top_k distinct experts per (token, layer), as the model would emit them ------
    if topk_fn is not None:
        # Phase 7 needs traces whose routing is a *known function* of a feature, so that a probe
        # returning no signal is distinguishable from a probe that is wired up wrongly. The
        # default random routing is the null case and cannot make that distinction.
        topk = np.ascontiguousarray(topk_fn(rng, tokens, spec), dtype=TOPK_DTYPE)
        if topk.shape != spec.topk_shape(n_tokens):
            raise ValueError(
                f"topk_fn returned shape {topk.shape}, expected {spec.topk_shape(n_tokens)}"
            )
    else:
        topk = np.empty(spec.topk_shape(n_tokens), dtype=TOPK_DTYPE)
        for t in range(n_tokens):
            for layer in range(spec.n_moe_layers):
                topk[t, layer] = rng.choice(spec.n_experts, size=spec.top_k, replace=False)

    # -- logits and hidden ------------------------------------------------------------------
    logits = rng.normal(0.0, 2.0, size=spec.logit_shape(n_tokens)).astype(LOGIT_DTYPE)
    hidden = rng.normal(0.0, 1.0, size=spec.hidden_shape(captured.size)).astype(HIDDEN_DTYPE)
    hidden_index = global_index[captured].astype(HIDDEN_INDEX_DTYPE)

    # -- splits: document level, deterministic ------------------------------------------------
    doc_ids = np.unique(tokens["doc_id"])
    cycle = int(n_docs_per_split_cycle)
    if cycle < 3:
        raise ValueError(f"n_docs_per_split_cycle must be >= 3, got {cycle}")
    doc_splits = {
        int(d): (
            "test"
            if i % cycle == cycle - 1
            else "val" if i % cycle == cycle - 2 else "train"
        )
        for i, d in enumerate(doc_ids)
    }

    # -- write shards ---------------------------------------------------------------------------
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    token_start = 0
    for shard_id, size in enumerate(shard_sizes):
        token_stop = token_start + size
        shard_dir = trace_dir / f"shard_{shard_id:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Selected by TOKEN position, not by index value: the index space is sparse now, so a
        # range test on the index would silently pick up a neighbouring shard's rows.
        rows = np.flatnonzero((captured >= token_start) & (captured < token_stop))

        _write(shard_dir / STREAM_FILES["tokens"], tokens[token_start:token_stop])
        _write(shard_dir / STREAM_FILES["topk"], topk[token_start:token_stop])
        _write(shard_dir / STREAM_FILES["logits"], logits[token_start:token_stop])
        _write(shard_dir / STREAM_FILES["hidden"], hidden[rows])
        # Indices stay GLOBAL so shards concatenate without rewriting (plan T2.3).
        _write(shard_dir / STREAM_FILES["hidden_index"], hidden_index[rows])

        doc_lo = int(tokens["doc_id"][token_start])
        doc_hi = int(tokens["doc_id"][token_stop - 1]) + 1

        manifest: dict[str, Any] = {
            "model": model,
            "checkpoint_status": "base",
            "gguf_sha256": "f" * 64,
            "llama_cpp_commit": "abc123",
            "run_config_sha256": run_config_sha256,
            "quant": "Q4_K_M",
            "router_dtype": "F32",
            "logit_tensor_used": "ffn_moe_logits",
            "corpus": corpus,
            "shard_id": shard_id,
            "shard_doc_range": [doc_lo, doc_hi],
            "n_tokens": int(size),
            "n_moe_layers": spec.n_moe_layers,
            "n_experts": spec.n_experts,
            "top_k": spec.top_k,
            "hidden_dim": spec.hidden_dim,
            "hidden_subsample_n": int(hidden_index.size),
            "n_captured": int(rows.size),
            "hidden_stride": int(hidden_every),
            "index_scheme": "doc_id*n_ctx+pos_in_doc",
            "index_doc_span": int(doc_index_span),
            "capture_flags": {"pre_topk": True, "pre_norm": True, "topk_captured": True},
            "layer_index_map": {
                f"trace_{i}": f"model_layer_{i}" for i in range(spec.n_moe_layers)
            },
            "device_plan": {"n_gpu": 1, "split_mode": "layer", "tensor_split": None},
            "file_sha256": {name: "0" * 64 for name in STREAM_FILES.values()},
            "kaggle_session_id": "synthetic",
            "collected_utc": now,
        }
        manifest.update(manifest_overrides or {})
        write_manifest(shard_dir, manifest)

        token_start = token_stop

    return SyntheticTrace(
        root=root,
        model=model,
        corpus=corpus,
        spec=spec,
        tokens=tokens,
        topk=topk,
        logits=logits,
        hidden=hidden,
        hidden_index=hidden_index,
        shard_sizes=list(shard_sizes),
        doc_splits=doc_splits,
    )


def _write(path: Path, array: np.ndarray) -> None:
    with open(path, "wb") as handle:
        handle.write(np.ascontiguousarray(array).tobytes())
