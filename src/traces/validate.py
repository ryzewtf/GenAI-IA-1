"""Post-collection per-shard validation — plan T5.3, plus the Python half of T3.1.

This is the *independent second witness* on a shard. ``moe_trace.cpp`` already asserts
three-stream lockstep per document and re-checks the size arithmetic in
``close_and_verify()``, but a harness cannot be its own witness: the failure mode this module
exists for is a harness bug, and a buggy harness's self-check is exactly as buggy. Everything
here runs from Python, against the bytes on disk, deriving every expected size from
:mod:`src.traces.format` rather than from anything the C++ wrote.

What it is checking for, in order of how badly it would corrupt the result if missed:

* **I12** — ``ffn_moe_topk`` is a strided view, and a contiguous read of it yields in-range,
  distinct, *wrong* expert indices for every token after the first. That passes the plan's
  own range/distinctness check, so range/distinctness is necessary and nowhere near
  sufficient. See :func:`check_topk_strided_view`.
* **I13** — if ``logit_tensor_used`` names an earlier node of the selection chain than the one
  top-k actually ran on, ``margins`` is not the margin that decided the flip and every T8.2
  number is measuring the wrong quantity. See :func:`check_selection_argsort_agreement`.
* **I15** — per-model truncation. ``max_doc_tokens`` is enforced under one reference tokenizer,
  so a document at the cap for OLMoE can exceed it for Gemma 4, and then the two models were
  never shown the same text. ``capture_stats.json`` counts it; this checks the count is 0.
* **I5 / I6** — a topk stream read as float, or a ubatch whose token count was taken from
  column 0, both land as a size or lockstep mismatch here.
* **I14** — not enforceable from the trace: whether ``n_experts`` counts only *routed* experts
  is a property of the model spec, checked in ``nodespec.py``. The load-balance index reported
  here inherits that definition and is meaningless if it was wrong.

Memory
------
Peak allocation is ``O(chunk_tokens * n_experts * 4 B)`` for the widest working array (one
layer of one logit chunk in float32), plus ``O(n_moe_layers * n_experts * 8 B)`` of
accumulators and ``O(sample_tokens * n_experts * 4 B)`` for the two sampled checks. **Nothing
allocated here is proportional to ``n_tokens``, ``n_captured`` or ``n_docs``** — the expert
histogram folds over chunks, the split check accumulates per-split token counters instead of a
doc-id set, and ``hidden_index`` is walked through :class:`~src.traces.reader.ShardHandle`'s
memmap in windows rather than via ``TraceReader.captured_token_ids()``, which does
materialise a full ``n_captured`` array. That is the same 4 GB budget T6.1 works under, and it
is what lets this run over a Qwen3 shard in a CPU session.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..metrics.information import entropy_mm
from .format import (
    FLAG_HIDDEN_CAPTURED,
    MANIFEST_NAME,
    SHARD_INVARIANT_KEYS,
    STREAM_FILES,
    TOKEN_DTYPE,
    FormatError,
    TraceSpec,
    expected_file_sizes,
    read_manifest,
)
from .reader import ShardHandle

__all__ = [
    "Finding",
    "ValidationReport",
    "STATS_NAME",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_SAMPLE_TOKENS",
    "CHECK_NAMES",
    "check_sizes",
    "check_lockstep",
    "check_topk_range_and_distinctness",
    "check_topk_strided_view",
    "check_expert_usage",
    "check_logits_sanity",
    "check_selection_argsort_agreement",
    "check_truncation",
    "check_split_coverage",
    "check_hidden_stride",
    "validate_shard",
    "validate_shards",
    "main",
]

#: Written by ``moe_trace`` next to the streams (``--stats`` defaults to ``<out>/…``).
STATS_NAME = "capture_stats.json"

#: Tokens per pass over a stream. 8192 x 128 experts x 4 B = 4 MB, so the working set is a few
#: MB regardless of shard size. Deliberately smaller than the reader's 65536 because the
#: validator holds several derived arrays of the same width at once.
DEFAULT_CHUNK_TOKENS = 8_192

#: Tokens used by the two argsort-based checks (I12, I13). These are the only checks that are
#: sampled rather than total: reconstructing a full argsort costs ``n_experts log n_experts``
#: per token, and both defects they detect are deterministic properties of the capture path, so
#: they are either present in the first few thousand tokens or not present at all.
DEFAULT_SAMPLE_TOKENS = 4_096

#: Bounded example lists — a finding is a diagnostic, not a data dump.
_MAX_EXAMPLES = 8

_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


# --------------------------------------------------------------------------------------
# findings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One validation outcome.

    ``detail`` carries the machine-readable numbers (rates, counts, offending indices) so
    that T9.1's aggregation can read a validation report without parsing English.
    """

    check: str
    severity: str
    message: str
    detail: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_RANK:
            raise ValueError(f"unknown severity {self.severity!r}")

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
        }
        if self.detail:
            out["detail"] = _jsonable(self.detail)
        return out


@dataclass
class ValidationReport:
    """Every finding for one shard (or, with ``shard_id = -1``, for a shard *set*)."""

    shard_dir: str
    shard_id: int
    model: str
    corpus: str = ""
    findings: list[Finding] = field(default_factory=list)
    counters: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------------------

    def add(self, findings: Iterable[Finding]) -> None:
        self.findings.extend(findings)

    # -- verdict ------------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """No error-severity findings. Warnings are for a human, not for the gate."""
        return not any(f.is_error for f in self.findings)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.is_error]

    def by_check(self, check: str) -> list[Finding]:
        return [f for f in self.findings if f.check == check]

    # -- serialisation ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_dir": self.shard_dir,
            "shard_id": self.shard_id,
            "model": self.model,
            "corpus": self.corpus,
            "ok": self.ok,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "counters": _jsonable(self.counters),
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def summary(self) -> str:
        name = Path(self.shard_dir).name or self.shard_dir
        verdict = "PASS" if self.ok else "FAIL"
        return (
            f"{verdict} {name} (shard {self.shard_id}, {self.model}) "
            f"{self.n_errors} error(s), {self.n_warnings} warning(s)"
        )


def _jsonable(value: Any) -> Any:
    """Findings go into JSON, and numpy scalars are not JSON — convert at the boundary."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


# --------------------------------------------------------------------------------------
# chunking helpers
# --------------------------------------------------------------------------------------


def _chunks(n: int, chunk_tokens: int) -> Iterator[slice]:
    step = max(1, int(chunk_tokens))
    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def _stream_size(handle: ShardHandle, stream: str) -> int:
    path = handle.dir / STREAM_FILES[stream]
    return path.stat().st_size if path.exists() else -1


def _sized_streams(handle: ShardHandle) -> set[str]:
    """Streams whose byte length matches the arithmetic, and are therefore safe to memmap.

    ``np.memmap`` with an explicit shape raises on a short file, so every content check has to
    know which streams survived :func:`check_sizes` — otherwise a truncated upload would come
    back as a ``ValueError`` traceback instead of as a finding.
    """
    want = expected_file_sizes(handle.spec, handle.n_tokens, handle.n_captured)
    ok: set[str] = set()
    for stream, filename in STREAM_FILES.items():
        path = handle.dir / filename
        if path.exists() and path.stat().st_size == want[filename]:
            ok.add(stream)
    return ok


def _skipped(check: str, stream: str) -> Finding:
    return Finding(
        check,
        "info",
        f"skipped: {STREAM_FILES[stream]} failed the size check, so it cannot be mapped",
        {"stream": stream},
    )


def _membership(sets: np.ndarray, n_experts: int) -> np.ndarray:
    """``(n, n_experts)`` bool from ``(n, k)`` indices, ignoring out-of-range values."""
    rows = np.zeros((sets.shape[0], n_experts), dtype=bool)
    idx = sets.astype(np.int64, copy=False)
    valid = (idx >= 0) & (idx < n_experts)
    r = np.repeat(np.arange(sets.shape[0]), sets.shape[1]).reshape(sets.shape)
    rows[r[valid], idx[valid]] = True
    return rows


def _set_agreement(pred: np.ndarray, true: np.ndarray, n_experts: int) -> float:
    """Mean ``|S_pred ∩ S_true| / k`` — Family A's definition (§1.2), used here as a
    consistency rate between two candidate readings of the same rows."""
    if pred.shape[0] == 0:
        return float("nan")
    mask = _membership(true, n_experts)
    hits = mask[np.arange(pred.shape[0])[:, None], np.clip(pred, 0, n_experts - 1)]
    inside = (pred >= 0) & (pred < n_experts)
    return float((hits & inside).sum() / (pred.shape[0] * true.shape[1]))


# --------------------------------------------------------------------------------------
# check: size arithmetic
# --------------------------------------------------------------------------------------


def check_sizes(handle: ShardHandle) -> Iterator[Finding]:
    """Every stream's byte length must equal the layout arithmetic exactly (T5.3).

    Off by a whole number of token rows is the signature of a killed session — the last
    document never got flushed. Off by a whole number of *layer* rows within a token is the
    signature of a mis-specified node name: one layer's tensor was never matched, so the
    writer emitted short rows. The delta is decomposed to say which of the two it is, because
    the remedies are opposite (re-run the shard vs. fix the node spec).
    """
    spec = handle.spec
    want = expected_file_sizes(spec, handle.n_tokens, handle.n_captured)
    per_row = {
        "tokens.bin": spec.token_stride,
        "topk.bin": spec.topk_stride,
        "logits.bin": spec.logit_stride,
        "hidden.bin": spec.hidden_stride,
        "hidden_index.bin": 4,
    }
    layer_row = {
        "topk.bin": spec.topk_stride // spec.n_moe_layers,
        "logits.bin": spec.logit_stride // spec.n_moe_layers,
        "hidden.bin": spec.hidden_stride // spec.n_moe_layers,
    }

    for stream, filename in STREAM_FILES.items():
        path = handle.dir / filename
        if not path.exists():
            yield Finding(
                "size_arithmetic",
                "error",
                f"{filename}: missing (stream {stream!r})",
                {"stream": stream, "file": filename},
            )
            continue
        got = path.stat().st_size
        expect = want[filename]
        if got == expect:
            continue

        delta = got - expect
        hint = "not a whole number of rows — the file is structurally broken"
        row = per_row[filename]
        if delta % row == 0:
            hint = f"{abs(delta) // row} whole token row(s) {'extra' if delta > 0 else 'short'}"
        elif filename in layer_row and delta % layer_row[filename] == 0:
            n = abs(delta) // layer_row[filename]
            hint = f"{n} layer-row(s) {'extra' if delta > 0 else 'short'} — check the node spec"
        yield Finding(
            "size_arithmetic",
            "error",
            f"{filename}: {got} B on disk, layout arithmetic says {expect} B "
            f"(delta {delta:+d}); {hint}",
            {
                "stream": stream,
                "file": filename,
                "size_bytes": got,
                "expected_bytes": expect,
                "delta_bytes": delta,
                "hint": hint,
                "n_tokens": handle.n_tokens,
                "n_captured": handle.n_captured,
            },
        )

    # hidden_index.bin is the authority on how many rows hidden.bin holds; the manifest's
    # `n_captured` is written before the last flush and disagrees exactly when a session died.
    #
    # NOT `hidden_subsample_n`: that is the T4.4 budget for the whole collection, converted to an
    # integer stride and then applied per shard, so equality with any one shard's row count would
    # be a coincidence. Comparing against it flagged a healthy shard as broken.
    declared = handle.manifest.get("n_captured")
    if declared is not None and int(declared) != handle.n_captured:
        yield Finding(
            "size_arithmetic",
            "error",
            f"manifest n_captured={int(declared)} but hidden_index.bin holds "
            f"{handle.n_captured} rows",
            {"declared": int(declared), "on_disk": handle.n_captured},
        )

    if handle.n_captured > handle.n_tokens:
        yield Finding(
            "size_arithmetic",
            "error",
            f"hidden_index.bin holds {handle.n_captured} rows for {handle.n_tokens} tokens; "
            "the subsample cannot be larger than what it subsamples",
            {"n_captured": handle.n_captured, "n_tokens": handle.n_tokens},
        )


# --------------------------------------------------------------------------------------
# check: three-stream lockstep (T3.1)
# --------------------------------------------------------------------------------------


def check_lockstep(
    handle: ShardHandle,
    *,
    stats: Mapping[str, Any] | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> Iterator[Finding]:
    """tokens / topk / logits must agree on the token count, and hidden must agree on the
    captured count (T3.1).

    The harness asserts this per document from inside the callback. This re-derives it from
    file lengths and from ``tokens.flags``, which is the point: a filter matching the wrong
    node for one stream, or a ubatch count taken from ``ne[0]`` instead of ``ne[1]`` (I6),
    produces plausible-but-wrong data that only a *count* comparison catches. A harness's own
    assertion cannot witness a bug in that harness.

    ``hidden_index`` is additionally required to be a strictly increasing subsequence of valid
    global token indices, and to agree row-for-row with the ``FLAG_HIDDEN_CAPTURED`` bits in
    ``tokens.bin``. Those two streams are written by different code paths, so agreement
    between them is real evidence; either alone is not.
    """
    spec = handle.spec
    rows = {
        "tokens": _stream_size(handle, "tokens") // spec.token_stride,
        "topk": _stream_size(handle, "topk") // spec.topk_stride,
        "logits": _stream_size(handle, "logits") // spec.logit_stride,
    }
    declared = handle.n_tokens
    disagree = {k: v for k, v in rows.items() if v != declared}
    if disagree:
        yield Finding(
            "lockstep",
            "error",
            f"three-stream lockstep failure: manifest n_tokens={declared}, rows on disk "
            f"{rows} — a stream is missing rows for at least one layer (T3.1)",
            {"declared_n_tokens": declared, "rows": rows},
        )

    hidden_rows = _stream_size(handle, "hidden") // spec.hidden_stride
    if hidden_rows != handle.n_captured:
        yield Finding(
            "lockstep",
            "error",
            f"hidden.bin holds {hidden_rows} rows but hidden_index.bin holds "
            f"{handle.n_captured}; the subsample streams are not in lockstep",
            {"hidden_rows": hidden_rows, "hidden_index_rows": handle.n_captured},
        )

    sized = _sized_streams(handle)
    if "hidden_index" not in sized:
        yield _skipped("lockstep", "hidden_index")
        return
    if handle.n_captured == 0:
        yield Finding(
            "lockstep", "info", "no hidden states captured in this shard", {"n_captured": 0}
        )
        return

    index = handle.hidden_index
    base = int(stats["global_token_base"]) if stats and "global_token_base" in stats else None

    # Strict monotonicity, walked in windows so that a 4M-token shard's index stream is never
    # resident. Indices are GLOBAL (plan T2.3) so shards concatenate unrewritten; a repeat or a
    # reversal here means either a re-shard rewrote them or two documents were flushed twice.
    prev: int | None = None
    bad_order: list[tuple[int, int]] = []
    for sl in _chunks(handle.n_captured, chunk_tokens):
        window = np.asarray(index[sl], dtype=np.int64)
        if prev is not None and window[0] <= prev:
            bad_order.append((sl.start, int(window[0])))
        step = np.diff(window)
        for offset in np.flatnonzero(step <= 0)[: _MAX_EXAMPLES]:
            bad_order.append((sl.start + int(offset) + 1, int(window[int(offset) + 1])))
        prev = int(window[-1])
    if bad_order:
        yield Finding(
            "lockstep",
            "error",
            f"hidden_index is not strictly increasing ({len(bad_order)} violation(s), first at "
            f"row {bad_order[0][0]} value {bad_order[0][1]}); global token indices must be "
            "ascending or hidden() resolves the wrong rows",
            {"violations": bad_order[:_MAX_EXAMPLES], "n_violations": len(bad_order)},
        )

    if "tokens" not in sized:
        yield _skipped("lockstep", "tokens")
        return

    # Cross-stream: the flag bits and the index stream are written independently, one per token
    # record and one per captured row. Disagreement means the subsample mask the callback used is
    # not the mask the writer recorded, and every F4/F5 lookup would take a neighbour's router
    # input.
    #
    # Only the COUNT is compared here, because it is the half that needs nothing but the two
    # streams. Comparing the index *values* needs to know how indices are formed, and
    # `check_hidden_stride` owns that: it recomputes `doc_id * n_ctx + pos_in_doc` per token and
    # checks every row, which is strictly stronger than the running-count-plus-base arithmetic
    # this loop used to do.
    cursor = 0
    for sl in _chunks(handle.n_tokens, chunk_tokens):
        flags = np.asarray(handle.tokens[sl]["flags"], dtype=np.uint32)
        cursor += int(np.count_nonzero(flags & FLAG_HIDDEN_CAPTURED))
    if cursor != handle.n_captured:
        yield Finding(
            "lockstep",
            "error",
            f"{cursor} token(s) carry FLAG_HIDDEN_CAPTURED but hidden_index.bin has "
            f"{handle.n_captured} row(s)",
            {"flagged_tokens": cursor, "hidden_index_rows": handle.n_captured},
        )


# --------------------------------------------------------------------------------------
# check: topk range + distinctness (I1, I12)
# --------------------------------------------------------------------------------------


def check_topk_range_and_distinctness(
    handle: ShardHandle, *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS
) -> Iterator[Finding]:
    """Every value in ``[0, n_experts)``, exactly ``top_k`` distinct experts per (token, layer).

    A cheap, *total* check on the labels — and this is where its usefulness stops.

    **Distinctness does not prove correctness, and must never be read as if it did.** I12:
    ``ffn_moe_topk`` is a strided view over the full ``[n_experts, n_tokens]`` argsort
    (``ne[0] = top_k`` while ``nb[1] = n_experts * 4``). A contiguous read of it returns
    token 0's first ``top_k`` experts, then the *rest of token 0's ranking*, and so on — every
    value in range, every row distinct, and every row after the first belonging to the wrong
    token. It passes this check completely. That is why :func:`check_topk_strided_view` exists,
    and why ``topk_layout`` from ``capture_stats.json`` is cross-checked in
    :func:`check_truncation`: three independent angles on the same defect, because it is the
    worst silent-corruption path in the plan (I5 makes the related float-cast defect loud; I12
    is silent).
    """
    if "topk" not in _sized_streams(handle):
        yield _skipped("topk_labels", "topk")
        return

    spec = handle.spec
    n_out = 0
    n_dup = 0
    out_examples: list[dict[str, int]] = []
    dup_examples: list[dict[str, int]] = []
    lo_seen = np.iinfo(np.int64).max
    hi_seen = np.iinfo(np.int64).min

    for sl in _chunks(handle.n_tokens, chunk_tokens):
        block = np.asarray(handle.topk[sl], dtype=np.int64)  # (c, L, k)
        if block.size == 0:
            continue
        lo_seen = min(lo_seen, int(block.min()))
        hi_seen = max(hi_seen, int(block.max()))

        bad = (block < 0) | (block >= spec.n_experts)
        if bad.any():
            n_out += int(bad.sum())
            for t, layer, slot in zip(*np.nonzero(bad)):
                if len(out_examples) >= _MAX_EXAMPLES:
                    break
                out_examples.append(
                    {
                        "token": sl.start + int(t),
                        "layer": int(layer),
                        "slot": int(slot),
                        "value": int(block[t, layer, slot]),
                    }
                )

        ordered = np.sort(block, axis=-1)
        dup_rows = (np.diff(ordered, axis=-1) == 0).any(axis=-1)
        if dup_rows.any():
            n_dup += int(dup_rows.sum())
            for t, layer in zip(*np.nonzero(dup_rows)):
                if len(dup_examples) >= _MAX_EXAMPLES:
                    break
                dup_examples.append(
                    {
                        "token": sl.start + int(t),
                        "layer": int(layer),
                        "row": [int(v) for v in block[t, layer]],
                    }
                )

    if n_out:
        yield Finding(
            "topk_labels",
            "error",
            f"{n_out} expert index/indices outside [0, {spec.n_experts}) in topk.bin "
            f"(observed range [{lo_seen}, {hi_seen}]); first: {out_examples[0]}. An I32 stream "
            "read as float, or a wrong n_experts, both land here (I5)",
            {"n_out_of_range": n_out, "examples": out_examples, "observed": [lo_seen, hi_seen]},
        )
    if n_dup:
        yield Finding(
            "topk_labels",
            "error",
            f"{n_dup} (token, layer) row(s) do not hold {spec.top_k} distinct experts; "
            f"first: {dup_examples[0]}",
            {"n_duplicate_rows": n_dup, "examples": dup_examples},
        )
    if not n_out and not n_dup:
        yield Finding(
            "topk_labels",
            "info",
            f"all {handle.n_tokens * spec.n_moe_layers} rows in range and distinct — necessary, "
            "NOT sufficient (see I12)",
            {"observed": [lo_seen, hi_seen], "n_rows": handle.n_tokens * spec.n_moe_layers},
        )


# --------------------------------------------------------------------------------------
# check: the I12 strided-view signature
# --------------------------------------------------------------------------------------


def check_topk_strided_view(
    handle: ShardHandle,
    *,
    sample_tokens: int = DEFAULT_SAMPLE_TOKENS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    strided_threshold: float = 0.95,
    margin_over_correct: float = 0.20,
    proxy_z: float = 8.0,
    proxy_excess: float = 0.15,
) -> Iterator[Finding]:
    """Positive test for the I12 contiguous-read corruption.

    **It is detectable, and it is detectable exactly, whenever ``logits.bin`` is usable.** The
    corruption is a deterministic function of the *same* underlying argsort that the correct
    read indexes into: the parent tensor is the full ``[n_experts, n_tokens]`` ranking, and a
    contiguous read takes flat positions ``[t*k, t*k+k)`` of it for token ``t`` instead of
    ``[t*n_experts, t*n_experts+k)``. ``logits.bin`` lets that parent be reconstructed
    (``argsort(-logits)``), so both hypotheses can be evaluated against the bytes actually on
    disk and the one that matches identified. Token 0 is identical under both hypotheses, which
    is precisely why an eyeball check of the first row proves nothing.

    Two things weaken the reconstruction and are why the proxy below exists as well: fp16
    resolution can reorder near-ties, and for GPT-OSS top-k ran on *biased* selection probs
    (I13) while ``logits.bin`` may hold the unbiased node. Both depress *every* hypothesis'
    agreement, so the check only fires when the strided hypothesis is both high in absolute
    terms and clearly better than the correct one.

    The **proxy** runs over the whole stream from ``topk.bin`` alone and needs no logits: under
    the corruption, consecutive tokens are consecutive segments of one permutation of
    ``[0, n_experts)``, so adjacent rows are *disjoint* far more often than independent routing
    would give. The test is one-sided in the right direction — real routers have temporal
    locality (that is feature F3), which pushes adjacent-token overlap *up*, never down — so an
    excess of adjacent disjointness cannot be explained by real routing behaviour.
    """
    spec = handle.spec
    sized = _sized_streams(handle)
    if "topk" not in sized:
        yield _skipped("topk_strided_view", "topk")
        return
    if spec.n_experts == spec.top_k:
        yield Finding(
            "topk_strided_view",
            "info",
            "n_experts == top_k, so the strided and contiguous readings coincide and the I12 "
            "corruption is not expressible",
            {"n_experts": spec.n_experts},
        )
        return

    # -- exact reconstruction, on a prefix ------------------------------------------------
    m = int(min(sample_tokens, handle.n_tokens))
    if "logits" in sized and m >= 8:
        per_layer: dict[int, dict[str, float]] = {}
        for layer in range(spec.n_moe_layers):
            lg = np.asarray(handle.logits[0:m, layer, :], dtype=np.float32)
            if not np.isfinite(lg).all():
                continue
            order = np.argsort(-lg, axis=1, kind="stable")  # the parent argsort, per token
            truth = np.asarray(handle.topk[0:m, layer, :], dtype=np.int64)
            correct = order[:, : spec.top_k]
            # What a contiguous read of the strided view WOULD have produced from this parent.
            strided = order.ravel()[: m * spec.top_k].reshape(m, spec.top_k)
            per_layer[layer] = {
                "correct": _set_agreement(truth, correct, spec.n_experts),
                "strided": _set_agreement(truth, strided, spec.n_experts),
            }

        if per_layer:
            correct_rate = float(np.mean([v["correct"] for v in per_layer.values()]))
            strided_rate = float(np.mean([v["strided"] for v in per_layer.values()]))
            detail = {
                "sample_tokens": m,
                "correct_hypothesis_agreement": correct_rate,
                "strided_hypothesis_agreement": strided_rate,
                "per_layer": {str(k): v for k, v in per_layer.items()},
            }
            if (
                strided_rate >= strided_threshold
                and strided_rate > correct_rate + margin_over_correct
            ):
                yield Finding(
                    "topk_strided_view",
                    "error",
                    f"topk.bin matches a CONTIGUOUS read of the strided ffn_moe_topk view "
                    f"(agreement {strided_rate:.4f}) far better than the correct de-strided "
                    f"read ({correct_rate:.4f}) over {m} tokens. This is I12: in-range, "
                    "distinct, wrong indices for every token after the first. The labels — and "
                    "therefore every result — are invalid; re-collect with a llama.cpp commit "
                    "whose topk layout the harness de-strides.",
                    detail,
                )
            else:
                yield Finding(
                    "topk_strided_view",
                    "info",
                    f"no I12 signature: strided-read hypothesis agrees {strided_rate:.4f} vs "
                    f"{correct_rate:.4f} for the correct read over {m} tokens",
                    detail,
                )
    else:
        yield Finding(
            "topk_strided_view",
            "info",
            "logits.bin unusable for exact I12 reconstruction; relying on the adjacency proxy "
            "alone, which is weaker",
            {"sample_tokens": m},
        )

    # -- proxy, total, topk.bin only ------------------------------------------------------
    n_experts, k = spec.n_experts, spec.top_k
    p_null = 1.0
    for i in range(k):
        p_null *= (n_experts - k - i) / (n_experts - i)
    if p_null <= 0.0:
        return  # 2k > n_experts: adjacent rows cannot be disjoint, statistic is degenerate

    disjoint = np.zeros(spec.n_moe_layers, dtype=np.int64)
    pairs = np.zeros(spec.n_moe_layers, dtype=np.int64)
    carry: np.ndarray | None = None  # (L, n_experts) membership of the previous chunk's last row
    for sl in _chunks(handle.n_tokens, chunk_tokens):
        block = np.asarray(handle.topk[sl], dtype=np.int64)
        for layer in range(spec.n_moe_layers):
            mask = _membership(block[:, layer, :], n_experts)
            if carry is not None:
                pairs[layer] += 1
                disjoint[layer] += int(not (carry[layer] & mask[0]).any())
            if mask.shape[0] > 1:
                overlap = (mask[:-1] & mask[1:]).sum(axis=1)
                pairs[layer] += overlap.size
                disjoint[layer] += int((overlap == 0).sum())
        carry = np.stack([_membership(block[-1:, layer, :], n_experts)[0] for layer in range(spec.n_moe_layers)])

    flagged: dict[str, dict[str, float]] = {}
    rates: dict[str, float] = {}
    for layer in range(spec.n_moe_layers):
        if pairs[layer] < 32:
            continue
        rate = float(disjoint[layer] / pairs[layer])
        rates[str(layer)] = rate
        z = (rate - p_null) / math.sqrt(p_null * (1.0 - p_null) / float(pairs[layer]))
        if z > proxy_z and rate > p_null + proxy_excess:
            flagged[str(layer)] = {"disjoint_rate": rate, "z": float(z)}

    if flagged:
        yield Finding(
            "topk_strided_view",
            "warning",
            f"adjacent-token expert sets are disjoint far more often than independent routing "
            f"predicts (p_null={p_null:.4f}) in {len(flagged)} layer(s): {flagged}. Consecutive "
            "segments of one permutation look exactly like this — the I12 proxy. Real temporal "
            "locality biases this statistic the other way, so it is not explainable by routing.",
            {"p_null": p_null, "flagged_layers": flagged, "n_pairs": int(pairs.max())},
        )
    elif rates:
        yield Finding(
            "topk_strided_view",
            "info",
            f"adjacent-token disjointness consistent with real routing (p_null={p_null:.4f})",
            {"p_null": p_null, "disjoint_rate_per_layer": rates},
        )


# --------------------------------------------------------------------------------------
# check: expert usage histogram + load balance (T5.3, T6.4, T9.4)
# --------------------------------------------------------------------------------------


def _worst_dead_layer(counts: Mapping[int, Any], spec: Any) -> str:
    """Which layer has the most never-selected experts, as `layer N (M dead)`.

    The full per-layer map goes in the finding's `detail`, not its message: at 128 experts and
    60 layers the map is thousands of characters, and a warning that has to be scrolled past is
    a warning that gets scrolled past. One number tells the reader whether to go look.
    """
    per_layer = {l: int((counts[l] == 0).sum()) for l in range(spec.n_moe_layers)}
    layer = max(per_layer, key=lambda l: per_layer[l])
    return f"{layer} ({per_layer[layer]} dead)"


def check_expert_usage(
    handle: ShardHandle,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    saturation_factor: float = 4.0,
    load_balance_warn_below: float = 0.5,
) -> Iterator[Finding]:
    """Per-layer expert selection histogram, dead/saturated experts, and load-balance index.

    A never-selected expert gives ``p(e) = 0`` exactly, which is the reason §1.2's ε-mix
    exists at all — without it the held-out CE is ``+inf`` the first time a predictor meets
    that expert in another split. It is reported as a *warning*, not an error: a dead expert
    can be a real property of the checkpoint (a genuine finding for T9.4), and the plan's
    requirement is that it be intentional rather than absent.

    **Load-balance index** = ``entropy_mm(counts) / log2(n_experts)`` per layer — T6.4's
    definition, computed with the same Miller-Madow estimator as :mod:`src.metrics.information`
    so the number here and the number in the results table cannot drift. 1.0 is perfectly
    uniform routing; low values mean the aux loss is not holding. This is the *measured*
    aux-loss instrument T9.4 needs — the config-declared coefficient says what was asked for,
    not what happened.

    Counted over slots, matching §1.2's random variable (a uniformly-chosen member of the
    selected set), so the normaliser really is ``log2(n_experts)``.
    """
    if "topk" not in _sized_streams(handle):
        yield _skipped("expert_usage", "topk")
        return

    spec = handle.spec
    counts = np.zeros((spec.n_moe_layers, spec.n_experts), dtype=np.int64)
    for sl in _chunks(handle.n_tokens, chunk_tokens):
        block = np.asarray(handle.topk[sl], dtype=np.int64)
        for layer in range(spec.n_moe_layers):
            values = block[:, layer, :].ravel()
            values = values[(values >= 0) & (values < spec.n_experts)]
            counts[layer] += np.bincount(values, minlength=spec.n_experts)

    dead: dict[str, list[int]] = {}
    saturated: dict[str, list[dict[str, float]]] = {}
    lbi: dict[str, float] = {}
    for layer in range(spec.n_moe_layers):
        row = counts[layer]
        total = int(row.sum())
        if total == 0:
            continue
        lbi[str(layer)] = float(entropy_mm(row) / math.log2(spec.n_experts))
        zero = np.flatnonzero(row == 0)
        if zero.size:
            dead[str(layer)] = [int(e) for e in zero[:_MAX_EXAMPLES]]
        uniform = total / spec.n_experts
        hot = np.flatnonzero(row > saturation_factor * uniform)
        if hot.size:
            saturated[str(layer)] = [
                {"expert": int(e), "share": float(row[e] / total), "x_uniform": float(row[e] / uniform)}
                for e in hot[:_MAX_EXAMPLES]
            ]

    if dead:
        n_dead = sum(int((counts[layer] == 0).sum()) for layer in range(spec.n_moe_layers))
        yield Finding(
            "expert_usage",
            "warning",
            f"{n_dead} dead expert slot(s) across {len(dead)} layer(s) — never selected in this "
            f"shard, so q(e) is exactly zero for them (this is what §1.2's epsilon-mix covers). "
            f"Worst layer {_worst_dead_layer(counts, spec)}; full map in "
            "detail['dead_experts_per_layer']. Confirm this is a property of the checkpoint, "
            "not a capture bug.",
            {"dead_experts_per_layer": dead, "n_dead": n_dead},
        )
    if saturated:
        yield Finding(
            "expert_usage",
            "warning",
            f"expert(s) selected more than {saturation_factor}x uniform in "
            f"{len(saturated)} layer(s); {sum(len(v) for v in saturated.values())} slot(s), "
            "full map in detail['saturated_per_layer']",
            {"saturated_per_layer": saturated, "saturation_factor": saturation_factor},
        )
    if lbi:
        worst = min(lbi.items(), key=lambda kv: kv[1])
        severity = "warning" if worst[1] < load_balance_warn_below else "info"
        yield Finding(
            "expert_usage",
            severity,
            f"load-balance index (Miller-Madow H / log2({spec.n_experts}), T6.4): "
            f"mean {float(np.mean(list(lbi.values()))):.4f}, worst layer {worst[0]} at "
            f"{worst[1]:.4f}",
            {
                "load_balance_index_per_layer": lbi,
                "load_balance_index_mean": float(np.mean(list(lbi.values()))),
                "worst_layer": int(worst[0]),
                "slots_per_layer": int(counts[0].sum()),
            },
        )


# --------------------------------------------------------------------------------------
# check: logits sanity
# --------------------------------------------------------------------------------------


def check_logits_sanity(
    handle: ShardHandle, *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS
) -> Iterator[Finding]:
    """No NaN/Inf anywhere in ``logits.bin``, and the k-th vs (k+1)-th margin is computable.

    A single NaN propagates into every entropy and margin aggregate as NaN and, because
    ``np.partition`` places NaN last, silently changes which value the margin is taken between.
    Scanned totally rather than sampled: it is one pass at fp16 and the failure is pointwise.

    The margin is what makes T8.2's flip analysis interpretable (flips at small margins are
    arithmetic, flips at large margins are a bug), so a trace where it cannot be formed is
    reported here rather than discovered in Phase 8.
    """
    if "logits" not in _sized_streams(handle):
        yield _skipped("logits_sanity", "logits")
        return

    spec = handle.spec
    n_nan = n_inf = 0
    first: dict[str, int] | None = None
    min_margin = math.inf
    for sl in _chunks(handle.n_tokens, chunk_tokens):
        block = np.asarray(handle.logits[sl], dtype=np.float32)
        if block.size == 0:
            continue
        nan = np.isnan(block)
        inf = np.isinf(block)
        if nan.any() or inf.any():
            n_nan += int(nan.sum())
            n_inf += int(inf.sum())
            if first is None:
                t, layer, expert = (int(a[0]) for a in np.nonzero(nan | inf))
                first = {
                    "token": sl.start + t,
                    "layer": layer,
                    "expert": expert,
                    "value": float(block[t, layer, expert]),
                }
        elif spec.top_k < spec.n_experts:
            part = np.partition(-block, kth=(spec.top_k - 1, spec.top_k), axis=-1)
            margin = -part[..., spec.top_k - 1] + part[..., spec.top_k]
            min_margin = min(min_margin, float(margin.min()))

    if n_nan or n_inf:
        yield Finding(
            "logits_sanity",
            "error",
            f"logits.bin holds {n_nan} NaN and {n_inf} Inf value(s); first at {first}. Every "
            "margin and entropy downstream becomes NaN, and np.partition sorts NaN last so the "
            "margin is taken between the wrong pair.",
            {"n_nan": n_nan, "n_inf": n_inf, "first": first},
        )
        return

    if spec.top_k >= spec.n_experts:
        yield Finding(
            "logits_sanity",
            "warning",
            f"top_k={spec.top_k} == n_experts={spec.n_experts}: there is no (k+1)-th logit, so "
            "the T8.2 margin is identically zero and flip analysis is not available",
            {"top_k": spec.top_k, "n_experts": spec.n_experts},
        )
    else:
        yield Finding(
            "logits_sanity",
            "info",
            f"logits finite; k-th vs (k+1)-th margin computable, minimum {min_margin:.6g}",
            {"min_margin": None if min_margin is math.inf else min_margin},
        )


def check_selection_argsort_agreement(
    handle: ShardHandle,
    *,
    sample_tokens: int = DEFAULT_SAMPLE_TOKENS,
    warn_below: float = 1.0,
) -> Iterator[Finding]:
    """Does ``argsort(-logits)[:k]`` reproduce ``topk.bin``? Report the rate (I13).

    This is the strongest on-disk evidence that ``logit_tensor_used`` names the **last** node of
    the selection chain. Top-k runs on ``selection_probs``; if the manifest names an earlier
    node — the raw ``ffn_moe_logits`` before a bias add, a sigmoid, or a group mask — then the
    ranking implied by ``logits.bin`` is not the ranking that chose the experts, agreement
    falls below 1.0, and every margin reported in T8.2 is the margin of the wrong quantity.

    Severity is **warning, never error**, and that is deliberate. Agreement below 1.0 is
    *expected* in two legitimate cases: GPT-OSS's router bias means the unbiased logits genuinely
    do not reproduce the selection (§1.6 — which is exactly why I1 forbids recomputing sets from
    logits), and at 128 experts the k-th/(k+1)-th gap is frequently inside fp16 resolution so
    near-ties reorder. A gate here would fail correct traces. The rate is *reported* so a human
    reads it against the model's known router; what is forbidden is passing silently.

    Sampled at evenly spaced positions rather than over a prefix, because a chain that is only
    wrong for some documents (a re-resolved node, I13's LLaMA-4 branch) would hide in a prefix.
    """
    sized = _sized_streams(handle)
    if not {"topk", "logits"} <= sized:
        yield _skipped("selection_argsort", "logits" if "logits" not in sized else "topk")
        return

    spec = handle.spec
    n = handle.n_tokens
    if n == 0:
        yield Finding("selection_argsort", "info", "empty shard", {"n_tokens": 0})
        return

    # Evenly spaced windows, so the sample spans the shard without materialising it.
    budget = int(min(sample_tokens, n))
    n_windows = max(1, min(8, n // max(1, budget // 8) if budget >= 8 else 1))
    per_window = max(1, budget // n_windows)
    starts = [int(round(i * (n - per_window) / max(1, n_windows - 1))) for i in range(n_windows)]

    per_layer: dict[str, float] = {}
    exact: dict[str, float] = {}
    n_mismatched = 0
    n_tied = 0
    n_rows = 0
    for layer in range(spec.n_moe_layers):
        hits = 0.0
        exact_hits = 0
        seen = 0
        for start in sorted(set(starts)):
            stop = min(n, start + per_window)
            lg = np.asarray(handle.logits[start:stop, layer, :], dtype=np.float32)
            truth = np.asarray(handle.topk[start:stop, layer, :], dtype=np.int64)
            order = np.argsort(-lg, axis=1, kind="stable")[:, : spec.top_k]
            hits += _set_agreement(truth, order, spec.n_experts) * truth.shape[0]
            agree = (np.sort(truth, axis=1) == np.sort(order, axis=1)).all(axis=1)
            exact_hits += int(agree.sum())
            seen += truth.shape[0]

            # Attribute each disagreement. `logits.bin` is fp16, so when the k-th and (k+1)-th
            # stored values are *exactly* equal the two sets are indistinguishable on disk and
            # either is a correct answer to the question this file can answer — llama.cpp broke
            # the tie in fp32, before the down-cast. Such a row carries no evidence about which
            # node the manifest names, so counting it against the chain would bury the signal
            # this check exists for under a floor of benign noise.
            bad = ~agree
            if bad.any():
                srt = np.sort(lg[bad], axis=1)[:, ::-1]
                n_mismatched += int(bad.sum())
                n_tied += int((srt[:, spec.top_k - 1] == srt[:, spec.top_k]).sum())
            n_rows += truth.shape[0]
        if seen:
            per_layer[str(layer)] = float(hits / seen)
            exact[str(layer)] = float(exact_hits / seen)

    if not per_layer:
        return
    rate = float(np.mean(list(per_layer.values())))
    detail = {
        "set_agreement": rate,
        "exact_match": float(np.mean(list(exact.values()))),
        "per_layer_set_agreement": per_layer,
        "per_layer_exact_match": exact,
        "sampled_tokens": int(min(budget, n)),
        "logit_tensor_used": handle.manifest.get("logit_tensor_used"),
        "rows_compared": n_rows,
        "rows_mismatched": n_mismatched,
        "rows_mismatched_at_an_fp16_tie": n_tied,
        "rows_mismatched_unexplained": n_mismatched - n_tied,
    }
    if n_mismatched and n_mismatched == n_tied:
        yield Finding(
            "selection_argsort",
            "info",
            f"argsort(-logits)[:k] disagrees with topk.bin on {n_mismatched} of {n_rows} rows, "
            f"and every one is an exact tie between the k-th and (k+1)-th stored fp16 value. "
            "The disagreement is storage precision, not the selection chain: llama.cpp ordered "
            "those experts in fp32 before the down-cast, so topk.bin is right and logits.bin "
            "simply cannot express the difference.",
            detail,
        )
        return
    if rate < warn_below:
        yield Finding(
            "selection_argsort",
            "warning",
            f"argsort(-logits)[:k] reproduces topk.bin at only set_agreement={rate:.4f} "
            f"(exact_match={detail['exact_match']:.4f}); logit_tensor_used="
            f"{handle.manifest.get('logit_tensor_used')!r}. Expected < 1.0 for a biased router "
            "(GPT-OSS) or fp16 near-ties; anything far below 1.0 for an unbiased router means "
            "the manifest names the wrong node of the selection chain (I13) and every T8.2 "
            "margin is the wrong quantity.",
            detail,
        )
    else:
        yield Finding(
            "selection_argsort",
            "info",
            f"argsort(-logits)[:k] reproduces topk.bin exactly (set_agreement={rate:.4f}); "
            "consistent with logit_tensor_used naming the selection tensor (I13)",
            detail,
        )


# --------------------------------------------------------------------------------------
# check: capture_stats.json — truncation (I15) and layout diagnostics
# --------------------------------------------------------------------------------------


def check_truncation(
    handle: ShardHandle, *, stats: Mapping[str, Any] | None = None
) -> Iterator[Finding]:
    """``n_docs_truncated`` must be 0 (I15), plus the rest of ``capture_stats.json``.

    I15 is the one invariant in this module that cannot be recovered from the streams at all:
    a truncated document produces a perfectly well-formed shard that is simply missing text.
    ``max_doc_tokens`` is enforced under one reference tokenizer while the panel's vocabularies
    span 50k-262k, so a document at the cap for OLMoE can exceed it for Gemma 4 — and then the
    cross-model comparison every Phase 9 number rests on is comparing two different corpora.
    The count only exists because the harness records it; if the stats file is gone the
    invariant is unverifiable and that is itself worth saying.

    ``topk_layout`` is cross-checked here as the third angle on I12: ``"mixed"`` means the
    harness saw the tensor both packed and strided within one shard, i.e. it misidentified a
    layout, and ``nodes_captured`` is compared against ``3 * n_moe_layers`` to catch the
    over-broad filter T5.3 warns about.
    """
    if stats is None:
        yield Finding(
            "capture_stats",
            "warning",
            f"{STATS_NAME} not found in {handle.dir}; I15 truncation, topk_layout and the "
            "matched-node count cannot be verified from the streams alone",
            {"shard_dir": str(handle.dir)},
        )
        return

    truncated = int(stats.get("n_docs_truncated", 0) or 0)
    if truncated:
        yield Finding(
            "truncation",
            "error",
            f"{truncated} document(s) exceeded n_ctx and were truncated, first doc_id "
            f"{stats.get('first_truncated_doc')}, {stats.get('n_tokens_dropped')} token(s) "
            "dropped. This model was not shown the same text a smaller-vocab model was, which "
            "invalidates every cross-model comparison (I15). Fix the corpus cap, re-collect.",
            {
                "n_docs_truncated": truncated,
                "first_truncated_doc": stats.get("first_truncated_doc"),
                "n_tokens_dropped": stats.get("n_tokens_dropped"),
            },
        )
    else:
        yield Finding("truncation", "info", "no documents truncated (I15)", {"n_docs_truncated": 0})

    exit_code = stats.get("exit_code")
    if exit_code is not None and int(exit_code) != 0:
        yield Finding(
            "capture_stats",
            "error",
            f"the capture process exited {int(exit_code)}; this shard is not complete and must "
            "not be marked collected",
            {"exit_code": int(exit_code)},
        )

    layout = stats.get("topk_layout")
    if layout == "mixed":
        yield Finding(
            "capture_stats",
            "error",
            "topk_layout='mixed': ffn_moe_topk arrived both packed and strided within one "
            "shard, so the de-striding logic misidentified a layout. Nothing in ggml should do "
            "that, and the labels cannot be trusted (I12).",
            {"topk_layout": layout},
        )
    elif layout in (None, "none"):
        yield Finding(
            "capture_stats",
            "warning",
            f"topk_layout={layout!r}: the harness recorded no topk tensor layout, so the I12 "
            "de-striding path was never exercised",
            {"topk_layout": layout},
        )
    else:
        yield Finding(
            "capture_stats", "info", f"topk_layout={layout!r}", {"topk_layout": layout}
        )

    nodes = stats.get("nodes_captured")
    want_nodes = 3 * handle.spec.n_moe_layers
    if nodes is not None and int(nodes) % want_nodes != 0:
        # One ubatch touches each of the three nodes once per MoE layer, so the total is always
        # a multiple of 3 * n_moe_layers. A non-multiple means the callback filter matched
        # something it should not have (T5.3's over-broad-regex check).
        yield Finding(
            "capture_stats",
            "warning",
            f"nodes_captured={int(nodes)} is not a multiple of 3 x n_moe_layers={want_nodes}; "
            "the callback filter may be matching more tensors than the three streams",
            {"nodes_captured": int(nodes), "nodes_per_ubatch": want_nodes},
        )

    for key, manifest_key in (
        ("n_tokens", "n_tokens"),
        ("n_moe_layers", "n_moe_layers"),
        ("n_experts", "n_experts"),
        ("top_k", "top_k"),
        ("hidden_dim", "hidden_dim"),
    ):
        if key in stats and manifest_key in handle.manifest:
            if int(stats[key]) != int(handle.manifest[manifest_key]):
                yield Finding(
                    "capture_stats",
                    "error",
                    f"{STATS_NAME} says {key}={int(stats[key])} but the manifest says "
                    f"{manifest_key}={int(handle.manifest[manifest_key])}",
                    {"key": key, "stats": int(stats[key]), "manifest": int(handle.manifest[manifest_key])},
                )
    if "n_captured" in stats and int(stats["n_captured"]) != handle.n_captured:
        yield Finding(
            "capture_stats",
            "error",
            f"{STATS_NAME} says n_captured={int(stats['n_captured'])} but hidden_index.bin "
            f"holds {handle.n_captured} rows",
            {"stats": int(stats["n_captured"]), "on_disk": handle.n_captured},
        )


# --------------------------------------------------------------------------------------
# check: split coverage
# --------------------------------------------------------------------------------------


def check_split_coverage(
    handle: ShardHandle,
    *,
    doc_splits: Mapping[int, str] | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    required_splits: Sequence[str] = ("train", "val", "test"),
) -> Iterator[Finding]:
    """Every doc_id in the trace has a split, and each required split has tokens in it.

    An unassigned document silently drops out of every Family-A and Family-B number, because
    ``split_mask`` raises and the usual fix is to filter. An empty **test** split is worse: `H`
    and `CE` are both estimated on test (§1.2), so every Family-B quantity becomes undefined
    and every Family-A number is computed on nothing. Both are errors.

    Per-split *token* counters are accumulated instead of a set of doc ids, so this check's
    memory is ``O(len(required_splits))`` rather than ``O(n_docs)`` — see the module docstring's
    peak-memory claim. Note the scope: a single shard covers only its own ``shard_doc_range``,
    so per-shard emptiness is expected and the trace-level verdict comes from
    :func:`validate_shards` folding these counters.
    """
    if doc_splits is None:
        yield Finding(
            "split_coverage",
            "info",
            "no doc_splits supplied; split coverage not checked (pass the mapping from the "
            "corpus file, plan T4.3)",
            None,
        )
        return
    if "tokens" not in _sized_streams(handle):
        yield _skipped("split_coverage", "tokens")
        return

    per_split: dict[str, int] = {s: 0 for s in required_splits}
    unknown_tokens = 0
    unknown_examples: list[int] = []
    for sl in _chunks(handle.n_tokens, chunk_tokens):
        doc_ids = np.asarray(handle.tokens[sl]["doc_id"], dtype=np.int64)
        # Unique within the chunk only: bounded by chunk_tokens, not by n_docs.
        for doc_id in np.unique(doc_ids):
            key = int(doc_id)
            split = doc_splits.get(key)
            count = int((doc_ids == doc_id).sum())
            if split is None:
                unknown_tokens += count
                if len(unknown_examples) < _MAX_EXAMPLES and key not in unknown_examples:
                    unknown_examples.append(key)
            else:
                per_split[split] = per_split.get(split, 0) + count

    if unknown_tokens:
        yield Finding(
            "split_coverage",
            "error",
            f"{unknown_tokens} token(s) belong to document(s) with no split assignment "
            f"(first doc_ids: {unknown_examples}); they would be silently dropped from every "
            "metric",
            {"unknown_tokens": unknown_tokens, "example_doc_ids": unknown_examples},
        )
    yield Finding(
        "split_coverage",
        "info",
        f"tokens per split in this shard: {per_split}",
        {"tokens_per_split": per_split, "unknown_tokens": unknown_tokens},
    )


def _split_totals_finding(
    totals: Mapping[str, int], required_splits: Sequence[str]
) -> Iterator[Finding]:
    """Trace-level split emptiness — the verdict :func:`check_split_coverage` cannot reach."""
    empty = [s for s in required_splits if int(totals.get(s, 0)) == 0]
    if empty:
        yield Finding(
            "split_coverage",
            "error",
            f"split(s) {empty} contain zero tokens across the whole shard set. H and CE are "
            "both estimated on test (§1.2), so an empty test split makes every Family-B number "
            "undefined and every Family-A number an average over nothing.",
            {"empty_splits": empty, "tokens_per_split": dict(totals)},
        )
    else:
        yield Finding(
            "split_coverage",
            "info",
            f"all required splits non-empty: {dict(totals)}",
            {"tokens_per_split": dict(totals)},
        )


# --------------------------------------------------------------------------------------
# check: hidden subsample stride
# --------------------------------------------------------------------------------------


def check_hidden_stride(
    handle: ShardHandle,
    *,
    stats: Mapping[str, Any] | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> Iterator[Finding]:
    """Every captured index must be exactly ``doc_id * n_ctx + pos_in_doc`` for a stride multiple.

    `moe_trace` reserves ``n_ctx`` index values per document and captures token ``i`` of document
    ``d`` when ``(d * n_ctx + i) % hidden_stride == 0``. Because the blocks are a pure function of
    the corpus, this check does not have to *infer* anything: ``tokens.bin`` carries ``doc_id`` and
    ``pos_in_doc`` for every token, so each row of ``hidden_index.bin`` is recomputed and compared
    individually.

    That exactness is why the runner's own count check is only a bound — it has the shard's token
    total but not the per-document split, and cannot do this. The failure being caught here is a
    hidden state silently belonging to a *different* token than the one F4/F5/FV will label it
    with: not a crash, a relabelled feature.

    Two independently-written things are cross-checked at once, since they can disagree:
    ``tokens.flags`` bit 0, set by the writer as it lays down rows, and ``hidden_index.bin``,
    written from the capture predicate.
    """
    sized = _sized_streams(handle)
    if "hidden_index" not in sized:
        yield _skipped("hidden_stride", "hidden_index")
        return
    if "tokens" not in sized:
        yield _skipped("hidden_stride", "tokens")
        return

    stride = int(stats.get("hidden_stride", 0)) if stats else 0
    doc_span = int(stats.get("index_doc_span", 0)) if stats else 0

    # np.memmap refuses a zero-length file, and a shard with no captured rows is legitimate
    # (T4.4's fallback ladder may drop the subsample entirely), so handle it before mapping.
    if handle.n_captured == 0:
        if stride > 0 and stride <= handle.n_tokens:
            yield Finding(
                "hidden_stride",
                "error",
                f"hidden_stride={stride} over {handle.n_tokens} token(s) should have captured "
                "at least one row, but hidden_index.bin is empty",
                {"hidden_stride": stride, "n_tokens": handle.n_tokens},
            )
        else:
            yield Finding("hidden_stride", "info", "no hidden states captured", {"n_captured": 0})
        return

    index = handle.hidden_index

    # Strict ascent is what the reader requires of the concatenated set (T2.3) and is checkable
    # without capture_stats, so it runs first and unconditionally.
    ascending = True
    for sl in _chunks(handle.n_captured, chunk_tokens):
        window = np.asarray(index[sl], dtype=np.int64)
        if window.size > 1 and (np.diff(window) <= 0).any():
            ascending = False
            break
        if sl.start and window[0] <= int(index[sl.start - 1]):
            ascending = False
            break
    if not ascending:
        yield Finding(
            "hidden_stride",
            "error",
            "hidden_index.bin is not strictly ascending within the shard; the reader concatenates "
            "shards without rewriting indices, so a non-monotone stream makes the whole set "
            "unreadable",
            {"n_captured": handle.n_captured},
        )
        return

    if doc_span <= 0 or stride <= 0:
        yield Finding(
            "hidden_stride",
            "warning",
            "hidden_stride/index_doc_span are unknown (no capture_stats), so captured indices "
            "were only checked for strict ascent, not against doc_id * n_ctx + pos_in_doc",
            {"n_captured": handle.n_captured},
        )
        return

    bad: list[dict[str, int]] = []
    flag_mismatch = 0
    cursor = 0
    for sl in _chunks(handle.n_tokens, chunk_tokens):
        rows = handle.tokens[sl]
        doc_ids = np.asarray(rows["doc_id"], dtype=np.int64)
        pos = np.asarray(rows["pos_in_doc"], dtype=np.int64)
        flags = np.asarray(rows["flags"], dtype=np.uint32)

        want_index = doc_ids * doc_span + pos
        selected = (want_index % stride) == 0
        flag_mismatch += int((selected != ((flags & FLAG_HIDDEN_CAPTURED) != 0)).sum())

        want = want_index[selected]
        got = np.asarray(index[cursor : cursor + want.size], dtype=np.int64)
        if got.size == want.size and len(bad) < _MAX_EXAMPLES:
            for j in np.flatnonzero(got != want)[: _MAX_EXAMPLES - len(bad)]:
                bad.append(
                    {"row": cursor + int(j), "got": int(got[int(j)]), "want": int(want[int(j)])}
                )
        cursor += int(want.size)

    if cursor != handle.n_captured:
        yield Finding(
            "hidden_stride",
            "error",
            f"the rule (doc_id * {doc_span} + pos_in_doc) % {stride} == 0 selects {cursor} of "
            f"this shard's {handle.n_tokens} tokens, but hidden_index.bin holds "
            f"{handle.n_captured} rows",
            {
                "hidden_stride": stride,
                "index_doc_span": doc_span,
                "expected_n_captured": cursor,
                "n_captured": handle.n_captured,
            },
        )
        return

    if flag_mismatch:
        yield Finding(
            "hidden_stride",
            "error",
            f"tokens.flags disagrees with the stride rule on {flag_mismatch} token(s); the "
            "writer and the capture predicate selected different tokens, so hidden.bin's rows "
            "are not the tokens tokens.bin says they are",
            {"n_mismatched": flag_mismatch, "hidden_stride": stride},
        )
        return

    if bad:
        yield Finding(
            "hidden_stride",
            "error",
            f"captured token indices are not doc_id * {doc_span} + pos_in_doc (first mismatch "
            f"{bad[0]}); F4/F5 would read a router input belonging to a different token",
            {"examples": bad, "hidden_stride": stride, "index_doc_span": doc_span},
        )
        return

    yield Finding(
        "hidden_stride",
        "info",
        f"all {handle.n_captured} captured indices equal doc_id * {doc_span} + pos_in_doc and "
        f"are exactly the stride-{stride} multiples; tokens.flags agrees",
        {"hidden_stride": stride, "index_doc_span": doc_span, "n_captured": handle.n_captured},
    )


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

#: Check name -> adapter. Names are stable: the CLI's ``--checks`` and T5.2's collection log
#: refer to them, and a caller running a subset (e.g. only the cheap total checks during
#: collection, the argsort checks afterwards) selects by name.
_CHECKS: dict[str, Callable[..., Iterator[Finding]]] = {
    "size_arithmetic": lambda h, o: check_sizes(h),
    "lockstep": lambda h, o: check_lockstep(h, stats=o["stats"], chunk_tokens=o["chunk_tokens"]),
    "topk_labels": lambda h, o: check_topk_range_and_distinctness(h, chunk_tokens=o["chunk_tokens"]),
    "topk_strided_view": lambda h, o: check_topk_strided_view(
        h, sample_tokens=o["sample_tokens"], chunk_tokens=o["chunk_tokens"]
    ),
    "expert_usage": lambda h, o: check_expert_usage(h, chunk_tokens=o["chunk_tokens"]),
    "logits_sanity": lambda h, o: check_logits_sanity(h, chunk_tokens=o["chunk_tokens"]),
    "selection_argsort": lambda h, o: check_selection_argsort_agreement(
        h, sample_tokens=o["sample_tokens"]
    ),
    "capture_stats": lambda h, o: check_truncation(h, stats=o["stats"]),
    "split_coverage": lambda h, o: check_split_coverage(
        h, doc_splits=o["doc_splits"], chunk_tokens=o["chunk_tokens"]
    ),
    "hidden_stride": lambda h, o: check_hidden_stride(
        h, stats=o["stats"], chunk_tokens=o["chunk_tokens"]
    ),
}

CHECK_NAMES = tuple(_CHECKS)


def read_stats(shard_dir: Path | str) -> dict[str, Any] | None:
    """Load ``capture_stats.json`` if the harness left one next to the streams."""
    path = Path(shard_dir) / STATS_NAME
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    # The runner keeps the stats file in per-shard scratch and does not upload it, but it copies
    # the parsed contents into the manifest under "capture_stats" (runner.build_shard_manifest).
    # For an uploaded shard that embedded copy is the only surviving record, and refusing to read
    # it would make I15 permanently unverifiable for every shard the study actually publishes.
    manifest_path = Path(shard_dir) / MANIFEST_NAME
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        embedded = manifest.get("capture_stats")
        if isinstance(embedded, dict):
            return embedded
    return None


def validate_shard(
    shard_dir: Path | str,
    *,
    stats: Mapping[str, Any] | None = None,
    doc_splits: Mapping[int, str] | None = None,
    sample_tokens: int = DEFAULT_SAMPLE_TOKENS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    checks: Sequence[str] | None = None,
) -> ValidationReport:
    """Run every T5.3 check against one shard directory.

    ``stats`` defaults to the shard's own ``capture_stats.json``. A malformed manifest, or a
    shard whose shapes cannot be read at all, comes back as a report holding one error rather
    than as an exception: the caller is a collection loop that must record the verdict for
    every shard, not stop at the first bad one.
    """
    shard_dir = Path(shard_dir)
    if stats is None:
        stats = read_stats(shard_dir)

    try:
        manifest = read_manifest(shard_dir)
        spec = TraceSpec.from_manifest(manifest)
        handle = ShardHandle(shard_dir, manifest, spec)
    except FormatError as exc:
        return ValidationReport(
            shard_dir=str(shard_dir),
            shard_id=int((stats or {}).get("shard_id", -1)),
            model=str((stats or {}).get("model_spec", "")),
            findings=[Finding("manifest", "error", str(exc), {"shard_dir": str(shard_dir)})],
        )

    report = ValidationReport(
        shard_dir=str(shard_dir),
        shard_id=handle.shard_id,
        model=str(manifest["model"]),
        corpus=str(manifest["corpus"]),
        counters={
            "n_tokens": handle.n_tokens,
            "n_captured": handle.n_captured,
            "n_moe_layers": spec.n_moe_layers,
            "n_experts": spec.n_experts,
            "top_k": spec.top_k,
            "hidden_dim": spec.hidden_dim,
        },
    )

    options = {
        "stats": stats,
        "doc_splits": doc_splits,
        "sample_tokens": sample_tokens,
        "chunk_tokens": chunk_tokens,
    }
    wanted = list(checks) if checks is not None else list(CHECK_NAMES)
    unknown = [c for c in wanted if c not in _CHECKS]
    if unknown:
        raise ValueError(f"unknown check(s) {unknown}; choose from {list(CHECK_NAMES)}")
    for name in wanted:
        report.add(_CHECKS[name](handle, options))

    # Fold the numbers the shard-set verdict needs out of the findings, so validate_shards does
    # not have to re-read anything.
    for finding in report.by_check("split_coverage"):
        if finding.detail and "tokens_per_split" in finding.detail:
            report.counters["tokens_per_split"] = dict(finding.detail["tokens_per_split"])
    for finding in report.by_check("expert_usage"):
        if finding.detail and "load_balance_index_per_layer" in finding.detail:
            report.counters["load_balance_index_per_layer"] = dict(
                finding.detail["load_balance_index_per_layer"]
            )
    handle.close()
    return report


def discover_shards(path: Path | str) -> list[Path]:
    """Shard directories under ``path``, or ``[path]`` if it is itself a shard."""
    path = Path(path)
    if (path / MANIFEST_NAME).exists():
        return [path]
    return sorted(d for d in path.rglob("shard_*") if (d / MANIFEST_NAME).exists())


def validate_shards(
    paths: Path | str | Sequence[Path | str],
    *,
    doc_splits: Mapping[int, str] | None = None,
    sample_tokens: int = DEFAULT_SAMPLE_TOKENS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    checks: Sequence[str] | None = None,
    required_splits: Sequence[str] = ("train", "val", "test"),
) -> list[ValidationReport]:
    """Validate a set of shards, then the properties only the *set* has.

    Returns one report per shard plus a final trace-level report with ``shard_id = -1``,
    carrying the two verdicts no single shard can reach: the T4.3 split coverage folded over
    all shards (a shard covers only its own ``shard_doc_range``, so per-shard emptiness is
    normal), and S.3/I2 shard compatibility — shards collected under a different
    ``run_config_sha256`` or ``logit_tensor_used`` are a *different experiment*, and merging
    them is a hard error rather than a warning.
    """
    if isinstance(paths, (str, Path)):
        shard_dirs = discover_shards(paths)
        root_label = str(paths)
    else:
        shard_dirs = [d for p in paths for d in discover_shards(p)]
        root_label = ", ".join(str(p) for p in paths)

    reports = [
        validate_shard(
            d,
            doc_splits=doc_splits,
            sample_tokens=sample_tokens,
            chunk_tokens=chunk_tokens,
            checks=checks,
        )
        for d in shard_dirs
    ]

    trace = ValidationReport(
        shard_dir=root_label,
        shard_id=-1,
        model=reports[0].model if reports else "",
        corpus=reports[0].corpus if reports else "",
        counters={"n_shards": len(reports)},
    )
    if not shard_dirs:
        trace.add([Finding("shard_set", "error", f"no shard directories found under {root_label}")])
        return reports + [trace]

    totals: dict[str, int] = {}
    for report in reports:
        for split, count in (report.counters.get("tokens_per_split") or {}).items():
            totals[split] = totals.get(split, 0) + int(count)
    trace.counters["tokens_per_split"] = totals
    trace.counters["n_tokens"] = sum(int(r.counters.get("n_tokens", 0)) for r in reports)
    if doc_splits is not None:
        trace.add(_split_totals_finding(totals, required_splits))

    manifests = []
    for directory in shard_dirs:
        try:
            manifests.append(read_manifest(directory))
        except FormatError as exc:
            trace.add([Finding("shard_set", "error", str(exc), {"shard_dir": str(directory)})])
    if len(manifests) > 1:
        for key in SHARD_INVARIANT_KEYS:
            values = {json.dumps(m.get(key), sort_keys=True) for m in manifests}
            if len(values) > 1:
                trace.add(
                    [
                        Finding(
                            "shard_set",
                            "error",
                            f"manifest key {key!r} differs across shards ({sorted(values)}); "
                            "these shards were collected under different conditions and are a "
                            "different experiment — merging them is a hard error (S.3, I2)",
                            {"key": key, "values": sorted(values)},
                        )
                    ]
                )
    ids = [r.shard_id for r in reports]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        trace.add(
            [
                Finding(
                    "shard_set",
                    "error",
                    f"duplicate shard id(s) {duplicates}; one shard would shadow the other",
                    {"duplicate_shard_ids": duplicates},
                )
            ]
        )
    return reports + [trace]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.traces.validate <shard-or-trace-dir> …`` — exit 1 on any error finding.

    Non-zero exit is what T5.2's collection loop keys on: a shard that fails validation must
    not be marked complete, and the notebook's loop reads the exit code rather than the log.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.traces.validate",
        description="Post-collection per-shard trace validation (plan T5.3 / T3.1).",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="shard dirs or a trace dir")
    parser.add_argument("--corpus", type=Path, help="corpus JSONL, for the doc_id -> split map")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--sample-tokens", type=int, default=DEFAULT_SAMPLE_TOKENS)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument(
        "--checks", help=f"comma-separated subset of: {','.join(CHECK_NAMES)}"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print info findings too, not just error/warning"
    )
    args = parser.parse_args(argv)

    doc_splits = None
    if args.corpus:
        from ..corpus.build import load_doc_splits  # imported lazily: the CLI is the only user

        doc_splits = load_doc_splits(args.corpus)

    reports = validate_shards(
        [*args.paths],
        doc_splits=doc_splits,
        sample_tokens=args.sample_tokens,
        chunk_tokens=args.chunk_tokens,
        checks=args.checks.split(",") if args.checks else None,
    )

    floor = 0 if args.verbose else 1
    for report in reports:
        print(report.summary())
        for finding in report.findings:
            if _SEVERITY_RANK[finding.severity] >= floor:
                print(f"    [{finding.severity:7s}] {finding.check}: {finding.message}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([r.to_dict() for r in reports], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    n_errors = sum(r.n_errors for r in reports)
    print(f"{len(reports) - 1} shard(s): {n_errors} error(s), "
          f"{sum(r.n_warnings for r in reports)} warning(s)")
    return 1 if n_errors else 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
