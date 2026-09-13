"""Binary trace format — plan T2.3.

THE single source of truth for the on-disk layout. The C++ capture binary
(``src/capture/moe_trace.cpp``) and the Python reader both derive every stride from this
module; :func:`emit_c_header` generates the constants the C++ side compiles against, so the
two implementations cannot drift apart silently.

Layout, per (model, corpus, shard) directory::

    shard_00007/
        tokens.bin          16 B per token, corpus order
        topk.bin            int32[top_k] per (token, MoE layer) — THE AUTHORITATIVE LABELS
        logits.bin          float16[n_experts] per (token, MoE layer)
        hidden.bin          float16[hidden_dim] per (subsampled token, MoE layer)
        hidden_index.bin    uint32 global token index per subsampled token, ascending
        manifest.json

Both ``topk.bin`` and ``logits.bin`` are **layer-major within token**: all layers for token 0,
then all layers for token 1. This is the order the callback produces them in during a forward
pass, so the writer never has to buffer or transpose.

Two rules this module exists to enforce
---------------------------------------
* **Expert sets come from ``topk.bin``** (plan §1.6, invariant I1). ``logits.bin`` is retained
  for margin and drift analysis only. GPT-OSS has a router bias, so top-k recomputed from the
  unbiased logits is *wrong* — and at 128 experts the k-th/(k+1)-th margin is frequently inside
  fp16 resolution, which would inject depth-correlated noise into the target variable.
* **``topk.bin`` is I32, not F32** (plan T2.2 rule 3). Casting that buffer to ``const float*``
  produces garbage without crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "TOKEN_DTYPE",
    "TOPK_DTYPE",
    "LOGIT_DTYPE",
    "HIDDEN_DTYPE",
    "HIDDEN_INDEX_DTYPE",
    "FLAG_HIDDEN_CAPTURED",
    "STREAM_FILES",
    "MANIFEST_NAME",
    "TraceSpec",
    "FormatError",
    "expected_file_sizes",
    "check_file_sizes",
    "read_manifest",
    "write_manifest",
    "emit_c_header",
]


# --------------------------------------------------------------------------------------
# dtypes — all explicitly little-endian; the format is not host-endianness dependent
# --------------------------------------------------------------------------------------

#: One 16-byte record per token, in corpus order.
TOKEN_DTYPE = np.dtype(
    [
        ("token_id", "<u4"),
        ("doc_id", "<u4"),
        ("pos_in_doc", "<u4"),
        ("flags", "<u4"),
    ]
)

TOPK_DTYPE = np.dtype("<i4")  # expert indices, as emitted by the model (I5)
LOGIT_DTYPE = np.dtype("<f2")
HIDDEN_DTYPE = np.dtype("<f2")
HIDDEN_INDEX_DTYPE = np.dtype("<u4")  # GLOBAL token index, so shards concatenate unrewritten

#: ``tokens.flags`` bit 0 — this token's router input was captured into ``hidden.bin``.
FLAG_HIDDEN_CAPTURED = 1 << 0

STREAM_FILES = {
    "tokens": "tokens.bin",
    "topk": "topk.bin",
    "logits": "logits.bin",
    "hidden": "hidden.bin",
    "hidden_index": "hidden_index.bin",
}

MANIFEST_NAME = "manifest.json"

#: Manifest keys that must be present and non-null on every shard (plan T2.3).
#
# `model_sha256` and `engine_build` are ENGINE-NEUTRAL and required on every engine: the study is
# collected on llama.cpp AND (post-2026-09) vLLM, and neither "which gguf" nor "which llama.cpp
# commit" is meaningful for a vLLM run that loads safetensors and has no commit. `model_sha256` is
# the weights hash in whatever format the engine loaded; `engine_build` names the engine and its
# version ("llama_cpp@7077abbe", "vllm@0.10.2"). The old `gguf_sha256`/`llama_cpp_commit` are kept
# as OPTIONAL fields (below) so existing llama.cpp readers keep working; they are no longer required
# because a vLLM shard cannot supply them.
REQUIRED_MANIFEST_KEYS = (
    "model",
    "checkpoint_status",
    "model_sha256",
    "engine_build",
    "run_config_sha256",
    "quant",
    "router_dtype",
    "logit_tensor_used",
    "corpus",
    "shard_id",
    "shard_doc_range",
    "n_tokens",
    "n_moe_layers",
    "n_experts",
    "top_k",
    "hidden_dim",
    "hidden_subsample_n",
    # How hidden.bin's rows are indexed and how many there are. Required, not optional: without
    # them T5.3's stride check degrades to "is it ascending", and the failure it exists to catch
    # is a hidden state labelled with the wrong token, which ascends perfectly well.
    "n_captured",
    "hidden_stride",
    "index_scheme",
    "index_doc_span",
    "capture_flags",
    "layer_index_map",
    "device_plan",
    "file_sha256",
    "collected_utc",
)

#: Manifest keys that must be IDENTICAL across every shard of one (model, corpus) trace.
#: A mismatch means the shards were collected under different conditions and are a different
#: experiment — plan S.3 makes merging them a hard error, not a warning.
SHARD_INVARIANT_KEYS = (
    "model",
    "corpus",
    "run_config_sha256",
    "model_sha256",
    "engine_build",
    "quant",
    "router_dtype",
    "logit_tensor_used",
    "n_moe_layers",
    "n_experts",
    "top_k",
    "hidden_dim",
    "checkpoint_status",
)


class FormatError(RuntimeError):
    """On-disk trace does not match its manifest, or a manifest is malformed."""


# --------------------------------------------------------------------------------------
# spec + strides
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceSpec:
    """Shape parameters that fix every stride in the format.

    ``n_moe_layers`` counts *MoE* layers, not model layers. For DeepSeek-V2-Lite,
    ``first_k_dense_replace: 1`` means model layer 0 is a dense FFN and the trace holds 26
    layers, not 27; the model-layer correspondence lives in the manifest's ``layer_index_map``
    rather than being re-derived here (plan §1.4, T3.5).
    """

    n_moe_layers: int
    n_experts: int
    top_k: int
    hidden_dim: int

    def __post_init__(self) -> None:
        for name in ("n_moe_layers", "n_experts", "top_k", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise FormatError(f"TraceSpec.{name} must be a positive int, got {value!r}")
        if self.top_k > self.n_experts:
            raise FormatError(f"top_k={self.top_k} exceeds n_experts={self.n_experts}")

    # -- per-token strides, in bytes ----------------------------------------------------

    @property
    def token_stride(self) -> int:
        return TOKEN_DTYPE.itemsize

    @property
    def topk_stride(self) -> int:
        """Bytes per token in ``topk.bin`` — all layers."""
        return self.n_moe_layers * self.top_k * TOPK_DTYPE.itemsize

    @property
    def logit_stride(self) -> int:
        """Bytes per token in ``logits.bin`` — all layers."""
        return self.n_moe_layers * self.n_experts * LOGIT_DTYPE.itemsize

    @property
    def hidden_stride(self) -> int:
        """Bytes per *captured* token in ``hidden.bin`` — all layers."""
        return self.n_moe_layers * self.hidden_dim * HIDDEN_DTYPE.itemsize

    # -- memmap shapes ------------------------------------------------------------------

    def topk_shape(self, n_tokens: int) -> tuple[int, int, int]:
        return (n_tokens, self.n_moe_layers, self.top_k)

    def logit_shape(self, n_tokens: int) -> tuple[int, int, int]:
        return (n_tokens, self.n_moe_layers, self.n_experts)

    def hidden_shape(self, n_captured: int) -> tuple[int, int, int]:
        return (n_captured, self.n_moe_layers, self.hidden_dim)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "TraceSpec":
        return cls(
            n_moe_layers=int(manifest["n_moe_layers"]),
            n_experts=int(manifest["n_experts"]),
            top_k=int(manifest["top_k"]),
            hidden_dim=int(manifest["hidden_dim"]),
        )


# --------------------------------------------------------------------------------------
# size arithmetic — plan T5.3 ("file sizes match manifest arithmetic exactly, per stream")
# --------------------------------------------------------------------------------------


def expected_file_sizes(spec: TraceSpec, n_tokens: int, n_captured: int) -> dict[str, int]:
    """Byte size each stream file must have, exactly."""
    return {
        "tokens.bin": n_tokens * spec.token_stride,
        "topk.bin": n_tokens * spec.topk_stride,
        "logits.bin": n_tokens * spec.logit_stride,
        "hidden.bin": n_captured * spec.hidden_stride,
        "hidden_index.bin": n_captured * HIDDEN_INDEX_DTYPE.itemsize,
    }


def check_file_sizes(
    shard_dir: Path, spec: TraceSpec, n_tokens: int, n_captured: int
) -> None:
    """Raise :class:`FormatError` on the first stream whose size is off by even one byte.

    A truncated upload that silently "succeeds" is the most likely way to lose a session's
    work (plan T5.3), and a partially-written stream is indistinguishable from a good one
    without this check.
    """
    expected = expected_file_sizes(spec, n_tokens, n_captured)
    for filename, want in expected.items():
        path = shard_dir / filename
        if not path.exists():
            raise FormatError(f"{path}: missing")
        got = path.stat().st_size
        if got != want:
            raise FormatError(
                f"{path}: size {got} B, manifest arithmetic says {want} B "
                f"(delta {got - want:+d}); n_tokens={n_tokens}, n_captured={n_captured}, "
                f"spec={spec}"
            )


# --------------------------------------------------------------------------------------
# manifest I/O
# --------------------------------------------------------------------------------------


def read_manifest(shard_dir: Path) -> dict[str, Any]:
    """Load and structurally validate one shard manifest."""
    path = Path(shard_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FormatError(f"{path}: missing manifest") from exc
    except json.JSONDecodeError as exc:
        raise FormatError(f"{path}: malformed JSON — {exc}") from exc

    missing = [k for k in REQUIRED_MANIFEST_KEYS if manifest.get(k) is None]
    if missing:
        raise FormatError(f"{path}: required manifest keys missing or null: {missing}")
    return manifest


def write_manifest(shard_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest atomically, after checking it has every required key."""
    missing = [k for k in REQUIRED_MANIFEST_KEYS if manifest.get(k) is None]
    if missing:
        raise FormatError(f"refusing to write manifest, keys missing or null: {missing}")

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------
# C header generation — keeps moe_trace.cpp in lockstep with this module
# --------------------------------------------------------------------------------------

_C_HEADER_TEMPLATE = """\
// GENERATED by src/traces/format.py - do not edit by hand.
// Regenerate:  python -m src.traces.format --emit-header src/capture/trace_format.h
//
// The Python reader and this header are produced from one definition so that the capture
// binary and the analysis code cannot disagree about the on-disk layout.

#pragma once
#include <stdint.h>

#define MOE_TRACE_FORMAT_VERSION {version}

// tokens.bin - one record per token, in corpus order.
typedef struct {{
    uint32_t token_id;
    uint32_t doc_id;
    uint32_t pos_in_doc;
    uint32_t flags;        // bit0: router input captured into hidden.bin
}} moe_token_record;

#define MOE_FLAG_HIDDEN_CAPTURED {flag_hidden}
#define MOE_TOKEN_RECORD_BYTES   {token_bytes}

// Element types, asserted per stream in the callback (plan T2.2 rule 3).
//   topk.bin    <- GGML_TYPE_I32   ({topk_bytes} B per index)
//   NOTE: casting the topk buffer to `const float *` produces garbage SILENTLY.
//   logits.bin  <- GGML_TYPE_F32 source, written as float16 ({logit_bytes} B)
//   hidden.bin  <- GGML_TYPE_F32 source, written as float16 ({hidden_bytes} B)
#define MOE_TOPK_ELEM_BYTES   {topk_bytes}
#define MOE_LOGIT_ELEM_BYTES  {logit_bytes}
#define MOE_HIDDEN_ELEM_BYTES {hidden_bytes}
#define MOE_HIDDEN_INDEX_ELEM_BYTES {hidden_index_bytes}

// Per-token strides. Both topk and logits are LAYER-MAJOR WITHIN TOKEN.
#define MOE_TOPK_STRIDE(n_layers, top_k)        ((size_t)(n_layers) * (top_k) * MOE_TOPK_ELEM_BYTES)
#define MOE_LOGIT_STRIDE(n_layers, n_experts)   ((size_t)(n_layers) * (n_experts) * MOE_LOGIT_ELEM_BYTES)
#define MOE_HIDDEN_STRIDE(n_layers, hidden_dim) ((size_t)(n_layers) * (hidden_dim) * MOE_HIDDEN_ELEM_BYTES)

static const char * const MOE_FILE_TOKENS       = "{f_tokens}";
static const char * const MOE_FILE_TOPK         = "{f_topk}";
static const char * const MOE_FILE_LOGITS       = "{f_logits}";
static const char * const MOE_FILE_HIDDEN       = "{f_hidden}";
static const char * const MOE_FILE_HIDDEN_INDEX = "{f_hidden_index}";
"""

FORMAT_VERSION = 1


def emit_c_header() -> str:
    """Render the C header that ``moe_trace.cpp`` includes."""
    return _C_HEADER_TEMPLATE.format(
        version=FORMAT_VERSION,
        flag_hidden=FLAG_HIDDEN_CAPTURED,
        token_bytes=TOKEN_DTYPE.itemsize,
        topk_bytes=TOPK_DTYPE.itemsize,
        logit_bytes=LOGIT_DTYPE.itemsize,
        hidden_bytes=HIDDEN_DTYPE.itemsize,
        hidden_index_bytes=HIDDEN_INDEX_DTYPE.itemsize,
        f_tokens=STREAM_FILES["tokens"],
        f_topk=STREAM_FILES["topk"],
        f_logits=STREAM_FILES["logits"],
        f_hidden=STREAM_FILES["hidden"],
        f_hidden_index=STREAM_FILES["hidden_index"],
    )


if __name__ == "__main__":  # pragma: no cover - tiny CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-header", type=Path, help="write the generated C header here")
    args = parser.parse_args()
    if args.emit_header:
        args.emit_header.parent.mkdir(parents=True, exist_ok=True)
        args.emit_header.write_text(emit_c_header(), encoding="utf-8")
        print(f"wrote {args.emit_header}")
    else:
        print(emit_c_header())
