"""Shard-aware trace reader — plan T6.1.

Presents a multi-shard trace as one continuous token axis. Shard boundaries are invisible to
callers, and files are never concatenated on disk — each shard is mapped separately and global
token indices are resolved internally.

Two hard constraints, both from the plan:

* **:meth:`TraceReader.topk_sets` reads ``topk.bin``. It does not recompute top-k from logits,
  and there is no code path in this module that could.** This is the API-level enforcement of
  §1.6 / invariant I1. GPT-OSS selects on biased logits, and at 128 experts the k-th/(k+1)-th
  margin is often inside fp16 resolution, so a recomputation would inject depth-correlated
  noise into the *target variable* — contaminating precisely the Phase 8 result.
* **No method materialises a full model's logit array.** Qwen3 is 128 experts × 48 layers ×
  2 B = 12 KB/token, so 1M tokens is ~12 GB. Everything is ``np.memmap`` plus explicit slicing;
  :meth:`iter_chunks` exists so metrics accumulate over chunks rather than over the corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ..runtime.config import assert_shards_compatible
from .format import (
    FLAG_HIDDEN_CAPTURED,
    HIDDEN_DTYPE,
    HIDDEN_INDEX_DTYPE,
    LOGIT_DTYPE,
    SHARD_INVARIANT_KEYS,
    STREAM_FILES,
    TOKEN_DTYPE,
    TOPK_DTYPE,
    FormatError,
    TraceSpec,
    check_file_sizes,
    read_manifest,
)

__all__ = ["TraceReader", "ShardHandle"]

DEFAULT_CHUNK_TOKENS = 65_536


class ShardHandle:
    """One shard's files, mapped lazily and kept mapped."""

    def __init__(self, directory: Path, manifest: Mapping[str, Any], spec: TraceSpec) -> None:
        self.dir = directory
        self.manifest = manifest
        self.spec = spec
        self.shard_id = int(manifest["shard_id"])
        self.n_tokens = int(manifest["n_tokens"])

        # n_captured is derived from hidden_index.bin rather than the manifest: the file is
        # the authority on how many rows hidden.bin actually holds, and a mismatch between the
        # two is exactly the truncation we want check_file_sizes to catch.
        index_path = self.dir / STREAM_FILES["hidden_index"]
        if index_path.exists():
            size = index_path.stat().st_size
            if size % HIDDEN_INDEX_DTYPE.itemsize:
                raise FormatError(f"{index_path}: size {size} is not a whole number of uint32")
            self.n_captured = size // HIDDEN_INDEX_DTYPE.itemsize
        else:
            self.n_captured = 0

        self.token_offset = 0  # set by TraceReader
        self.hidden_offset = 0
        self._maps: dict[str, np.memmap] = {}

    # -- lazy memmaps --------------------------------------------------------------------

    def _map(self, stream: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
        cached = self._maps.get(stream)
        if cached is None:
            path = self.dir / STREAM_FILES[stream]
            cached = np.memmap(path, dtype=dtype, mode="r", shape=shape)
            self._maps[stream] = cached
        return cached

    @property
    def tokens(self) -> np.memmap:
        return self._map("tokens", TOKEN_DTYPE, (self.n_tokens,))

    @property
    def topk(self) -> np.memmap:
        return self._map("topk", TOPK_DTYPE, self.spec.topk_shape(self.n_tokens))

    @property
    def logits(self) -> np.memmap:
        return self._map("logits", LOGIT_DTYPE, self.spec.logit_shape(self.n_tokens))

    @property
    def hidden(self) -> np.memmap:
        return self._map("hidden", HIDDEN_DTYPE, self.spec.hidden_shape(self.n_captured))

    @property
    def hidden_index(self) -> np.memmap:
        return self._map("hidden_index", HIDDEN_INDEX_DTYPE, (self.n_captured,))

    def close(self) -> None:
        self._maps.clear()

    def __repr__(self) -> str:
        return f"ShardHandle(id={self.shard_id}, n_tokens={self.n_tokens})"


class TraceReader:
    """Read one (model, corpus) trace assembled from any number of shards.

    Parameters
    ----------
    root:
        Trace root; shards live at ``root/<model>/<corpus>/shard_*/``.
    doc_splits:
        Optional ``{doc_id: "train"|"val"|"test"}`` mapping, needed only by
        :meth:`split_mask`. Splits are assigned at document level and written into the corpus
        file (plan T4.3), so they come from there rather than from the trace.
    validate_sizes:
        Check every stream's size against manifest arithmetic at construction (plan T5.3).
        On by default; it is O(number of files), not O(bytes).
    """

    def __init__(
        self,
        root: Path | str,
        model: str,
        corpus: str,
        *,
        doc_splits: Mapping[int, str] | None = None,
        validate_sizes: bool = True,
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.corpus = corpus
        self._doc_splits = dict(doc_splits) if doc_splits is not None else None

        trace_dir = self.root / model / corpus
        if not trace_dir.is_dir():
            raise FormatError(f"{trace_dir}: no such trace directory")

        shard_dirs = sorted(d for d in trace_dir.iterdir() if d.is_dir() and d.name.startswith("shard_"))
        if not shard_dirs:
            raise FormatError(f"{trace_dir}: contains no shard_* directories")

        manifests = [read_manifest(d) for d in shard_dirs]

        # Plan S.3: shards collected under different conditions are different experiments.
        assert_shards_compatible(manifests, SHARD_INVARIANT_KEYS)

        self.spec = TraceSpec.from_manifest(manifests[0])
        if self.spec.n_experts > np.iinfo(np.int16).max:
            raise FormatError(
                f"n_experts={self.spec.n_experts} does not fit the int16 expert-index "
                "convention used by topk_sets(); widen the dtype before proceeding"
            )

        handles = [ShardHandle(d, m, self.spec) for d, m in zip(shard_dirs, manifests)]
        handles.sort(key=lambda h: h.shard_id)

        ids = [h.shard_id for h in handles]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise FormatError(f"{trace_dir}: duplicate shard ids {sorted(duplicates)}")

        token_offset = hidden_offset = 0
        for handle in handles:
            handle.token_offset = token_offset
            handle.hidden_offset = hidden_offset
            token_offset += handle.n_tokens
            hidden_offset += handle.n_captured
            if validate_sizes:
                check_file_sizes(handle.dir, self.spec, handle.n_tokens, handle.n_captured)

        self._shards = handles
        self.n_tokens = token_offset
        self.n_captured = hidden_offset
        self._hidden_tokens: np.ndarray | None = None
        self._captured: np.ndarray | None = None

    # -- metadata -------------------------------------------------------------------------

    @property
    def manifests(self) -> list[Mapping[str, Any]]:
        return [h.manifest for h in self._shards]

    @property
    def shard_ids(self) -> list[int]:
        return [h.shard_id for h in self._shards]

    @property
    def n_moe_layers(self) -> int:
        return self.spec.n_moe_layers

    @property
    def n_experts(self) -> int:
        return self.spec.n_experts

    @property
    def top_k(self) -> int:
        return self.spec.top_k

    @property
    def run_config_sha256(self) -> str:
        return self._shards[0].manifest["run_config_sha256"]

    @property
    def logit_tensor_used(self) -> str:
        """``ffn_moe_logits`` or ``ffn_moe_logits_biased`` — see plan §1.6."""
        return self._shards[0].manifest["logit_tensor_used"]

    def model_layer(self, trace_layer: int) -> int:
        """Map a trace layer index to the model's own layer index.

        DeepSeek-V2-Lite's ``first_k_dense_replace: 1`` makes model layer 0 a dense FFN, so
        trace layer 0 is model layer 1. The mapping is recorded explicitly in the manifest
        rather than re-derived (plan T3.5) — an off-by-one here corrupts every per-depth result.
        """
        layer_map = self._shards[0].manifest.get("layer_index_map")
        if layer_map is None:
            raise FormatError("manifest has no layer_index_map")

        # A list is the canonical form the runner writes (position = trace layer, value = model
        # layer), because that is what `nodespec.NodeSpec.layer_map` is. The keyed form is also
        # accepted: it was what the synthetic fixtures wrote, and reading only one of the two
        # meant a real collection's manifest was unreadable by the code meant to analyse it.
        if isinstance(layer_map, (list, tuple)):
            if not 0 <= trace_layer < len(layer_map):
                raise FormatError(
                    f"layer_index_map has {len(layer_map)} entries, no trace layer {trace_layer}"
                )
            return int(layer_map[trace_layer])

        key = f"trace_{trace_layer}"
        if key in layer_map:
            value = layer_map[key]
            if isinstance(value, str) and value.startswith("model_layer_"):
                return int(value.rsplit("_", 1)[1])
            return int(value)
        raise FormatError(f"layer_index_map has no entry for {key}")

    # -- index plumbing ---------------------------------------------------------------------

    def _check_layer(self, layer: int) -> int:
        if not 0 <= layer < self.spec.n_moe_layers:
            raise IndexError(
                f"layer {layer} out of range for {self.spec.n_moe_layers} MoE layers"
            )
        return layer

    def _normalize(self, token_slice: slice | int | None) -> slice:
        if token_slice is None:
            return slice(0, self.n_tokens)
        if isinstance(token_slice, (int, np.integer)):
            index = int(token_slice)
            if index < 0:
                index += self.n_tokens
            return slice(index, index + 1)
        start, stop, step = token_slice.indices(self.n_tokens)
        if step != 1:
            raise ValueError("strided token slices are not supported; read a range and stride it")
        return slice(start, max(start, stop))

    def _spans(self, window: slice) -> Iterator[tuple[ShardHandle, int, int, int, int]]:
        """Yield ``(shard, local_start, local_stop, out_start, out_stop)`` for a global slice."""
        for handle in self._shards:
            begin = max(window.start, handle.token_offset)
            end = min(window.stop, handle.token_offset + handle.n_tokens)
            if begin < end:
                yield (
                    handle,
                    begin - handle.token_offset,
                    end - handle.token_offset,
                    begin - window.start,
                    end - window.start,
                )

    def iter_chunks(self, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> Iterator[slice]:
        """Slices covering the corpus, for chunk-composable metric accumulation."""
        for start in range(0, self.n_tokens, chunk_tokens):
            yield slice(start, min(start + chunk_tokens, self.n_tokens))

    # -- streams ------------------------------------------------------------------------------

    def topk_sets(self, layer: int, token_slice: slice | int | None = None) -> np.ndarray:
        """Model-emitted expert sets, ``(n, top_k)`` int16.

        **Read directly from ``topk.bin``** — the indices llama.cpp's ``ffn_moe_topk-<il>``
        node emitted, bias-inclusive by construction and immune to fp16 near-ties. Order is
        the model's own; sort the result yourself if you need canonical ordering.
        """
        self._check_layer(layer)
        window = self._normalize(token_slice)
        out = np.empty((window.stop - window.start, self.spec.top_k), dtype=np.int16)
        for handle, ls, le, os_, oe in self._spans(window):
            out[os_:oe] = handle.topk[ls:le, layer, :]
        return out

    def logits(self, layer: int, token_slice: slice | int | None = None) -> np.ndarray:
        """Raw router logits, ``(n, n_experts)`` float32.

        For **margin and drift analysis only** (plan §1.6). Do not derive expert sets here —
        use :meth:`topk_sets`. Which tensor these came from is in :attr:`logit_tensor_used`.
        """
        self._check_layer(layer)
        window = self._normalize(token_slice)
        out = np.empty((window.stop - window.start, self.spec.n_experts), dtype=np.float32)
        for handle, ls, le, os_, oe in self._spans(window):
            out[os_:oe] = handle.logits[ls:le, layer, :].astype(np.float32)
        return out

    def margins(self, layer: int, token_slice: slice | int | None = None) -> np.ndarray:
        """k-th minus (k+1)-th largest logit per token, ``(n,)`` float32.

        This is what makes the Phase 8 flip-rate analysis interpretable: flips concentrated at
        small margins are expected arithmetic, flips at large margins are a bug (plan T8.2).
        """
        values = self.logits(layer, token_slice)
        k = self.spec.top_k
        if k >= self.spec.n_experts:
            return np.zeros(values.shape[0], dtype=np.float32)
        # Partition on the negated array so index k-1 is the k-th largest and k the (k+1)-th.
        part = np.partition(-values, kth=(k - 1, k), axis=1)
        return (-part[:, k - 1] + part[:, k]).astype(np.float32)

    def tokens(self, token_slice: slice | int | None = None) -> np.ndarray:
        """Token records, structured ``(token_id, doc_id, pos_in_doc, flags)``."""
        window = self._normalize(token_slice)
        out = np.empty(window.stop - window.start, dtype=TOKEN_DTYPE)
        for handle, ls, le, os_, oe in self._spans(window):
            out[os_:oe] = handle.tokens[ls:le]
        return out

    # -- hidden states -------------------------------------------------------------------------

    def _captured_rows(self) -> np.ndarray:
        """Token **row positions** (0..n_tokens-1, corpus order) that carry a hidden state.

        This is the reader's currency for the subsample, and it is deliberately not the value
        stored in ``hidden_index.bin``. That file holds ``doc_id * n_ctx + pos_in_doc`` — a
        sparse, per-document reserved block, chosen so shards concatenate without rewriting
        (T2.3) and so which tokens are captured does not depend on how documents were grouped.
        Every consumer, though, works in row positions: ``topk_sets(layer)[rows]``,
        ``index.pos_in_doc[rows]``. Handing them sparse indices produced no error, just an empty
        intersection — the FV probe reported "no usable rows after exclusions" on a healthy trace.

        The rows come from ``tokens.flags``, so this also cross-checks the two streams: the flag
        bits and ``hidden_index.bin`` are written by different code paths and must agree in count.
        """
        if self._captured is None:
            parts = []
            for handle in self._shards:
                flags = np.asarray(handle.tokens["flags"], dtype=np.uint32)
                local = np.flatnonzero(flags & FLAG_HIDDEN_CAPTURED)
                if local.size != handle.n_captured:
                    raise FormatError(
                        f"shard {handle.shard_id}: {local.size} token(s) carry "
                        f"FLAG_HIDDEN_CAPTURED but hidden.bin holds {handle.n_captured} row(s)"
                    )
                parts.append(local.astype(np.int64) + handle.token_offset)
            self._captured = (
                np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            )
        return self._captured

    def _hidden_token_ids(self) -> np.ndarray:
        """The stored global indices of every captured token, ascending.

        Not used for lookup — :meth:`_captured_rows` is — but the ascent is the invariant that
        lets a shard set be concatenated at all, so it is checked whenever anything touches the
        hidden streams.
        """
        if self._hidden_tokens is None:
            parts = [np.asarray(h.hidden_index) for h in self._shards if h.n_captured]
            ids = np.concatenate(parts).astype(np.int64) if parts else np.empty(0, np.int64)
            if ids.size and np.any(np.diff(ids) <= 0):
                raise FormatError(
                    "hidden_index is not strictly ascending across shards; indices must be a "
                    "pure function of the corpus (doc_id * n_ctx + pos_in_doc) so concatenation "
                    "needs no rewriting (plan T2.3)"
                )
            self._hidden_tokens = ids
        return self._hidden_tokens

    def captured_rows(self) -> np.ndarray:
        """Copy of the token row positions that have hidden states."""
        self._hidden_token_ids()  # assert the on-disk invariant before handing rows out
        return self._captured_rows().copy()

    def captured_token_ids(self) -> np.ndarray:
        """Deprecated alias for :meth:`captured_rows`, kept so callers fail loudly, not subtly."""
        return self.captured_rows()

    def hidden(self, layer: int, token_idxs: Sequence[int] | np.ndarray) -> np.ndarray:
        """Router-input vectors at ``layer`` for the given token **row positions**.

        ``(n, hidden_dim)`` float32.

        .. note::
           The plan's T6.1 signature is ``hidden(self, token_idxs)`` with no layer argument,
           but F4/F5 condition on the router input at layer ℓ−1 and FV on layer ℓ, so a layer
           selector is required. Deviation recorded here deliberately.

        Every requested row must have been captured — asking for an uncaptured token is a bug in
        the caller's subsample handling, not something to silently drop.
        """
        self._check_layer(layer)
        wanted = np.asarray(token_idxs, dtype=np.int64).ravel()
        available = self._captured_rows()

        rows = np.searchsorted(available, wanted)
        bad = (rows >= available.size) | (available[np.minimum(rows, available.size - 1)] != wanted)
        if available.size == 0 or np.any(bad):
            missing = wanted[bad] if available.size else wanted
            raise KeyError(
                f"{missing.size} requested token(s) have no captured hidden state "
                f"(first few: {missing[:5].tolist()}); check the subsample mask"
            )

        out = np.empty((wanted.size, self.spec.hidden_dim), dtype=np.float32)
        for handle in self._shards:
            if not handle.n_captured:
                continue
            lo, hi = handle.hidden_offset, handle.hidden_offset + handle.n_captured
            sel = (rows >= lo) & (rows < hi)
            if not sel.any():
                continue
            local = rows[sel] - lo
            out[sel] = handle.hidden[local, layer, :].astype(np.float32)
        return out

    # -- splits ---------------------------------------------------------------------------------

    def split_mask(self, split: str) -> np.ndarray:
        """Boolean mask over tokens for a document-level split.

        Splits are assigned per document, 80/10/10, stratified by (domain, lang), and written
        into the corpus file so they are identical across models (plan T4.3). Token-level
        splits would leak through adjacency, which is why F3 in particular needs this.
        """
        if self._doc_splits is None:
            raise ValueError(
                "split_mask requires doc_splits; construct TraceReader with the mapping from "
                "the corpus file (plan T4.3)"
            )
        doc_ids = self.tokens()["doc_id"]
        # One np.unique with inverse indices, not three passes plus a per-token dict lookup: this
        # runs over every token of every shard, and Qwen3 at 1M tokens makes the difference visible.
        uniq, inverse = np.unique(doc_ids, return_inverse=True)

        unknown = [int(d) for d in uniq if int(d) not in self._doc_splits]
        if unknown:
            raise KeyError(
                f"{len(unknown)} doc_id(s) in the trace have no split assignment "
                f"(first few: {sorted(unknown)[:5]})"
            )
        lookup = np.array([self._doc_splits[int(d)] == split for d in uniq], dtype=bool)
        return lookup[inverse.reshape(doc_ids.shape)]

    # -- lifecycle --------------------------------------------------------------------------------

    def close(self) -> None:
        for handle in self._shards:
            handle.close()

    def __enter__(self) -> "TraceReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"TraceReader(model={self.model!r}, corpus={self.corpus!r}, "
            f"shards={len(self._shards)}, tokens={self.n_tokens}, "
            f"layers={self.spec.n_moe_layers}, experts={self.spec.n_experts}, "
            f"top_k={self.spec.top_k})"
        )
