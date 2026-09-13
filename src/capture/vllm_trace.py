"""vLLM router-trace capture — the engine port of ``moe_trace.cpp`` (plan §1.6, P1).

Why this module exists
----------------------
The study was collected on llama.cpp with a C++ ``cb_eval`` callback (``moe_trace.cpp``). The
30-min-per-session llama.cpp build made paid Kaggle GPU time expensive, so the collection engine
is being ported to vLLM. This module produces the SAME five files ``moe_trace.cpp`` did, byte for
byte, so nothing downstream of the trace directory (Phase-6 readers, Phase-8 analysis) has to
change:

    tokens.bin  topk.bin  logits.bin  hidden.bin  hidden_index.bin  manifest.json

Byte-identity is delegated, not re-implemented: every stride, dtype and the size arithmetic come
from :mod:`src.traces.format`, the one source of truth the C++ header is also generated from.
numpy's ``astype(np.float16)`` is IEEE round-to-nearest-even, matching the hand-written
``f32_to_f16`` in ``trace_writer.hpp`` (verified element-wise in the tests), so ``logits.bin`` and
``hidden.bin`` land the same bits either engine produced them.

The three streams (plan §1.6), and where vLLM exposes each
----------------------------------------------------------
``models.yaml`` records llama.cpp *node* names (``ffn_moe_probs``, ``ffn_norm``); those do not
exist in vLLM. The SEMANTICS carry over, captured with PyTorch module hooks on the router:

* **topk.bin** — the authoritative expert labels (I1). Recomputed here by argsort of the SAME
  gating tensor vLLM's ``select_experts`` consumes, then asserted equal to vLLM's own selection on
  every token (:class:`SelectionMismatch` is a HALT, mirroring llama.cpp's T1.4). Recompute-not-
  read is deliberate: it is the only construction that also works for the TP>1 models later, and
  the equality assert is what proves it faithful.
* **logits.bin** — the margin stream. ``logit_tensor_used`` is preserved from the llama.cpp study:
  for the whole panel it is ``ffn_moe_probs``, i.e. the POST-gating probabilities, not the raw
  logits. So the gating op (softmax for OLMoE/Qwen/Gemma; +bias then softmax for GPT-OSS) is
  applied before writing, keeping margins comparable with every trace already collected.
* **hidden.bin** — the router-input hidden state (F4/F5/FV), captured on the subsampled tokens via
  a ``forward_pre_hook`` on the router module: its input is exactly the normalized hidden state the
  router matmul consumes.

Scope of THIS file (P1, TP=1)
-----------------------------
Python forward hooks only reach modules in the driver process, so this path is for the five models
that fit on one card (OLMoE x3, DeepSeek-V2-Lite, GPT-OSS-20B). The two TP=2 models (Qwen3-30B,
Gemma-4) need vLLM's in-worker routed-experts capturer and are a later step; the writer, manifest
and selection-check code here are engine-agnostic and will be shared with it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# The on-disk contract. Reused, never re-derived: this is what guarantees the vLLM traces are
# byte-identical to the llama.cpp ones the analysis already trusts.
from src.traces.format import (
    HIDDEN_INDEX_DTYPE,
    LOGIT_DTYPE,
    TOPK_DTYPE,
    TraceSpec,
)

__all__ = [
    "CaptureError",
    "SelectionMismatch",
    "GatingOp",
    "RouterCapture",
    "DocumentTrace",
    "recompute_topk",
    "gating_from_config",
    "discover_router_and_experts",
    "worker_install_capture",
    "worker_reset_capture",
    "worker_drain_capture",
    "worker_remove_capture",
]


class CaptureError(RuntimeError):
    """The capture harness is misconfigured or a hook produced an impossible tensor."""


class SelectionMismatch(CaptureError):
    """Recomputed top-k disagrees with vLLM's own expert selection.

    This is a HALT, not a warning. topk.bin is the study's target variable (I1); a single wrong
    label per layer per token silently corrupts every downstream entropy/MI quantity, and unlike a
    size mismatch nothing later would catch it. Mirrors llama.cpp T1.4's recompute-and-compare gate.
    """


# --------------------------------------------------------------------------------------
# gating — reproducing what the model's router does BEFORE top-k, per models.yaml
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GatingOp:
    """How raw router logits become the tensor top-k is taken over, for one model.

    ``logit_tensor_used`` in models.yaml says which node the study's ``logits.bin`` came from. For
    the whole panel that is ``ffn_moe_probs`` — the softmax (GPT-OSS: bias-then-softmax) output.
    Softmax is monotonic so it does not change WHICH experts win (topk.bin is unaffected), but the
    k/(k+1) MARGIN differs, and the margin is the measurement (T8.2). So logits.bin must hold the
    same transform the study recorded.

    ``has_router_bias`` (GPT-OSS' ``gate_inp_b``) is added to the raw logits BEFORE softmax and
    before selection — it is not order-preserving, so it changes topk.bin too, and recomputed top-k
    that ignored it would name experts the model never routed. It is applied in both places here.
    """

    softmax: bool
    has_router_bias: bool
    logit_tensor_used: str

    def selection_scores(self, logits: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
        """The tensor top-k is actually taken over (float64, for a stable argsort)."""
        scores = logits.astype(np.float64, copy=True)
        if self.has_router_bias:
            if bias is None:
                raise CaptureError(
                    "has_router_bias is set but no router bias tensor was captured; GPT-OSS "
                    "selects on logits+gate_inp_b and recomputing top-k without it is wrong"
                )
            scores = scores + bias.astype(np.float64)
        return scores

    def written_logits(self, logits: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
        """The tensor to store in logits.bin — matches ``logit_tensor_used``.

        For every panel model that is the post-gating probabilities: softmax over the (optionally
        biased) logits. Kept in float64 through the softmax; the caller casts to float16 exactly
        once, at write time, so the rounding happens in one place.
        """
        scores = self.selection_scores(logits, bias)
        if not self.softmax:
            return scores
        scores = scores - scores.max(axis=-1, keepdims=True)  # numerically stable softmax
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        return scores


def gating_from_config(config: Mapping[str, Any]) -> GatingOp:
    """Build the :class:`GatingOp` for a model from its ``models.yaml`` card.

    The panel's ``post_topk`` field describes what happens AFTER selection (normalization over the
    chosen experts) and does not affect either stream here. What matters for logits.bin is that
    ``logit_tensor_used`` is ``ffn_moe_probs`` (softmax output) for every model, and that GPT-OSS
    additionally has a pre-selection router bias.
    """
    used = config.get("logit_tensor_used")
    if not used:
        raise CaptureError(
            "models.yaml card has no logit_tensor_used; which tensor logits.bin came from must be "
            "recorded so the margin is interpretable (plan §1.6, T8.2)"
        )
    if used not in ("ffn_moe_probs", "ffn_moe_logits"):
        raise CaptureError(
            f"logit_tensor_used={used!r} is neither the raw logits nor the softmax probs; the "
            "vLLM hook path only knows how to reproduce those two. Extend GatingOp before using it."
        )
    return GatingOp(
        softmax=(used == "ffn_moe_probs"),
        has_router_bias=bool(config.get("has_router_bias", False)),
        logit_tensor_used=used,
    )


# --------------------------------------------------------------------------------------
# top-k recomputation + the faithfulness gate
# --------------------------------------------------------------------------------------


def recompute_topk(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Top-``k`` expert indices per row, as int32, HIGHEST score first.

    Ties are broken by lowest index (stable), matching what a descending argsort of the selection
    tensor does. This is compared against vLLM's own selection on every token; a mismatch halts, so
    the tie rule only has to agree with vLLM's on the ties that actually occur (fp resolution makes
    exact ties between distinct experts essentially never happen at F32 router precision).
    """
    if scores.ndim != 2:
        raise CaptureError(f"scores must be [n_rows, n_experts], got shape {scores.shape}")
    n_experts = scores.shape[1]
    if top_k > n_experts:
        raise CaptureError(f"top_k={top_k} exceeds n_experts={n_experts}")
    # argsort is ascending and stable; negate for descending-by-score, stable in index on ties.
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :top_k].astype(TOPK_DTYPE)


def _assert_selection_matches(
    recomputed: np.ndarray,
    vllm_topk: np.ndarray,
    *,
    layer: int,
) -> None:
    """Compare recomputed vs vLLM selection as SETS per token (order within top-k is free)."""
    if recomputed.shape != vllm_topk.shape:
        raise SelectionMismatch(
            f"layer {layer}: recomputed top-k shape {recomputed.shape} != vLLM {vllm_topk.shape}"
        )
    a = np.sort(recomputed, axis=1)
    b = np.sort(vllm_topk.astype(TOPK_DTYPE), axis=1)
    if not np.array_equal(a, b):
        bad = int(np.argmax(np.any(a != b, axis=1)))
        raise SelectionMismatch(
            f"layer {layer}, first differing token (doc-relative row {bad}): recomputed experts "
            f"{sorted(recomputed[bad].tolist())} != vLLM {sorted(vllm_topk[bad].tolist())}. "
            "topk.bin is the study's labels (I1); refusing to write a trace whose labels disagree "
            "with the engine that produced them."
        )


# --------------------------------------------------------------------------------------
# per-document staging — final on-disk layout, layer-major within token
# --------------------------------------------------------------------------------------


class DocumentTrace:
    """Accumulates one document's three streams in final layout, then hands out flat byte buffers.

    Mirrors ``trace_writer.hpp``'s per-document staging: both topk and logits are LAYER-MAJOR
    WITHIN TOKEN (all layers of token 0, then all layers of token 1, ...). vLLM hands us a whole
    prompt's rows for ONE layer at a time (the transpose of file order), exactly like the llama.cpp
    callback, so we stage in RAM and emit one contiguous buffer per stream at document end.

    ``capture_mask`` marks which doc-relative tokens get a router-input vector in hidden.bin; the
    hidden buffer is sized to exactly that count (sparse subsample, plan O2/T4.4).
    """

    def __init__(
        self,
        spec: TraceSpec,
        *,
        n_tokens: int,
        capture_mask: Sequence[bool],
        gating: GatingOp,
    ) -> None:
        if len(capture_mask) != n_tokens:
            raise CaptureError(
                f"capture_mask has {len(capture_mask)} entries for {n_tokens} tokens"
            )
        self.spec = spec
        self.n_tokens = int(n_tokens)
        self.gating = gating

        # doc-relative token -> row in the hidden buffer, or -1 if not subsampled.
        self._hidden_slot = np.full(n_tokens, -1, dtype=np.int64)
        captured = 0
        for i, take in enumerate(capture_mask):
            if take:
                self._hidden_slot[i] = captured
                captured += 1
        self.n_captured = captured

        L, E, K, H = spec.n_moe_layers, spec.n_experts, spec.top_k, spec.hidden_dim
        # Staged in native precision; cast to the on-disk dtype happens once, in to_buffers().
        self._topk = np.zeros((n_tokens, L, K), dtype=TOPK_DTYPE)
        self._logits = np.zeros((n_tokens, L, E), dtype=np.float64)
        self._hidden = np.zeros((captured, L, H), dtype=np.float64)
        self._layer_seen = np.zeros(L, dtype=bool)

    def put_layer(
        self,
        trace_layer: int,
        *,
        logits: np.ndarray,
        vllm_topk: np.ndarray,
        router_input: np.ndarray,
        bias: np.ndarray | None = None,
    ) -> None:
        """Ingest one MoE layer's rows for the whole document.

        ``logits`` is [n_tokens, n_experts] raw router output; ``vllm_topk`` is [n_tokens, top_k]
        vLLM's own selection; ``router_input`` is [n_tokens, hidden_dim] the router's input hidden.
        ``bias`` is the [n_experts] router bias (GPT-OSS) or None.
        """
        L = self.spec.n_moe_layers
        if not (0 <= trace_layer < L):
            raise CaptureError(f"trace_layer {trace_layer} out of range [0,{L})")
        if self._layer_seen[trace_layer]:
            raise CaptureError(f"trace_layer {trace_layer} ingested twice for one document")
        for name, arr, cols in (
            ("logits", logits, self.spec.n_experts),
            ("router_input", router_input, self.spec.hidden_dim),
        ):
            if arr.shape != (self.n_tokens, cols):
                raise CaptureError(
                    f"layer {trace_layer} {name}: expected shape "
                    f"{(self.n_tokens, cols)}, got {tuple(arr.shape)}"
                )
        if vllm_topk.shape != (self.n_tokens, self.spec.top_k):
            raise CaptureError(
                f"layer {trace_layer} vllm_topk: expected {(self.n_tokens, self.spec.top_k)}, "
                f"got {tuple(vllm_topk.shape)}"
            )

        # Faithfulness gate FIRST: recompute selection on the tensor vLLM selects over, compare as
        # sets against vLLM's own topk, halt on any disagreement (before anything is staged).
        scores = self.gating.selection_scores(np.asarray(logits), bias)
        recomputed = recompute_topk(scores, self.spec.top_k)
        _assert_selection_matches(recomputed, np.asarray(vllm_topk), layer=trace_layer)

        # topk.bin stores the authoritative labels. Use vLLM's own selection order — the equality
        # check above proved the SET is identical; storing the engine's order keeps topk.bin exactly
        # what the engine routed, not a re-sorted view of it.
        self._topk[:, trace_layer, :] = np.asarray(vllm_topk, dtype=TOPK_DTYPE)
        self._logits[:, trace_layer, :] = self.gating.written_logits(np.asarray(logits), bias)

        rows = self._hidden_slot >= 0
        if rows.any():
            slots = self._hidden_slot[rows]
            self._hidden[slots, trace_layer, :] = np.asarray(router_input)[rows]
        self._layer_seen[trace_layer] = True

    def _assert_complete(self) -> None:
        missing = np.where(~self._layer_seen)[0]
        if missing.size:
            raise CaptureError(
                f"document flushed with {missing.size} MoE layer(s) never ingested: "
                f"{missing.tolist()[:10]}{'...' if missing.size > 10 else ''}. Every layer's hook "
                "must fire once per document, or topk/logits.bin would have zero-filled layers."
            )

    def to_buffers(
        self,
        *,
        token_ids: Sequence[int],
        doc_id: int,
        global_token_base: int,
    ) -> dict[str, bytes]:
        """Freeze the document to the five stream byte-blobs, in on-disk dtype and layout.

        ``global_token_base`` is the corpus-relative index of this document's first token, so
        hidden_index.bin holds GLOBAL indices and shards concatenate without rewriting (T2.3).
        """
        self._assert_complete()
        if len(token_ids) != self.n_tokens:
            raise CaptureError(
                f"token_ids has {len(token_ids)} entries for {self.n_tokens} tokens"
            )

        tokens = np.zeros(self.n_tokens, dtype=[
            ("token_id", "<u4"), ("doc_id", "<u4"), ("pos_in_doc", "<u4"), ("flags", "<u4"),
        ])
        tokens["token_id"] = np.asarray(token_ids, dtype="<u4")
        tokens["doc_id"] = np.uint32(doc_id)
        tokens["pos_in_doc"] = np.arange(self.n_tokens, dtype="<u4")
        captured = self._hidden_slot >= 0
        tokens["flags"] = captured.astype("<u4")  # FLAG_HIDDEN_CAPTURED == 1

        # hidden_index.bin: GLOBAL indices of the captured tokens, ascending by construction.
        captured_positions = np.where(captured)[0]
        hidden_index = (global_token_base + captured_positions).astype(HIDDEN_INDEX_DTYPE)

        return {
            "tokens.bin": tokens.tobytes(),
            # cast to on-disk dtype ONCE here; float16 via numpy is IEEE RNE == f32_to_f16.
            "topk.bin": self._topk.astype(TOPK_DTYPE, copy=False).tobytes(),
            "logits.bin": self._logits.astype(LOGIT_DTYPE).tobytes(),
            "hidden.bin": self._hidden.astype(np.float16).tobytes(),
            "hidden_index.bin": hidden_index.tobytes(),
        }


# --------------------------------------------------------------------------------------
# router hooks — the vLLM-specific glue (TP=1)
# --------------------------------------------------------------------------------------


def _to_2d_numpy(t: Any) -> np.ndarray:
    """Detach a torch tensor to a [rows, cols] numpy array on CPU (float32, or int for indices)."""
    import torch

    if torch.is_floating_point(t):
        arr = t.detach().to("cpu", dtype=torch.float32).numpy()
    else:
        arr = t.detach().to("cpu").numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise CaptureError(f"router tensor has {arr.ndim} dims, expected 2 ([rows, cols])")
    return arr


class RouterCapture:
    """Captures the three router streams per MoE layer for the in-flight document, from vLLM.

    The caller resolves the actual modules against the LIVE vLLM module tree and passes them in —
    vLLM's internal names differ from HuggingFace's and must be confirmed on the box, never assumed
    (that confirmation is the P1 notebook's discovery cell). In vLLM the router is a
    ``ReplicatedLinear`` (``.gate``) whose OUTPUT is the raw router logits the block consumes, and
    the FusedMoE experts module (``.experts``) computes the selection inside ``select_experts``.

    Streams and where each comes from:

    * router INPUT hidden (hidden.bin) — ``forward_pre_hook`` on ``.gate``: its first positional
      arg is the normalized hidden state the router matmul consumes.
    * raw router logits (logits.bin, after :class:`GatingOp`) — ``forward_hook`` on ``.gate``:
      its output.
    * vLLM's OWN expert selection (topk.bin, and the faithfulness gate's independent side) —
      a wrapper around ``FusedMoE.select_experts``, capturing the returned ``topk_ids``. This is
      what makes the gate a real test rather than a tautology: topk comes from vLLM's kernel path,
      not from our recomputation of the same logits.

    ``select_experts`` is a **staticmethod** that vLLM invokes as ``FusedMoE.select_experts(...)``
    through the CLASS (from ``UnquantizedFusedMoEMethod.forward_cuda`` and the quantized methods'
    apply), never as ``self.select_experts(...)`` — so it must be wrapped on the class object, not
    on an instance (an instance attribute is simply bypassed and captures nothing). The wrapper
    records each call's ``topk_ids`` in CALL ORDER; for a single-document prefill the MoE layers run
    sequentially, so the k-th call is layer k. That alignment is not taken on faith: the per-layer
    :meth:`DocumentTrace.put_layer` gate recomputes top-k from layer L's own logits and compares it
    to the k-th captured selection, so any order skew surfaces as :class:`SelectionMismatch`.

    ``experts_modules`` is optional. Without it, only inputs/outputs are captured and the caller
    must supply the comparison topk some other way (e.g. the TP>1 routed-experts capturer).
    """

    def __init__(
        self,
        router_modules: Sequence[Any],
        experts_modules: Sequence[Any] | None = None,
    ):
        if not router_modules:
            raise CaptureError("no router modules given; nothing to hook")
        if experts_modules is not None and len(experts_modules) != len(router_modules):
            raise CaptureError(
                f"experts_modules has {len(experts_modules)} entries but there are "
                f"{len(router_modules)} routers; they must align by layer"
            )
        self._routers = list(router_modules)
        self._experts = list(experts_modules) if experts_modules is not None else None
        self._handles: list[Any] = []
        self._patched: list[tuple[Any, str, Any]] = []  # (owner, attr, original) for restore
        # trace_layer -> captured tensors for the in-flight document
        self.inputs: dict[int, np.ndarray] = {}
        self.outputs: dict[int, np.ndarray] = {}
        # vLLM's own selections, in CALL ORDER within the in-flight document (see class docstring).
        self._topk_calls: list[np.ndarray] = []

    @property
    def topk_ids(self) -> dict[int, np.ndarray]:
        """vLLM's captured selections keyed by call order (0-based), i.e. by MoE layer in prefill.

        A property, not a stored dict, so it always reflects the ordered calls the wrapper recorded
        for the current document. ``topk_ids[L]`` is [n_tokens, top_k] for layer ``L``.
        """
        return {i: arr for i, arr in enumerate(self._topk_calls)}

    def register(self) -> None:
        for trace_layer, router in enumerate(self._routers):
            def pre_hook(_module, args, _layer=trace_layer):
                self.inputs[_layer] = _to_2d_numpy(args[0])

            def hook(_module, _args, output, _layer=trace_layer):
                logits = output[0] if isinstance(output, (tuple, list)) else output
                self.outputs[_layer] = _to_2d_numpy(logits)

            self._handles.append(router.register_forward_pre_hook(pre_hook))
            self._handles.append(router.register_forward_hook(hook))

        if self._experts is not None:
            self._wrap_select_experts(self._experts)

    @staticmethod
    def _defining_class(cls: type, attr: str) -> type | None:
        """The class in ``cls``'s MRO that actually defines ``attr`` (where the name binds)."""
        for c in cls.__mro__:
            if attr in c.__dict__:
                return c
        return None

    def _wrap_select_experts(self, experts_modules: Sequence[Any]) -> None:
        """Wrap ``FusedMoE.select_experts`` on the CLASS to stash each call's topk_ids in order.

        vLLM's ``select_experts`` is a staticmethod called as ``FusedMoE.select_experts(...)`` via
        the class, so wrapping an instance attribute would never fire (that was the original bug —
        it silently captured nothing). We patch the defining class once, appending each returned
        ``topk_ids`` to :attr:`_topk_calls`. The wrapper returns the original result untouched, so
        the model's numerics are unchanged.
        """
        seen: set[type] = set()
        for experts in experts_modules:
            owner = self._defining_class(type(experts), "select_experts")
            if owner is None:
                raise CaptureError(
                    f"experts module {type(experts).__name__} has no select_experts anywhere in "
                    "its MRO; cannot capture vLLM's own selection. Pass the FusedMoE modules, or "
                    "omit experts_modules and supply topk another way."
                )
            if owner in seen:
                continue
            seen.add(owner)

            raw = owner.__dict__["select_experts"]  # the staticmethod descriptor, to restore later
            # Underlying function whether it is a staticmethod (expected) or a plain function.
            original_fn = raw.__func__ if isinstance(raw, staticmethod) else raw
            calls = self._topk_calls

            def wrapped(*args, _orig=original_fn, _calls=calls, **kwargs):
                result = _orig(*args, **kwargs)
                topk_ids = result[1] if isinstance(result, (tuple, list)) else result
                _calls.append(_to_2d_numpy(topk_ids).astype(TOPK_DTYPE))
                return result

            setattr(owner, "select_experts", staticmethod(wrapped))
            self._patched.append((owner, "select_experts", raw))

    def reset(self) -> None:
        """Clear the per-document buffers. Call before each document's prefill."""
        self.inputs.clear()
        self.outputs.clear()
        self._topk_calls.clear()

    def drain(self) -> dict[str, dict[int, np.ndarray]]:
        """Return the current document's captured streams as plain numpy, keyed by layer.

        Everything here is already numpy (the hooks convert on capture), so the result is
        picklable — which is what lets a worker process hand its capture back to the driver across
        the ``collective_rpc`` boundary on the TP>1 path. Returns copies of the dicts, not the live
        buffers, so a subsequent :meth:`reset` cannot mutate what the caller received.
        """
        return {
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "topk_ids": self.topk_ids,
        }

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for module, attr, original in self._patched:
            setattr(module, attr, original)
        self._patched.clear()

    def __enter__(self) -> "RouterCapture":
        self.register()
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()


# --------------------------------------------------------------------------------------
# module discovery — shared by the TP=1 driver path and the TP=2 worker path
# --------------------------------------------------------------------------------------


def discover_router_and_experts(
    model: Any,
    *,
    router_suffix: str = ".mlp.gate",
    experts_suffix: str = ".mlp.experts",
    n_expected: int | None = None,
) -> tuple[list[Any], list[Any], list[str], list[str]]:
    """Find the per-layer router and experts modules on a live vLLM model, ordered by layer.

    vLLM's internal module names differ from HuggingFace's and must be confirmed on the box, never
    assumed (the T1.4 analogue) — so this MATCHES by suffix against the live module tree and returns
    both the modules and their names, for the caller to print and sanity-check. The suffixes come
    from the model card (``.mlp.gate``/``.mlp.experts`` for OLMoE/Qwen; Gemma-4 routes through
    ``.router``). Ordering is by the integer in ``layers.<i>.`` so index 0 is the lowest MoE layer.
    """
    import re

    named = dict(model.named_modules())

    def layer_of(name: str) -> int:
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    gate_names = sorted(
        (n for n in named if n.endswith(router_suffix) and layer_of(n) >= 0), key=layer_of
    )
    expert_names = sorted(
        (n for n in named if n.endswith(experts_suffix) and layer_of(n) >= 0), key=layer_of
    )
    if not gate_names:
        raise CaptureError(
            f"no modules end with router_suffix {router_suffix!r}; the vLLM name differs on this "
            "model — print model.named_modules() and pass the right suffix"
        )
    if len(gate_names) != len(expert_names):
        raise CaptureError(
            f"found {len(gate_names)} routers ({router_suffix!r}) but {len(expert_names)} experts "
            f"({experts_suffix!r}); they must pair one-to-one per MoE layer"
        )
    if n_expected is not None and len(gate_names) != n_expected:
        raise CaptureError(
            f"expected {n_expected} MoE layers, discovered {len(gate_names)} "
            f"({router_suffix!r}); check the model card's n_moe_layers or the suffix"
        )
    return (
        [named[n] for n in gate_names],
        [named[n] for n in expert_names],
        gate_names,
        expert_names,
    )


# --------------------------------------------------------------------------------------
# worker-side entrypoints — the TP>1 path (Python hooks can't cross the worker boundary)
# --------------------------------------------------------------------------------------
#
# At tensor_parallel_size>1 vLLM runs each rank in its own process, so a RouterCapture built in the
# driver reaches nothing. These four functions are passed to Executor.collective_rpc, which
# cloudpickles them to every worker and calls them with the worker as the first argument. They
# install / reset / drain / remove a RouterCapture that lives ON the worker, stashed as an
# attribute. The router is a ReplicatedLinear, so router_logits (and thus select_experts' topk) are
# the FULL global selection on every rank — rank 0's drain is complete on its own. (Verify on-box
# that the captured expert IDs are GLOBAL, not TP-local: vllm-ascend #15451 shows that failure mode
# for the built-in capturer; the drain-side selection gate would also catch it as a mismatch.)
#
# For collective_rpc to resolve these by reference in a spawned worker, the repo must be importable
# there — set PYTHONPATH to include it BEFORE constructing the LLM (the probe notebook does this).


def worker_install_capture(
    worker: Any,
    *,
    router_suffix: str = ".mlp.gate",
    experts_suffix: str = ".mlp.experts",
    n_moe_layers: int | None = None,
) -> dict[str, Any]:
    """Build and register a RouterCapture on this worker's model. Returns a small ack per rank."""
    model = worker.model_runner.model
    routers, experts, gate_names, _ = discover_router_and_experts(
        model,
        router_suffix=router_suffix,
        experts_suffix=experts_suffix,
        n_expected=n_moe_layers,
    )
    cap = RouterCapture(routers, experts_modules=experts)
    cap.register()
    worker._moe_capture = cap  # stash so later RPCs on this worker can reach it
    return {"rank": getattr(worker, "rank", None), "n_gates": len(gate_names),
            "first_gate": gate_names[0]}


def worker_reset_capture(worker: Any) -> None:
    """Clear the per-document buffers on this worker. Call before each document's prefill."""
    cap = getattr(worker, "_moe_capture", None)
    if cap is not None:
        cap.reset()


def worker_drain_capture(worker: Any) -> dict[str, Any] | None:
    """Return this worker's captured streams (numpy, picklable) for the current document."""
    cap = getattr(worker, "_moe_capture", None)
    if cap is None:
        return None
    return {"rank": getattr(worker, "rank", None), **cap.drain()}


def worker_remove_capture(worker: Any) -> None:
    """Restore the hooks and the patched staticmethod on this worker."""
    cap = getattr(worker, "_moe_capture", None)
    if cap is not None:
        cap.remove()
        worker._moe_capture = None
