"""vLLM collection loop — the S.3 shard loop for the vLLM engine (plan T5.2, the vLLM port).

Why this exists separately from ``src/runtime/runner.py``
--------------------------------------------------------
``runner.py`` is the collection loop for llama.cpp: it spawns ``moe_trace`` **once per shard** as a
subprocess (``SubprocessInvoker``). vLLM is the opposite shape — the model is expensive to load and
stays resident while we iterate documents in-process — so a subprocess-per-shard model would throw
away the load. This module is the vLLM-shaped loop. It does **not** re-implement the S.3 contract
machinery (``runner.py``'s docstring warns against that); it **calls** the same engine-agnostic
helpers:

* ``plan_shards`` (runner) — splits the corpus into shard JSONLs + ``ShardPlan``s with the
  ``hidden_stride`` and ``ref_token_offset`` every shard needs.
* ``ShardState`` (state) — the resumption ledger; a shard is recorded only after its upload
  round-trip verifies.
* ``upload_shard`` (upload) — upload + round-trip verify + mark-complete + delete-local, in that
  fixed order (S.3 step d).
* ``SessionBudget`` (session), ``run_preflight`` (preflight), ``write_manifest`` / ``TraceSpec`` /
  ``check_file_sizes`` (format), and the proven capture primitives in ``vllm_trace`` (``RouterCapture``,
  ``discover_router_and_experts``, the ``worker_*`` collective_rpc entrypoints, ``DocumentTrace``).

The traces this writes are byte-identical in layout to the llama.cpp ones (same ``format.py``); they
differ only in ``run_config_sha256`` (a different engine is a different experiment, S.3) and carry the
engine-neutral manifest keys ``model_sha256`` / ``engine_build``.

Scope
-----
Proven end-to-end (P3 notebook) on the two already-validated models: ``olmoe-0125`` (fp16, TP=1) and
``qwen3-30b-a3b`` (GPTQ-Int4, TP=2). Models with a router bias (GPT-OSS) are refused here until bias
capture is wired — recomputing top-k without the bias would name experts the model never routed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..corpus.build import load_corpus
from ..traces.format import (
    HIDDEN_INDEX_DTYPE,
    STREAM_FILES,
    TraceSpec,
    check_file_sizes,
    write_manifest,
)
from ..runtime.config import RunConfig
from ..runtime.runner import (
    ShardPlan,
    _check_n_captured,
    append_log_row,
    assert_not_kaggle_working,
    load_model_meta,
    plan_shards,
)
from ..runtime.session import SessionBudget
from ..runtime.state import ShardState
from ..runtime.upload import StorageBackend, UploadError, sha256_file, upload_shard
from .vllm_trace import (
    CaptureError,
    DocumentTrace,
    GatingOp,
    gating_from_config,
)

__all__ = [
    "CollectError",
    "VLLMCaptureEngine",
    "collect_shard",
    "validate_vllm_stats",
    "build_vllm_manifest",
    "run_vllm_collection",
    "spec_and_gating_for",
    "main",
]

INDEX_SCHEME = "doc_id*n_ctx+pos_in_doc"  # same scheme moe_trace uses; hidden_index = doc_id*n_ctx+pos


class CollectError(RuntimeError):
    """The vLLM collection run cannot proceed, or a shard failed validation."""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------
# model card -> TraceSpec + GatingOp
# --------------------------------------------------------------------------------------


def resolve_model_sha256(model_id: str, revision: str | None = None) -> str:
    """The weights identity for the manifest — the HF repo's commit sha (immutable per revision).

    vLLM loads safetensors from a HF repo at a revision; that revision's commit sha uniquely and
    immutably identifies the exact weights, so it is the vLLM analogue of ``gguf.sha256`` and the
    right value for the engine-neutral ``model_sha256`` manifest key. Hashing multi-GB safetensors
    would be equivalent but far slower. Requires ``huggingface_hub`` (present in the collection env).
    """
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 - only the collection env has it
    except ImportError as exc:
        raise CollectError(
            "huggingface_hub is needed to resolve model_sha256 (the HF revision). Install it, or "
            "fill vllm.model_sha256 in the model card."
        ) from exc
    info = HfApi().model_info(model_id, revision=revision)
    sha = getattr(info, "sha", None)
    if not sha:
        raise CollectError(f"could not resolve a commit sha for {model_id!r} (revision={revision!r})")
    return str(sha)


def spec_and_gating_for(meta: Mapping[str, Any]) -> tuple[TraceSpec, GatingOp]:
    """Build the on-disk :class:`TraceSpec` and the :class:`GatingOp` from a models.yaml card.

    ``logit_tensor_used`` and ``has_router_bias`` come from the card; the gate reproduces exactly
    what the study recorded on llama.cpp so ``logits.bin`` stays comparable (softmax for the panel;
    GPT-OSS additionally biases before selection).
    """
    spec = TraceSpec(
        n_moe_layers=int(meta["n_moe_layers"]),
        n_experts=int(meta["n_experts"]),
        top_k=int(meta["top_k"]),
        hidden_dim=int(meta["hidden_dim"]),
    )
    gating = gating_from_config(meta)
    return spec, gating


# --------------------------------------------------------------------------------------
# the engine — loads the model ONCE, captures one document at a time (TP=1 or TP=2)
# --------------------------------------------------------------------------------------


class VLLMCaptureEngine:
    """Loads a vLLM ``LLM`` once and captures the three router streams per document.

    TP=1 uses driver-side Python hooks (:class:`RouterCapture`); TP>1 injects the same capture into
    each worker via ``collective_rpc`` and drains rank 0 (the router is a ReplicatedLinear, so rank
    0's selection is the full, global one). Both paths are proven in the P1/P2 notebooks.

    vLLM/torch are imported lazily inside :meth:`load`, so this module imports on a CPU-only box and
    the collection loop is testable with a fake engine.
    """

    def __init__(
        self,
        *,
        model_id: str,
        tensor_parallel_size: int,
        spec: TraceSpec,
        gating: GatingOp,
        router_suffix: str = ".mlp.gate",
        experts_suffix: str = ".mlp.experts",
        max_model_len: int,
        gpu_memory_utilization: float,
        dtype: str = "float16",
        seed: int = 0,
    ) -> None:
        if gating.has_router_bias:
            # GPT-OSS: selection runs on logits + gate bias. RouterCapture does not yet capture the
            # bias tensor, and recomputing top-k without it would name experts the model never
            # routed (§1.6). Refuse loudly rather than write a plausible, wrong topk.bin.
            raise CollectError(
                f"{model_id}: has_router_bias is set, but vLLM bias capture is not implemented yet. "
                "Wire the router-bias stream before collecting a biased-router model."
            )
        self.model_id = model_id
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.spec = spec
        self.gating = gating
        self.router_suffix = router_suffix
        self.experts_suffix = experts_suffix
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.dtype = dtype
        self.seed = int(seed)

        self.llm: Any = None
        self.tokenizer: Any = None
        self._cap: Any = None          # TP=1 RouterCapture
        self._executor: Any = None     # TP>1 executor for collective_rpc
        self._sampling: Any = None

    # -- lifecycle ----------------------------------------------------------------------

    def load(self) -> "VLLMCaptureEngine":
        """Construct the LLM and install capture. Sets the Turing-required env first."""
        os.environ["VLLM_USE_V1"] = "0"
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from vllm import LLM, SamplingParams  # noqa: PLC0415 - GPU-only, lazy on purpose

        self.llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            enforce_eager=True,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            seed=self.seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._sampling = SamplingParams(max_tokens=1, temperature=0.0)
        self._install()
        return self

    def _install(self) -> None:
        from . import vllm_trace as vt  # noqa: PLC0415

        if self.tensor_parallel_size == 1:
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            routers, experts, gate_names, _ = vt.discover_router_and_experts(
                model,
                router_suffix=self.router_suffix,
                experts_suffix=self.experts_suffix,
                n_expected=self.spec.n_moe_layers,
            )
            self._cap = vt.RouterCapture(routers, experts_modules=experts)
            self._cap.register()
        else:
            self._executor = self.llm.llm_engine.model_executor
            acks = self._executor.collective_rpc(
                vt.worker_install_capture,
                kwargs=dict(
                    router_suffix=self.router_suffix,
                    experts_suffix=self.experts_suffix,
                    n_moe_layers=self.spec.n_moe_layers,
                ),
            )
            bad = [a for a in acks if int(a.get("n_gates", -1)) != self.spec.n_moe_layers]
            if bad:
                raise CollectError(
                    f"{self.model_id}: worker capture install found the wrong module count on some "
                    f"rank(s): {acks}. Expected n_gates={self.spec.n_moe_layers} on every rank."
                )

    def remove(self) -> None:
        from . import vllm_trace as vt  # noqa: PLC0415

        if self._cap is not None:
            self._cap.remove()
            self._cap = None
        if self._executor is not None:
            self._executor.collective_rpc(vt.worker_remove_capture)

    def __enter__(self) -> "VLLMCaptureEngine":
        return self.load()

    def __exit__(self, *exc: object) -> None:
        self.remove()

    # -- capture ------------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def capture_document(self, token_ids: Sequence[int]) -> dict[int, tuple[Any, Any, Any]]:
        """Prefill one document and return ``{trace_layer: (logits, topk, router_input)}``.

        Each array holds the last ``len(token_ids)`` rows for that layer (prefill packs the prompt
        contiguously). TP>1 drains rank 0, whose capture is the full global selection.
        """
        n = len(token_ids)
        if n == 0:
            raise CaptureError("document tokenized to zero tokens; the corpus has an unusable doc")

        from . import vllm_trace as vt  # noqa: PLC0415

        if self.tensor_parallel_size == 1:
            self._cap.reset()
            self.llm.generate([{"prompt_token_ids": list(token_ids)}], self._sampling)
            outputs, inputs, topk = self._cap.outputs, self._cap.inputs, self._cap.topk_ids
        else:
            self._executor.collective_rpc(vt.worker_reset_capture)
            self.llm.generate([{"prompt_token_ids": list(token_ids)}], self._sampling)
            drained = self._executor.collective_rpc(vt.worker_drain_capture)
            r0 = drained[0]
            if r0 is None:
                raise CaptureError("rank 0 returned no capture; worker_install_capture did not run")
            outputs, inputs, topk = r0["outputs"], r0["inputs"], r0["topk_ids"]

        for stream, got in (("gate output", outputs), ("gate input", inputs),
                            ("select_experts topk", topk)):
            if sorted(got) != list(range(self.spec.n_moe_layers)):
                raise CaptureError(
                    f"{stream} fired for layers {sorted(got)}, expected 0..{self.spec.n_moe_layers-1}"
                )
        return {
            L: (outputs[L][-n:], topk[L][-n:], inputs[L][-n:])
            for L in range(self.spec.n_moe_layers)
        }


# --------------------------------------------------------------------------------------
# one shard: iterate documents, stage, write the five .bin files, return stats
# --------------------------------------------------------------------------------------


def _capture_mask(doc_id: int, n_tokens: int, n_ctx: int, hidden_stride: int) -> list[bool]:
    """Which tokens get a hidden.bin row — replicates moe_trace's rule exactly.

    ``capture[i] = hidden_stride > 0 and (doc_id * n_ctx + i) % hidden_stride == 0``. The index is
    the same ``doc_id * n_ctx + pos`` block every other shard reserves, so the subsample is a pure
    function of the corpus and identical across sessions and re-shardings (T2.3/T4.4).
    """
    if hidden_stride <= 0:
        return [False] * n_tokens
    base = doc_id * n_ctx
    return [((base + i) % hidden_stride) == 0 for i in range(n_tokens)]


def _assert_hidden_index_ascending(path: Path, shard_id: int) -> None:
    """Refuse a shard whose hidden_index.bin is not strictly ascending (the T5.3 lockstep rule).

    Reads the uint32 stream and raises :class:`CollectError` on the first non-increasing step, so a
    misordered or mis-indexed shard fails BEFORE upload. Uses numpy (a hard dependency of the trace
    format) and holds only the one stream, which is tiny (4 bytes per captured token).
    """
    import numpy as np

    idx = np.fromfile(path, dtype=HIDDEN_INDEX_DTYPE)
    if idx.size < 2:
        return
    bad = np.flatnonzero(np.diff(idx.astype(np.int64)) <= 0)
    if bad.size:
        first = int(bad[0])
        raise CollectError(
            f"shard {shard_id}: hidden_index.bin is not strictly ascending "
            f"({bad.size} violation(s), first at row {first + 1}: "
            f"{int(idx[first])} -> {int(idx[first + 1])}); documents must be emitted in doc_id "
            "order so the global index doc_id*n_ctx+pos is monotone. NOT uploading this shard."
        )


def collect_shard(
    engine: Any,
    plan: ShardPlan,
    *,
    spec: TraceSpec,
    gating: GatingOp,
    n_ctx: int,
    out_dir: Path | str,
) -> dict[str, Any]:
    """Capture every document in one shard, write the five stream files, return the stats dict.

    Documents are processed in ``doc_id`` order and their per-document byte blobs concatenated —
    ``DocumentTrace`` emits each stream layer-major-within-token, so concatenation in doc order is
    the shard file. The faithfulness gate lives inside :meth:`DocumentTrace.put_layer` (recomputed
    top-k vs vLLM's selection); a mismatch raises before anything is written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_corpus(plan.corpus_path)
    hidden_stride = int(plan.hidden_stride)

    blobs: dict[str, list[bytes]] = {name: [] for name in STREAM_FILES.values()}
    n_tokens_total = 0
    n_captured_total = 0
    n_docs_truncated = 0
    n_tokens_dropped = 0
    first_truncated_doc: int | None = None

    # Documents MUST be emitted in ascending doc_id order: the global index is
    # ``doc_id * n_ctx + pos`` and the reader requires hidden_index strictly ascending within a
    # shard. plan_shards preserves corpus (interleaved) order, so sort here — the canonical corpus
    # scatters doc_ids across shards, and iterating file order writes a non-monotone index stream.
    for doc in sorted(docs, key=lambda d: int(d.doc_id)):
        ids = engine.tokenize(doc.text)
        n = len(ids)
        if n > n_ctx:
            # Invariant I15: the corpus was capped under a reference tokenizer, so a doc at the cap
            # for a 50k-vocab model can exceed it for a 262k-vocab one. Do NOT silently truncate —
            # count it and let validation refuse the shard (this model was not shown the same text).
            n_docs_truncated += 1
            n_tokens_dropped += n - n_ctx
            if first_truncated_doc is None:
                first_truncated_doc = int(doc.doc_id)
            continue

        captured = engine.capture_document(ids)
        mask = _capture_mask(int(doc.doc_id), n, n_ctx, hidden_stride)
        trace = DocumentTrace(spec, n_tokens=n, capture_mask=mask, gating=gating)
        for L in range(spec.n_moe_layers):
            logits, topk, router_input = captured[L]
            trace.put_layer(L, logits=logits, vllm_topk=topk, router_input=router_input)
        bufs = trace.to_buffers(
            token_ids=ids, doc_id=int(doc.doc_id), global_token_base=int(doc.doc_id) * n_ctx
        )
        for name, blob in bufs.items():
            blobs[name].append(blob)
        n_tokens_total += n
        n_captured_total += trace.n_captured

    for name, parts in blobs.items():
        (out_dir / name).write_bytes(b"".join(parts))

    # Safety net: hidden_index MUST be strictly ascending within the shard, or the reader resolves
    # the wrong rows and T5.3 rejects the whole set. This catches ANY indexing regression (bad doc
    # order, a wrong global base, a stride bug) HERE — before upload — instead of after a paid
    # session's traces are already on HF. Cheap: one pass over the uint32 index stream.
    _assert_hidden_index_ascending(out_dir / STREAM_FILES["hidden_index"], plan.shard_id)

    # The size arithmetic is the last local gate before upload; a short stream never leaves here.
    check_file_sizes(out_dir, spec, n_tokens_total, n_captured_total)

    return {
        "engine": "vllm",
        "vllm_version": getattr(engine, "vllm_version", None),
        "tensor_parallel_size": int(getattr(engine, "tensor_parallel_size", 1)),
        "shard_id": int(plan.shard_id),
        "n_docs": len(docs) - n_docs_truncated,
        "n_docs_in_shard": len(docs),
        "n_tokens": n_tokens_total,
        "n_captured": n_captured_total,
        "n_moe_layers": spec.n_moe_layers,
        "n_experts": spec.n_experts,
        "top_k": spec.top_k,
        "hidden_dim": spec.hidden_dim,
        "hidden_stride": hidden_stride,
        "index_scheme": INDEX_SCHEME,
        "index_doc_span": int(n_ctx),
        "n_docs_truncated": n_docs_truncated,
        "n_tokens_dropped": n_tokens_dropped,
        "first_truncated_doc": first_truncated_doc,
        "selection_gate": "passed",  # DocumentTrace.put_layer would have raised otherwise
        "exit_code": 0,
    }


# --------------------------------------------------------------------------------------
# stats validation — the vLLM analogue of runner.validate_stats
# --------------------------------------------------------------------------------------


def validate_vllm_stats(
    stats: Mapping[str, Any], plan: ShardPlan, *, spec: TraceSpec, n_ctx: int
) -> None:
    """Raise :class:`CollectError` unless the stats describe THIS plan, collected cleanly.

    The vLLM path has none of llama.cpp's ``topk_layout``/``nodes_captured`` failure modes (topk
    faithfulness is enforced per layer inside :meth:`DocumentTrace.put_layer`), so this checks the
    engine-independent invariants: the stats belong to this shard, every document was processed and
    none truncated (I15), the shapes match the spec, and ``n_captured`` is inside the closed-form
    bound. The ``n_captured`` bound is reused from ``runner._check_n_captured`` — one source of
    truth for the subsample arithmetic.
    """
    problems: list[str] = []

    if int(stats.get("shard_id", -1)) != plan.shard_id:
        problems.append(f"stats are for shard {stats.get('shard_id')}, this is shard {plan.shard_id}")
    if str(stats.get("index_scheme", "")) != INDEX_SCHEME:
        problems.append(f"index_scheme is {stats.get('index_scheme')!r}, expected {INDEX_SCHEME!r}")
    if int(stats.get("index_doc_span", -1)) != int(n_ctx):
        problems.append(f"index_doc_span is {stats.get('index_doc_span')}, but n_ctx is {n_ctx}")

    if int(stats.get("n_docs", -1)) != plan.n_docs:
        problems.append(f"processed {stats.get('n_docs')} documents, the shard has {plan.n_docs}")
    if int(stats.get("n_docs_in_shard", -1)) != plan.n_docs:
        problems.append(
            f"the shard file holds {stats.get('n_docs_in_shard')} documents, the plan wrote "
            f"{plan.n_docs}"
        )

    n_trunc = int(stats.get("n_docs_truncated", 0))
    if n_trunc:
        problems.append(
            f"{n_trunc} document(s) exceeded n_ctx and were dropped "
            f"({int(stats.get('n_tokens_dropped', 0))} tokens, first doc_id "
            f"{stats.get('first_truncated_doc')}) — invariant I15: this model was not shown the "
            "same text as the rest of the panel. Fix the corpus cap, do not keep the shard"
        )

    for key, want in (
        ("n_moe_layers", spec.n_moe_layers),
        ("n_experts", spec.n_experts),
        ("top_k", spec.top_k),
        ("hidden_dim", spec.hidden_dim),
    ):
        if int(stats.get(key, -1)) != want:
            problems.append(f"{key}={stats.get(key)} but the spec says {want}")

    n_tokens = int(stats.get("n_tokens", 0))
    if n_tokens < plan.n_docs:
        problems.append(f"n_tokens={n_tokens} for {plan.n_docs} documents; each yields >= 1 token")
    problems += _check_n_captured(stats, plan, n_tokens=n_tokens)

    if problems:
        raise CollectError(
            f"shard {plan.shard_id} failed capture validation; NOT marking it complete:\n  - "
            + "\n  - ".join(problems)
        )


# --------------------------------------------------------------------------------------
# manifest — engine-neutral keys (model_sha256 / engine_build)
# --------------------------------------------------------------------------------------


def build_vllm_manifest(
    shard_dir: Path,
    plan: ShardPlan,
    stats: Mapping[str, Any],
    *,
    config: RunConfig,
    spec: TraceSpec,
    model: str,
    corpus: str,
    model_meta: Mapping[str, Any],
    n_ctx: int,
) -> dict[str, Any]:
    """Write ``manifest.json`` beside the streams and return it.

    Uses the engine-neutral required keys: ``model_sha256`` (the vLLM weights hash from the model
    card) and ``engine_build`` (``vllm@<version>`` from the config). The llama.cpp aliases
    (``gguf_sha256`` / ``llama_cpp_commit``) are deliberately absent — a vLLM shard has neither.
    """
    vllm = dict(model_meta.get("vllm") or {})
    tp = int(vllm.get("tensor_parallel", stats.get("tensor_parallel_size", 1)))
    offset = int(model_meta.get("moe_layer_offset", 0))

    manifest: dict[str, Any] = {
        "model": model,
        "corpus": corpus,
        "checkpoint_status": model_meta.get("checkpoint_status"),
        "model_sha256": vllm.get("model_sha256"),
        "engine_build": config.engine_build(),
        "run_config_sha256": config.sha256,
        "quant": vllm.get("quant"),
        "router_dtype": model_meta.get("router_dtype"),
        "logit_tensor_used": model_meta.get("logit_tensor_used"),
        "shard_id": int(plan.shard_id),
        # Half-open [min, max+1). plan.doc_range assumes doc_ids arrive sorted; the canonical corpus
        # interleaves them across shards, so derive the span from the actual id set instead.
        "shard_doc_range": [int(min(plan.doc_ids)), int(max(plan.doc_ids)) + 1],
        "n_docs": int(stats["n_docs"]),
        "n_tokens": int(stats["n_tokens"]),
        "n_captured": int(stats.get("n_captured", 0)),
        "n_moe_layers": spec.n_moe_layers,
        "n_experts": spec.n_experts,
        "top_k": spec.top_k,
        "hidden_dim": spec.hidden_dim,
        # trace layer i -> model layer i + moe_layer_offset (DeepSeek's dense layer 0, T3.5).
        "layer_index_map": [offset + i for i in range(spec.n_moe_layers)],
        "index_scheme": INDEX_SCHEME,
        "index_doc_span": int(n_ctx),
        "ref_token_offset": plan.ref_token_offset,
        "hidden_stride": int(plan.hidden_stride),
        "hidden_subsample_n": config.capture.get("hidden_subsample_n"),
        "capture_flags": {"pre_topk": True, "pre_norm": True, "topk_captured": True},
        "device_plan": {
            "engine": "vllm",
            "tensor_parallel": tp,
            "n_gpu": tp,
            "gpu_arch": config.gpu_arch,
        },
        "collected_utc": _utc_now(),
        "kaggle_session_id": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local"),
        "capture_stats": dict(stats),
        "file_sha256": {
            name: sha256_file(shard_dir / name)
            for name in STREAM_FILES.values()
            if (shard_dir / name).is_file()
        },
    }
    write_manifest(shard_dir, manifest)
    return manifest


# --------------------------------------------------------------------------------------
# the collection loop
# --------------------------------------------------------------------------------------


@dataclass
class VLLMShardResult:
    shard_id: int
    status: str  # "complete" | "skipped" | "failed"
    n_docs: int = 0
    n_tokens: int = 0
    n_captured: int = 0
    wall_s: float = 0.0
    upload_verified: bool = False
    error: str = ""

    @property
    def exit_code(self) -> int | None:
        return 0 if self.status in ("complete", "skipped") else 1

    @property
    def tokens_per_s(self) -> float:
        return self.n_tokens / self.wall_s if self.wall_s > 0 else 0.0


@dataclass
class VLLMCollectionResult:
    model: str
    corpus: str
    run_config_sha256: str
    results: list[VLLMShardResult]
    planned: list[int]
    stopped_early: bool = False

    @property
    def completed(self) -> list[int]:
        return [r.shard_id for r in self.results if r.status == "complete"]

    @property
    def failed(self) -> list[int]:
        return [r.shard_id for r in self.results if r.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        state = "stopped on session budget" if self.stopped_early else "finished the shard list"
        n_skip = sum(1 for r in self.results if r.status == "skipped")
        n_tok = sum(r.n_tokens for r in self.results if r.status == "complete")
        return (
            f"{self.model}/{self.corpus}: {len(self.completed)} collected, {n_skip} already "
            f"complete, {len(self.failed)} failed, {n_tok} tokens — {state}"
        )


def run_vllm_collection(
    plans: Sequence[ShardPlan],
    *,
    engine: Any,
    config: RunConfig,
    backend: StorageBackend,
    ledger: ShardState,
    spec: TraceSpec,
    gating: GatingOp,
    model: str,
    corpus: str,
    model_meta: Mapping[str, Any],
    scratch_root: Path | str,
    remote_root: str,
    budget: SessionBudget | None = None,
    log_path: Path | str | None = None,
    verbose: bool = True,
) -> VLLMCollectionResult:
    """The vLLM shard loop — mirrors ``runner.run_collection``'s S.3 ordering.

    Per shard: skip if already in the ledger, else capture → validate → manifest →
    upload+round-trip-verify → record → delete-local → log. The session budget is consulted only
    BETWEEN shards; the first failure stops the run (every failure this loop sees is a property of
    the run, not one shard's luck).
    """
    config.assert_collection_ready()
    if config.sha256 != ledger.run_config_sha256:
        raise CollectError(
            f"run config {config.short} does not match the ledger's "
            f"{ledger.run_config_sha256[:12]} (invariant I2)"
        )
    scratch_root = assert_not_kaggle_working(scratch_root, what="vLLM shard scratch")
    budget = budget or SessionBudget.from_config(config)
    n_ctx = int(config.inference["max_model_len"])

    result = VLLMCollectionResult(
        model=model, corpus=corpus, run_config_sha256=config.sha256,
        results=[], planned=[p.shard_id for p in plans],
    )

    for index, plan in enumerate(plans):
        if index and budget.should_stop():
            result.stopped_early = True
            if verbose:
                print(f"# session budget reached ({budget}); stopping between shards")
            break

        if ledger.is_complete(plan.shard_id):
            rec = ledger.record(plan.shard_id)
            result.results.append(VLLMShardResult(
                shard_id=plan.shard_id, status="skipped", n_docs=plan.n_docs,
                n_tokens=rec.n_tokens if rec else 0, n_captured=rec.n_captured if rec else 0,
                upload_verified=True,
            ))
            if verbose:
                print(f"# shard {plan.shard_id}: skipped (already complete)")
            continue

        started = time.monotonic()
        out_dir = assert_not_kaggle_working(
            Path(scratch_root) / f"shard_{plan.shard_id:05d}", what=f"shard {plan.shard_id}"
        )
        try:
            stats = collect_shard(engine, plan, spec=spec, gating=gating, n_ctx=n_ctx, out_dir=out_dir)
            validate_vllm_stats(stats, plan, spec=spec, n_ctx=n_ctx)
            build_vllm_manifest(
                out_dir, plan, stats, config=config, spec=spec, model=model, corpus=corpus,
                model_meta=model_meta, n_ctx=n_ctx,
            )
            upload = upload_shard(
                out_dir, backend,
                remote_prefix=f"{str(remote_root).strip('/')}/shard_{plan.shard_id:05d}",
                verify=True, delete_local_on_success=True, state=ledger,
            )
            shard_result = VLLMShardResult(
                shard_id=plan.shard_id, status="complete", n_docs=int(stats["n_docs"]),
                n_tokens=int(stats["n_tokens"]), n_captured=int(stats.get("n_captured", 0)),
                wall_s=time.monotonic() - started, upload_verified=upload.verified,
            )
        except (CollectError, CaptureError, UploadError, OSError, ValueError, KeyError) as exc:
            shard_result = VLLMShardResult(
                shard_id=plan.shard_id, status="failed", n_docs=plan.n_docs,
                wall_s=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
            )

        result.results.append(shard_result)
        if log_path is not None and shard_result.status != "skipped":
            append_log_row(log_path, shard_result, model=model, config=config)
        if verbose:
            print(f"# shard {plan.shard_id}: {shard_result.status} {shard_result.error}".rstrip())
        if shard_result.status == "failed":
            break

    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_shard_filter(text: str | None) -> set[int] | None:
    if not text:
        return None
    out: set[int] = set()
    for piece in (p.strip() for p in text.split(",")):
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(piece))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="vLLM capture shard loop (S.3, T5.2 — vLLM port)")
    parser.add_argument("--model", required=True, help="model key in configs/models.yaml")
    parser.add_argument("--corpus", required=True, type=Path, help="corpus JSONL (T4.2 output)")
    parser.add_argument("--corpus-name", default=None)
    parser.add_argument("--models", type=Path, default=repo_root / "configs" / "models.yaml")
    parser.add_argument("--run-config", type=Path, default=repo_root / "configs" / "run_vllm.yaml")
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--backend", choices=("local", "hf"), default="local")
    parser.add_argument("--local-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--public", action="store_true",
                        help="create the HF trace repo as PUBLIC (world-readable) instead of "
                             "private — uses the account's large public storage quota. Only "
                             "applies when the repo is first created; existing repos keep their "
                             "visibility.")
    parser.add_argument("--log", type=Path, default=repo_root / "results" / "collection_log.csv")
    parser.add_argument("--shards", default=None, help="restrict to '0-19' or '3,7,11'")
    parser.add_argument("--subsample-n", type=int, default=None,
                        help="override capture.hidden_subsample_n (P3 proof: exercise hidden on a "
                             "tiny corpus); default is the config value")
    parser.add_argument("--dry-run", action="store_true", help="plan shards, load nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = RunConfig.load(args.run_config)
    if config.engine != "vllm":
        print(f"{args.run_config} is not a vLLM config (engine={config.engine!r})", file=sys.stderr)
        return 2
    meta = load_model_meta(args.models, args.model)
    vllm = dict(meta.get("vllm") or {})
    if not vllm.get("model_id"):
        print(f"{args.model} has no vllm.model_id in models.yaml", file=sys.stderr)
        return 2
    corpus_name = args.corpus_name or args.corpus.stem
    spec, gating = spec_and_gating_for(meta)

    scratch = args.scratch or Path((config.unhashed.get("paths") or {}).get("scratch", "."))
    scratch_root = assert_not_kaggle_working(Path(scratch) / args.model / corpus_name)
    subsample_n = args.subsample_n if args.subsample_n is not None else config.capture.get("hidden_subsample_n")
    plans = plan_shards(args.corpus, out_root=scratch_root / "shards", subsample_n=subsample_n)
    wanted = _parse_shard_filter(args.shards)
    if wanted is not None:
        plans = [p for p in plans if p.shard_id in wanted]
        if not plans:
            print(f"no shards match --shards {args.shards}", file=sys.stderr)
            return 2

    if args.dry_run:
        print(f"# {args.model} ({vllm['model_id']}, TP={vllm.get('tensor_parallel', 1)}): "
              f"{len(plans)} shard(s), run_config {config.short}")
        for p in plans:
            print(f"#   shard {p.shard_id}: {p.n_docs} docs, ~{p.n_tokens_ref} ref tokens")
        return 0

    # model_sha256 (the weights identity) is the HF revision sha; resolve it if the card left it
    # null (the T1.1-equivalent gap), so the manifest is attributable. build_vllm_manifest reads it
    # from meta["vllm"]. Done only for a real run (network), never for --dry-run above.
    if not vllm.get("model_sha256"):
        vllm["model_sha256"] = resolve_model_sha256(vllm["model_id"], meta.get("revision"))
        meta = {**meta, "vllm": vllm}
        print(f"# resolved model_sha256 for {args.model}: {vllm['model_sha256']}")

    ledger = ShardState.load_or_create(
        scratch_root / "state.json", args.model, corpus_name, config.sha256
    )
    if args.backend == "hf":
        if not args.repo_id:
            print("--backend hf requires --repo-id", file=sys.stderr)
            return 2
        from ..runtime.upload import HFBackend  # noqa: PLC0415

        backend: StorageBackend = HFBackend(args.repo_id, create=True, private=not args.public)
    else:
        if not args.local_root:
            print("--backend local requires --local-root", file=sys.stderr)
            return 2
        from ..runtime.upload import LocalDirBackend  # noqa: PLC0415

        backend = LocalDirBackend(args.local_root)

    engine = VLLMCaptureEngine(
        model_id=vllm["model_id"],
        tensor_parallel_size=int(vllm.get("tensor_parallel", 1)),
        spec=spec, gating=gating,
        router_suffix=vllm.get("router_suffix", ".mlp.gate"),
        experts_suffix=vllm.get("experts_suffix", ".mlp.experts"),
        max_model_len=int(config.inference["max_model_len"]),
        gpu_memory_utilization=float(
            vllm.get("gpu_memory_utilization")
            or (config.unhashed.get("inference") or {}).get("gpu_memory_utilization", 0.90)
        ),
        dtype=str(config.build.get("dtype", "float16")),
        seed=int(config.inference.get("seed", 0)),
    )
    engine.vllm_version = config.build.get("vllm_version")
    engine.load()
    try:
        outcome = run_vllm_collection(
            plans, engine=engine, config=config, backend=backend, ledger=ledger,
            spec=spec, gating=gating, model=args.model, corpus=corpus_name, model_meta=meta,
            scratch_root=scratch_root,
            remote_root=args.remote_root or f"traces/{args.model}/{corpus_name}",
            log_path=args.log,
        )
    finally:
        engine.remove()

    print(outcome.summary())
    for f in (r for r in outcome.results if r.status == "failed"):
        print(f"shard {f.shard_id}: {f.error}", file=sys.stderr)
    return 0 if outcome.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
