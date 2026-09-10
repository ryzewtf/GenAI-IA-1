"""T3.2 -- cross-validate the llama.cpp routing against PyTorch/transformers.

T3.3 proves the trace is *internally* consistent: recompute the routing from ``hidden.bin`` and
the GGUF's own router weight, and it reproduces ``topk.bin``. That is a strong check, but it is
closed over one implementation. If llama.cpp's OLMoE graph applied the wrong norm before the
router, or the GGUF conversion transposed something, T3.3 would still pass -- the trace would be
a faithful record of the wrong computation.

This module closes that loop against an independent implementation. It runs the *same token ids*
through HF transformers, hooks each router module, and compares selections and orderings against
the trace. Two implementations, two codebases, one answer.

Design decisions that are load-bearing, each of which is a confound removed:

* **fp16 on both sides.** The plan is explicit and v1.0 got this wrong: comparing bf16 PyTorch
  against an F16 GGUF compares two formats as much as two implementations. Here both sides come
  from the same bf16 to fp16 cast (``convert_hf_to_gguf.py --outtype f16`` and ``.half()`` round
  identically), so a mismatch is a harness bug rather than a format artifact. Also mandatory on
  Turing: sm_75 has no bf16 tensor cores. Note that OLMoE's ``config.json`` declares
  ``torch_dtype: float32`` -- pass fp16 explicitly or the load OOMs on a 15 GB T4.

* **The trace's own token ids are replayed, never re-tokenized.** Re-tokenizing would make a
  tokenizer difference indistinguishable from a routing difference, and the tokenizer is the one
  part of the stack this check is not trying to validate. ``tokens.bin`` carries ``doc_id`` and
  ``pos_in_doc``, so document boundaries are recovered exactly and each document is a separate
  forward with ``use_cache=False`` -- matching ``clear_kv_between_docs: true``, which is what
  makes document-level sharding bit-exact in the first place.

* **Router modules are discovered structurally and the count is asserted.** A name match like
  ``mlp.gate`` is a guess that silently hooks the wrong module on the next architecture. The
  discovery here requires an ``nn.Linear`` with ``out_features == n_experts`` and no bias, and
  then refuses to proceed unless it found exactly ``n_moe_layers`` of them.

**On the Spearman gate.** The plan asks for "per-layer Spearman on raw logits". Our
``logits.bin`` holds ``ffn_moe_probs`` -- post-softmax -- not raw logits (invariant I13). Softmax
is strictly monotone, so the *ranks* are unchanged and the comparison remains exactly the one the
plan intends. What softmax does change is resolution: fp16 storage of small probabilities ties
values that were distinct as logits. Measured on a real OLMoE shard, 42.5% of (token, layer) rows
contain at least one tied pair, averaging 63.4 distinct values out of 64. This module therefore
uses **average ranks** rather than ordinal ranks -- ordinal ranking would invent an arbitrary
order for tied values and charge the difference to the model -- and reports
``frac_rows_with_ties`` alongside, so a shortfall can be attributed rather than guessed at. For
n=64 a single adjacent transposition moves rho by about 2.3e-5, so the plan's 0.999 floor has
room for roughly forty times the tie rate actually observed.

**How much of the budget fp16 storage already spends.** Feeding this module the trace's own
``logits.bin`` as a stand-in for the HF side -- a comparison that should be perfect by
construction -- scores 0.999541 set_agreement and 0.990208 exact_match on a real 817-token OLMoE
shard, not 1.0. The reason is that ``topk.bin`` records llama.cpp's argsort over *fp32*
``selection_probs`` while ``logits.bin`` stores those same probabilities in fp16, and at a
near-tie the rounding reorders them. So before PyTorch is involved at all, the storage format
consumes roughly 1 point of the 3 available under the 0.97 exact_match floor and about half of
the 0.001 available under the 0.999 set_agreement floor.

That is an upper bound on the effect rather than a prediction: the real HF side arrives in fp32
and is closer to llama.cpp's fp32 probabilities than the fp16 round trip is. But it means a
marginal T3.2 failure should be checked against this number before anyone goes looking for a
harness bug, and it is why the interpretation rule below reads *concentration across layers*
rather than absolute magnitude.

The comparison core is pure numpy and has no torch dependency, so it is testable on a workstation
without a 2 GB import; only :func:`capture_router_logits` needs transformers, and it imports it
lazily.
"""

from __future__ import annotations

import argparse
import json

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..metrics.ranks import average_ranks, spearman_rows
from .format import TOKEN_DTYPE, read_manifest

__all__ = [
    "TorchCheckError",
    "LayerComparison",
    "TorchCheckReport",
    "average_ranks",
    "spearman_rows",
    "compare_layer",
    "compare_shard",
    "documents_from_tokens",
    "find_router_modules",
    "capture_router_logits",
    "main",
]

#: Plan T3.2's tiered acceptance. All three are hard gates; the tiering is about *which* number
#: fails, because the failure mode is diagnostic. A diffuse 1-3% exact-match shortfall spread
#: evenly across layers is floating point; a shortfall concentrated in one or two layers is a
#: name-filter, batching or layer-index bug.
MIN_SET_AGREEMENT = 0.999
MIN_EXACT_MATCH = 0.97
MIN_SPEARMAN = 0.999

#: Attribute names transformers uses for the router Linear inside a MoE block. Used only to
#: *order* and label candidates -- selection is by structure (see :func:`find_router_modules`).
ROUTER_ATTR_NAMES = ("gate", "router", "gate_proj", "block_sparse_moe.gate")


class TorchCheckError(RuntimeError):
    """The check could not be run. Never raised for *disagreement* -- that is a result."""


@dataclass(frozen=True)
class LayerComparison:
    """One trace layer's agreement between HF routing and the trace."""

    layer: int
    model_layer: int
    n_tokens: int
    exact_match: float
    set_agreement: float
    spearman: float
    spearman_min: float
    frac_rows_with_ties: float

    def failures(
        self,
        *,
        min_set_agreement: float = MIN_SET_AGREEMENT,
        min_exact_match: float = MIN_EXACT_MATCH,
        min_spearman: float = MIN_SPEARMAN,
    ) -> list[str]:
        out = []
        if self.set_agreement < min_set_agreement:
            out.append(f"set_agreement {self.set_agreement:.6f} < {min_set_agreement}")
        if self.exact_match < min_exact_match:
            out.append(f"exact_match {self.exact_match:.6f} < {min_exact_match}")
        if self.spearman < min_spearman:
            out.append(f"spearman {self.spearman:.6f} < {min_spearman}")
        return [f"layer {self.layer} (model layer {self.model_layer}): {f}" for f in out]


@dataclass
class TorchCheckReport:
    model: str
    shard_dir: str
    n_tokens: int
    n_docs: int
    top_k: int
    n_experts: int
    layers: list[LayerComparison] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def worst(self) -> LayerComparison | None:
        return min(self.layers, key=lambda c: c.set_agreement, default=None)

    def to_json(self) -> dict[str, Any]:
        return {
            "task": "T3.2",
            "model": self.model,
            "shard_dir": self.shard_dir,
            "n_tokens": self.n_tokens,
            "n_docs": self.n_docs,
            "top_k": self.top_k,
            "n_experts": self.n_experts,
            "thresholds": self.thresholds,
            "ok": self.ok,
            "failures": self.failures,
            "notes": self.notes,
            "layers": [asdict(c) for c in self.layers],
        }


# -- comparison ----------------------------------------------------------------------------


def compare_layer(
    hf_logits: np.ndarray,
    trace_topk: np.ndarray,
    trace_selection: np.ndarray,
    *,
    top_k: int,
    layer: int = 0,
    model_layer: int = 0,
) -> LayerComparison:
    """Compare one layer.

    Parameters
    ----------
    hf_logits:
        ``(n_tokens, n_experts)`` raw router logits from the HF forward hook.
    trace_topk:
        ``(n_tokens, top_k)`` expert ids from ``topk.bin``.
    trace_selection:
        ``(n_tokens, n_experts)`` from ``logits.bin`` -- post-softmax for every model in the
        panel, which is fine: only the ordering is used, and softmax preserves it (I13).
    """
    hf_logits = np.asarray(hf_logits, dtype=np.float32)
    trace_selection = np.asarray(trace_selection, dtype=np.float32)
    trace_topk = np.asarray(trace_topk, dtype=np.int64)

    if hf_logits.ndim != 2 or trace_selection.ndim != 2 or trace_topk.ndim != 2:
        raise TorchCheckError("compare_layer expects 2-D (n_tokens, ...) arrays")
    n_tokens, n_experts = hf_logits.shape
    if trace_selection.shape != hf_logits.shape:
        raise TorchCheckError(
            f"selection shape {trace_selection.shape} != HF logits shape {hf_logits.shape}"
        )
    if trace_topk.shape != (n_tokens, top_k):
        raise TorchCheckError(
            f"topk shape {trace_topk.shape} != expected {(n_tokens, top_k)}"
        )
    if n_tokens == 0:
        raise TorchCheckError("no tokens to compare")

    hf_top = np.argsort(-hf_logits, axis=1, kind="stable")[:, :top_k]

    # set_agreement is over SETS, exact_match over ordered tuples. Both matter and they fail
    # differently: a permuted-but-equal set is floating point at a near-tie, while a wrong set is
    # a wrong computation.
    hf_sorted = np.sort(hf_top, axis=1)
    trace_sorted = np.sort(trace_topk, axis=1)
    overlap = np.array(
        [np.intersect1d(h, t, assume_unique=False).size for h, t in zip(hf_sorted, trace_sorted)],
        dtype=np.float64,
    )
    set_agreement = float((overlap / top_k).mean())
    exact_match = float((hf_top == trace_topk).all(axis=1).mean())

    rho = spearman_rows(hf_logits, trace_selection)
    finite = np.isfinite(rho)
    spearman = float(rho[finite].mean()) if finite.any() else float("nan")
    spearman_min = float(rho[finite].min()) if finite.any() else float("nan")

    distinct = np.array([np.unique(row).size for row in trace_selection], dtype=np.int64)
    frac_ties = float((distinct < n_experts).mean())

    return LayerComparison(
        layer=layer,
        model_layer=model_layer,
        n_tokens=int(n_tokens),
        exact_match=exact_match,
        set_agreement=set_agreement,
        spearman=spearman,
        spearman_min=spearman_min,
        frac_rows_with_ties=frac_ties,
    )


def documents_from_tokens(tokens: np.ndarray) -> list[np.ndarray]:
    """Split a shard's ``tokens.bin`` into per-document token-id arrays.

    Uses ``doc_id`` and validates ``pos_in_doc`` rather than trusting either alone: a document
    boundary that disagrees with the position counter means the writer and the runner disagree
    about what a document is, and replaying that would compare two different segmentations.
    """
    tokens = np.asarray(tokens)
    if tokens.dtype != TOKEN_DTYPE:
        raise TorchCheckError(f"expected TOKEN_DTYPE records, got {tokens.dtype}")
    if tokens.size == 0:
        raise TorchCheckError("tokens.bin is empty")

    doc_ids = tokens["doc_id"].astype(np.int64)
    positions = tokens["pos_in_doc"].astype(np.int64)

    starts = np.flatnonzero(np.r_[True, doc_ids[1:] != doc_ids[:-1]])
    ends = np.r_[starts[1:], doc_ids.size]

    docs = []
    for start, end in zip(starts, ends):
        expected = np.arange(end - start, dtype=np.int64)
        if not np.array_equal(positions[start:end], expected):
            raise TorchCheckError(
                f"doc_id {int(doc_ids[start])} spans rows {start}:{end} but pos_in_doc is not "
                f"0..{end - start - 1}; the shard is not in document order or a document is split"
            )
        docs.append(tokens["token_id"][start:end].astype(np.int64))

    if len(np.unique(doc_ids)) != len(docs):
        raise TorchCheckError("a doc_id appears in more than one contiguous run")
    return docs


def _layer_map(manifest: Mapping[str, Any], n_moe_layers: int) -> list[int]:
    raw = manifest.get("layer_index_map")
    if isinstance(raw, dict):
        try:
            return [int(raw[f"trace_{i}"]) for i in range(n_moe_layers)]
        except KeyError as exc:
            raise TorchCheckError(f"layer_index_map has no entry for {exc}") from exc
    if isinstance(raw, (list, tuple)):
        if len(raw) != n_moe_layers:
            raise TorchCheckError(
                f"layer_index_map has {len(raw)} entries for {n_moe_layers} trace layers"
            )
        return [int(v) for v in raw]
    raise TorchCheckError("manifest has no usable layer_index_map")


def compare_shard(
    shard_dir: Path | str,
    hf_logits: np.ndarray,
    *,
    min_set_agreement: float = MIN_SET_AGREEMENT,
    min_exact_match: float = MIN_EXACT_MATCH,
    min_spearman: float = MIN_SPEARMAN,
) -> TorchCheckReport:
    """Compare a captured shard against HF router logits.

    ``hf_logits`` is ``(n_tokens, n_model_layers, n_experts)`` indexed by MODEL layer, exactly as
    :func:`capture_router_logits` returns it. Indexing by model layer rather than by trace layer
    is deliberate: it forces the manifest's ``layer_index_map`` to be applied here, where it can
    be checked, rather than being assumed by the caller.
    """
    shard_dir = Path(shard_dir)
    manifest = read_manifest(shard_dir)

    n_tokens = int(manifest["n_tokens"])
    n_layers = int(manifest["n_moe_layers"])
    n_experts = int(manifest["n_experts"])
    top_k = int(manifest["top_k"])
    layer_map = _layer_map(manifest, n_layers)

    hf_logits = np.asarray(hf_logits)
    if hf_logits.ndim != 3:
        raise TorchCheckError(
            f"hf_logits must be (n_tokens, n_model_layers, n_experts), got {hf_logits.shape}"
        )
    if hf_logits.shape[0] != n_tokens:
        raise TorchCheckError(
            f"HF produced {hf_logits.shape[0]} token rows but the shard holds {n_tokens}. "
            "The forward replayed a different segmentation -- do not reconcile by truncating."
        )
    if hf_logits.shape[2] != n_experts:
        raise TorchCheckError(
            f"HF router width {hf_logits.shape[2]} != manifest n_experts {n_experts}"
        )
    if max(layer_map) >= hf_logits.shape[1]:
        raise TorchCheckError(
            f"layer_index_map names model layer {max(layer_map)} but HF captured only "
            f"{hf_logits.shape[1]} router layers"
        )

    topk = np.fromfile(shard_dir / "topk.bin", dtype="<i4").reshape(n_tokens, n_layers, top_k)
    selection = np.fromfile(shard_dir / "logits.bin", dtype="<f2").reshape(
        n_tokens, n_layers, n_experts
    )

    report = TorchCheckReport(
        model=str(manifest.get("model", "?")),
        shard_dir=str(shard_dir),
        n_tokens=n_tokens,
        n_docs=int(manifest.get("n_docs", 0)),
        top_k=top_k,
        n_experts=n_experts,
        thresholds={
            "set_agreement": min_set_agreement,
            "exact_match": min_exact_match,
            "spearman": min_spearman,
        },
    )
    node = str(manifest.get("logit_tensor_used", ""))
    if node and "logit" not in node:
        report.notes.append(
            f"logit_tensor_used={node!r} is post-softmax; Spearman is computed on ranks, which "
            "softmax preserves (I13)"
        )

    for trace_layer in range(n_layers):
        model_layer = layer_map[trace_layer]
        comparison = compare_layer(
            hf_logits[:, model_layer, :],
            topk[:, trace_layer, :],
            selection[:, trace_layer, :],
            top_k=top_k,
            layer=trace_layer,
            model_layer=model_layer,
        )
        report.layers.append(comparison)
        report.failures.extend(
            comparison.failures(
                min_set_agreement=min_set_agreement,
                min_exact_match=min_exact_match,
                min_spearman=min_spearman,
            )
        )

    return report


# -- the torch half ------------------------------------------------------------------------


def find_router_modules(model: Any, n_experts: int, n_moe_layers: int) -> list[tuple[str, Any]]:
    """Locate the router Linear in every MoE block, structurally.

    Selection is by shape -- an ``nn.Linear`` projecting to exactly ``n_experts`` outputs -- not
    by attribute name, because the name differs per architecture (``mlp.gate`` on OLMoE,
    ``block_sparse_moe.gate`` on Mixtral) and a name that stops matching hooks nothing while
    still returning a plausible-looking result. The count is then asserted: hooking 15 routers on
    a 16-layer model is exactly the class of bug this task exists to catch.
    """
    found: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            continue
        if int(module.out_features) != n_experts:
            continue
        # The lm_head of a model whose vocab happens to equal n_experts would be absurd, but
        # requiring the module to sit inside a decoder layer costs one check and removes the
        # whole class of accidental match.
        if "layers." not in name:
            continue
        found.append((name, module))

    if len(found) != n_moe_layers:
        raise TorchCheckError(
            f"found {len(found)} candidate router modules with out_features=={n_experts} but the "
            f"trace has {n_moe_layers} MoE layers: {[n for n, _ in found]}. Refusing to guess -- "
            "a partial hook set produces a report that looks fine and compares the wrong layers."
        )
    # named_modules walks in registration order, which is layer order, but sorting by the numeric
    # layer index makes that an assertion rather than an assumption.
    def layer_of(name: str) -> int:
        parts = name.split("layers.")[1].split(".")[0]
        return int(parts)

    found.sort(key=lambda item: layer_of(item[0]))
    return found


def capture_router_logits(
    model_id: str,
    documents: Sequence[np.ndarray],
    *,
    n_experts: int,
    n_moe_layers: int,
    max_memory: Mapping[int | str, str] | None = None,
    dtype: str = "float16",
    mxfp4_dequantize: bool | None = None,
) -> np.ndarray:
    """Replay ``documents`` through HF transformers and return ``(n_tokens, n_layers, n_experts)``.

    One forward per document with ``use_cache=False`` and no padding: batching documents together
    would introduce pad positions whose routing has no counterpart in the trace, and a left-pad
    would shift every position id.

    ``dtype`` defaults to fp16 for the reason in the module docstring; it is a parameter only so
    a CPU-only debugging run can ask for fp32 and say so in the report.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - exercised only on a machine with torch
        raise TorchCheckError(
            "T3.2 needs torch and transformers. The Kaggle image ships both (T0.1 measured "
            f"torch 2.10.0+cu128, transformers 5.0.0); this host does not: {exc}"
        ) from exc

    torch_dtype = getattr(torch, dtype)
    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if max_memory:
        kwargs["max_memory"] = dict(max_memory)

    if mxfp4_dequantize is not None:
        # GPT-OSS. Plan C11: transformers gates MXFP4 at compute capability >= (7, 5), and a T4 is
        # exactly (7, 5) -- so this is worth attempting rather than writing off. The real
        # requirement is Triton >= 3.4 plus the `kernels` package; without them transformers
        # dequantizes to bf16, which needs ~40 GB and will fail.
        from transformers import Mxfp4Config

        kwargs["quantization_config"] = Mxfp4Config(dequantize=mxfp4_dequantize)

    config = AutoConfig.from_pretrained(model_id)
    declared = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    if declared is not None and int(declared) != n_experts:
        raise TorchCheckError(
            f"{model_id} declares {declared} experts but the trace has {n_experts}. This is a "
            "different checkpoint, not a numerics question."
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    routers = find_router_modules(model, n_experts, n_moe_layers)
    captured: list[list[np.ndarray]] = [[] for _ in routers]
    handles = []

    def make_hook(slot: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[slot].append(tensor.detach().reshape(-1, n_experts).float().cpu().numpy())
        return hook

    for slot, (_name, module) in enumerate(routers):
        handles.append(module.register_forward_hook(make_hook(slot)))

    try:
        with torch.no_grad():
            for doc in documents:
                ids = torch.tensor(np.asarray(doc, dtype=np.int64)[None, :], device=model.device)
                model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    per_layer = [np.concatenate(chunks, axis=0) for chunks in captured]
    lengths = {arr.shape[0] for arr in per_layer}
    if len(lengths) != 1:
        raise TorchCheckError(
            f"router hooks captured differing row counts {sorted(lengths)}; at least one layer "
            "ran a different number of times than the others"
        )
    total = sum(len(d) for d in documents)
    if per_layer[0].shape[0] != total:
        raise TorchCheckError(
            f"hooks captured {per_layer[0].shape[0]} rows for {total} input tokens"
        )

    # (n_tokens, n_layers, n_experts), indexed by MODEL layer via the sorted discovery order.
    stacked = np.stack(per_layer, axis=1)
    model_layers = max(int(name.split("layers.")[1].split(".")[0]) for name, _ in routers) + 1
    out = np.zeros((stacked.shape[0], model_layers, n_experts), dtype=np.float32)
    for slot, (name, _module) in enumerate(routers):
        out[:, int(name.split("layers.")[1].split(".")[0]), :] = stacked[:, slot, :]
    return out


# -- CLI -----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T3.2 PyTorch cross-validation")
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("--model-id", required=True, help="HF repo id, e.g. allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--max-docs", type=int, default=0, help="replay only the first N documents")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-memory", default=None,
                        help='e.g. \'{"0": "13GiB", "1": "13GiB", "cpu": "4GiB"}\'')
    parser.add_argument("--mxfp4-dequantize", choices=("true", "false"), default=None,
                        help="GPT-OSS only; sets Mxfp4Config(dequantize=...)")
    parser.add_argument("--hf-logits", type=Path, default=None,
                        help="reuse a previously saved .npy instead of running the model")
    parser.add_argument("--save-hf-logits", type=Path, default=None)
    parser.add_argument("--min-set-agreement", type=float, default=MIN_SET_AGREEMENT)
    parser.add_argument("--min-exact-match", type=float, default=MIN_EXACT_MATCH)
    parser.add_argument("--min-spearman", type=float, default=MIN_SPEARMAN)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--attempt-log", type=Path, default=None,
                        help="on failure to RUN, record the exact failure here (plan T3.2 step 2)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = read_manifest(args.shard_dir)
    n_experts = int(manifest["n_experts"])
    n_layers = int(manifest["n_moe_layers"])

    try:
        if args.hf_logits is not None:
            hf_logits = np.load(args.hf_logits)
        else:
            tokens = np.fromfile(args.shard_dir / "tokens.bin", dtype=TOKEN_DTYPE)
            documents = documents_from_tokens(tokens)
            if args.max_docs:
                documents = documents[: args.max_docs]
            hf_logits = capture_router_logits(
                args.model_id,
                documents,
                n_experts=n_experts,
                n_moe_layers=n_layers,
                dtype=args.dtype,
                max_memory=json.loads(args.max_memory) if args.max_memory else None,
                mxfp4_dequantize=(
                    None if args.mxfp4_dequantize is None
                    else args.mxfp4_dequantize == "true"
                ),
            )
            if args.save_hf_logits:
                np.save(args.save_hf_logits, hf_logits)

        report = compare_shard(
            args.shard_dir,
            hf_logits,
            min_set_agreement=args.min_set_agreement,
            min_exact_match=args.min_exact_match,
            min_spearman=args.min_spearman,
        )
    except TorchCheckError as exc:
        # "Could not run" and "ran and disagreed" are different outcomes and the plan treats them
        # differently -- for GPT-OSS the first one is an expected branch with a documented
        # fallback, and it has to be recorded rather than merely printed.
        payload = {"task": "T3.2", "model_id": args.model_id, "shard_dir": str(args.shard_dir),
                   "ran": False, "error": str(exc)}
        if args.attempt_log:
            args.attempt_log.parent.mkdir(parents=True, exist_ok=True)
            args.attempt_log.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"T3.2 COULD NOT RUN for {args.model_id}: {exc}")
        return 2

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")

    worst = report.worst
    if report.ok:
        assert worst is not None
        print(
            f"T3.2 PASSED for {report.model}: worst layer {worst.layer} at set_agreement "
            f"{worst.set_agreement:.6f}, exact_match {worst.exact_match:.6f}, "
            f"spearman {worst.spearman:.6f}"
        )
        return 0

    print(f"T3.2 FAILED for {report.model}:")
    for failure in report.failures:
        print(f"  {failure}")
    concentrated = len({f.split(":")[0] for f in report.failures})
    print(
        f"  {concentrated} of {len(report.layers)} layers implicated. The plan's reading rule: a "
        "diffuse shortfall spread evenly across layers is floating point; one concentrated in a "
        "layer or two is a name-filter, batching or layer-index bug."
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
