"""Two-trace equivalence and drift: Phase 8's precision ladder, plus the T3.7/T3.8 gates.

Four tasks in the plan reduce to one operation — collect the same corpus twice with exactly one
pinned variable changed, then compare the routing:

===== ==================================== ============================ =========================
Task   Variable                             Acceptance                   Judged on
===== ==================================== ============================ =========================
T3.7   router placement (GPU vs CPU)        >= 99.9% set agreement       worst layer
T3.8   batching (prefill vs decode)         >= 99.9% set agreement       worst layer
T8.2   quantization (F16 vs Q8_0/Q4_K_M)    a measured bound, not a pass  mean, banded by T8.3
T8.5   tensor split (0.5/0.5 vs 0.7/0.3)    >= 99.9% set agreement       worst layer
===== ==================================== ============================ =========================

The three gates share :func:`check_equivalence`; T8.3 keeps its own banded reading because it is
not asking pass/fail, it is quantifying a limitation. The split between "worst layer" and "mean" is
deliberate and is the one place these tasks genuinely differ — see :func:`check_equivalence`.

Phase 8 turns "quantization probably doesn't matter" into a measured bound. Every model in the
panel except OLMoE is collected quantized because nothing else fits, so the size of the
quantization effect is a limitation the paper has to state either way; the only question is
whether it is stated as a number or as a hope.

The comparison is between two traces of the **same model on the same corpus** that differ in one
pinned variable. Everything here is that one comparison, plus the arithmetic that reads a decision
off it.

**What is compared, and why it is `topk.bin` on both sides.** Set agreement is computed from the
model-emitted expert indices, never from a recomputation off the stored logits. `topk.bin` is what
llama.cpp actually routed to; recomputing from `logits.bin` would measure the fp16 storage of the
logits as much as the quantization of the weights, and those two effects are the same order (T3.2
measured the storage term alone at ~1 point of exact-match on OLMoE). Only the Spearman leg reads
`logits.bin`, because ordering is the thing it is asking about.

**Margins are binned by quantile, not by absolute value.** The plan asks for flip rate conditioned
on the k-th/(k+1)-th margin, which is the right idea: flips concentrated at small margins are
expected arithmetic, flips at large margins are a bug. But our margins come from ``logits.bin``,
which holds post-softmax probabilities (I13), and a softmax is normalized over the expert
dimension — so a margin of 0.01 at 64 experts is a different amount of separation than 0.01 at 128,
and a fixed bin edge would make the panel's expert-count comparison meaningless. Since T8.2's
stated prediction is precisely that drift scales with expert count, binning that confounds expert
count with margin scale would answer the question by construction. Quantile bins are computed
per (layer, comparison) and are directly comparable across both.

**Alignment is checked, not assumed.** Two traces of the same corpus must contain byte-identical
``tokens.bin`` token ids: the tokenizer depends on neither the weights' quantization, the router's
placement, nor the batching. If they differ, one of the runs saw different text and every number
below is meaningless, so it is a hard error rather than a low score. This is the counterpart of
T3.2's segmentation check and it costs one array comparison.

**A comparison where nothing actually varied is refused.** Every one of these tasks scores 100% if
the two legs are secretly identical — a `--override-tensor` pattern that matched no tensor, a
`--decode-mode` flag the binary ignored, a second collection that reused the first's config. That
is the worst failure mode a validation gate can have, because it is indistinguishable from a pass.
:func:`check_equivalence` therefore takes the name of the field that was supposed to differ and
refuses to return a verdict when it did not.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..metrics.ranks import spearman_rows
from ..traces.format import FormatError
from ..traces.reader import TraceReader

__all__ = [
    "PrecisionError",
    "LayerDrift",
    "LadderComparison",
    "InterpretationGate",
    "EquivalenceResult",
    "DEFAULT_MARGIN_QUANTILES",
    "EQUIVALENCE_FLOOR",
    "EQUIVALENCE_TASKS",
    "compare_traces",
    "interpret",
    "check_equivalence",
    "check_device_split",
    "main",
]

#: Plan T8.3. Read as: agreement at or above the first number is a bounded limitation; below the
#: second, the paper's contribution changes. These are the plan's numbers and are not tunable.
GATE_BOUNDED = 0.97
GATE_SEVERE = 0.90

#: Plan acceptance shared by T3.7, T3.8 and T8.5. All three ask the same question — "is this
#: knob invisible to the routing?" — and the plan gives all three the same floor.
EQUIVALENCE_FLOOR = 0.999
DEVICE_SPLIT_FLOOR = EQUIVALENCE_FLOOR  # kept as the T8.5 spelling

#: For each gate: the manifest field that MUST differ between the two legs, and what to say when a
#: failure means. `varies` is not decoration -- it is the check that the experiment happened at all.
EQUIVALENCE_TASKS: dict[str, dict[str, str]] = {
    "T3.7": {
        "varies": "override_tensor",
        "label": "router placement (GPU-resident vs --override-tensor ...=CPU)",
        "on_fail": (
            "The router's device placement changes its own output. §0b assumes it does not, and "
            "the collection config leaves the router on GPU on that basis. Report the measured "
            "figure and pin the placement."
        ),
    },
    "T3.8": {
        "varies": "decode_mode",
        "label": "batching (prefill vs decode)",
        "on_fail": (
            "Prefill routing is NOT decode routing. Every result in the paper is a prefill "
            "measurement being read as a claim about decode-time prefetching, so this is a "
            "validity threat to the central interpretation, not a footnote. §0b asserts this "
            "equivalence; the assertion is wrong and the framing has to change."
        ),
    },
    "T8.5": {
        "varies": "tensor_split",
        "label": "device split (0.5/0.5 vs 0.7/0.3)",
        "on_fail": (
            "Pair T results are split-dependent: every Qwen3 and Gemma 4 number must be collected "
            "under one pinned split, and that fact stated prominently."
        ),
    },
}

#: Quantile edges for the margin conditioning. Five bins, weighted toward the small-margin end
#: because that is where the interesting structure is: the top bin exists mainly to show that it
#: is empty of flips, which is the actual claim being tested.
DEFAULT_MARGIN_QUANTILES = (0.0, 0.05, 0.20, 0.50, 0.80, 1.0)

#: Manifest keys that MUST agree between the two sides. `quant` is deliberately absent: it is the
#: variable. `run_config_sha256` is absent for the same reason -- it hashes the quant.
COMPARABLE_KEYS = ("model", "corpus", "n_moe_layers", "n_experts", "top_k", "hidden_dim",
                   "llama_cpp_commit", "checkpoint_status")


class PrecisionError(RuntimeError):
    """The two traces cannot be compared. Never raised for *drift* — that is a result."""


@dataclass(frozen=True)
class LayerDrift:
    """One layer's drift between a reference trace and a candidate."""

    layer: int
    model_layer: int
    depth: float
    n_tokens: int
    set_agreement: float
    exact_match: float
    spearman: float
    flip_rate: float
    #: Flip rate within each margin quantile bin, smallest margin first.
    flip_rate_by_margin: tuple[float, ...] = ()
    #: Right edge of each margin bin, in the reference trace's own margin units.
    margin_bin_edges: tuple[float, ...] = ()
    n_by_margin: tuple[int, ...] = ()


@dataclass
class LadderComparison:
    """A full reference-vs-candidate comparison across every layer."""

    model: str
    corpus: str
    reference: str
    candidate: str
    reference_quant: str
    candidate_quant: str
    n_tokens: int
    top_k: int
    n_experts: int
    layers: list[LayerDrift] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: `capture_variant` from each side's manifest -- what was actually pinned differently. Empty
    #: for a trace collected before the field existed, which check_equivalence treats as "unknown"
    #: rather than "same": see its require_variation argument.
    reference_variant: dict[str, Any] = field(default_factory=dict)
    candidate_variant: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_set_agreement(self) -> float:
        return float(np.mean([l.set_agreement for l in self.layers])) if self.layers else float("nan")

    @property
    def worst(self) -> LayerDrift | None:
        return min(self.layers, key=lambda l: l.set_agreement, default=None)

    def depth_trend(self) -> float:
        """Spearman of flip rate against normalized depth.

        T8.2's testable prediction is that flip rate *increases* with depth, because error
        accumulates through the quantized layers beneath the router. Returning the correlation
        rather than a fitted slope keeps it scale-free, which matters because flip rates across
        the panel span orders of magnitude.
        """
        if len(self.layers) < 3:
            return float("nan")
        depth = np.array([[l.depth for l in self.layers]], dtype=np.float64)
        flips = np.array([[l.flip_rate for l in self.layers]], dtype=np.float64)
        return float(spearman_rows(depth, flips)[0])

    def to_json(self) -> dict[str, Any]:
        return {
            "task": "T8.2",
            "model": self.model,
            "corpus": self.corpus,
            "reference": self.reference,
            "candidate": self.candidate,
            "reference_quant": self.reference_quant,
            "candidate_quant": self.candidate_quant,
            "n_tokens": self.n_tokens,
            "top_k": self.top_k,
            "n_experts": self.n_experts,
            "mean_set_agreement": self.mean_set_agreement,
            "depth_trend_spearman": self.depth_trend(),
            "reference_variant": self.reference_variant,
            "candidate_variant": self.candidate_variant,
            "notes": self.notes,
            "layers": [asdict(l) for l in self.layers],
        }


# -- comparison ------------------------------------------------------------------------------


def _check_comparable(reference: TraceReader, candidate: TraceReader) -> list[str]:
    ref_m, cand_m = reference.manifests[0], candidate.manifests[0]
    mismatched = [k for k in COMPARABLE_KEYS if ref_m.get(k) != cand_m.get(k)]
    if mismatched:
        detail = ", ".join(f"{k}: {ref_m.get(k)!r} vs {cand_m.get(k)!r}" for k in mismatched)
        raise PrecisionError(
            f"the two traces disagree on {mismatched}, which must be identical for a precision "
            f"ladder comparison ({detail}). Only the quantization may differ."
        )
    if reference.n_tokens != candidate.n_tokens:
        raise PrecisionError(
            f"reference has {reference.n_tokens} tokens, candidate has {candidate.n_tokens}. "
            "A drift number computed over different amounts of text is not a drift number."
        )

    notes: list[str] = []
    ref_q, cand_q = str(ref_m.get("quant", "?")), str(cand_m.get("quant", "?"))
    if ref_q == cand_q:
        notes.append(
            f"BOTH SIDES ARE {ref_q}: this comparison measures run-to-run nondeterminism, not "
            "quantization drift. Useful as a floor, misleading as a ladder rung — label it."
        )
    return notes


def _check_same_tokens(reference: TraceReader, candidate: TraceReader, window: slice) -> None:
    ref_ids = reference.tokens(window)["token_id"]
    cand_ids = candidate.tokens(window)["token_id"]
    if not np.array_equal(ref_ids, cand_ids):
        first = int(np.flatnonzero(ref_ids != cand_ids)[0]) + window.start
        raise PrecisionError(
            f"token ids diverge at corpus position {first} ({int(ref_ids[first - window.start])} "
            f"vs {int(cand_ids[first - window.start])}). Quantization does not change the "
            "tokenizer, so the two runs saw different text and no drift number below is real."
        )


def _quantile_bins(
    margins: np.ndarray, quantiles: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row to a margin bin, returning ``(bin_index, right_edges)``.

    Degenerate margin distributions (many exact ties at zero after the fp16 cast) collapse
    adjacent quantile edges; ``np.unique`` folds them so an empty bin is not reported as a
    zero-flip bin, which would read as evidence of the very thing being tested.
    """
    edges = np.unique(np.quantile(margins, np.asarray(quantiles, dtype=np.float64)))
    if edges.size < 2:
        return np.zeros(margins.shape[0], dtype=np.int64), np.asarray([margins.max()])
    # Interior edges only; searchsorted with "right" puts a value equal to an edge in the lower bin.
    idx = np.searchsorted(edges[1:-1], margins, side="right")
    return idx.astype(np.int64), edges[1:]


def compare_traces(
    reference: TraceReader,
    candidate: TraceReader,
    *,
    margin_quantiles: Sequence[float] = DEFAULT_MARGIN_QUANTILES,
    chunk_tokens: int = 50_000,
    max_tokens: int | None = None,
) -> LadderComparison:
    """Compare a candidate trace against a reference, per layer (plan T8.2).

    Accumulated in chunks so a 4M-token trace does not need a 4M x n_experts float array
    resident; the margin binning needs the margins for the whole layer at once, so those are
    accumulated as a float32 vector (4 bytes per token per layer, ~16 MB for OLMoE at 1M tokens).
    """
    notes = _check_comparable(reference, candidate)

    n_tokens = reference.n_tokens if max_tokens is None else min(max_tokens, reference.n_tokens)
    if n_tokens <= 0:
        raise PrecisionError("no tokens to compare")
    if max_tokens is not None and max_tokens < reference.n_tokens:
        notes.append(f"compared the first {n_tokens} of {reference.n_tokens} tokens")

    ref_m = reference.manifests[0]
    cand_m = candidate.manifests[0]
    top_k = reference.top_k
    n_layers = reference.n_moe_layers

    comparison = LadderComparison(
        model=str(ref_m.get("model", "?")),
        corpus=str(ref_m.get("corpus", "?")),
        reference=str(reference.root / reference.model / reference.corpus),
        candidate=str(candidate.root / candidate.model / candidate.corpus),
        reference_quant=str(ref_m.get("quant", "?")),
        candidate_quant=str(cand_m.get("quant", "?")),
        n_tokens=n_tokens,
        top_k=top_k,
        n_experts=reference.n_experts,
        notes=notes,
        reference_variant=_variant_block(ref_m),
        candidate_variant=_variant_block(cand_m),
    )

    windows = []
    start = 0
    while start < n_tokens:
        stop = min(start + chunk_tokens, n_tokens)
        windows.append(slice(start, stop))
        start = stop

    for window in windows:
        _check_same_tokens(reference, candidate, window)

    for layer in range(n_layers):
        overlap_sum = 0.0
        exact_sum = 0
        rho_sum = 0.0
        rho_n = 0
        flips = np.empty(n_tokens, dtype=bool)
        margins = np.empty(n_tokens, dtype=np.float32)

        for window in windows:
            ref_sets = np.sort(reference.topk_sets(layer, window).astype(np.int64), axis=1)
            cand_sets = np.sort(candidate.topk_sets(layer, window).astype(np.int64), axis=1)

            # Both sides are sorted, so a row-wise set intersection is a sorted-array merge; doing
            # it with a membership matrix keeps it vectorized and is O(n * k * k) with tiny k.
            same = (ref_sets[:, :, None] == cand_sets[:, None, :]).any(axis=2)
            overlap = same.sum(axis=1)
            overlap_sum += float(overlap.sum())
            exact_sum += int((overlap == top_k).sum())
            flips[window] = overlap != top_k

            margins[window] = reference.margins(layer, window)

            rho = spearman_rows(reference.logits(layer, window), candidate.logits(layer, window))
            finite = np.isfinite(rho)
            rho_sum += float(rho[finite].sum())
            rho_n += int(finite.sum())

        bins, edges = _quantile_bins(margins, margin_quantiles)
        n_bins = edges.size
        flip_by_bin = []
        n_by_bin = []
        for b in range(n_bins):
            mask = bins == b
            count = int(mask.sum())
            n_by_bin.append(count)
            flip_by_bin.append(float(flips[mask].mean()) if count else float("nan"))

        comparison.layers.append(LayerDrift(
            layer=layer,
            model_layer=reference.model_layer(layer),
            depth=(layer / (n_layers - 1)) if n_layers > 1 else 0.0,
            n_tokens=n_tokens,
            set_agreement=overlap_sum / (n_tokens * top_k),
            exact_match=exact_sum / n_tokens,
            spearman=(rho_sum / rho_n) if rho_n else float("nan"),
            flip_rate=float(flips.mean()),
            flip_rate_by_margin=tuple(flip_by_bin),
            margin_bin_edges=tuple(float(e) for e in edges),
            n_by_margin=tuple(n_by_bin),
        ))

    return comparison


# -- T8.3 ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InterpretationGate:
    band: str            # "bounded" | "prominent" | "severe"
    mean_set_agreement: float
    worst_layer: int
    worst_set_agreement: float
    depth_trend_spearman: float
    action: str

    @property
    def proceed_as_planned(self) -> bool:
        return self.band == "bounded"

    def to_json(self) -> dict[str, Any]:
        return {"task": "T8.3", **asdict(self), "proceed_as_planned": self.proceed_as_planned}


def interpret(comparison: LadderComparison) -> InterpretationGate:
    """Apply plan T8.3's three-band interpretation.

    The bands are read off the **mean** agreement across layers because that is what the plan
    states, but the worst layer is carried alongside and named in the action text. A panel where
    one deep layer sits at 0.85 while the mean is 0.98 is not the same situation as a uniform
    0.98, and the difference decides whether a depth-resolved caveat is needed.
    """
    mean = comparison.mean_set_agreement
    worst = comparison.worst
    trend = comparison.depth_trend()
    worst_layer = worst.layer if worst else -1
    worst_value = worst.set_agreement if worst else float("nan")

    if mean >= GATE_BOUNDED:
        band = "bounded"
        action = (
            f"Quantization is a bounded, reportable limitation ({mean:.4f} mean set agreement "
            f">= {GATE_BOUNDED}). Proceed as planned; report the bound in the limitations section."
        )
    elif mean >= GATE_SEVERE:
        band = "prominent"
        action = (
            f"{mean:.4f} mean set agreement falls in the {GATE_SEVERE}-{GATE_BOUNDED} band. Report "
            "the bound PROMINENTLY, re-run the primary analysis on F16 OLMoE, and show the "
            "conclusions are unchanged."
        )
    else:
        band = "severe"
        action = (
            f"{mean:.4f} mean set agreement is below {GATE_SEVERE}. Plan T8.3: THE PAPER CHANGES. "
            "The finding becomes 'MoE routing predictability studies are confounded by inference "
            "precision', which is a legitimate and arguably more valuable contribution. Do not "
            "bury this outcome."
        )

    if worst and worst.set_agreement < GATE_SEVERE <= mean:
        action += (
            f" NOTE: layer {worst.layer} alone sits at {worst.set_agreement:.4f}, below the "
            "severe band, while the mean does not. A depth-resolved caveat is required."
        )
    if trend == trend and trend > 0.5:
        action += (
            f" Flip rate rises with depth (rho={trend:.2f}), matching T8.2's prediction that error "
            "accumulates through the quantized layers beneath the router."
        )

    return InterpretationGate(
        band=band, mean_set_agreement=mean, worst_layer=worst_layer,
        worst_set_agreement=worst_value, depth_trend_spearman=trend, action=action,
    )


# -- T8.5 ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceResult:
    task: str
    label: str
    passed: bool
    mean_set_agreement: float
    worst_layer: int
    worst_set_agreement: float
    floor: float
    verdict: str

    def to_json(self) -> dict[str, Any]:
        return {"task": self.task, **asdict(self)}


#: Back-compatible alias: T8.5 was written first and named its result type after itself.
DeviceSplitResult = EquivalenceResult


def _variant_block(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The knobs a validation gate may legitimately vary, gathered from where each actually lives.

    They are not in one place and pretending otherwise would be the bug: `override_tensor` and
    `decode_mode` are capture-time flags outside `run_config_sha256` and live in `capture_variant`,
    while `tensor_split` is a pinned `run.yaml` value the runner records under `device_plan`. A
    resolver that looked in only one of them would silently read `None` for the other and conclude
    "these legs are identical" about a comparison that varied exactly as intended.
    """
    block = dict(manifest.get("capture_variant") or {})
    device = manifest.get("device_plan") or {}
    split = device.get("tensor_split")
    block.setdefault("tensor_split", tuple(split) if isinstance(split, (list, tuple)) else split)
    block.setdefault("quant", manifest.get("quant"))
    return block


def _variant_values(comparison: LadderComparison, field_name: str) -> tuple[Any, Any]:
    return (comparison.reference_variant.get(field_name),
            comparison.candidate_variant.get(field_name))


def check_equivalence(
    comparison: LadderComparison,
    *,
    task: str = "T8.5",
    floor: float = EQUIVALENCE_FLOOR,
    require_variation: bool = True,
) -> EquivalenceResult:
    """Apply a >= ``floor`` set-agreement gate to a two-leg comparison (T3.7, T3.8, T8.5).

    **Judged on the worst layer, not the mean**, and that is a deliberate departure from how T8.3
    reads its bands. These three are not measuring a quantity to report as a bound; they are asking
    whether a knob is a confound at all. One layer that lands on a different device, or whose
    kernel selection flips at batch size 1, is exactly the failure mode — and a 16-layer mean
    dilutes it by sixteen, a 61-layer Qwen3 mean by sixty-one.

    **A comparison where nothing varied is refused rather than passed.** Each of these gates scores
    a perfect 100% when the two legs are secretly identical: a `--override-tensor` pattern that
    matched nothing, a flag the binary silently ignored, a second collection that reused the first
    config. `moe_trace` now refuses a no-op override at the source, but a gate that cannot tell
    "invisible knob" from "knob never turned" is not a gate, so the manifests are checked too.
    """
    spec = EQUIVALENCE_TASKS.get(task)
    if spec is None:
        raise PrecisionError(f"unknown equivalence task {task!r}; have {sorted(EQUIVALENCE_TASKS)}")

    worst = comparison.worst
    if worst is None:
        raise PrecisionError("comparison has no layers")

    if require_variation:
        field_name = spec["varies"]
        ref_value, cand_value = _variant_values(comparison, field_name)
        if ref_value is None and cand_value is None:
            raise PrecisionError(
                f"{task}: neither manifest records {field_name}, so it cannot be confirmed that "
                "the two legs differ at all — and this gate scores ~100% when they do not. "
                "Re-collect through src.runtime.runner so the variant is recorded, or pass "
                "require_variation=False if you have verified it some other way."
            )
        if ref_value == cand_value:
            raise PrecisionError(
                f"{task}: both legs report {field_name}={ref_value!r}, so the variable this gate "
                f"exists to test was never changed. The comparison would score ~100% and the gate "
                f"would pass without having measured anything. Check that the second leg was "
                f"actually collected with a different {field_name}."
            )

    passed = worst.set_agreement >= floor
    if passed:
        verdict = (
            f"PASS: {spec['label']} is invisible to the routing — worst layer {worst.layer} agrees "
            f"at {worst.set_agreement:.6f} >= {floor}. Report the measured figure in the methods "
            "section; it upgrades an assumption into a verified claim."
        )
    else:
        verdict = (
            f"FAIL: {spec['label']} changes the routing — layer {worst.layer} agrees at only "
            f"{worst.set_agreement:.6f} < {floor}. {spec['on_fail']}"
        )

    return EquivalenceResult(
        task=task, label=spec["label"], passed=passed,
        mean_set_agreement=comparison.mean_set_agreement, worst_layer=worst.layer,
        worst_set_agreement=worst.set_agreement, floor=floor, verdict=verdict,
    )


def check_device_split(
    comparison: LadderComparison, *, floor: float = EQUIVALENCE_FLOOR,
    require_variation: bool = True,
) -> EquivalenceResult:
    """Plan T8.5 — the device-split spelling of :func:`check_equivalence`."""
    return check_equivalence(comparison, task="T8.5", floor=floor,
                             require_variation=require_variation)


# -- CLI ---------------------------------------------------------------------------------------


def _open(root: Path, model: str, corpus: str) -> TraceReader:
    return TraceReader(root, model, corpus, validate_sizes=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 precision ladder (T8.2/T8.3/T8.5)")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--candidate-model", default=None,
                        help="if the candidate trace is filed under a different model key")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--mode", choices=("ladder", "device-split", "T3.7", "T3.8", "T8.5"),
                        default="ladder",
                        help="ladder applies T8.3's banded reading; the task names apply that "
                             "task's >=99.9%% floor on the worst layer")
    parser.add_argument("--floor", type=float, default=EQUIVALENCE_FLOOR)
    parser.add_argument("--allow-unvaried", action="store_true",
                        help="skip the check that the two legs differ in the knob under test. Only "
                             "for traces collected before capture_variant existed.")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        reference = _open(args.reference_root, args.model, args.corpus)
        candidate = _open(args.candidate_root, args.candidate_model or args.model, args.corpus)
        comparison = compare_traces(reference, candidate, max_tokens=args.max_tokens)
    except (PrecisionError, FormatError, OSError) as exc:
        # A missing or malformed trace is "could not run" (exit 2), never "ran and found
        # drift" (exit 1). A caller scripting the ladder must be able to tell a rung that
        # has not been collected yet from one that failed the gate.
        print(f"Phase 8 COULD NOT RUN: {exc}")
        return 2

    payload: dict[str, Any] = {"comparison": comparison.to_json()}
    if args.mode != "ladder":
        task = "T8.5" if args.mode == "device-split" else args.mode
        try:
            outcome = check_equivalence(comparison, task=task, floor=args.floor,
                                        require_variation=not args.allow_unvaried)
        except PrecisionError as exc:
            # Refusing to judge is "could not run" (exit 2), not a failed gate (exit 1). A caller
            # scripting the three gates has to be able to tell "this knob is a confound" from
            # "this comparison never tested the knob".
            print(f"{task} COULD NOT RUN: {exc}")
            return 2
        payload["equivalence"] = outcome.to_json()
        print(f"{task} {outcome.verdict}")
        passed = outcome.passed
    else:
        gate = interpret(comparison)
        payload["gate"] = gate.to_json()
        print(f"T8.2 mean set agreement {comparison.mean_set_agreement:.6f} over "
              f"{len(comparison.layers)} layers, depth trend rho={comparison.depth_trend():.3f}")
        print(f"T8.3 [{gate.band}] {gate.action}")
        passed = gate.proceed_as_planned

    for note in comparison.notes:
        print(f"  note: {note}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
