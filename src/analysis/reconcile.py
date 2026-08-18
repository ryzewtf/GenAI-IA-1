"""T9.5 — literature reconciliation under one definition (plan §1.7).

The field reports incompatible headline numbers for MoE routing predictability, and §1.7 pins down
why: the numbers measure different quantities. The Mixtral paper's Table 5 figure is a
**consecutive-token expert repetition rate**; the prefetching literature's figure is a **cache-hit
rate at some capacity**. Setting "28%" against "99%" as a contradiction compares a single-step
repetition statistic against a cache-hit rate — and that mismatch is the paper's motivation, not a
puzzle to be resolved by picking one of the two.

What this module does, and refuses to do
----------------------------------------
It places *this study's* measured statistics on the same axis as each literature figure and names
the definitional choice that accounts for the gap. It does not adjudicate, and it does not rescale
one definition into another: an axis this study cannot measure is reported as a gap.

Every literature figure is a named constant carrying the definition it was measured under, its
source, and the plan section that states it. A bare float is not admissible — "28%" without "first
choice, layer 15, consecutive tokens, against a 12.5% random baseline" is exactly the artefact this
module exists to stop propagating. Where §1.7 does not state a figure precisely enough to encode,
the constant is explicitly ``None`` with a reason, and :func:`reconcile` lists it under
``unpinned`` instead of comparing against an approximation.

**O3 is closed** (§1.7): ``core12345/MoE_expert_selection_trace`` contains Llama-4-Maverick,
DeepSeek-R1, Kimi-K2-Thinking and Qwen3-235B — no Mixtral. There is no Mixtral anchor in this
panel, so no comparison here claims one; see :data:`UNAVAILABLE_ANCHORS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .tables import (
    MissingCell,
    ResultSet,
    Table,
    cell_metric,
    normalized_depth,
)

__all__ = [
    "LiteratureFigure",
    "DefinitionalAxis",
    "StudyPoint",
    "Comparison",
    "ReconciliationReport",
    "LITERATURE_FIGURES",
    "DEFINITIONAL_AXES",
    "UNAVAILABLE_ANCHORS",
    "THIS_STUDY_REPETITION_DEFINITION",
    "reconcile",
]


# -- literature figures --------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteratureFigure:
    """One published number, with the definition it was measured under attached.

    ``value`` is a point estimate and ``value_range`` a low/high pair; a source that reports a
    range gets the range, and inventing a midpoint for it would be fabricating a figure nobody
    measured. ``random_baseline`` is the source's own chance level — without it the raw rate is
    uninterpretable across models with different ``n_experts``/``top_k``.

    ``gap_reason`` is set when §1.7 does not pin the figure down. Such a figure is *not* compared
    against; it is reported as a gap.
    """

    name: str
    quantity: str
    definition: str
    source: str
    plan_reference: str
    model: str | None = None
    n_experts: int | None = None
    top_k: int | None = None
    layer: int | None = None
    n_layers: int | None = None
    value: float | None = None
    value_range: tuple[float, float] | None = None
    random_baseline: float | None = None
    random_baseline_gap: str | None = None
    gap_reason: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pinned = self.value is not None or self.value_range is not None
        if pinned and self.gap_reason is not None:
            raise ValueError(
                f"{self.name}: a figure cannot be both pinned and a gap; drop one"
            )
        if not pinned and self.gap_reason is None:
            raise ValueError(
                f"{self.name}: a figure with no value must say why §1.7 does not pin it down. "
                "Approximating it is not an option (T9.5)."
            )
        if pinned and not (self.definition and self.source):
            raise ValueError(f"{self.name}: a pinned figure needs a definition and a source")

    @property
    def is_pinned(self) -> bool:
        return self.value is not None or self.value_range is not None

    @property
    def normalized_depth(self) -> float | None:
        """The figure's position on ℓ/(L−1), the only depth axis comparable across models (§1.4)."""
        if self.layer is None or self.n_layers is None:
            return None
        return normalized_depth(self.layer, self.n_layers)

    def excess_over_random(self) -> tuple[float, float] | None:
        """Range of (rate − chance). ``None`` when the source's chance level is not pinned."""
        if self.random_baseline is None:
            return None
        lo, hi = self.span() or (None, None)
        if lo is None:
            return None
        return (lo - self.random_baseline, hi - self.random_baseline)

    def span(self) -> tuple[float, float] | None:
        if self.value_range is not None:
            return (float(self.value_range[0]), float(self.value_range[1]))
        if self.value is not None:
            return (float(self.value), float(self.value))
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "definition": self.definition,
            "source": self.source,
            "plan_reference": self.plan_reference,
            "model": self.model,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "layer": self.layer,
            "n_layers": self.n_layers,
            "normalized_depth": self.normalized_depth,
            "value": self.value,
            "value_range": list(self.value_range) if self.value_range else None,
            "random_baseline": self.random_baseline,
            "random_baseline_gap": self.random_baseline_gap,
            "is_pinned": self.is_pinned,
            "gap_reason": self.gap_reason,
            "notes": list(self.notes),
        }


_MIXTRAL_SOURCE = (
    "Mixtral of Experts, Table 5 — as restated in plan §1.7 (correction C5). Mixtral-8x7B: "
    "8 experts, top-2. The 12.5% first-choice chance level stated in §1.7 is 1/8, which is what "
    "fixes n_experts=8; top_k=2 is what makes 'first-or-second choice' the whole selected set. "
    "n_layers=32 follows from §1.7's table indexing layers 0, 15 and 31."
)

_MIXTRAL_FIRST_CHOICE_DEFINITION = (
    "Fraction of consecutive token pairs (i, i+1) at one layer for which the SAME expert is the "
    "router's first choice for both tokens. Top-1 identity repetition, single step, per layer. "
    "This is a temporal-locality statistic — the phenomenon this study's F3 feature generalizes — "
    "and NOT an exact-set-match predictability ceiling (§1.7)."
)

_MIXTRAL_FIRST_OR_SECOND_DEFINITION = (
    "Fraction of consecutive token pairs (i, i+1) at one layer for which an expert in token i's "
    "selected top-2 set is also selected for token i+1. Set-membership repetition over the full "
    "top-2 set, single step, per layer (§1.7)."
)

LITERATURE_FIGURES: tuple[LiteratureFigure, ...] = (
    LiteratureFigure(
        name="mixtral_first_choice_layer0",
        quantity="consecutive_token_first_choice_repetition",
        definition=_MIXTRAL_FIRST_CHOICE_DEFINITION,
        source=_MIXTRAL_SOURCE,
        plan_reference="§1.7 table, row 'first choice repeats', column 'Layer 0'",
        model="Mixtral-8x7B",
        n_experts=8,
        top_k=2,
        layer=0,
        n_layers=32,
        value_range=(0.136, 0.149),
        random_baseline=0.125,
        notes=(
            "§1.7 reports a range, not a point; the range is what is encoded. At layer 0 the rate "
            "is barely above chance.",
        ),
    ),
    LiteratureFigure(
        name="mixtral_first_choice_layer15",
        quantity="consecutive_token_first_choice_repetition",
        definition=_MIXTRAL_FIRST_CHOICE_DEFINITION,
        source=_MIXTRAL_SOURCE,
        plan_reference="§1.7 table, row 'first choice repeats', column 'Layer 15'; correction C5",
        model="Mixtral-8x7B",
        n_experts=8,
        top_k=2,
        layer=15,
        n_layers=32,
        value_range=(0.236, 0.284),
        random_baseline=0.125,
        notes=(
            "This is the source of the widely-quoted '28%'. The quoted number is the top of a "
            "23.6-28.4% range at ONE layer, against a 12.5% chance level (§1.7 / C5).",
        ),
    ),
    LiteratureFigure(
        name="mixtral_first_choice_layer31",
        quantity="consecutive_token_first_choice_repetition",
        definition=_MIXTRAL_FIRST_CHOICE_DEFINITION,
        source=_MIXTRAL_SOURCE,
        plan_reference="§1.7 table, row 'first choice repeats', column 'Layer 31'",
        model="Mixtral-8x7B",
        n_experts=8,
        top_k=2,
        layer=31,
        n_layers=32,
        value_range=(0.197, 0.263),
        random_baseline=0.125,
    ),
    LiteratureFigure(
        name="mixtral_first_or_second_choice_layer15",
        quantity="consecutive_token_topk_set_membership_repetition",
        definition=_MIXTRAL_FIRST_OR_SECOND_DEFINITION,
        source=_MIXTRAL_SOURCE,
        plan_reference="§1.7 table, row 'first-or-second-choice repeats', column 'Layer 15'",
        model="Mixtral-8x7B",
        n_experts=8,
        top_k=2,
        layer=15,
        n_layers=32,
        value_range=(0.616, 0.670),
        random_baseline=None,
        random_baseline_gap=(
            "§1.7 gives this row's chance level only as '~46%'. A tilde is not a measured "
            "baseline, so no excess-over-chance is computed for this figure; the raw range is "
            "reported and the baseline is listed as a gap (T9.5)."
        ),
        notes=(
            "Reported because it is the figure most often confused with a set-level "
            "predictability ceiling. It is still a single-step repetition statistic.",
        ),
    ),
    LiteratureFigure(
        name="prefetch_cache_hit_rate",
        quantity="expert_cache_hit_rate_at_fixed_capacity",
        definition=(
            "Fraction of expert activations served from a resident cache of some capacity, under "
            "some prefetch policy, pooled over layers and decode steps. NOT a per-layer, "
            "single-step repetition rate — which is the whole point of §1.7."
        ),
        source=(
            "§1.7 refers to 'systems papers' 99%' as the counterpart to Mixtral's 28%, without "
            "naming a paper, system, cache capacity, model, or corpus."
        ),
        plan_reference="§1.7, closing paragraph; T9.5",
        gap_reason=(
            "§1.7 states the numeral only as a rhetorical contrast. Every parameter needed to "
            "place it on a shared axis is absent: which system, what cache capacity (in experts or "
            "bytes), per-layer or any-layer accounting, which model, which corpus, and whether the "
            "hit rate counts prefetched-and-used or merely resident experts. Encoding 0.99 with "
            "any of those guessed would manufacture a comparison. Left as a gap until the source "
            "papers are cited directly in the write-up."
        ),
        notes=(
            "This study CAN measure the cache-style axis under its own definition — recall@m at a "
            "fixed budget m is a per-layer, single-step cache-hit rate at capacity m — and "
            "reconcile() reports that curve. What it cannot do is compare it against an unpinned "
            "figure.",
        ),
    ),
)
"""Named, sourced, definition-carrying literature figures from plan §1.7. Never bare floats."""


UNAVAILABLE_ANCHORS: tuple[dict[str, str], ...] = (
    {
        "anchor": "Mixtral-8x7B trace anchor via core12345/MoE_expert_selection_trace",
        "status": "unavailable — O3 closed (§1.7)",
        "reason": (
            "That dataset (199 GB, supplement to arXiv 2510.05497) contains Llama-4-Maverick, "
            "DeepSeek-R1, Kimi-K2-Thinking and Qwen3-235B. It has no Mixtral-8x7B, so the v1.0/v2.0 "
            "plan to recompute Mixtral's statistic under this study's definition was never "
            "executable. Do not promise adjudication the panel cannot deliver."
        ),
    },
)
"""Comparisons the panel cannot make. Listed so the write-up does not promise them."""


THIS_STUDY_REPETITION_DEFINITION = (
    "|S_t ∩ S_{t-1}| / k at one MoE layer, averaged over non-document-initial tokens of the test "
    "split, with S taken from the model's own emitted top-k indices (invariant I1). Chance level is "
    "k / n_experts. Computed by src.probes.features.consecutive_repetition_rate; the same statistic "
    "is stored next to F3 in the sweep documents."
)


# -- definitional axes ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefinitionalAxis:
    """One definitional choice that moves a headline number, per §1.7 / T9.5."""

    axis: str
    literature_side: str
    this_study_side: str
    plan_reference: str

    def to_json(self) -> dict[str, str]:
        return {
            "axis": self.axis,
            "literature_side": self.literature_side,
            "this_study_side": self.this_study_side,
            "plan_reference": self.plan_reference,
        }


DEFINITIONAL_AXES: dict[str, DefinitionalAxis] = {
    "top1_vs_topk_set": DefinitionalAxis(
        axis="top-1 identity vs top-k set overlap",
        literature_side=(
            "Mixtral's 'first choice repeats' is top-1 identity; its 'first-or-second-choice' row "
            "is membership in a top-2 set"
        ),
        this_study_side=(
            "|S_t ∩ S_{t-1}| / k over the full emitted top-k set, so the same routing behaviour "
            "yields a different rate purely from k"
        ),
        plan_reference="§1.7, §1.2 (Family A definitions)",
    ),
    "chance_level": DefinitionalAxis(
        axis="chance level moves with k and n_experts",
        literature_side="12.5% for Mixtral's top-1 over 8 experts",
        this_study_side=(
            "k / n_experts, i.e. 12.5% for OLMoE (8/64) and GPT-OSS (4/32) but 6.25% for a "
            "top-8-of-128 model. Raw rates are not comparable across the panel; excess over chance "
            "is the axis on which they are."
        ),
        plan_reference="§1.7 (random columns), §1.5 (matched activated density)",
    ),
    "per_layer_vs_any_layer": DefinitionalAxis(
        axis="per-layer vs pooled-over-layers accounting",
        literature_side=(
            "Mixtral's figures are per layer and vary strongly with depth (13.6-14.9% at layer 0 "
            "against 23.6-28.4% at layer 15); a cache-hit rate is usually pooled over all layers"
        ),
        this_study_side=(
            "Every statistic is reported per MoE layer and aggregated only on normalized depth "
            "ℓ/(L−1), never pooled across depths without saying so"
        ),
        plan_reference="§1.4 (normalized depth), T9.4 (depth), §1.7",
    ),
    "capacity_vs_single_step": DefinitionalAxis(
        axis="cache capacity vs single-step repetition",
        literature_side=(
            "A cache-hit rate is a function of cache capacity and eviction policy over a whole "
            "decode trajectory"
        ),
        this_study_side=(
            "recall@m at a fixed budget m ∈ {k, 2k, 4k} is the capacity-m, per-layer, single-step "
            "version of the same question; the repetition rate is the budget-free single-step "
            "statistic"
        ),
        plan_reference="T9.5, §1.2 (Family A)",
    ),
    "raw_statistic_vs_learned_predictor": DefinitionalAxis(
        axis="raw statistic vs learned predictor",
        literature_side="Mixtral's Table 5 counts repetitions; nothing is fitted",
        this_study_side=(
            "F3 fits a predictor on the same conditioning variable, so F3 ≥ the raw rate is the "
            "expected relationship and the gap between them is the interesting quantity"
        ),
        plan_reference="T7.4, T9.5",
    ),
}
"""The axes §1.7 and T9.5 name. Nothing here is invented: each one is a stated difference."""


# -- this study's side ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyPoint:
    """This study's measurement, aligned to a literature figure by normalized depth."""

    model: str
    layer: int
    n_moe_layers: int
    n_experts: int
    top_k: int
    definition: str
    repetition_rate: float | None
    exact_set_repeat_rate: float | None
    random_baseline: float | None
    depth_alignment_error: float | None
    n_rows: int | None = None
    unavailable_reason: str | None = None

    @property
    def normalized_depth(self) -> float:
        return normalized_depth(self.layer, self.n_moe_layers)

    @property
    def excess_over_random(self) -> float | None:
        if self.repetition_rate is None or self.random_baseline is None:
            return None
        return self.repetition_rate - self.random_baseline

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "layer": self.layer,
            "normalized_depth": self.normalized_depth,
            "n_moe_layers": self.n_moe_layers,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "definition": self.definition,
            "repetition_rate": self.repetition_rate,
            "exact_set_repeat_rate": self.exact_set_repeat_rate,
            "random_baseline": self.random_baseline,
            "excess_over_random": self.excess_over_random,
            "depth_alignment_error": self.depth_alignment_error,
            "n_rows": self.n_rows,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class Comparison:
    """One literature figure, this study's points on the same axis, and the axes that differ."""

    figure: LiteratureFigure
    points: tuple[StudyPoint, ...]
    axes: tuple[DefinitionalAxis, ...]
    caveats: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "figure": self.figure.to_json(),
            "literature_excess_over_random": (
                list(self.figure.excess_over_random())
                if self.figure.excess_over_random() is not None
                else None
            ),
            "this_study": [p.to_json() for p in self.points],
            "definitional_axes": [a.to_json() for a in self.axes],
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class ReconciliationReport:
    """T9.5 output: comparisons under one definition, plus what could not be pinned down."""

    comparisons: tuple[Comparison, ...]
    unpinned: tuple[LiteratureFigure, ...]
    cache_budget_axis: Table
    unavailable_anchors: tuple[dict[str, str], ...] = UNAVAILABLE_ANCHORS
    notes: tuple[str, ...] = ()

    @property
    def gaps(self) -> tuple[dict[str, str | None], ...]:
        """Figures §1.7 does not pin down, and the baselines it leaves approximate."""
        out: list[dict[str, str | None]] = [
            {"figure": f.name, "kind": "figure", "reason": f.gap_reason} for f in self.unpinned
        ]
        for comparison in self.comparisons:
            if comparison.figure.random_baseline_gap:
                out.append(
                    {
                        "figure": comparison.figure.name,
                        "kind": "random_baseline",
                        "reason": comparison.figure.random_baseline_gap,
                    }
                )
        return tuple(out)

    def to_json(self) -> dict[str, Any]:
        return {
            "this_study_definition": THIS_STUDY_REPETITION_DEFINITION,
            "comparisons": [c.to_json() for c in self.comparisons],
            "unpinned_figures": [f.to_json() for f in self.unpinned],
            "gaps": [dict(g) for g in self.gaps],
            "unavailable_anchors": [dict(a) for a in self.unavailable_anchors],
            "cache_budget_axis": self.cache_budget_axis.to_json(),
            "notes": list(self.notes),
        }

    def to_table(self) -> Table:
        columns = [
            "figure",
            "quantity",
            "lit_low",
            "lit_high",
            "lit_random",
            "lit_excess_low",
            "lit_excess_high",
            "lit_normalized_depth",
            "model",
            "our_layer",
            "our_normalized_depth",
            "depth_alignment_error",
            "our_repetition_rate",
            "our_exact_set_repeat_rate",
            "our_random",
            "our_excess",
            "definitional_axes",
        ]
        rows: list[list[Any]] = []
        for comparison in self.comparisons:
            figure = comparison.figure
            span = figure.span() or (None, None)
            excess = figure.excess_over_random() or (None, None)
            axes = ", ".join(a.axis for a in comparison.axes)
            for point in comparison.points:
                rows.append(
                    [
                        figure.name,
                        figure.quantity,
                        span[0],
                        span[1],
                        figure.random_baseline
                        if figure.random_baseline is not None
                        else MissingCell("metric_absent", figure.random_baseline_gap or ""),
                        excess[0]
                        if excess[0] is not None
                        else MissingCell("metric_absent", "chance level not pinned in §1.7"),
                        excess[1]
                        if excess[1] is not None
                        else MissingCell("metric_absent", "chance level not pinned in §1.7"),
                        figure.normalized_depth,
                        point.model,
                        point.layer,
                        point.normalized_depth,
                        point.depth_alignment_error,
                        point.repetition_rate
                        if point.repetition_rate is not None
                        else MissingCell("metric_absent", point.unavailable_reason or ""),
                        point.exact_set_repeat_rate
                        if point.exact_set_repeat_rate is not None
                        else MissingCell("metric_absent", point.unavailable_reason or ""),
                        point.random_baseline,
                        point.excess_over_random
                        if point.excess_over_random is not None
                        else MissingCell("metric_absent", point.unavailable_reason or ""),
                        axes,
                    ]
                )
        notes = [
            f"this study's definition: {THIS_STUDY_REPETITION_DEFINITION}",
            "Rows are aligned to each literature figure by normalized depth ℓ/(L−1); "
            "depth_alignment_error is how far the nearest available layer sits from the figure's "
            "depth (§1.4).",
            *self.notes,
        ]
        notes.extend(f"GAP — {g['figure']} ({g['kind']}): {g['reason']}" for g in self.gaps)
        notes.extend(f"UNAVAILABLE — {a['anchor']}: {a['reason']}" for a in self.unavailable_anchors)
        return Table(
            name="T9.5 reconciliation — literature figures vs this study, one definition at a time",
            columns=columns,
            rows=rows,
            notes=notes,
            meta={"gaps": [dict(g) for g in self.gaps]},
        )


def _repetition_from_resultset(resultset: ResultSet) -> dict[str, dict[int, Mapping[str, Any]]]:
    """Pull ``mixtral_table5_statistic`` out of the F3 cells the sweep already wrote (T7.4)."""
    out: dict[str, dict[int, Mapping[str, Any]]] = {}
    for name in resultset.model_names:
        model = resultset.models[name]
        per_layer: dict[int, Mapping[str, Any]] = {}
        for layer, record in (model.cells.get("F3") or {}).items():
            stat = record.get("mixtral_table5_statistic")
            if isinstance(stat, Mapping):
                per_layer[int(layer)] = stat
        if per_layer:
            out[name] = per_layer
    return out


def _nearest_layer(
    available: Iterable[int], n_moe_layers: int, target_depth: float
) -> tuple[int | None, float | None]:
    best: tuple[int | None, float | None] = (None, None)
    for layer in sorted(available):
        error = abs(normalized_depth(layer, n_moe_layers) - target_depth)
        if best[1] is None or error < best[1]:
            best = (layer, error)
    return best


def _cache_budget_table(resultset: ResultSet, features: Sequence[str]) -> Table:
    """recall@{k,2k,4k} per (model, feature) — the cache-style axis, under our definition only.

    A cache-hit rate at capacity m is recall@m per layer and step. This is the axis on which a
    prefetching system's headline number *would* be comparable; §1.7 pins no such figure, so this
    table stands alone and claims no comparison.
    """
    metrics = ("recall@k", "recall@2k", "recall@4k", "set_agreement@k", "exact_match")
    columns = ["model", "top_k", "n_experts", "feature", "layer", *metrics]
    rows: list[list[Any]] = []
    for name in resultset.model_names:
        model = resultset.models[name]
        for feature in features:
            if feature not in model.cells:
                continue
            for layer in model.expected_layers():
                values = [cell_metric(model, feature, layer, m) for m in metrics]
                if all(isinstance(v, MissingCell) for v in values):
                    continue
                rows.append([name, model.top_k, model.n_experts, feature, layer, *values])
    return Table(
        name="T9.5 cache-style budget axis — recall@m under this study's definition",
        columns=columns,
        rows=rows,
        notes=[
            "recall@m at budget m is the per-layer, single-step form of a cache-hit rate at "
            "capacity m (T9.5). Budgets are k, 2k and 4k, so the absolute capacity differs per "
            "model: 4/8/16 for GPT-OSS and 8/16/32 for OLMoE at matched 12.5% density (§1.5).",
            "No literature figure is attached: §1.7 does not pin down a cache-hit rate precisely "
            "enough to compare (see the report's gaps).",
        ],
    )


def reconcile(
    resultset: ResultSet,
    repetition_stats: Mapping[str, Mapping[Any, Mapping[str, Any]]] | None = None,
    *,
    figures: Sequence[LiteratureFigure] = LITERATURE_FIGURES,
    models: Sequence[str] | None = None,
    features: Sequence[str] = ("F1", "F3", "F4"),
) -> ReconciliationReport:
    """Place this study's numbers on the same axis as each §1.7 figure — plan T9.5.

    ``repetition_stats`` is ``model -> {layer -> stat}`` where ``stat`` is what
    :func:`src.probes.features.consecutive_repetition_rate` returns. It defaults to the copies the
    sweep already stored next to F3, so the reconciliation reads the same numbers the results
    directory reports.

    Alignment is by **normalized depth**: Mixtral's layer 15 of 32 sits at 15/31, and the nearest
    layer of a 16-layer or 48-layer model is chosen with the residual reported. Absolute layer
    indices would compare unrelated positions in the stack (§1.4).

    A figure §1.7 does not pin down is **not** compared against. It lands in ``unpinned`` and in
    ``gaps``, with the reason. That is the whole point of the exercise: the field's apparent
    contradiction is substantially a metric mismatch, and closing it with an approximation would
    reproduce the error the section documents.
    """
    stats = repetition_stats if repetition_stats is not None else _repetition_from_resultset(resultset)
    names = list(models) if models is not None else resultset.model_names
    resultset.require(names)

    comparisons: list[Comparison] = []
    unpinned: list[LiteratureFigure] = []

    for figure in figures:
        if not figure.is_pinned:
            unpinned.append(figure)
            continue

        target = figure.normalized_depth
        points: list[StudyPoint] = []
        for name in names:
            model = resultset.models[name]
            per_layer = {int(k): v for k, v in (stats.get(name) or {}).items()}
            if target is None or not per_layer:
                points.append(
                    StudyPoint(
                        model=name,
                        layer=-1,
                        n_moe_layers=model.n_moe_layers,
                        n_experts=model.n_experts,
                        top_k=model.top_k,
                        definition=THIS_STUDY_REPETITION_DEFINITION,
                        repetition_rate=None,
                        exact_set_repeat_rate=None,
                        random_baseline=None,
                        depth_alignment_error=None,
                        unavailable_reason=(
                            f"no consecutive-repetition statistic available for {name}"
                            if per_layer == {}
                            else f"{figure.name} carries no layer/depth to align to"
                        ),
                    )
                )
                continue
            layer, error = _nearest_layer(per_layer, model.n_moe_layers, target)
            stat = per_layer[int(layer)]
            points.append(
                StudyPoint(
                    model=name,
                    layer=int(layer),
                    n_moe_layers=model.n_moe_layers,
                    n_experts=model.n_experts,
                    top_k=model.top_k,
                    definition=THIS_STUDY_REPETITION_DEFINITION,
                    repetition_rate=(
                        float(stat["repetition_rate"])
                        if stat.get("repetition_rate") is not None
                        else None
                    ),
                    exact_set_repeat_rate=(
                        float(stat["exact_set_repeat_rate"])
                        if stat.get("exact_set_repeat_rate") is not None
                        else None
                    ),
                    random_baseline=(
                        float(stat["random_baseline"])
                        if stat.get("random_baseline") is not None
                        else float(model.top_k) / float(model.n_experts)
                    ),
                    depth_alignment_error=error,
                    n_rows=int(stat["n_rows"]) if stat.get("n_rows") is not None else None,
                )
            )

        axes = [
            DEFINITIONAL_AXES["top1_vs_topk_set"],
            DEFINITIONAL_AXES["chance_level"],
            DEFINITIONAL_AXES["per_layer_vs_any_layer"],
            DEFINITIONAL_AXES["raw_statistic_vs_learned_predictor"],
        ]
        caveats = [
            "The literature figure and this study's statistic are different quantities; the axes "
            "column names every difference. The comparison is legitimate only after they are read "
            "(§1.7).",
        ]
        if figure.quantity == "consecutive_token_first_choice_repetition":
            caveats.append(
                "The figure is top-1 identity repetition; this study reports top-k set overlap, "
                "which is a different statistic on the same behaviour. This study does not measure "
                "a top-1-only repetition rate, so no like-for-like value is emitted here."
            )
        if figure.random_baseline is None:
            caveats.append(
                "This figure's chance level is not pinned in §1.7, so excess over chance is left "
                "empty on the literature side."
            )
        comparisons.append(
            Comparison(
                figure=figure,
                points=tuple(points),
                axes=tuple(axes),
                caveats=tuple(caveats),
            )
        )

    return ReconciliationReport(
        comparisons=tuple(comparisons),
        unpinned=tuple(unpinned),
        cache_budget_axis=_cache_budget_table(resultset, features),
        notes=(
            "§1.7: the field's '28% vs 99%' contradiction is substantially a metric mismatch — a "
            "consecutive-token repetition rate against a cache-hit rate. This report states both "
            "under one definition rather than adjudicating between them.",
            "O3 is closed: no Mixtral anchor exists for this panel (see unavailable_anchors). The "
            "claim the panel supports is Pair G's: does routing predictability scale with expert "
            "granularity at matched activated density (§1.5, §1.7)?",
        ),
    )
