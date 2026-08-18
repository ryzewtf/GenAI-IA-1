"""Phase 9 reporting — result files in, tables out (plan T9.1–T9.4, T9.6).

This module is the *only* place a number crosses from ``results/*.json`` into the write-up, and it
is deliberately dumb: every value it emits is read out of a result document written by
:func:`src.probes.train.sweep`. It computes reductions over layers and differences between models;
it never estimates a metric, never fills a hole, and never rounds a value it renders.

Three policies it enforces rather than documents
------------------------------------------------
**I2 — one run config or nothing.** ``load_results`` refuses to build a :class:`ResultSet` from
documents whose ``run_config_sha256`` disagrees. Merging traces from two run configs (or two
platforms, I3) silently averages two experiments; there is no "mostly the same run" and no warning
path. The same refusal covers ``n_experts`` / ``top_k`` / ``n_moe_layers`` disagreeing for one
model, and ``logit_tensor_used`` is surfaced per model because a different selection tensor is a
different label stream (I13, §1.6).

**I8 — ``Î`` is signed, everywhere.** Negative cells are a required output of the study (§1.2), so
:func:`negative_mi_cells` reports the count *and* the full listing, and no function in this module
takes a "clamp" or "floor" argument. ``load_results`` additionally re-derives ``H − CE`` from each
document and refuses the document if the stored ``mi_bits`` disagrees — an upstream clamp would show
up here as a mismatch rather than as a plausible table.

**A blank is not a zero.** A cell that does not exist — an ``UndefinedFeature`` skip (F2 at layer 0),
or a layer a killed session never reached — becomes a :class:`MissingCell` and renders as
:data:`MISSING_MARKER`. Every renderer keeps that marker distinct from ``0.0``, and every table
carries the count. Reading a truncated sweep as a floor of zeros is the failure mode that would make
a killed session look like an absence of routing information.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "MISSING_MARKER",
    "MissingCell",
    "Table",
    "CellRef",
    "ModelResults",
    "ResultSet",
    "NegativeMICells",
    "ShapeExpectation",
    "ShapeCheckReport",
    "load_results",
    "primary_table",
    "negative_mi_cells",
    "pair_table",
    "confound_table",
    "expected_shape_check",
    "normalized_depth",
    "METRICS",
    "LAYER_REDUCTIONS",
]

MISSING_MARKER = "--"
"""Rendered in place of a cell that was never computed. Never ``0``, never blank."""

_MI_CONSISTENCY_TOL = 1e-9


# -- missing cells and the table container ------------------------------------------------------


@dataclass(frozen=True)
class MissingCell:
    """A cell with no value, and why.

    ``kind`` is one of ``"not_reached"`` (the layer is absent from the document — a killed session,
    plan S.3), ``"skipped"`` (the sweep recorded ``UndefinedFeature``, e.g. F2 at layer 0),
    ``"feature_absent"`` (no document for this (model, feature)), ``"metric_absent"`` (the cell ran
    but does not carry this metric) or ``"no_layers"`` (a reduction had nothing to reduce).
    """

    kind: str
    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return MISSING_MARKER

    def to_json(self) -> dict[str, str]:
        return {"missing": self.kind, "reason": self.reason}


def _is_missing(value: Any) -> bool:
    return isinstance(value, MissingCell)


def _render_scalar(value: Any) -> str:
    """Text for one cell. Floats use ``repr``, which round-trips exactly — no truncation."""
    if _is_missing(value):
        return MISSING_MARKER
    if value is None:
        return MISSING_MARKER
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_render_scalar(v) for v in value)
    return str(value)


@dataclass
class Table:
    """Column names, rows, and three renderers.

    Values stay in the rows as Python objects (``float``, ``str``, :class:`MissingCell`); rendering
    happens at the edge so that ``to_csv`` and ``to_json`` cannot disagree with ``to_markdown``
    about what the number was.
    """

    name: str
    columns: list[str]
    rows: list[list[Any]]
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for i, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"table {self.name!r}: row {i} has {len(row)} cells against "
                    f"{len(self.columns)} columns"
                )

    @property
    def n_missing(self) -> int:
        """Count of cells with no value. Reported next to every table (a blank is not a zero)."""
        return sum(1 for row in self.rows for cell in row if _is_missing(cell))

    def missing_cells(self) -> list[dict[str, Any]]:
        """Where the holes are, addressed by row index and column name."""
        out: list[dict[str, Any]] = []
        for r, row in enumerate(self.rows):
            for c, cell in enumerate(row):
                if _is_missing(cell):
                    out.append(
                        {
                            "row": r,
                            "column": self.columns[c],
                            "kind": cell.kind,
                            "reason": cell.reason,
                        }
                    )
        return out

    def column(self, name: str) -> list[Any]:
        return [row[self.columns.index(name)] for row in self.rows]

    def to_markdown(self) -> str:
        header = "| " + " | ".join(self.columns) + " |"
        rule = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join(_render_scalar(c).replace("|", "\\|") for c in row) + " |"
            for row in self.rows
        ]
        lines = [f"**{self.name}**", "", header, rule, *body, ""]
        lines.append(f"Missing cells ({MISSING_MARKER}): {self.n_missing}")
        lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.columns)
        for row in self.rows:
            writer.writerow([_render_scalar(c) for c in row])
        return buffer.getvalue()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "rows": [
                [cell.to_json() if _is_missing(cell) else cell for cell in row]
                for row in self.rows
            ],
            "n_missing": self.n_missing,
            "missing_cells": self.missing_cells(),
            "notes": list(self.notes),
            "meta": self.meta,
            "missing_marker": MISSING_MARKER,
        }


# -- the loaded result set ----------------------------------------------------------------------


@dataclass(frozen=True)
class CellRef:
    """One (model, feature, layer) address, used by the negative-``Î`` listing and the checks."""

    model: str
    feature: str
    layer: int
    model_layer: int | None = None


@dataclass
class ModelResults:
    """Every (feature, layer) cell for one model, plus the identity fields that gate merging."""

    model: str
    run_config_sha256: str
    logit_tensor_used: str | None
    n_experts: int
    top_k: int
    n_moe_layers: int
    entropy_estimator: str | None = None
    shard_ids: list[int] = field(default_factory=list)
    cells: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return sorted(self.cells)

    def expected_layers(self) -> list[int]:
        """``0 .. n_moe_layers-1``.

        The honest denominator: a layer a killed session never reached is a hole in the results,
        not a shorter model. DeepSeek's trace layer i is model layer i+1 (`moe_layer_offset`), so
        the trace index is the axis here and ``model_layer`` is carried alongside for reference.
        """
        return list(range(self.n_moe_layers))

    def cell(self, feature: str, layer: int) -> dict[str, Any] | MissingCell:
        if feature not in self.cells:
            return MissingCell("feature_absent", f"no {feature} document for {self.model}")
        record = self.cells[feature].get(int(layer))
        if record is None:
            return MissingCell(
                "not_reached", f"{self.model}/{feature}/layer {layer} absent from the document"
            )
        if record.get("status") != "ok":
            return MissingCell(
                "skipped", str(record.get("reason") or record.get("status") or "skipped")
            )
        return record


@dataclass
class ResultSet:
    """All models that share one ``run_config_sha256``."""

    models: dict[str, ModelResults] = field(default_factory=dict)
    run_config_sha256: str | None = None
    out_dir: str | None = None

    def __iter__(self):
        return iter(self.models.values())

    @property
    def model_names(self) -> list[str]:
        return sorted(self.models)

    def features(self) -> list[str]:
        seen: set[str] = set()
        for m in self.models.values():
            seen.update(m.features)
        return sorted(seen)

    def logit_tensors(self) -> dict[str, str | None]:
        """``logit_tensor_used`` per model — surfaced because a differing value is a differing
        experiment (§1.6/I13), and no reduction in this module can detect that on its own."""
        return {name: m.logit_tensor_used for name, m in sorted(self.models.items())}

    def require(self, models: Sequence[str]) -> None:
        absent = [m for m in models if m not in self.models]
        if absent:
            raise KeyError(
                f"result set has no results for {absent}; present: {self.model_names}. A pair "
                "table over an absent member would silently become a one-model table."
            )


def _identity(doc: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(doc["n_experts"]), int(doc["top_k"]), int(doc["n_moe_layers"]))


def _check_unclamped(doc: Mapping[str, Any], model: str, feature: str, layer: str) -> None:
    """Refuse a document whose stored ``mi_bits`` is not ``H − CE`` (invariant I8).

    A clamp applied anywhere upstream shows up here as an arithmetic inconsistency instead of as a
    plausible non-negative table. TASKS.md I8 is asserted at the source in T6.3; this is the same
    assertion at the reporting boundary, because that is where a clamp would actually pay off.
    """
    family_b = ((doc.get("layers") or {}).get(layer) or {}).get("metrics", {}).get("family_b")
    if not family_b:
        return
    stored = family_b.get("mi_bits")
    if stored is None:
        return
    derived = float(family_b["entropy_bits"]) - float(family_b["cross_entropy_bits"])
    if math.isnan(derived) and math.isnan(float(stored)):
        return
    if abs(float(stored) - derived) > _MI_CONSISTENCY_TOL:
        raise ValueError(
            f"{model}/{feature}/layer {layer}: stored mi_bits {stored!r} is not "
            f"entropy_bits - cross_entropy_bits ({derived!r}). Î is reported signed and is never "
            "clamped (plan §1.2, invariant I8); this document has been altered."
        )


def load_results(
    out_dir: Path | str,
    *,
    pattern: str = "*__*.json",
    require_uniform_run_config: bool = True,
) -> ResultSet:
    """Read every ``{model}__{feature}.json`` under ``out_dir`` into one :class:`ResultSet`.

    Refuses, as hard errors:

    * two documents with different ``run_config_sha256`` (invariant I2, and I3 for the platform
      half of the run config). ``require_uniform_run_config=False`` relaxes this **only** across
      distinct models — within one model a mismatch is always fatal, because those documents
      describe shards of one trace.
    * ``n_experts`` / ``top_k`` / ``n_moe_layers`` disagreeing between two documents for the same
      model. Those come from the trace manifest, so a disagreement means two different traces were
      swept into one results directory.
    """
    out_dir = Path(out_dir)
    resultset = ResultSet(out_dir=str(out_dir))
    config_sources: dict[str, str] = {}

    for path in sorted(out_dir.glob(pattern)):
        doc = json.loads(path.read_text(encoding="utf-8"))
        model = str(doc["model"])
        feature = str(doc["feature"])
        sha = str(doc["run_config_sha256"])

        for other_sha, source in config_sources.items():
            if other_sha == sha:
                continue
            same_model = source.split("::", 1)[0] == model
            if same_model or require_uniform_run_config:
                raise ValueError(
                    f"refusing to combine results from two run configs: {source} has "
                    f"run_config_sha256 {other_sha} and {model}::{feature} ({path.name}) has "
                    f"{sha}. Merging mismatched run configs is a hard error, not a warning "
                    "(invariant I2; the platform is part of the run config, I3)."
                )
        config_sources[sha] = f"{model}::{feature}"

        existing = resultset.models.get(model)
        if existing is None:
            existing = ModelResults(
                model=model,
                run_config_sha256=sha,
                logit_tensor_used=doc.get("logit_tensor_used"),
                n_experts=int(doc["n_experts"]),
                top_k=int(doc["top_k"]),
                n_moe_layers=int(doc["n_moe_layers"]),
                entropy_estimator=doc.get("entropy_estimator"),
                shard_ids=list(doc.get("shard_ids") or []),
            )
            resultset.models[model] = existing
        else:
            if _identity(doc) != (existing.n_experts, existing.top_k, existing.n_moe_layers):
                raise ValueError(
                    f"{model}: {feature} declares (n_experts, top_k, n_moe_layers)="
                    f"{_identity(doc)} but earlier documents declare "
                    f"({existing.n_experts}, {existing.top_k}, {existing.n_moe_layers}). Two "
                    "traces have been swept into one results directory."
                )
            if doc.get("logit_tensor_used") != existing.logit_tensor_used:
                raise ValueError(
                    f"{model}: logit_tensor_used differs between documents "
                    f"({existing.logit_tensor_used!r} vs {doc.get('logit_tensor_used')!r}). The "
                    "selection tensor names the label stream (§1.6/I13) — these are different "
                    "experiments."
                )

        layers: dict[int, dict[str, Any]] = {}
        for key, record in (doc.get("layers") or {}).items():
            _check_unclamped(doc, model, feature, key)
            layers[int(key)] = record
        existing.cells[feature] = layers
        existing.paths[feature] = str(path)

    if config_sources:
        shas = set(config_sources)
        resultset.run_config_sha256 = shas.pop() if len(shas) == 1 else None
    return resultset


# -- metric extraction --------------------------------------------------------------------------

# Metric name -> (family, key template). "{k}", "{2k}", "{4k}" resolve against the model's top_k,
# which is why extraction needs the model and not just the record: recall@k on OLMoE is recall@8
# and on GPT-OSS is recall@4, and reporting them in one column without that substitution would
# compare a 4-expert budget against an 8-expert one (plan T9.5).
METRICS: dict[str, tuple[str, str]] = {
    "ce_normalized": ("family_b", "ce_normalized"),
    "mi_bits": ("family_b", "mi_bits"),
    "mi_ratio": ("family_b", "ratio"),
    "entropy_bits": ("family_b", "entropy_bits"),
    "cross_entropy_bits": ("family_b", "cross_entropy_bits"),
    "entropy_normalized": ("family_b", "entropy_normalized"),
    "set_agreement@k": ("family_a", "set_agreement@k"),
    "exact_match": ("family_a", "exact_match"),
    "recall@k": ("family_a", "recall@{k}"),
    "recall@2k": ("family_a", "recall@{2k}"),
    "recall@4k": ("family_a", "recall@{4k}"),
}
"""Reportable metric names. ``mi_ratio`` is §1.2's ``Î/H``; it is signed, like ``mi_bits``."""

LAYER_REDUCTIONS: tuple[str, ...] = ("per_layer", "mean", "per_normalized_depth")


def _metric_from_record(record: Mapping[str, Any], metric: str, top_k: int) -> Any:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; have {sorted(METRICS)}")
    family, template = METRICS[metric]
    key = template.format(**{"k": top_k, "2k": 2 * top_k, "4k": 4 * top_k})
    payload = (record.get("metrics") or {}).get(family)
    if not payload or key not in payload:
        return MissingCell("metric_absent", f"{family}.{key} not present in this cell")
    value = payload[key]
    return float(value) if value is not None else MissingCell("metric_absent", f"{key} is null")


def cell_metric(
    model_results: ModelResults, feature: str, layer: int, metric: str
) -> Any:
    """One metric from one (feature, layer) cell, or a :class:`MissingCell`. Never clamped."""
    record = model_results.cell(feature, layer)
    if _is_missing(record):
        return record
    return _metric_from_record(record, metric, model_results.top_k)


def normalized_depth(layer: int, n_moe_layers: int) -> float:
    """``ℓ / (L − 1)`` ∈ [0, 1] — plan §1.4.

    The **only** cross-model-comparable depth axis in this panel. Absolute layer index is not:
    layer 15 is mid-stack in OLMoE's 16 MoE layers and early in Qwen3's 48, so a "layer 15" column
    spanning both models compares two unrelated positions in the network. A single-MoE-layer model
    has no interior depth to normalize, so it maps to 0.0 rather than dividing by zero.
    """
    if n_moe_layers <= 1:
        return 0.0
    return float(layer) / float(n_moe_layers - 1)


def _depth_bin(depth: float, n_bins: int) -> tuple[int, str]:
    """Bin a normalized depth onto a shared grid.

    Binning is what actually puts a 16-layer and a 48-layer model on one axis: their normalized
    depths are both in [0, 1] but never coincide exactly (1/15 vs 1/47), so a join on the raw
    value would produce two disjoint sets of rows.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    idx = min(int(depth * n_bins), n_bins - 1)
    lo, hi = idx / n_bins, (idx + 1) / n_bins
    closer = "]" if idx == n_bins - 1 else ")"
    return idx, f"[{lo:.2f},{hi:.2f}{closer}"


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


# -- T9.1 primary table -------------------------------------------------------------------------


def primary_table(
    resultset: ResultSet,
    *,
    metric: str = "ce_normalized",
    layer_reduction: str = "mean",
    features: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    depth_bins: int = 5,
) -> Table:
    """Models × features for one metric — plan T9.1.

    ``layer_reduction``:

    * ``"mean"`` — one row per model, averaged over the layers that have a value. Pooled per-model
      averages are summaries; T9.4's domain-stratified breakdown is the controlled cross-model
      comparison (plan T9.1), so this reduction is a headline, not the argument.
    * ``"per_layer"`` — one row per (model, trace layer). Comparable **within** a model only.
    * ``"per_normalized_depth"`` — one row per (model, depth bin) on ``ℓ/(L−1)``. The only reduction
      whose rows line up across models of different depth (§1.4); see :func:`normalized_depth`.

    Layers with no value are never imputed: they are counted, listed in ``meta["coverage"]``, and
    a cell with no layers at all renders as :data:`MISSING_MARKER`.
    """
    if layer_reduction not in LAYER_REDUCTIONS:
        raise ValueError(
            f"unknown layer_reduction {layer_reduction!r}; have {list(LAYER_REDUCTIONS)}"
        )
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; have {sorted(METRICS)}")

    names = list(models) if models is not None else resultset.model_names
    resultset.require(names)
    feature_cols = list(features) if features is not None else resultset.features()

    notes = [
        f"metric = {metric}; layer_reduction = {layer_reduction}",
        "Î and Î/H are reported signed; negative cells are listed by negative_mi_cells() "
        "(plan §1.2, invariant I8).",
        "aux_loss_coef (config-declared), the measured load-balance index and checkpoint_status "
        "are in confound_table() (plan T9.1/T9.4) — they need configs/models.yaml, which this "
        "table does not read.",
        f"logit_tensor_used per model: {resultset.logit_tensors()}",
    ]
    meta: dict[str, Any] = {
        "metric": metric,
        "layer_reduction": layer_reduction,
        "coverage": {},
        "partial_cells": [],
    }

    if layer_reduction == "mean":
        columns = ["model", "n_experts", "top_k", "n_moe_layers", *feature_cols]
    elif layer_reduction == "per_layer":
        columns = ["model", "layer", "model_layer", *feature_cols]
    else:
        columns = ["model", "depth_bin", "normalized_depth_mid", "n_layers_in_bin", *feature_cols]

    rows: list[list[Any]] = []
    for name in names:
        model = resultset.models[name]
        expected = model.expected_layers()

        if layer_reduction == "mean":
            row: list[Any] = [name, model.n_experts, model.top_k, model.n_moe_layers]
            for feature in feature_cols:
                values, holes = [], []
                for layer in expected:
                    value = cell_metric(model, feature, layer, metric)
                    (holes if _is_missing(value) else values).append(
                        value if _is_missing(value) else float(value)
                    )
                meta["coverage"][f"{name}::{feature}"] = {
                    "n_used": len(values),
                    "n_expected": len(expected),
                }
                if not values:
                    row.append(
                        MissingCell(
                            "no_layers",
                            f"no layer of {name}/{feature} carries {metric} "
                            f"({len(holes)} missing of {len(expected)})",
                        )
                    )
                    continue
                if holes:
                    meta["partial_cells"].append(
                        {
                            "model": name,
                            "feature": feature,
                            "n_used": len(values),
                            "n_expected": len(expected),
                            "kinds": sorted({h.kind for h in holes}),
                        }
                    )
                row.append(_mean(values))
            rows.append(row)

        elif layer_reduction == "per_layer":
            for layer in expected:
                model_layer: Any = MissingCell("not_reached", "no cell reached for this layer")
                for feature in model.features:
                    record = model.cells[feature].get(layer)
                    if record and record.get("model_layer") is not None:
                        model_layer = record["model_layer"]
                        break
                rows.append(
                    [name, layer, model_layer]
                    + [cell_metric(model, feature, layer, metric) for feature in feature_cols]
                )

        else:
            bins: dict[int, list[int]] = {}
            for layer in expected:
                idx, _ = _depth_bin(normalized_depth(layer, model.n_moe_layers), depth_bins)
                bins.setdefault(idx, []).append(layer)
            for idx in range(depth_bins):
                layers_in_bin = bins.get(idx, [])
                _, label = _depth_bin((idx + 0.5) / depth_bins, depth_bins)
                row = [name, label, (idx + 0.5) / depth_bins, len(layers_in_bin)]
                for feature in feature_cols:
                    values = [
                        float(v)
                        for v in (
                            cell_metric(model, feature, layer, metric) for layer in layers_in_bin
                        )
                        if not _is_missing(v)
                    ]
                    row.append(
                        _mean(values)
                        if values
                        else MissingCell(
                            "no_layers",
                            f"{name}/{feature} has no value in depth bin {label}",
                        )
                    )
                rows.append(row)

    if layer_reduction == "per_normalized_depth":
        notes.append(
            "Rows are keyed by normalized depth ℓ/(L−1) binned onto a shared grid, so a "
            "16-MoE-layer and a 48-MoE-layer model occupy the same rows (§1.4). n_layers_in_bin "
            "differs between them by construction and is reported."
        )

    return Table(
        name=f"T9.1 primary — {metric} ({layer_reduction})",
        columns=columns,
        rows=rows,
        notes=notes,
        meta=meta,
    )


# -- T9.1 negative Î cells ----------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeMICells:
    """Every cell with ``Î < 0``, as measured — plan §1.2, invariant I8.

    **There is no clamped view of this object and no way to ask for one.** It is frozen, it holds
    the measured ``mi_bits``, and this module exposes no floor, clip or ``abs`` on that path. A
    negative ``Î`` reads as *"no information detected beyond the marginal, and the probe is
    overfitting"*; it is a required output of the study, and the count is what disarms a reviewer
    attacking estimator bias. ``n_cells_examined`` is carried so the count has a denominator.
    """

    cells: tuple[dict[str, Any], ...]
    n_cells_examined: int
    by_model: dict[str, int]
    by_feature: dict[str, int]
    n_missing_cells: int = 0

    @property
    def count(self) -> int:
        return len(self.cells)

    def to_json(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "n_cells_examined": self.n_cells_examined,
            "n_missing_cells": self.n_missing_cells,
            "by_model": dict(self.by_model),
            "by_feature": dict(self.by_feature),
            "cells": [dict(c) for c in self.cells],
            "policy": "reported as measured; never clamped (plan §1.2, invariant I8)",
        }

    def to_table(self) -> Table:
        columns = [
            "model",
            "feature",
            "layer",
            "model_layer",
            "mi_bits",
            "mi_ratio",
            "entropy_bits",
            "cross_entropy_bits",
        ]
        rows = [[cell.get(c, MissingCell("metric_absent", c)) for c in columns] for cell in self.cells]
        return Table(
            name="T9.1 negative Î cells (reported as measured)",
            columns=columns,
            rows=rows,
            notes=[
                f"{self.count} of {self.n_cells_examined} evaluated cells have Î < 0.",
                f"by model: {dict(self.by_model)}; by feature: {dict(self.by_feature)}",
                "Negative Î means no information was detected beyond the marginal and the probe "
                "is overfitting (plan §1.2). Values are not clamped and not omitted.",
            ],
            meta=self.to_json(),
        )


def negative_mi_cells(resultset: ResultSet) -> NegativeMICells:
    """Count **and** full listing of cells with ``Î < 0`` — plan T9.1, invariant I8.

    This is a reported result, not a diagnostic to be hidden, so the listing is complete and the
    denominator travels with it. The returned object cannot be asked for a clamped or filtered
    view: see :class:`NegativeMICells`.
    """
    cells: list[dict[str, Any]] = []
    by_model: dict[str, int] = {}
    by_feature: dict[str, int] = {}
    examined = 0
    n_missing = 0

    for name in resultset.model_names:
        model = resultset.models[name]
        for feature in model.features:
            for layer in model.expected_layers():
                record = model.cell(feature, layer)
                if _is_missing(record):
                    n_missing += 1
                    continue
                family_b = (record.get("metrics") or {}).get("family_b")
                if not family_b or family_b.get("mi_bits") is None:
                    n_missing += 1
                    continue
                examined += 1
                mi = float(family_b["mi_bits"])
                if not (mi < 0.0):
                    continue
                cells.append(
                    {
                        "model": name,
                        "feature": feature,
                        "layer": int(layer),
                        "model_layer": record.get("model_layer"),
                        "mi_bits": mi,
                        "mi_ratio": (
                            float(family_b["ratio"]) if family_b.get("ratio") is not None else None
                        ),
                        "entropy_bits": float(family_b["entropy_bits"]),
                        "cross_entropy_bits": float(family_b["cross_entropy_bits"]),
                        "n_experts": model.n_experts,
                        "estimator": family_b.get("estimator"),
                    }
                )
                by_model[name] = by_model.get(name, 0) + 1
                by_feature[feature] = by_feature.get(feature, 0) + 1

    return NegativeMICells(
        cells=tuple(cells),
        n_cells_examined=examined,
        by_model=by_model,
        by_feature=by_feature,
        n_missing_cells=n_missing,
    )


# -- T9.2 / T9.3 pair tables --------------------------------------------------------------------


def pair_table(
    resultset: ResultSet,
    pair_name: str,
    models_config: Mapping[str, Any],
    *,
    metric: str = "mi_ratio",
    layer_reduction: str = "mean",
    depth_bins: int = 5,
) -> Table:
    """One pair from ``configs/models.yaml``, with its framing attached to every row — T9.2/T9.3.

    ``holds_fixed``, ``not_matched`` and ``framing`` are **columns**, not a caption. §1.5 is
    explicit that no pair except Pair A is controlled in the strict sense, so the qualification has
    to travel in the same row as the delta: a reader who copies the numbers out of this table
    copies the framing with them. Dropping the framing to save a column would reintroduce exactly
    the "controlled pair" claim v2.1 removed.

    Raises if either member has no results — a pair table silently reduced to one model would read
    as a comparison.
    """
    pairs = models_config.get("pairs") or {}
    if pair_name not in pairs:
        raise KeyError(f"models.yaml has no pair {pair_name!r}; have {sorted(pairs)}")
    pair = pairs[pair_name]
    members = list(pair.get("members") or [])
    if len(members) != 2:
        raise ValueError(f"pair {pair_name!r} has {len(members)} members; expected exactly 2")
    resultset.require(members)

    framing = str(pair.get("framing") or "")
    holds_fixed = list(pair.get("holds_fixed") or [])
    not_matched = list(pair.get("not_matched") or [])
    controls = list(pair.get("controls") or [])

    a, b = members
    base = primary_table(
        resultset,
        metric=metric,
        layer_reduction=layer_reduction,
        models=members,
        depth_bins=depth_bins,
    )
    feature_cols = [c for c in base.columns if c in resultset.features()]

    if layer_reduction == "mean":
        axis_cols: list[str] = []
        keys: list[tuple[Any, ...]] = [()]
        by_key = {(): {row[0]: row for row in base.rows}}
    elif layer_reduction == "per_layer":
        axis_cols = ["layer"]
        keys = []
        by_key = {}
        for row in base.rows:
            key = (row[base.columns.index("layer")],)
            by_key.setdefault(key, {})[row[0]] = row
            if key not in keys:
                keys.append(key)
    else:
        axis_cols = ["depth_bin"]
        keys = []
        by_key = {}
        for row in base.rows:
            key = (row[base.columns.index("depth_bin")],)
            by_key.setdefault(key, {})[row[0]] = row
            if key not in keys:
                keys.append(key)

    columns = [
        "pair",
        *axis_cols,
        "feature",
        f"{a} ({metric})",
        f"{b} ({metric})",
        "delta_b_minus_a",
        "holds_fixed",
        "not_matched",
        "framing",
    ]
    rows: list[list[Any]] = []
    for key in keys:
        rowset = by_key.get(key, {})
        for feature in feature_cols:
            values = []
            for member in (a, b):
                row = rowset.get(member)
                if row is None:
                    values.append(
                        MissingCell("not_reached", f"{member} has no row at {key}")
                    )
                else:
                    values.append(row[base.columns.index(feature)])
            va, vb = values
            delta = (
                float(vb) - float(va)
                if not _is_missing(va) and not _is_missing(vb)
                else MissingCell("no_layers", "delta needs both members")
            )
            rows.append(
                [
                    pair_name,
                    *key,
                    feature,
                    va,
                    vb,
                    delta,
                    ", ".join(holds_fixed),
                    ", ".join(not_matched),
                    framing,
                ]
            )

    notes = [
        f"framing (§1.5, verbatim from configs/models.yaml): {framing}",
        f"holds fixed: {holds_fixed}",
        f"NOT matched: {not_matched}",
        f"metric = {metric}; layer_reduction = {layer_reduction}",
    ]
    if controls:
        notes.append(
            f"declared controls for this pair: {controls} — report them on every metric (T9.2); "
            "the size of that delta bounds how much of the pair gap is attributable to the "
            "control axis rather than to the pair's own variable."
        )
    if pair_name in {"pair_g", "pair_t"}:
        notes.append(
            "Tokenizer caveat (T9.2/T9.3): the members have different tokenizers, so F1 does not "
            "condition on an identical variable. Report F1 as Î/H and alongside the word-level "
            "control; F2–F5 are tokenizer-invariant. Pair A carries the fully tokenizer-controlled "
            "version of the F1 claim (§1.5)."
        )
    if layer_reduction == "per_layer":
        notes.append(
            "per_layer rows are NOT aligned across members when their depths differ; use "
            "per_normalized_depth for the cross-model read (§1.4)."
        )
    return Table(
        name=f"T9.2/T9.3 pair {pair_name} — {metric} ({layer_reduction})",
        columns=columns,
        rows=rows,
        notes=notes,
        meta={
            "pair": pair_name,
            "members": members,
            "holds_fixed": holds_fixed,
            "not_matched": not_matched,
            "framing": framing,
            "controls": controls,
            "metric": metric,
            "layer_reduction": layer_reduction,
        },
    )


# -- T9.4 confound table ------------------------------------------------------------------------


def confound_table(
    resultset: ResultSet,
    models_config: Mapping[str, Any],
    *,
    load_balance: Mapping[str, Any] | None = None,
) -> Table:
    """Per-model confound axes — plan T9.4.

    ``load_balance_measured`` (T6.4, an entropy ratio computed on the traces) and
    ``aux_loss_coef_declared`` (a field in a released ``config.json``) are **different
    quantities**, kept in different columns on purpose. The declared coefficient describes a
    training procedure that was never published for most of the panel, and HF config fields are not
    always the values used in training; the measured index is the instrument. The point of putting
    them side by side is that they are free to disagree — that disagreement is the T9.4 result, so
    nothing here derives one from the other, fills one in from the other, or reports a single
    "load balance" column.

    ``load_balance`` is an injected plain mapping (``model -> float`` or ``model -> {layer: float}``)
    so this module stays decoupled from whatever produces it. A model with no measured value gets
    :data:`MISSING_MARKER`, never a default.
    """
    configs = models_config.get("models") or {}
    columns = [
        "model",
        "checkpoint_status",
        "n_moe_layers",
        "n_experts",
        "top_k",
        "activated_density",
        "shared_experts",
        "parallel_dense_mlp",
        "aux_loss_key",
        "aux_loss_coef_declared",
        "load_balance_measured",
        "load_balance_n_layers",
        "logit_tensor_used",
    ]
    rows: list[list[Any]] = []
    for name in resultset.model_names:
        model = resultset.models[name]
        config = configs.get(name) or {}
        aux = config.get("aux_loss") or {}

        measured = (load_balance or {}).get(name)
        n_layers_measured: Any = MissingCell("metric_absent", "no measured load balance supplied")
        if measured is None:
            measured_cell: Any = MissingCell(
                "metric_absent",
                f"no measured load-balance index supplied for {name} (T6.4 input)",
            )
        elif isinstance(measured, Mapping):
            values = [float(v) for v in measured.values()]
            measured_cell = _mean(values) if values else MissingCell("no_layers", "empty mapping")
            n_layers_measured = len(values)
        else:
            measured_cell = float(measured)
            n_layers_measured = 1

        declared = aux.get("value")
        rows.append(
            [
                name,
                config.get("checkpoint_status")
                or MissingCell("metric_absent", "checkpoint_status not set in models.yaml"),
                model.n_moe_layers,
                model.n_experts,
                model.top_k,
                float(model.top_k) / float(model.n_experts) if model.n_experts else MissingCell(
                    "metric_absent", "n_experts is zero"
                ),
                config.get("shared_experts")
                if config.get("shared_experts") is not None
                else MissingCell("metric_absent", "shared_experts not in models.yaml"),
                bool(config.get("parallel_dense_mlp", False)),
                aux.get("key")
                or MissingCell("metric_absent", "aux-loss coefficient absent from config"),
                float(declared)
                if declared is not None
                else MissingCell("metric_absent", "aux-loss coefficient absent from config"),
                measured_cell,
                n_layers_measured,
                model.logit_tensor_used
                or MissingCell("metric_absent", "logit_tensor_used not recorded"),
            ]
        )

    return Table(
        name="T9.4 confounds — measured load balance vs config-declared aux loss",
        columns=columns,
        rows=rows,
        notes=[
            "load_balance_measured is H_marginal / log2(n_experts) on the traces (T6.4). "
            "aux_loss_coef_declared is a config.json field, marked config-declared: the training "
            "procedure was not published and the field is not necessarily the value used in "
            "training. They are different quantities and may disagree; the measured index carries "
            "the argument (T9.4).",
            "Normalization ratios are a framing choice, not a fix: aux loss affects numerator and "
            "denominator non-proportionally, so this table states the confound rather than "
            "claiming it is controlled (T9.4).",
            "shared_experts and parallel_dense_mlp are always-on capacity: excluded from every "
            "entropy/MI quantity (invariant I14), and n_experts is the routed count.",
            "n_experts and top_k are read from the trace manifests in the result documents; the "
            "remaining columns are read from configs/models.yaml.",
        ],
        meta={"load_balance_supplied": sorted((load_balance or {}).keys())},
    )


# -- T9.6 expected-shape check -------------------------------------------------------------------

VERDICTS: tuple[str, ...] = ("agrees", "mixed", "disagrees", "not_evaluable")
"""The full verdict vocabulary. ``mixed`` is a first-class outcome, not a failure to decide."""

FEATURE_LADDER: tuple[str, ...] = ("F0", "F1", "F2", "F4")
"""The ordering the feature ladder implies (§1.3): the marginal, then token identity, then the
previous layer's set, then the router input's linear ceiling. Each conditions on strictly more
than the one before it, so ``Î/H`` is expected to be non-decreasing along it. F3 is temporal
rather than a superset of F2 and F5 differs from F4 in model class, not in conditioning, so
neither sits on this ladder."""


@dataclass(frozen=True)
class ShapeExpectation:
    """One stated expectation, evaluated per model, with the evidence that produced the verdict."""

    name: str
    statement: str
    plan_reference: str
    verdict: str
    per_model: dict[str, dict[str, Any]]
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statement": self.statement,
            "plan_reference": self.plan_reference,
            "verdict": self.verdict,
            "per_model": self.per_model,
            "note": self.note,
        }


@dataclass(frozen=True)
class ShapeCheckReport:
    """Agreement/disagreement per expectation per model — plan T9.6.

    T9.6 asks for a check written for the likely outcome: *"token-ID explains 40–60% depending on
    layer and frequency bucket"* — a nuance result, not a clean dichotomy. So this report has no
    boolean anywhere and no aggregate verdict. ``verdict_counts`` is a distribution over
    :data:`VERDICTS`; a caller who wants "did it pass" gets a histogram instead, because collapsing
    a per-model, per-layer pattern into pass/fail is the reading T9.6 exists to prevent.
    """

    expectations: tuple[ShapeExpectation, ...]
    metric: str

    @property
    def verdict_counts(self) -> dict[str, int]:
        counts = {v: 0 for v in VERDICTS}
        for expectation in self.expectations:
            counts[expectation.verdict] += 1
        return counts

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "expectations": [e.to_json() for e in self.expectations],
            "verdict_counts": self.verdict_counts,
            "verdict_vocabulary": list(VERDICTS),
            "note": (
                "Per-expectation, per-model verdicts. There is deliberately no aggregate "
                "pass/fail: T9.6 is written for a nuance result, and a single verdict would "
                "misreport a pattern that varies by model and by layer."
            ),
        }

    def to_table(self) -> Table:
        columns = ["expectation", "verdict", "model", "model_verdict", "evidence", "statement"]
        rows: list[list[Any]] = []
        for expectation in self.expectations:
            if not expectation.per_model:
                rows.append(
                    [
                        expectation.name,
                        expectation.verdict,
                        MissingCell("no_layers", "no model was evaluable"),
                        "not_evaluable",
                        "",
                        expectation.statement,
                    ]
                )
            for model, detail in sorted(expectation.per_model.items()):
                rows.append(
                    [
                        expectation.name,
                        expectation.verdict,
                        model,
                        detail.get("verdict", "not_evaluable"),
                        json.dumps(
                            {k: v for k, v in detail.items() if k != "verdict"}, sort_keys=True
                        ),
                        expectation.statement,
                    ]
                )
        return Table(
            name="T9.6 expected-shape check",
            columns=columns,
            rows=rows,
            notes=[
                f"verdict counts over expectations: {self.verdict_counts}",
                "Verdicts are reported per expectation and per model. No aggregate pass/fail is "
                "emitted (T9.6).",
            ],
            meta=self.to_json(),
        )


def _ladder_verdict(
    model: ModelResults, metric: str, ladder: Sequence[str], tolerance: float
) -> dict[str, Any]:
    """Per-layer check of the ladder ordering, reported as a fraction, not a boolean."""
    present = [f for f in ladder if f in model.cells]
    if len(present) < 2:
        return {
            "verdict": "not_evaluable",
            "reason": f"fewer than two ladder features present (have {present})",
            "ladder": present,
        }
    holds, breaks, evaluated = [], [], 0
    for layer in model.expected_layers():
        values = [cell_metric(model, f, layer, metric) for f in present]
        if any(_is_missing(v) for v in values):
            continue
        evaluated += 1
        ordered = all(
            float(b) - float(a) >= -tolerance for a, b in zip(values, values[1:])
        )
        (holds if ordered else breaks).append(
            {"layer": layer, "values": {f: float(v) for f, v in zip(present, values)}}
        )
    if not evaluated:
        return {
            "verdict": "not_evaluable",
            "reason": "no layer has a value for every ladder feature",
            "ladder": present,
        }
    fraction = len(holds) / evaluated
    if fraction == 1.0:
        verdict = "agrees"
    elif fraction == 0.0:
        verdict = "disagrees"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "ladder": present,
        "n_layers_evaluated": evaluated,
        "n_layers_ordered": len(holds),
        "fraction_ordered": fraction,
        "layers_out_of_order": [b["layer"] for b in breaks],
        "tolerance": tolerance,
    }


def _depth_verdict(model: ModelResults, metric: str, feature: str, flat_tol: float) -> dict[str, Any]:
    """Does the metric vary with normalized depth at all, and where does it peak?

    The plan states that predictability *varies* with depth (T9.4 tests LayerScope's layer-group
    claim on a wider panel) but does not state a direction, so no direction is asserted here. The
    reported quantities are the spread and the location of the extremes on ℓ/(L−1); the direction
    is left to the write-up rather than invented by the checker.
    """
    if feature not in model.cells:
        return {"verdict": "not_evaluable", "reason": f"{feature} absent for {model.model}"}
    points = []
    for layer in model.expected_layers():
        value = cell_metric(model, feature, layer, metric)
        if _is_missing(value):
            continue
        points.append((normalized_depth(layer, model.n_moe_layers), layer, float(value)))
    if len(points) < 3:
        return {
            "verdict": "not_evaluable",
            "reason": f"{len(points)} usable layers; a depth shape needs at least 3",
            "feature": feature,
        }
    values = [p[2] for p in points]
    spread = max(values) - min(values)
    peak = max(points, key=lambda p: p[2])
    trough = min(points, key=lambda p: p[2])
    first, last = values[0], values[-1]
    if spread <= flat_tol:
        verdict = "disagrees"  # flat within tolerance: no depth structure to report
    elif 0.0 < peak[0] < 1.0 and peak[2] - max(first, last) > flat_tol:
        verdict = "agrees"  # interior extremum: depth structure, direction left to the write-up
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "feature": feature,
        "n_layers": len(points),
        "spread": spread,
        "peak_normalized_depth": peak[0],
        "peak_layer": peak[1],
        "peak_value": peak[2],
        "trough_normalized_depth": trough[0],
        "trough_layer": trough[1],
        "trough_value": trough[2],
        "shallowest_value": first,
        "deepest_value": last,
        "flat_tolerance": flat_tol,
    }


def _combine(per_model: Mapping[str, Mapping[str, Any]]) -> str:
    verdicts = {d.get("verdict", "not_evaluable") for d in per_model.values()}
    verdicts.discard("not_evaluable")
    if not verdicts:
        return "not_evaluable"
    if verdicts == {"agrees"}:
        return "agrees"
    if verdicts == {"disagrees"}:
        return "disagrees"
    return "mixed"


def expected_shape_check(
    resultset: ResultSet,
    *,
    metric: str = "mi_ratio",
    ladder: Sequence[str] = FEATURE_LADDER,
    depth_feature: str = "F1",
    tolerance: float = 0.0,
    flat_tolerance: float = 1e-3,
) -> ShapeCheckReport:
    """Evaluate the plan's stated expectations about the *shape* of the result — plan T9.6.

    Two expectations, both stated by the plan rather than by this function:

    * **feature ladder** — each feature in :data:`FEATURE_LADDER` conditions on strictly more than
      the previous one (§1.3, "every other feature must beat" the marginal), so the metric should
      be non-decreasing along it.
    * **depth** — predictability varies with normalized depth (T9.4's test of LayerScope's
      layer-group claim). No direction is asserted; the check reports spread and extremum location.

    The verdict vocabulary is :data:`VERDICTS`, and ``mixed`` is the outcome T9.6 says to expect:
    *"token-ID explains 40–60% depending on layer and frequency bucket"* is a nuance result, so a
    per-model, per-layer disagreement is reported as such and never editorialised into a
    refutation. Nothing here returns a boolean.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; have {sorted(METRICS)}")

    ladder_per_model = {
        name: _ladder_verdict(resultset.models[name], metric, ladder, tolerance)
        for name in resultset.model_names
    }
    depth_per_model = {
        name: _depth_verdict(resultset.models[name], metric, depth_feature, flat_tolerance)
        for name in resultset.model_names
    }

    expectations = (
        ShapeExpectation(
            name="feature_ladder_ordering",
            statement=(
                f"{' < '.join(ladder)} in {metric}: each feature conditions on strictly more than "
                "the previous one, and every feature must beat the F0 marginal"
            ),
            plan_reference="§1.3, T9.1",
            verdict=_combine(ladder_per_model),
            per_model=dict(ladder_per_model),
            note=(
                "Evaluated per layer; the per-model verdict is 'mixed' whenever the ordering holds "
                "on some layers and not others, which is the outcome T9.6 says to write for."
            ),
        ),
        ShapeExpectation(
            name="depth_variation",
            statement=(
                f"{metric} for {depth_feature} varies with normalized depth ℓ/(L−1) rather than "
                "being flat across the stack"
            ),
            plan_reference="T9.4 (depth), §1.4 (normalized depth)",
            verdict=_combine(depth_per_model),
            per_model=dict(depth_per_model),
            note=(
                "The plan states that predictability varies with depth but not in which "
                "direction, so no direction is checked. Spread and extremum location are reported "
                "for the write-up to interpret."
            ),
        ),
    )
    return ShapeCheckReport(expectations=expectations, metric=metric)
