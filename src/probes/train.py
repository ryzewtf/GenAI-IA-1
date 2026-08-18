"""The probe sweep — plan T7.8.

Sweeps (layer × feature) for one model, writing **one JSON per (model, feature)** rather than per
(model, layer, feature): a 48-layer × 8-feature sweep would be 384 files against Kaggle's ~500
file cap on ``/kaggle/working``, so per-layer files would blow the cap on the second model.

Three properties the plan requires and this module implements literally:

*Idempotent and resumable.* Every completed (feature, layer) is a key in that feature's JSON, and
the file is rewritten atomically after each one. A killed session loses at most the layer in
flight; a resumed session skips what is already there unless ``--force``.

*Session-budget aware.* Wired to the same :class:`~src.runtime.session.SessionBudget` as capture.
When the budget says stop, the sweep stops between layers with everything so far on disk and
``stopped_early`` set — it does not try to squeeze one more fit into the reserve.

*Layer by layer.* The outer loop is the layer, so at most one layer's tables and one layer's
router-input block are resident. Qwen3's F1 table alone is 3.7 GB across 48 layers (plan T7.3),
which is the whole reason the loops are nested in this order and not the other one.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..runtime.session import SessionBudget
from ..traces.reader import TraceReader
from .base import UndefinedFeature
from .counts import CountTablePredictor, frequency_buckets
from .evaluate import evaluate_on_test
from .features import (
    CPU_FEATURES,
    FEATURES,
    CorpusIndex,
    build_features,
    consecutive_repetition_rate,
)
from .linear import SoftmaxProbe
from .marginal import MarginalPredictor
from .mlp import MLPProbe

__all__ = ["SweepResult", "sweep", "output_path", "FV_GATE"]

FV_GATE = 0.99
"""Plan T7.7 / T3.3: FV's ``set_agreement@k`` must reach this. **Validation only, never a finding.**"""

# Order matters: F1 is fitted before F6, which consumes it as a feature block. F5 follows F4 so
# that the linear ceiling for a layer is on disk before the nonlinear one is attempted — if a
# session dies mid-layer, F4 alone is still a reportable result and F5 alone is not.
_FEATURE_ORDER: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "FV")


def output_path(out_dir: Path | str, model: str, feature: str) -> Path:
    return Path(out_dir) / f"{_slug(model)}__{feature}.json"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class SweepResult:
    model: str
    features: list[str]
    layers_done: list[int] = field(default_factory=list)
    layers_skipped: list[int] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None
    paths: dict[str, str] = field(default_factory=dict)
    skipped_cells: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "features": self.features,
            "layers_done": self.layers_done,
            "layers_skipped": self.layers_skipped,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "paths": self.paths,
            "skipped_cells": self.skipped_cells,
        }


def _make_predictor(feature: str, *, n_experts: int, vocab_size: int, layer: int, seed: int):
    if feature == "F0":
        return MarginalPredictor(n_experts=n_experts, layer=layer)
    if feature == "F1":
        return CountTablePredictor(n_experts=n_experts, vocab_size=vocab_size, layer=layer)
    if feature == "F5":
        # The one nonlinear probe (plan T7.5). Available on request, never a CPU-session default —
        # see CPU_FEATURES.
        return MLPProbe(n_experts=n_experts, feature=feature, layer=layer, seed=seed)
    return SoftmaxProbe(n_experts=n_experts, feature=feature, layer=layer, seed=seed)


def sweep(
    reader: TraceReader,
    *,
    model: str,
    out_dir: Path | str,
    features: Sequence[str] = CPU_FEATURES,
    layers: Iterable[int] | None = None,
    budget: SessionBudget | None = None,
    force: bool = False,
    seed: int = 0,
    vocab_size: int | None = None,
    n_frequency_buckets: int = 10,
    estimator: str = "miller_madow",
    probe_kwargs: Mapping[str, Any] | None = None,
) -> SweepResult:
    """Fit and evaluate every (layer, feature) not already on disk."""
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"unknown feature(s) {unknown}; have {FEATURES}")

    out_dir = Path(out_dir)
    ordered = [f for f in _FEATURE_ORDER if f in features]
    layer_list = list(range(reader.n_moe_layers)) if layers is None else [int(l) for l in layers]
    index = CorpusIndex.from_reader(reader, vocab_size=vocab_size)

    documents: dict[str, dict[str, Any]] = {}
    for feature in ordered:
        path = output_path(out_dir, model, feature)
        if path.exists() and not force:
            documents[feature] = json.loads(path.read_text(encoding="utf-8"))
        else:
            documents[feature] = {
                "model": model,
                "feature": feature,
                "run_config_sha256": reader.run_config_sha256,
                "logit_tensor_used": reader.logit_tensor_used,
                "n_moe_layers": reader.n_moe_layers,
                "n_experts": reader.n_experts,
                "top_k": reader.top_k,
                "shard_ids": reader.shard_ids,
                "entropy_estimator": estimator,
                "layers": {},
            }
        documents[feature].setdefault("layers", {})

    result = SweepResult(model=model, features=ordered)
    result.paths = {f: str(output_path(out_dir, model, f)) for f in ordered}

    train_ids = index.token_ids[index.rows("train")]

    for layer in layer_list:
        pending = [f for f in ordered if str(layer) not in documents[f]["layers"]]
        if not pending:
            result.layers_skipped.append(layer)
            continue
        if budget is not None and budget.should_stop():
            result.stopped_early = True
            result.stop_reason = f"session budget exhausted before layer {layer}: {budget.summary()}"
            break

        f1_cache: CountTablePredictor | None = None
        touched = False

        for feature in pending:
            try:
                record, f1_fitted = _fit_one(
                    reader=reader,
                    index=index,
                    feature=feature,
                    layer=layer,
                    seed=seed,
                    estimator=estimator,
                    train_ids=train_ids,
                    n_frequency_buckets=n_frequency_buckets,
                    f1=f1_cache,
                    probe_kwargs=probe_kwargs or {},
                )
            except UndefinedFeature as exc:
                record = {
                    "layer": layer,
                    "model_layer": reader.model_layer(layer),
                    "status": "skipped",
                    "reason": str(exc),
                }
                f1_fitted = None
                result.skipped_cells.append(
                    {"feature": feature, "layer": layer, "reason": str(exc)}
                )
            if f1_fitted is not None:
                f1_cache = f1_fitted

            documents[feature]["layers"][str(layer)] = record
            documents[feature]["updated_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            _atomic_write_json(output_path(out_dir, model, feature), documents[feature])
            touched = True

        if touched:
            result.layers_done.append(layer)

    return result


def _fit_one(
    *,
    reader: TraceReader,
    index: CorpusIndex,
    feature: str,
    layer: int,
    seed: int,
    estimator: str,
    train_ids: np.ndarray,
    n_frequency_buckets: int,
    f1: CountTablePredictor | None,
    probe_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], CountTablePredictor | None]:
    """Fit one cell on train, early-stop on val, evaluate on test."""
    needs_f1 = feature == "F6"
    if needs_f1 and f1 is None:
        # Cheap to refit: F1 is closed-form and one pass over train, so a resume that has F1 on
        # disk but not F6 does not need F1's result file to be re-readable.
        f1 = _fit_f1(reader, index, layer, train_ids)

    train = build_features(feature, reader, index, layer, "train", f1=f1)
    val = build_features(
        feature, reader, index, layer, "val", f1=f1, standardizer=train.standardizer
    )
    test = build_features(
        feature, reader, index, layer, "test", f1=f1, standardizer=train.standardizer
    )

    predictor = _make_predictor(
        feature,
        n_experts=reader.n_experts,
        vocab_size=index.vocab_size,
        layer=layer,
        seed=seed,
    )
    if isinstance(predictor, (SoftmaxProbe, MLPProbe)):
        # Validated, not blind setattr. A typo'd hyperparameter would otherwise attach itself to the
        # probe, never be applied, and still be reported in `hyperparams` as though it had been --
        # so the result file would claim a setting the fit never used. That got sharper once two
        # probe classes with different field sets started sharing this path: `hidden_width` is real
        # for MLPProbe and meaningless on SoftmaxProbe, and a mixed F4+F5 sweep passes both.
        # A key unknown to EVERY probe class is a typo and is fatal. A key that is real for another
        # probe class is skipped silently, because one sweep legitimately passes the union: F4 wants
        # `lr`, F5 wants `lr` and `hidden_width`, and requiring the caller to split them per feature
        # would be a worse interface than ignoring the inapplicable half.
        applicable = {f.name for f in fields(predictor)}
        known_anywhere = {f.name for f in fields(SoftmaxProbe)} | {f.name for f in fields(MLPProbe)}
        for key, value in probe_kwargs.items():
            if key not in known_anywhere:
                raise ValueError(
                    f"probe_kwargs[{key!r}] is not a field of any probe class; known: "
                    f"{sorted(known_anywhere)}"
                )
            if key in applicable:
                setattr(predictor, key, value)
    predictor.fit(train.X, train.y, val.X, val.y)

    buckets = None
    bucket_meta: dict[str, Any] | None = None
    if feature == "F1":
        buckets, bucket_meta = frequency_buckets(
            train_ids,
            test.token_ids,
            n_buckets=n_frequency_buckets,
            vocab_size=index.vocab_size,
        )

    metrics = evaluate_on_test(
        predictor,
        test.X,
        test.y,
        n_experts=reader.n_experts,
        top_k=reader.top_k,
        estimator=estimator,
        buckets=buckets,
        bucket_meta=bucket_meta,
    )

    report = getattr(predictor, "report", None)
    record: dict[str, Any] = {
        "layer": layer,
        "model_layer": reader.model_layer(layer),
        "status": "ok",
        "fit": report.to_json() if report is not None else None,
        "metrics": metrics,
        "rows": {
            "train": train.n_rows,
            "val": val.n_rows,
            "test": test.n_rows,
        },
        "excluded": {
            "train": train.n_excluded,
            "val": val.n_excluded,
            "test": test.n_excluded,
            "reason": train.exclusion_reason,
        },
        "meta": train.meta,
    }

    if feature == "F3":
        # Plan T7.4: report F3 next to the raw statistic it generalizes, on the same split.
        record["mixtral_table5_statistic"] = consecutive_repetition_rate(
            reader, index, layer, split="test"
        )

    if feature == "FV":
        agreement = float(metrics["family_a"]["set_agreement@k"])
        record["validation_gate"] = {
            "threshold": FV_GATE,
            "set_agreement@k": agreement,
            "pass": agreement >= FV_GATE,
            "appendix_only": True,
            "note": (
                "Data-alignment test, not a result. The router at this layer is a linear map on "
                "this exact input, so a linear probe must reach ~100%; a low value means the "
                "trace rows are misaligned. Never report FV as a finding (plan T7.7)."
            ),
        }

    return record, f1 if feature == "F1" else None


def _fit_f1(
    reader: TraceReader, index: CorpusIndex, layer: int, train_ids: np.ndarray
) -> CountTablePredictor:
    train = build_features("F1", reader, index, layer, "train")
    val = build_features("F1", reader, index, layer, "val")
    predictor = CountTablePredictor(
        n_experts=reader.n_experts, vocab_size=index.vocab_size, layer=layer
    )
    predictor.fit(train.X, train.y, val.X, val.y)
    return predictor


# -- CLI ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7 probe sweep (plan T7.8)")
    parser.add_argument("--trace-root", required=True, help="shards live at ROOT/model/corpus/")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True, help="results directory")
    parser.add_argument("--splits", required=True, help="JSON mapping doc_id -> split (plan T4.3)")
    parser.add_argument("--features", nargs="+", default=list(CPU_FEATURES))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="refit cells already on disk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wall-limit-s", type=float, default=None)
    parser.add_argument("--reserve-s", type=float, default=1800.0)
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    doc_splits = {int(k): str(v) for k, v in raw.items()}

    budget = (
        SessionBudget(wall_limit_s=args.wall_limit_s, reserve_s=args.reserve_s)
        if args.wall_limit_s
        else None
    )

    with TraceReader(
        Path(args.trace_root), args.model, args.corpus, doc_splits=doc_splits
    ) as reader:
        result = sweep(
            reader,
            model=args.model,
            out_dir=args.out,
            features=args.features,
            layers=args.layers,
            budget=budget,
            force=args.force,
            seed=args.seed,
            vocab_size=args.vocab_size,
        )
    print(json.dumps(result.to_json(), indent=2))
    return 1 if result.stopped_early else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
