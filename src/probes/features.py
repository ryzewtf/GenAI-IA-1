"""Feature construction from a trace — plan §1.3, T7.3, T7.4, T7.5, T7.7.

Everything here targets **routing at layer ℓ** and differs only in what it conditions on:

===== ============================================ ==========================================
ID    Conditioning variable                        Undefined when
===== ============================================ ==========================================
F0    nothing                                      never
F1    ``token_id`` of the token being routed       never
F2    expert set at layer **ℓ−1**, same token      ℓ = 0
F3    expert set at layer **ℓ**, previous token    first token of a document
F4    router input at layer **ℓ−1**                ℓ = 0, or token not in the hidden subsample
F5    router input at layer **ℓ−1** (MLP)          same as F4 — identical rows, nonlinear probe
F6    F1 ⊕ F2 ⊕ F3                                 inherits F2's and F3's exclusions
FV    router input at layer **ℓ**                  token not in the hidden subsample
===== ============================================ ==========================================

Two structural points that are easy to get wrong and impossible to notice afterwards:

**F3 does not leak across the split boundary.** Its conditioning token is the previous token *of
the same document*, and plan T4.3 makes splits document-level precisely so that this holds. A
token-level split would put a token's own predecessor in the training set, and F3 — the feature
that generalizes the Mixtral Table 5 statistic — would be measuring memorisation. The exclusion
of document-initial tokens is what guarantees the predecessor is in-document, and therefore
in-split.

**Exclusion counts are recorded, not silently applied.** F2 at ℓ = 0 and F3 at document-initial
tokens are dropped, and the count travels in the :class:`FitReport` so that F3's numbers are
comparable against a raw repetition rate computed over the same rows (plan T7.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from ..traces.reader import TraceReader
from .base import Design, DenseBlock, MultiHotBlock, Standardizer, UndefinedFeature
from .counts import CountTablePredictor

__all__ = [
    "FEATURES",
    "CPU_FEATURES",
    "CorpusIndex",
    "LayerFeatures",
    "build_features",
    "consecutive_repetition_rate",
]

FEATURES: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "FV")
"""Features implemented in NumPy. F5's predictor is :mod:`src.probes.mlp`, the rest :mod:`src.probes.linear`."""

CPU_FEATURES: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F6")
"""The subset plan §Phase S routes to CPU sessions so they cost no GPU quota.

F4/F5/FV are absent by design, and F5 most deliberately: plan §Phase S budgets it a GPU session,
so it must be requested explicitly rather than arrive as a default in a CPU-budgeted sweep.
"""

_HIDDEN_FEATURES = frozenset({"F4", "F5", "FV"})


@dataclass
class CorpusIndex:
    """Token metadata and split row-sets, built once and reused across all layers.

    Building this per layer would re-read the token stream 48 times for Qwen3. It is small —
    16 bytes per token, so 16 MB at 1M tokens — so it is held whole while the layer-by-layer
    fitting of plan T7.3 keeps the *tables* out of RAM.
    """

    token_ids: np.ndarray
    doc_ids: np.ndarray
    pos_in_doc: np.ndarray
    split_rows: dict[str, np.ndarray]
    captured: np.ndarray
    vocab_size: int

    @property
    def n_tokens(self) -> int:
        return int(self.token_ids.shape[0])

    @classmethod
    def from_reader(
        cls,
        reader: TraceReader,
        *,
        splits: Iterable[str] = ("train", "val", "test"),
        vocab_size: int | None = None,
    ) -> "CorpusIndex":
        records = reader.tokens()
        token_ids = np.asarray(records["token_id"], dtype=np.int64)
        rows = {s: np.flatnonzero(reader.split_mask(s)).astype(np.int64) for s in splits}
        for split, idx in rows.items():
            if not idx.size:
                raise ValueError(f"split {split!r} contains no tokens")
        return cls(
            token_ids=token_ids,
            doc_ids=np.asarray(records["doc_id"], dtype=np.int64),
            pos_in_doc=np.asarray(records["pos_in_doc"], dtype=np.int64),
            split_rows=rows,
            captured=reader.captured_token_ids(),
            vocab_size=int(vocab_size) if vocab_size is not None else int(token_ids.max()) + 1,
        )

    def rows(self, split: str) -> np.ndarray:
        if split not in self.split_rows:
            raise KeyError(f"unknown split {split!r}; have {sorted(self.split_rows)}")
        return self.split_rows[split]


@dataclass
class LayerFeatures:
    """One (feature, layer, split) ready to hand to a :class:`~src.probes.base.Predictor`.

    ``X`` is whatever that feature's predictor expects: ``None`` for F0, a ``(n,)`` token-id
    vector for F1, a :class:`Design` otherwise.
    """

    feature: str
    layer: int
    split: str
    X: Any
    y: np.ndarray
    token_ids: np.ndarray
    row_index: np.ndarray
    n_excluded: int = 0
    exclusion_reason: str | None = None
    standardizer: Standardizer | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.y.size)


def _kept_rows(index: CorpusIndex, feature: str, layer: int, split: str) -> tuple[np.ndarray, int, str | None]:
    """Rows of the split this feature is defined on, plus the exclusion count and its reason."""
    rows = index.rows(split)
    total = rows.size

    if feature in ("F2", "F4", "F5") and layer == 0:
        raise UndefinedFeature(
            f"{feature} conditions on layer {layer - 1}, which does not exist (plan T7.4)"
        )
    if feature == "F6" and layer == 0:
        raise UndefinedFeature(
            "F6 contains F2, which is undefined at layer 0 (plan T7.4); reported as skipped "
            "rather than silently degraded to F1 + F3, which would not be the same feature"
        )

    reasons: list[str] = []
    keep = np.ones(total, dtype=bool)

    if feature in ("F3", "F6"):
        doc_initial = index.pos_in_doc[rows] == 0
        keep &= ~doc_initial
        reasons.append(f"document-initial tokens ({int(doc_initial.sum())})")

    if feature in _HIDDEN_FEATURES:
        has_hidden = np.isin(rows, index.captured, assume_unique=False)
        keep &= has_hidden
        reasons.append(f"tokens outside the hidden subsample ({int((~has_hidden).sum())})")

    kept = rows[keep]
    if not kept.size:
        raise UndefinedFeature(
            f"{feature} at layer {layer} on split {split!r} has no usable rows after exclusions"
        )
    return kept, int(total - kept.size), "; ".join(reasons) or None


def build_features(
    feature: str,
    reader: TraceReader,
    index: CorpusIndex,
    layer: int,
    split: str,
    *,
    f1: CountTablePredictor | None = None,
    standardizer: Standardizer | None = None,
) -> LayerFeatures:
    """Assemble one (feature, layer, split).

    ``f1`` is required for F6 and must already be fitted on train. ``standardizer`` is required
    for F4/F5/FV on any split other than train; on train it is fitted here from the train rows
    and returned on the result, which is the mechanism enforcing plan T7.5's "standardize inputs
    using train-split statistics only" — a caller cannot reach test-fitted statistics by mistake.
    """
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}; have {FEATURES}")

    rows, n_excluded, reason = _kept_rows(index, feature, layer, split)
    y = reader.topk_sets(layer)[rows].astype(np.int32)
    token_ids = index.token_ids[rows]
    n_experts = reader.n_experts
    meta: dict[str, Any] = {"model_layer": reader.model_layer(layer)}

    X: Any
    out_standardizer: Standardizer | None = None

    if feature == "F0":
        X = None

    elif feature == "F1":
        X = token_ids

    elif feature in ("F2", "F3", "F6"):
        blocks: list[Any] = []
        if feature == "F6":
            if f1 is None:
                raise ValueError("F6 requires a fitted F1 predictor for its log-probability block")
            blocks.append(DenseBlock(f1.log_proba(token_ids), name="f1_log_proba"))
        if feature in ("F2", "F6"):
            lower = reader.topk_sets(layer - 1)[rows].astype(np.int32)
            blocks.append(MultiHotBlock(lower, n_experts, name="f2_lower_layer_set"))
        if feature in ("F3", "F6"):
            prev = reader.topk_sets(layer)[rows - 1].astype(np.int32)
            blocks.append(MultiHotBlock(prev, n_experts, name="f3_previous_token_set"))
        X = Design(tuple(blocks))

    else:  # F4, F5, FV — F4 and F5 share this design exactly and differ only in the predictor
        hidden_layer = layer - 1 if feature in ("F4", "F5") else layer
        values = reader.hidden(hidden_layer, rows)
        if standardizer is None:
            if split != "train":
                raise ValueError(
                    f"{feature} on split {split!r} needs the train-fitted standardizer; refusing "
                    "to fit one on non-train rows (plan T7.5)"
                )
            standardizer = Standardizer.fit(values)
        out_standardizer = standardizer
        meta["hidden_layer"] = hidden_layer
        meta["hidden_model_layer"] = reader.model_layer(hidden_layer)
        X = Design((DenseBlock(values, standardizer=standardizer, name="router_input"),))

    return LayerFeatures(
        feature=feature,
        layer=layer,
        split=split,
        X=X,
        y=y,
        token_ids=token_ids,
        row_index=rows,
        n_excluded=n_excluded,
        exclusion_reason=reason,
        standardizer=out_standardizer,
        meta=meta,
    )


# -- the Mixtral Table 5 statistic ------------------------------------------------------------------


def consecutive_repetition_rate(
    reader: TraceReader,
    index: CorpusIndex,
    layer: int,
    split: str | None = None,
) -> dict[str, float | int]:
    """Raw consecutive-token expert repetition at ``layer`` — plan T7.4, §1.7.

    ``|S_t ∩ S_{t-1}| / k`` averaged over non-document-initial tokens. This is the statistic the
    Mixtral paper reports (23.6–28.4% at layer 15 against a 12.5% random baseline), and F3 is its
    learned generalization: F3 fits a linear map on the same conditioning variable, so F3 ≥ this
    rate is the expected relationship and a large gap is the interesting result. Reporting them
    side by side is what makes the two literatures' numbers comparable (plan T9.5).
    """
    rows = index.rows(split) if split is not None else np.arange(index.n_tokens, dtype=np.int64)
    keep = index.pos_in_doc[rows] != 0
    rows = rows[keep]
    if not rows.size:
        raise ValueError("no non-document-initial tokens in the requested rows")

    sets = reader.topk_sets(layer)
    cur = sets[rows].astype(np.intp)
    prev = sets[rows - 1].astype(np.intp)
    k, n_experts = reader.top_k, reader.n_experts

    mask = np.zeros((rows.size, n_experts), dtype=bool)
    mask[np.arange(rows.size)[:, None], prev] = True
    overlap = np.take_along_axis(mask, cur, axis=1).sum(axis=1) / k

    return {
        "layer": int(layer),
        "model_layer": int(reader.model_layer(layer)),
        "split": split if split is not None else "all",
        "n_rows": int(rows.size),
        "n_excluded_doc_initial": int((~keep).sum()),
        "repetition_rate": float(overlap.mean()),
        "random_baseline": float(k / n_experts),
        "exact_set_repeat_rate": float((overlap == 1.0).mean()),
    }
