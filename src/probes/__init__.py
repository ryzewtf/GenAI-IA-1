"""Phase 7 — predictors and probes.

Feature definitions live in :mod:`src.probes.features`; each has a predictor:

* F0 → :class:`~src.probes.marginal.MarginalPredictor` (train marginal, constant)
* F1 → :class:`~src.probes.counts.CountTablePredictor` (closed-form smoothed count table)
* F2 / F3 / F4 / F6 / FV → :class:`~src.probes.linear.SoftmaxProbe` (AdamW, cosine, early stop)
* F5 → :class:`~src.probes.mlp.MLPProbe` (2-layer MLP, GELU; same optimizer, schedule and
  early-stopping-with-restore as the linear probe, so the F5−F4 gap is a difference in model class
  and not in training effort). It is the one probe plan §Phase S budgets a GPU session for, so it
  is **absent from** :data:`~src.probes.features.CPU_FEATURES` and must be requested explicitly.

:func:`~src.probes.evaluate.evaluate_on_test` produces both metric families on test;
:func:`~src.probes.train.sweep` is the resumable (layer × feature) driver.
"""

from __future__ import annotations

__all__ = ["base", "counts", "evaluate", "features", "linear", "marginal", "mlp", "train"]
