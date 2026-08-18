"""Phase 4 — corpus composition, fetching and assembly.

Three modules, split so the two parts that must be reproducible offline do not depend on the one that
needs a network:

* :mod:`src.corpus.spec` — the *declaration*. ``CorpusSpec`` / ``SourceSpec``, ``TARGET_SHARES``
  (25/25/20/30 prose/code/math/multilingual), and the token budgets derived from them
  (``domain_token_targets`` / ``source_token_targets``). ``role="parallel_control"`` is what makes
  the FLORES-200 cap structural rather than a comment (T4.1).
* :mod:`src.corpus.fetch` — the *acquisition*. Reads each source to its budget through the
  ``DocumentSource`` seam (``HFDatasetSource``, always ``streaming=True``; ``InMemorySource`` for a
  cached snapshot), cleans, deduplicates and truncates, and returns ``FetchResult`` — documents plus
  requested-vs-delivered-vs-dropped-by-reason accounting. Imports ``datasets`` lazily, so nothing
  else in the package needs it.
* :mod:`src.corpus.build` — the *assembly*. Takes any ``list[Document]`` and produces
  ``corpora/<name>.jsonl`` plus its sidecar: document-level stratified splits (T4.3), domain-
  interleaved write order, ~50k-token shards that never straddle a document (T4.2), and the T4.4
  hidden-state budget ladder.

Token counts everywhere come from a ``TokenCounter`` — ``CharRatioCounter`` is the reference proxy the
shared corpus file is budgeted under, never a real tokenizer; the per-model realized shares are
measured afterwards by ``realized_shares``.
"""

from __future__ import annotations

__all__ = ["build", "fetch", "spec"]
