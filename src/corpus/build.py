"""Assemble, split, shard and validate a corpus file — plan T4.2 / T4.3 / T4.4.

**There is no dataset download in this module.** Fetching from the Hub is a separate later task; this
module takes whatever :class:`Document` list it is handed and turns it into the artifact the rest of
the pipeline consumes: ``corpora/<name>.jsonl`` plus a ``.meta.json`` sidecar. That separation is
deliberate — the split assignment and shard table are the parts the analysis depends on, so they have
to be testable offline, in seconds, without a network or a tokenizer.

Four rules here are load-bearing:

1. **Shards never straddle a document** (T4.2). ``moe_trace`` tokenizes one document per
   ``llama_decode`` sequence and writes one shard directory at a time; a document split over two
   shards would have its second half tokenized without its first half's context, so the router
   inputs for those tokens would not be the ones the model would really see. A document longer than
   ``shard_tokens`` therefore gets a shard to itself rather than being cut.

2. **Splits are document level and live in the corpus file** (T4.3). ``TraceReader.split_mask``
   reads the ``doc_id -> split`` mapping produced by :func:`load_doc_splits`, so every model in the
   panel is evaluated on the same documents. Token-level splits would leak through adjacency (F3
   conditions on the previous token's routing), and a per-model split would make the cross-model
   comparison of T9.1/T9.4 uncontrolled.

3. **Write order is domain-interleaved.** Kaggle kills a session at the 12 h cap and T5.1 may stop
   part way through a model, leaving shards 0..j collected and the rest not. If the corpus were
   ordered domain by domain, that partial collection would be silently biased — all prose, no math —
   and the domain-stratified metrics that T9.4 makes the *primary* comparison would be computed on a
   corpus that no longer has the composition the manifest claims. Interleaving makes any prefix of
   shards approximately representative, so a truncated run degrades in precision rather than in
   validity.

4. **``topk.bin`` is never dropped** (T4.4, fallback ladder). It is the labels. Every rung of
   :func:`hidden_budget_report`'s ladder reduces the hidden subsample first and drops ``logits.bin``
   second; there is no rung that drops ``topk.bin``, and :data:`FALLBACK_LADDER` is asserted against
   that on construction.

Token counts in this module come from a :class:`TokenCounter`, never from a real tokenizer, because
the corpus is shared by seven checkpoints whose vocabularies span 50k to 262k. See
:class:`CharRatioCounter`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .spec import DOMAINS, CorpusSpec, CorpusSpecError

__all__ = [
    "SPLIT_ORDER",
    "FALLBACK_LADDER",
    "TokenCounter",
    "CharRatioCounter",
    "Document",
    "BuildResult",
    "assign_splits",
    "interleave_documents",
    "assign_shards",
    "shard_table",
    "write_corpus",
    "load_corpus",
    "load_doc_splits",
    "realized_shares",
    "hidden_budget_report",
    "build",
]

SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")
"""Canonical order. Used for rounding tie-breaks so the assignment cannot depend on the iteration
order of the ratios mapping the caller happened to pass."""


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can say how many tokens a string is."""

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class CharRatioCounter:
    """The **reference** counter: ``max(1, ceil(len(text) / chars_per_token))``.

    This is a proxy, not a measurement, and it is the only kind of count that can go *into* the
    corpus file. The panel's seven tokenizers span 50,304 (OLMoE, GPT-NeoX-style) to 262,144
    (Gemma 4) entries, so one document is a materially different number of tokens for each
    checkpoint — most sharply on the 30% non-Latin share, where OLMoE fragments toward bytes and
    Gemma does not. But T4.3 requires *identical* splits and shard ids across models, which means
    one shared file and therefore one shared token count.

    So the budget is set once here, and the numbers that go in the paper are the *realized* per-model
    shares measured afterwards by :func:`realized_shares`, which takes any :class:`TokenCounter` and
    so accepts a real tokenizer without any change to this module (T4.1's per-model token-share
    table). Nothing downstream may treat ``n_tokens_ref`` as a true token count.

    Deterministic and offline by construction: no tokenizer files, no network, same answer on every
    machine, which is what lets the shard table be reproduced from the spec alone.
    """

    chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        if self.chars_per_token <= 0:
            raise CorpusSpecError("chars_per_token must be positive")

    def count(self, text: str) -> int:
        # max(1, ...) because a document that counts as zero tokens would be invisible to the
        # budget and to sharding, but moe_trace treats a zero-token document as a hard error.
        return max(1, math.ceil(len(text) / self.chars_per_token))


@dataclass
class Document:
    """One corpus line. ``split`` and ``shard_id`` are ``None`` until :func:`build` fills them."""

    doc_id: int
    text: str
    domain: str
    lang: str
    """ISO code for the multilingual domain; ``"en"`` for English prose/math, ``""`` for code, where
    the natural-language label is meaningless and would create empty strata."""
    source: str
    n_tokens_ref: int
    split: str | None = None
    shard_id: int | None = None

    @property
    def stratum(self) -> tuple[str, str]:
        """The (domain, lang) pair T4.3 stratifies on."""
        return (self.domain, self.lang)


def _doc_key(doc_id: int, seed: int, purpose: str) -> str:
    """Deterministic ordering key for one document.

    A hash of (seed, doc_id) rather than a seeded RNG walked over the input list: the assignment has
    to be a pure function of document *identity*, so that re-running the build with the documents in
    a different order — a different fetch order, a dict that iterated differently — reproduces the
    same splits. A shared corpus file whose splits quietly depended on fetch order would be the
    worst kind of failure here, because nothing downstream could detect it.
    """
    return hashlib.blake2b(
        f"{purpose}|{seed}|{doc_id}".encode("utf-8"), digest_size=16
    ).hexdigest()


def _strata(docs: Sequence[Document], *, seed: int, purpose: str) -> dict[tuple[str, str], list[Document]]:
    """Group by (domain, lang), each group in deterministic hash order."""
    groups: dict[tuple[str, str], list[Document]] = {}
    for doc in docs:
        groups.setdefault(doc.stratum, []).append(doc)
    return {
        key: sorted(groups[key], key=lambda d: (_doc_key(d.doc_id, seed, purpose), d.doc_id))
        for key in sorted(groups)
    }


def _allocate(n: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Split ``n`` documents across splits.

    Largest remainder, then a **minimum-one repair**: if a split would get zero while the stratum has
    at least one document per split, one document is taken from the currently largest split. Plain
    largest remainder gives a 3-document stratum train=3/val=0/test=0, which silently deletes that
    (domain, lang) cell from the test set — and per-stratum test cells are exactly what T9.4 reports,
    so an absent cell would read as "not measured" for a stratum we *did* collect. The repair is
    skipped when ``n < len(ratios)``, where no allocation can cover every split.
    """
    order = [s for s in SPLIT_ORDER if s in ratios] + sorted(set(ratios) - set(SPLIT_ORDER))
    exact = {s: n * ratios[s] for s in order}
    alloc = {s: int(math.floor(exact[s])) for s in order}
    left = n - sum(alloc.values())
    for split in sorted(order, key=lambda s: (-(exact[s] - alloc[s]), order.index(s)))[:left]:
        alloc[split] += 1

    if n >= len(order):
        for split in order:
            if alloc[split]:
                continue
            donor = max(order, key=lambda s: (alloc[s], -order.index(s)))
            if alloc[donor] <= 1:
                break
            alloc[donor] -= 1
            alloc[split] += 1
    return alloc


def assign_splits(
    docs: Sequence[Document], *, ratios: Mapping[str, float], seed: int
) -> None:
    """Assign ``split`` in place, document level, stratified by (domain, lang) — T4.3.

    Mutates ``docs``; returns nothing, so there is one copy of the assignment and no chance of a
    caller writing the pre-split list. Allocation within a stratum follows :func:`_allocate` (largest
    remainder plus minimum-one repair) applied to that stratum's documents in :func:`_doc_key` order,
    which makes the result a pure function of (doc_id, seed) and independent of the order ``docs``
    arrives in.
    """
    for stratum, members in _strata(docs, seed=seed, purpose="split").items():
        alloc = _allocate(len(members), ratios)
        cursor = 0
        for split in [s for s in SPLIT_ORDER if s in alloc] + sorted(set(alloc) - set(SPLIT_ORDER)):
            for doc in members[cursor : cursor + alloc[split]]:
                doc.split = split
            cursor += alloc[split]
        assert cursor == len(members), stratum


def interleave_documents(docs: Sequence[Document], *, seed: int = 0) -> list[Document]:
    """Return the documents in the order they will be written — rule 3 in the module docstring.

    Token-weighted round robin over (domain, lang) strata: repeatedly emit from whichever stratum has
    delivered the smallest fraction of its own token total so far, ties broken by stratum key. That
    keeps every prefix close to the corpus's composition *in tokens*, which is the quantity T4.1
    budgets — a doc-count round robin would over-represent whichever domain has short documents
    (code and FLORES sentences) in early shards.

    ``seed`` only sets the within-stratum order, so re-ordering the input cannot change the result.
    """
    strata = _strata(docs, seed=seed, purpose="order")
    totals = {k: sum(d.n_tokens_ref for d in v) or 1 for k, v in strata.items()}
    cursors = {k: 0 for k in strata}
    emitted = {k: 0 for k in strata}

    out: list[Document] = []
    for _ in range(len(docs)):
        live = [k for k in strata if cursors[k] < len(strata[k])]
        pick = min(live, key=lambda k: (emitted[k] / totals[k], k))
        doc = strata[pick][cursors[pick]]
        cursors[pick] += 1
        emitted[pick] += doc.n_tokens_ref
        out.append(doc)
    return out


def assign_shards(docs: Sequence[Document], *, shard_tokens: int) -> None:
    """Assign contiguous ``shard_id`` in place by greedy fill over the given order — T4.2.

    A shard closes when the next document would take it past ``shard_tokens``, so shard sizes land
    under the target rather than on it, and a document larger than ``shard_tokens`` occupies a shard
    alone. Never splits a document: see rule 1 in the module docstring.
    """
    if shard_tokens <= 0:
        raise CorpusSpecError("shard_tokens must be positive")

    shard = 0
    used = 0
    for doc in docs:
        if used and used + doc.n_tokens_ref > shard_tokens:
            shard += 1
            used = 0
        doc.shard_id = shard
        used += doc.n_tokens_ref
        if used >= shard_tokens:
            # Already at or past the target (including the single-oversized-document case), so the
            # next document starts a new shard instead of pushing this one further over.
            shard += 1
            used = 0


def shard_table(docs: Sequence[Document]) -> list[dict[str, Any]]:
    """Per-shard token and document counts, for the sidecar and for T4.2's ±20% acceptance check."""
    table: dict[int, dict[str, Any]] = {}
    for doc in docs:
        if doc.shard_id is None:
            raise CorpusSpecError(f"doc {doc.doc_id} has no shard_id; run assign_shards first")
        row = table.setdefault(
            doc.shard_id,
            {"shard_id": doc.shard_id, "n_docs": 0, "n_tokens_ref": 0, "domains": {}},
        )
        row["n_docs"] += 1
        row["n_tokens_ref"] += doc.n_tokens_ref
        row["domains"][doc.domain] = row["domains"].get(doc.domain, 0) + doc.n_tokens_ref
    return [table[k] for k in sorted(table)]


def realized_shares(docs: Sequence[Document], counter: TokenCounter) -> dict[str, Any]:
    """Realized token share per domain and per (domain, lang) under ``counter`` — T4.1.

    The per-model table the plan makes mandatory. ``counter`` is a parameter and not a module
    constant precisely so a real tokenizer can be passed once one is loaded, without touching this
    file: the seven realized tables and the reference table are then the *same* computation over
    different counters, which is what makes them comparable.
    """
    by_domain: dict[str, int] = {}
    by_stratum: dict[str, int] = {}
    for doc in docs:
        n = counter.count(doc.text)
        by_domain[doc.domain] = by_domain.get(doc.domain, 0) + n
        key = f"{doc.domain}/{doc.lang}" if doc.lang else doc.domain
        by_stratum[key] = by_stratum.get(key, 0) + n
    total = sum(by_domain.values())
    return {
        "total_tokens": total,
        "by_domain": {d: by_domain[d] for d in sorted(by_domain)},
        "by_domain_share": {d: (by_domain[d] / total if total else 0.0) for d in sorted(by_domain)},
        "by_stratum": {k: by_stratum[k] for k in sorted(by_stratum)},
        "by_stratum_share": {
            k: (by_stratum[k] / total if total else 0.0) for k in sorted(by_stratum)
        },
    }


# The T4.4 fallback ladder, in order. ``hidden_scale`` multiplies the requested subsample;
# ``hidden_subsample`` of 0 means no hidden states at all. ``keep_topk`` is True on every rung and
# there is no rung where it is not: topk.bin is the labels, so dropping it does not degrade the
# experiment, it deletes it. Rung 2 of the plan's ladder (random projection to 512 dims) and rung 3
# (subset of layers) are deliberately absent -- a projection has to be applied at capture time, not
# chosen here, and a layer subset breaks the per-normalized-depth analysis of T9.4.
FALLBACK_LADDER: tuple[dict[str, Any], ...] = (
    {"step": 0, "hidden_tokens": None, "keep_logits": True, "keep_topk": True,
     "note": "requested subsample, all streams"},
    {"step": 1, "hidden_tokens": 32_000, "keep_logits": True, "keep_topk": True,
     "note": "T4.4 rung 1: hidden subsample 50k -> 32k; costs F5 capacity"},
    {"step": 2, "hidden_tokens": 16_000, "keep_logits": True, "keep_topk": True,
     "note": "hidden subsample 16k; F5 becomes a weak lower bound"},
    {"step": 3, "hidden_tokens": 16_000, "keep_logits": False, "keep_topk": True,
     "note": "drop logits.bin; T8.2 margin analysis is lost, labels are not"},
    {"step": 4, "hidden_tokens": 0, "keep_logits": False, "keep_topk": True,
     "note": "topk.bin only; F1/F2/F3/F6 still computable, F4/F5 are not"},
)
assert all(rung["keep_topk"] for rung in FALLBACK_LADDER), "topk.bin is never dropped (T4.4)"


def _multiples_below(x: int, stride: int) -> int:
    """How many multiples of ``stride`` are in ``[0, x)``."""
    return (x + stride - 1) // stride


def hidden_budget_report(
    docs: Sequence[Document],
    *,
    n_experts: int,
    hidden_dim: int,
    subsample_n: int,
    n_moe_layers: int,
    top_k: int = 8,
    max_bytes_per_shard: int = 500 * 1024**2,
) -> dict[str, Any]:
    """Per-shard capture bytes and the T4.4 fallback recommendation — resolves O2.

    Hidden state is fp16, ``hidden_dim`` per MoE layer per *captured* token, and the subsample is
    taken by global stride (``moe_trace`` keys ``capture[i]`` on the global token index, so which
    tokens carry hidden state is a property of corpus position and survives re-sharding). Captured
    counts per shard are therefore computed from each shard's global token range, not assumed
    proportional — a shard of long documents and a shard of short ones both get their true count.

    Returns the ladder with each rung's worst-shard bytes and whether it clears
    ``max_bytes_per_shard``, plus ``recommended``: the first rung that clears it. ``keep_topk`` is
    True on every rung by construction (:data:`FALLBACK_LADDER`).
    """
    if subsample_n <= 0:
        raise CorpusSpecError("subsample_n must be positive")
    table = shard_table(docs)
    total_tokens = sum(row["n_tokens_ref"] for row in table)
    if not total_tokens:
        raise CorpusSpecError("cannot budget an empty corpus")

    hidden_bytes_per_token = n_moe_layers * hidden_dim * 2
    logits_bytes_per_token = n_moe_layers * n_experts * 2
    topk_bytes_per_token = n_moe_layers * top_k * 4

    def shards_for(hidden_tokens: int, keep_logits: bool) -> list[dict[str, Any]]:
        stride = max(1, total_tokens // hidden_tokens) if hidden_tokens else 0
        rows = []
        start = 0
        for row in table:
            end = start + row["n_tokens_ref"]
            captured = (
                _multiples_below(end, stride) - _multiples_below(start, stride) if stride else 0
            )
            hidden = captured * hidden_bytes_per_token
            logits = row["n_tokens_ref"] * logits_bytes_per_token if keep_logits else 0
            topk = row["n_tokens_ref"] * topk_bytes_per_token
            rows.append(
                {
                    "shard_id": row["shard_id"],
                    "n_tokens_ref": row["n_tokens_ref"],
                    "n_captured": captured,
                    "hidden_bytes": hidden,
                    "logits_bytes": logits,
                    "topk_bytes": topk,
                    "total_bytes": hidden + logits + topk,
                    "within_cap": hidden + logits + topk <= max_bytes_per_shard,
                }
            )
            start = end
        return rows

    ladder: list[dict[str, Any]] = []
    for rung in FALLBACK_LADDER:
        if rung["hidden_tokens"] is None:
            hidden_tokens = subsample_n
        elif rung["hidden_tokens"] == 0:
            hidden_tokens = 0
        else:
            # A rung never *raises* the subsample: if the caller already asked for less than the
            # rung's cap, stepping down the ladder must not silently ask for more capture.
            hidden_tokens = min(subsample_n, rung["hidden_tokens"])
        rows = shards_for(hidden_tokens, rung["keep_logits"])
        ladder.append(
            {
                **rung,
                "hidden_subsample": hidden_tokens,
                "worst_shard_bytes": max((r["total_bytes"] for r in rows), default=0),
                "shards_over_cap": [r["shard_id"] for r in rows if not r["within_cap"]],
                "within_cap": all(r["within_cap"] for r in rows),
            }
        )

    baseline = shards_for(subsample_n, True)
    recommended = next((r for r in ladder if r["within_cap"]), None)
    return {
        "max_bytes_per_shard": max_bytes_per_shard,
        "total_tokens_ref": total_tokens,
        "bytes_per_token": {
            "hidden": hidden_bytes_per_token,
            "logits": logits_bytes_per_token,
            "topk": topk_bytes_per_token,
        },
        "shards": baseline,
        "shards_over_cap": [r["shard_id"] for r in baseline if not r["within_cap"]],
        "within_cap": all(r["within_cap"] for r in baseline),
        "ladder": ladder,
        "recommended": recommended,
        # Stated in the payload as well as in the code so a downstream reader of the JSON sidecar
        # cannot conclude from silence that dropping the labels is an option (T4.4).
        "topk_never_dropped": True,
    }


# Key order matters. ``moe_trace``'s parser (src/capture/moe_trace.cpp, parse_jsonl_line) finds the
# fields by searching the raw line for "doc_id" and "text", so both must appear before any field
# whose *value* could contain those substrings. doc_id first, text second, everything else after --
# extra keys are then simply never reached and are tolerated.
CORPUS_FIELDS: tuple[str, ...] = (
    "doc_id",
    "text",
    "domain",
    "lang",
    "source",
    "n_tokens_ref",
    "split",
    "shard_id",
)


def write_corpus(path: str | Path, docs: Sequence[Document], spec: CorpusSpec) -> Path:
    """Write ``<path>`` as JSONL and ``<path>.meta.json`` as the sidecar manifest.

    ``ensure_ascii=True``: the C++ parser is byte-oriented, decodes ``\\uXXXX`` itself, and a line is
    a hard error if it does not parse. Escaping everything non-ASCII keeps the file inside the subset
    that parser is known to handle and keeps it diffable, which matters because 30% of the corpus is
    non-Latin.

    Non-BMP characters are safe. They are escaped here as surrogate *pairs*, and the C++ parser
    recombines them into 4-byte UTF-8 and rejects an unpaired surrogate. That was not true when this
    module was first written -- the parser decoded each ``\\uXXXX`` independently, which turned every
    emoji or CJK-extension character into CESU-8 mojibake rather than a parse error, biased toward
    the 30% multilingual share. Fetching therefore does **not** need to strip non-BMP text; if this
    file is ever pointed at an older harness binary, that stops being true.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for doc in docs:
            row = {k: getattr(doc, k) for k in CORPUS_FIELDS}
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")

    meta = {
        "spec": spec.to_dict(),
        "n_docs": len(docs),
        "realized": realized_shares(docs, CharRatioCounter()),
        "split_counts": _split_counts(docs),
        "shards": shard_table(docs),
        "counter": "CharRatioCounter(chars_per_token=4.0) -- reference proxy, not a real tokenizer",
    }
    sidecar = path.with_name(path.name + ".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_corpus(path: str | Path) -> list[Document]:
    """Read a corpus file back. Blank lines are tolerated; anything else that fails is fatal."""
    docs: list[Document] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusSpecError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
        missing = set(CORPUS_FIELDS) - set(row)
        if missing:
            raise CorpusSpecError(f"{path}:{lineno}: missing field(s) {sorted(missing)}")
        docs.append(Document(**{k: row[k] for k in CORPUS_FIELDS}))
    return docs


def load_doc_splits(path: str | Path) -> dict[int, str]:
    """``doc_id -> split``, the mapping ``TraceReader(doc_splits=...)`` wants (T4.3).

    Read from the corpus file rather than recomputed, so that a change to the splitting code can
    never silently disagree with the splits the traces were collected under.
    """
    out: dict[int, str] = {}
    for doc in load_corpus(path):
        if doc.split is None:
            raise CorpusSpecError(f"doc {doc.doc_id} has no split; the corpus file is unusable")
        if doc.doc_id in out:
            raise CorpusSpecError(f"duplicate doc_id {doc.doc_id} in {path}")
        out[doc.doc_id] = doc.split
    return out


def _split_counts(docs: Sequence[Document]) -> dict[str, Any]:
    per_split: dict[str, int] = {}
    per_split_tokens: dict[str, int] = {}
    per_stratum: dict[str, dict[str, int]] = {}
    for doc in docs:
        split = doc.split or "unassigned"
        per_split[split] = per_split.get(split, 0) + 1
        per_split_tokens[split] = per_split_tokens.get(split, 0) + doc.n_tokens_ref
        key = f"{doc.domain}/{doc.lang}" if doc.lang else doc.domain
        per_stratum.setdefault(key, {})[split] = per_stratum.setdefault(key, {}).get(split, 0) + 1
    return {"docs": per_split, "tokens_ref": per_split_tokens, "by_stratum": per_stratum}


@dataclass
class BuildResult:
    """What :func:`build` produced. ``docs`` are in write order."""

    spec: CorpusSpec
    docs: list[Document]
    shards: list[dict[str, Any]]
    realized: dict[str, Any]
    split_counts: dict[str, Any]

    @property
    def n_shards(self) -> int:
        return len(self.shards)

    @property
    def total_tokens_ref(self) -> int:
        return sum(d.n_tokens_ref for d in self.docs)


def build(spec: CorpusSpec, docs: Iterable[Document]) -> BuildResult:
    """Validate, split, interleave and shard — the whole of T4.2/T4.3 in one call.

    Validation is fatal, never a warning (:class:`CorpusSpecError`): the corpus is written once and
    then seven models are traced against it, so anything wrong here is discovered after the GPU
    quota is spent. In particular a document over ``spec.max_doc_tokens`` is refused by name, because
    ``moe_trace`` would silently truncate it at ``-c 2048`` and the trace would then disagree with
    ``n_tokens_ref`` with nothing in the output saying so.
    """
    docs = list(docs)
    if not docs:
        raise CorpusSpecError("cannot build a corpus from zero documents")

    seen: set[int] = set()
    for doc in docs:
        if doc.doc_id in seen:
            raise CorpusSpecError(f"duplicate doc_id {doc.doc_id}")
        seen.add(doc.doc_id)
        if doc.domain not in DOMAINS:
            raise CorpusSpecError(f"doc {doc.doc_id}: unknown domain {doc.domain!r}")
        if doc.n_tokens_ref <= 0:
            raise CorpusSpecError(
                f"doc {doc.doc_id}: n_tokens_ref is {doc.n_tokens_ref}; moe_trace treats a "
                "zero-token document as a hard error"
            )
        if doc.n_tokens_ref > spec.max_doc_tokens:
            raise CorpusSpecError(
                f"doc {doc.doc_id} (source {doc.source!r}, domain {doc.domain!r}) is "
                f"{doc.n_tokens_ref} reference tokens, over max_doc_tokens "
                f"({spec.max_doc_tokens}); truncate it at fetch time -- capture at "
                f"-c {spec.max_doc_tokens} would drop the tail silently"
            )
        if not doc.text:
            raise CorpusSpecError(f"doc {doc.doc_id}: empty text")

    assign_splits(docs, ratios=spec.split_ratios, seed=spec.seed)
    ordered = interleave_documents(docs, seed=spec.seed)
    assign_shards(ordered, shard_tokens=spec.shard_tokens)

    return BuildResult(
        spec=spec,
        docs=ordered,
        shards=shard_table(ordered),
        realized=realized_shares(ordered, CharRatioCounter()),
        split_counts=_split_counts(ordered),
    )
