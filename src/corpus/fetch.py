"""Corpus fetch stage — plan T4.1 (the half :mod:`src.corpus.spec` and :mod:`src.corpus.build` leave
open). Turns a :class:`~src.corpus.spec.CorpusSpec` into the ``list[Document]`` that
:func:`~src.corpus.build.build` accepts.

``spec.py`` declares *what* the corpus is; ``build.py`` splits, interleaves and shards *whatever it is
handed*. This module is the only place that talks to a dataset, and five rules in it are load-bearing:

1. **The real reader streams.** ``HFDatasetSource`` passes ``streaming=True`` unconditionally. The
   volume sources (fineweb-edu ``sample-10BT``, starcoderdata, open-web-math, CulturaX) are hundreds
   of gigabytes; a non-streaming ``load_dataset`` would try to materialise them into a Kaggle session
   with ~57 GB of scratch and ~20 GB of writable output, and the session would die during *download*,
   before a single document existed. We need ~1M reference tokens, i.e. a few thousand documents, so
   streaming is not an optimisation here — it is the difference between working and not.

2. **The reader is a seam, not a hard dependency.** Every filter, every budget rule and the whole of
   :func:`fetch_corpus` is exercised offline through :class:`InMemorySource`, which is a shipped
   implementation and not a test double: it is also how a corpus is rebuilt from a cached snapshot of
   raw texts without touching the Hub again (T10.1 reproducibility). ``datasets`` is imported *inside*
   :meth:`HFDatasetSource._load`, so importing this module never requires it.

3. **Fetch must guarantee ``max_doc_tokens``, not hope for it.** ``build()`` treats an over-cap
   document as a hard :class:`CorpusSpecError`, and I15 says truncation is per model anyway. Over-cap
   documents are therefore **truncated here, never dropped** — see :func:`truncate_to_tokens` for why
   dropping them would silently change the corpus into "the internet's short documents".

4. **A parallel control running dry is normal; a volume source running dry is a finding.**
   FLORES-200 devtest is ~1012 sentences per language, so ``flores200`` cannot fill even its capped
   share of a 4M-token corpus. That shortfall is recorded and reallocated to the domain's volume
   sources. A ``volume`` source running dry instead changes the composition the paper claims, so it
   lands in :attr:`FetchResult.problems`, flips :attr:`FetchResult.ok`, warns, and fails the CLI.

5. **Every dropped document is countable by reason.** :class:`SourceReport.dropped` is broken out by
   pipeline stage. A 40% drop rate is a fact about the corpus that has to be visible *before* the GPU
   quota is spent, not inferred afterwards from a token count that came in low.

Nothing here strips non-BMP text: :func:`~src.corpus.build.write_corpus` escapes it as surrogate
pairs and the C++ parser recombines them correctly (see that function's docstring).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from .build import CharRatioCounter, Document, TokenCounter, build, realized_shares, write_corpus
from .spec import DOMAINS, MIXED_V1, MIXED_V1_SCALE, CorpusSpec, SourceSpec

__all__ = [
    "DROP_REASONS",
    "FetchError",
    "FetchWarning",
    "DocumentSource",
    "InMemorySource",
    "HFDatasetSource",
    "NearDuplicateFilter",
    "SourceReport",
    "FetchResult",
    "normalise_whitespace",
    "truncate_to_tokens",
    "within_length_bounds",
    "shingle_signature",
    "looks_like_language",
    "fetch_corpus",
    "main",
]


class FetchError(Exception):
    """Fetching is impossible or the result is unusable. Fatal, like :class:`CorpusSpecError`."""


class FetchWarning(UserWarning):
    """A volume source ran dry, or realized shares drifted from the target. Never silent."""


DROP_REASONS: tuple[str, ...] = (
    "empty_after_normalise",
    "too_short",
    "over_max_tokens",
    "lang_mismatch",
    "duplicate_exact",
    "duplicate_near",
    "parallel_row_incomplete",
)
"""Every reason a fetched record can fail to become a :class:`Document`. Reports carry all of these
keys whether or not they fired, so a zero is a measured zero and not a missing field."""

# A document shorter than this contributes almost nothing and costs a doc-level split slot. F3/F6
# condition on the previous token *in the same document* and exclude doc-initial rows (I11), so an
# n-token document yields n-1 usable conditioned rows -- at n=8 that is mostly overhead, and short web
# records are overwhelmingly navigation boilerplate, which is also what the dedup stage would then
# spend its sketch budget on.
DEFAULT_MIN_DOC_TOKENS = 24


# --------------------------------------------------------------------------------------------------
# The source seam
# --------------------------------------------------------------------------------------------------


@runtime_checkable
class DocumentSource(Protocol):
    """Yields raw text records for one :class:`SourceSpec`.

    Deliberately tiny: one method, no lifecycle, no cursor. Everything that makes a corpus a corpus
    (budgets, cleaning, dedup, doc_id assignment) lives in :func:`fetch_corpus`, so a new backend is
    an iterator over strings and nothing else.

    ``lang`` is ``None`` for a source with no language dimension (``source.langs`` empty) and
    otherwise one member of ``source.langs``; the caller drives the language loop so that a parallel
    control can be read index-major across languages (see :func:`fetch_corpus`). ``limit`` is an
    upper bound on records, not a request for exactly that many.

    The returned metadata dict is provenance only -- dataset revision, record index, original id --
    and is never required to contain anything. It exists so a surprising document can be traced back
    to its record without re-deriving the stream.
    """

    def iter_texts(
        self, source: SourceSpec, *, lang: str | None = None, limit: int | None = None
    ) -> Iterator[tuple[str, dict]]: ...


@dataclass
class InMemorySource:
    """A :class:`DocumentSource` backed by pre-supplied texts.

    Shipped, not a mock. Two uses: it is how the offline tests drive the production path through
    :func:`fetch_corpus`, and it is how a corpus is rebuilt from a cached snapshot of raw records
    (``{source_name: {lang: [text, ...]}}``) without hitting the Hub, which is what makes the T10.1
    artifact reproducible by someone who cannot get CulturaX access.

    ``texts`` maps source name to either a flat sequence (no language dimension) or a mapping of
    language to sequence. A source asked for a language it has no entry for yields nothing, which
    :func:`fetch_corpus` records as a shortfall rather than treating as an error -- a snapshot
    legitimately may not cover every language in the spec.
    """

    texts: Mapping[str, Sequence[str] | Mapping[str, Sequence[str]]]

    def iter_texts(
        self, source: SourceSpec, *, lang: str | None = None, limit: int | None = None
    ) -> Iterator[tuple[str, dict]]:
        entry = self.texts.get(source.name)
        if entry is None:
            return
        if lang is None:
            if isinstance(entry, Mapping):
                raise FetchError(
                    f"snapshot for {source.name!r} is keyed by language but the source declares no "
                    "langs; one of the two is wrong and guessing would silently drop text"
                )
            records: Sequence[str] = entry
        else:
            if not isinstance(entry, Mapping):
                raise FetchError(
                    f"source {source.name!r} declares langs {list(source.langs)} but its snapshot is "
                    "a flat sequence; a per-language source needs {lang: [text, ...]} or every "
                    "language would be handed the same text and dedup would delete all but the first"
                )
            records = entry.get(lang, ())
        for index, text in enumerate(records):
            if limit is not None and index >= limit:
                return
            yield text, {"source": source.name, "lang": lang, "index": index, "origin": "in_memory"}


# FLORES-200 configs are three-letter ISO 639-3 plus a script tag, not the two-letter codes the spec
# uses for the volume sources, and there is no way to derive one from the other. The mapping is data,
# so it lives here rather than in spec.py, which must stay a pure declaration.
#
# ``load_dataset("facebook/flores", "<code>", split="devtest")`` takes exactly one of these per call
# (the hyphenated ``eng_Latn-xxx_Yyyy`` pairs and ``all`` are the other two forms, and neither is what
# a per-language read wants). Getting a code wrong is not a soft failure: an unknown config raises at
# collection time, and a *valid but wrong* one -- ``zho_Hant`` for ``zh``, ``arb_Latn`` for ``ar`` --
# silently fills the parallel control with the wrong script, which is precisely the variable T9.4
# measures. Codes verified against facebookresearch/flores' flores200 language list.
FLORES_LANG_CONFIGS: dict[str, str] = {
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "en": "eng_Latn",
}

# Field holding the document text, per dataset. Checked in order; the first present wins.
TEXT_FIELDS: tuple[str, ...] = ("text", "content", "sentence", "raw_content")


@dataclass
class HFDatasetSource:
    """The real reader: ``datasets.load_dataset(..., streaming=True)``.

    Streaming is not configurable. See rule 1 in the module docstring -- a non-streaming read of
    ``sample-10BT`` or CulturaX does not fit in the session that would be doing the reading, and the
    failure would come as an out-of-disk hours in, with nothing collected.

    ``datasets`` is imported inside :meth:`_load` so that importing :mod:`src.corpus.fetch` (and
    therefore running the offline tests, and the CLI's ``--help``) never needs it. If it is absent the
    error says what to install rather than surfacing as a bare ``ModuleNotFoundError`` from a module
    the caller never mentioned.
    """

    revision: str | None = None
    lang_configs: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """Per-source override, ``{source_name: {spec_lang: hf_config}}``, consulted before
    :meth:`_lang_config` falls back on the source's role. Keyed by source *name* because it exists to
    handle a dataset this module does not know about; anything it does know about belongs in a
    role-keyed default, not here."""
    text_fields: Mapping[str, str] = field(default_factory=dict)
    """``{source_name: field}`` override for a dataset whose text is not in :data:`TEXT_FIELDS`."""
    trust_remote_code: bool = False

    def _lang_config(self, source: SourceSpec, lang: str) -> str:
        """The dataset's own name for ``lang``. Resolved by *role*, not by source name.

        A previous version keyed the FLORES table on the source name ``"flores200"``, which meant the
        translation only fired for a spec that happened to use that exact string: rename the source,
        or point a second spec at FLORES under another name, and the bare ``zh`` from
        :mod:`~src.corpus.spec` went to the Hub unchanged. That is not a crash -- ``zh`` is simply not
        a FLORES config, so it raises at collection time on Kaggle, hours in and after the volume
        sources have already been streamed -- and for a code that *is* a valid config it would not
        even do that, it would quietly return a different language's sentences.

        The naming is a property of the *kind* of source, so that is what it keys on: in this plan a
        ``parallel_control`` is FLORES-200 (spec.py rule 1 -- that role exists for it), and FLORES-200
        names its configs in ISO 639-3 + script. A ``volume`` source keeps the spec's own code, which
        is what CulturaX and mC4 use. :attr:`lang_configs` overrides both for a source that does
        neither.
        """
        override = self.lang_configs.get(source.name)
        if override and lang in override:
            return override[lang]
        if source.role == "parallel_control":
            try:
                return FLORES_LANG_CONFIGS[lang]
            except KeyError:
                raise FetchError(
                    f"parallel control {source.name!r} declares language {lang!r}, which has no "
                    f"FLORES-200 config in FLORES_LANG_CONFIGS. Add it (ISO 639-3 + script, e.g. "
                    f"'zho_Hans') rather than letting {lang!r} through: a spec-level code is not a "
                    "FLORES config, and passing one through either fails at collection time or, if "
                    "it collides with a real config, fills the control with the wrong language."
                ) from None
        return lang

    def config_for(self, source: SourceSpec, lang: str | None) -> str | None:
        """Which HF config to open. A per-language source needs one config *per language*.

        CulturaX and FLORES both expose language as the config, not as a column, so iterating
        languages means re-opening the dataset. ``hf_config`` may contain ``{lang}`` for datasets that
        template it (mC4's ``multilingual``-style names) so a substitution does not need code.
        """
        if lang is None:
            return source.hf_config
        mapped = self._lang_config(source, lang)
        if source.hf_config and "{lang}" in source.hf_config:
            return source.hf_config.format(lang=mapped)
        return mapped

    def _load(self, source: SourceSpec, lang: str | None) -> Iterable[Mapping[str, Any]]:
        # Resolved before the try so that an unresolvable language surfaces as its own FetchError
        # rather than being swallowed by the "could not open" handler below, which would report it as
        # a Hub problem and send the reader looking in the wrong place.
        config = self.config_for(source, lang)
        try:
            from datasets import load_dataset  # noqa: PLC0415 -- lazy on purpose, see class docstring
        except ImportError as exc:
            raise FetchError(
                "HFDatasetSource needs the 'datasets' package, which is not installed in this "
                "environment (it is deliberately absent from requirements-local.txt: the local venv "
                "runs the offline tests only). Install it in the Kaggle CPU session with "
                "`pip install datasets`, or rebuild the corpus offline from a cached snapshot with "
                "InMemorySource, which drives the identical fetch path."
            ) from exc
        try:
            return load_dataset(
                source.hf_dataset,
                config,
                split=source.hf_split,
                streaming=True,  # rule 1: never negotiable
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001 -- gated datasets, missing configs, network, all fatal
            raise FetchError(
                f"could not open {source.hf_dataset!r} "
                f"(config={config!r}, split={source.hf_split!r}): {exc}. "
                "If this is an access-gated dataset, record the substitution in spec.py's "
                "SourceSpec.substituted_for rather than swapping it silently (T10.1)."
            ) from exc

    def _text_of(self, source: SourceSpec, record: Mapping[str, Any]) -> str:
        override = self.text_fields.get(source.name)
        keys = (override,) if override else TEXT_FIELDS
        for key in keys:
            if key in record:
                return str(record[key])
        raise FetchError(
            f"source {source.name!r} ({source.hf_dataset}): no text field among {list(keys)}; "
            f"record has {sorted(record)}. Add an entry to HFDatasetSource.text_fields."
        )

    def iter_texts(
        self, source: SourceSpec, *, lang: str | None = None, limit: int | None = None
    ) -> Iterator[tuple[str, dict]]:
        stream = self._load(source, lang)
        config = self.config_for(source, lang)
        for index, record in enumerate(stream):
            if limit is not None and index >= limit:
                return
            yield (
                self._text_of(source, record),
                {
                    "source": source.name,
                    "hf_dataset": source.hf_dataset,
                    "hf_config": config,
                    "hf_split": source.hf_split,
                    "revision": self.revision,
                    "lang": lang,
                    "index": index,
                },
            )


# --------------------------------------------------------------------------------------------------
# Cleaning pipeline. One named function per rule so each is reviewable and testable on its own.
# --------------------------------------------------------------------------------------------------

_BLANK_RUN = re.compile(r"\n{3,}")
# Control characters are stripped except tab and newline. A stray NUL or 0x1a in a crawled record ends
# up inside a JSON string literal that the byte-oriented C++ parser (src/capture/moe_trace.cpp) reads
# without validating, and the failure mode is a mis-parsed line rather than an error.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def normalise_whitespace(text: str) -> str:
    """Normalise line endings, trailing space and blank runs. **Never collapses indentation.**

    The obvious one-liner -- ``" ".join(text.split())`` -- is wrong here, and wrong in a way that
    would not show up in any token count. Code is 25% of the corpus (T4.1); flattening a Python or
    Rust file's leading whitespace destroys the block structure that makes it code at all, so the
    "code" condition would become "code-flavoured word salad" and every domain-stratified claim in
    T9.4 about code routing would be about a domain we never actually collected. Math is worse still:
    aligned LaTeX environments carry meaning in their columns.

    So: CRLF/CR to LF, NFC (the same string arriving in two normal forms would defeat dedup and split
    one document's tokens across two byte sequences), control characters out, per-line *trailing*
    whitespace out, runs of blank lines squeezed to one, outer blank lines out. Intra-line whitespace,
    and in particular leading indentation, is left exactly as it was.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip("\n")


def truncate_to_tokens(
    text: str, counter: TokenCounter, max_tokens: int, *, boundary_slack: float = 0.15
) -> tuple[str, bool]:
    """Return ``(text, was_truncated)`` with ``counter.count(text) <= max_tokens``.

    **Truncate, do not drop.** Three reasons, in order of how badly dropping would hurt:

    * The plan says so (T4.2: "Documents truncated to <= 2048 tokens (approx, by character
      heuristic)"), and I15 says truncation happens per model anyway, so a document at the reference
      cap is already going to be cut for the wider-vocabulary checkpoints. Refusing over-cap
      documents here would not avoid truncation, it would only move it.
    * Dropping is a *selection* on length, and length correlates with everything we care about. Web
      prose, open-web-math and starcoderdata documents are mostly well over 2048 reference tokens;
      keeping only the ones that happen to fit means keeping stubs, navigation pages and one-function
      files. The corpus would then differ from the plan's composition in a way no share table shows,
      because the shares would still be exactly 25/25/20/30.
    * ``build()`` refuses an over-cap document by name, so "guarantee, do not hope" is the only
      contract that lets the fetcher's own output be buildable. The tests assert this by calling
      ``build()`` on the fetch result.

    The counter is a :class:`TokenCounter` and therefore not invertible, so the cut point is found by
    binary search on the character prefix -- correct for any monotone counter, including a real
    tokenizer, not just :class:`CharRatioCounter`'s character ratio. The cut is then pulled back to
    the last newline (else space) within ``boundary_slack`` of the end, so a truncated code file ends
    at a line boundary rather than mid-identifier; pulling *back* only shortens, so the bound holds.
    """
    if max_tokens <= 0:
        raise FetchError("max_tokens must be positive")
    if counter.count(text) <= max_tokens:
        return text, False

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]

    floor = int(len(cut) * (1.0 - boundary_slack))
    for sep in ("\n", " "):
        idx = cut.rfind(sep)
        if idx >= floor and idx > 0:
            cut = cut[:idx]
            break
    trimmed = cut.rstrip()
    # A document that is all whitespace up to the cut would be emptied here; keep the raw prefix and
    # let within_length_bounds reject it, so the drop is attributed to a length rule and not silently
    # to truncation.
    return (trimmed or cut, True)


def within_length_bounds(
    text: str, counter: TokenCounter, *, min_tokens: int, max_tokens: int
) -> str | None:
    """``None`` if acceptable, else the drop reason. Assumes :func:`truncate_to_tokens` already ran."""
    if not text.strip():
        return "empty_after_normalise"
    n = counter.count(text)
    if n < min_tokens:
        return "too_short"
    if n > max_tokens:
        # Unreachable while :func:`truncate_to_tokens` runs first, and reported as a counted drop
        # rather than raised so that a future reordering of the pipeline shows up in the report
        # instead of as a crashed fetch -- but it must never be zero-cost to reach, because build()
        # rejects an over-cap document outright and the whole point of this stage is that it cannot.
        return "over_max_tokens"
    return None


# Near-duplicate detection: bottom-k sketch of word shingles, indexed in bands.
_SHINGLE_WORDS = 5
_SHINGLE_CHARS = 12
_SKETCH_SIZE = 16
_BAND_SIZE = 4
_WORD = re.compile(r"\w+", re.UNICODE)
_WS = re.compile(r"\s+")

# Scripts written without word spacing. Word shingles degenerate on these -- ``\w+`` returns one
# enormous "word" for a Chinese paragraph, so a word-shingle sketch would collapse to a single hash and
# near-duplicate detection would silently become exact matching on exactly the part of the corpus it
# matters most for (the non-Latin multilingual share, 30% of the total). Those documents get character
# shingles instead.
_UNSPACED_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK
    (0xF900, 0xFAFF),  # CJK compatibility
    (0x0E00, 0x0E7F),  # Thai
    (0x20000, 0x2A6DF),  # CJK ext B
)
UNSPACED_SCRIPT_FRACTION = 0.20


def _hash64(payload: str) -> int:
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "big")


def _is_unspaced(text: str) -> bool:
    body = [ch for ch in text if not ch.isspace()]
    if not body:
        return False
    hits = sum(1 for ch in body if any(lo <= ord(ch) <= hi for lo, hi in _UNSPACED_RANGES))
    return hits / len(body) >= UNSPACED_SCRIPT_FRACTION


def shingle_signature(text: str) -> tuple[int, ...]:
    """Bottom-:data:`_SKETCH_SIZE` sketch of shingle hashes — an order-invariant fingerprint.

    A bottom-k sketch rather than k independent MinHashes: one pass, one hash function, and the same
    expected Jaccard estimate, which matters because this runs on every fetched record in a CPU
    session with no budget for 128 permutations.

    The shingle unit is 5 words for space-delimited text and :data:`_SHINGLE_CHARS` characters for
    text that is mostly an unspaced script (see :data:`_UNSPACED_RANGES`). Documents too short for
    either fall back to a single whole-text hash, i.e. exact matching only -- near-duplicate detection
    on a four-word string is noise, and such documents are below :data:`DEFAULT_MIN_DOC_TOKENS` anyway.
    """
    lowered = text.lower()
    words = _WORD.findall(lowered)
    if _is_unspaced(lowered) or len(words) < _SHINGLE_WORDS:
        squeezed = _WS.sub(" ", lowered).strip()
        if len(squeezed) >= 2 * _SHINGLE_CHARS:
            units = {
                squeezed[i : i + _SHINGLE_CHARS]
                for i in range(len(squeezed) - _SHINGLE_CHARS + 1)
            }
        else:
            return (_hash64(squeezed),)
    else:
        units = {
            " ".join(words[i : i + _SHINGLE_WORDS])
            for i in range(len(words) - _SHINGLE_WORDS + 1)
        }
    return tuple(sorted(_hash64(u) for u in units)[:_SKETCH_SIZE])


@dataclass
class NearDuplicateFilter:
    """Exact hash plus banded bottom-k sketch. ``check`` returns a drop reason or ``None``.

    **Why exact hashing alone is not enough here.** CulturaX is built from mC4 and OSCAR, so it
    already contains the same crawled pages twice with different boilerplate; fineweb-edu and CulturaX
    both draw on Common Crawl; starcoderdata contains vendored and forked copies of the same files
    with a changed header. All of those are byte-*different* and semantically identical. Repetition is
    the one thing that inflates every predictability number in this paper at once: F1's count table
    memorises a repeated document's token IDs, F3's previous-token conditioning becomes trivial across
    a duplicated span, and the T9.4 multilingual comparison -- the reason the multilingual share is
    30% -- turns into a measurement of how often a boilerplate footer recurs. A duplicate in the
    corpus does not add noise, it adds a *bias in the direction of the hypothesis*.

    **Accepted false-negative rate.** With a 16-value sketch read as 4 bands of 4, a pair is caught
    when any band matches: detection is roughly ``1 - (1 - J**4)**4`` in the Jaccard similarity ``J``
    of their shingle sets -- about 98% at J=0.9, about 63% at J=0.7, about 6% at J=0.4. So we catch
    reprints, template pages and header-swapped forks, and we knowingly miss pairs that merely share
    a topic or a footer (J below ~0.5). That is the intended operating point: chasing the tail would
    need a real MinHash-LSH index over hundreds of thousands of records, and the residual risk is
    bounded from the other side by the fact that a J=0.4 pair contributes mostly distinct token
    sequences. Bottom-k also makes the estimate slightly length-biased for very short documents, which
    is a second, smaller source of false negatives on the FLORES sentences -- and there a false
    negative is harmless, because those sentences are unique by construction.

    False *positives* are the cost worth watching: a band collision drops a legitimate document. At
    this sketch size that needs 4 shared bottom-k hashes, so it takes near-identity, and every such
    drop is counted under ``duplicate_near`` where an implausible rate is visible before collection.
    """

    exact: set[str] = field(default_factory=set)
    bands: dict[tuple[int, tuple[int, ...]], int] = field(default_factory=dict)
    n_docs: int = 0

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def _band_keys(text: str) -> list[tuple[int, tuple[int, ...]]]:
        signature = shingle_signature(text)
        return [
            (i // _BAND_SIZE, tuple(signature[i : i + _BAND_SIZE]))
            for i in range(0, len(signature), _BAND_SIZE)
            if len(signature[i : i + _BAND_SIZE]) == _BAND_SIZE
        ] or [(0, signature)]

    def check(self, text: str) -> str | None:
        """Drop reason, or ``None``. Side-effect free, so a caller that may reject the document for a
        later reason (a parallel row losing one language) does not poison the index against text it
        never emitted."""
        if self._digest(text) in self.exact:
            return "duplicate_exact"
        if any(key in self.bands for key in self._band_keys(text)):
            return "duplicate_near"
        return None

    def add(self, text: str) -> None:
        self.exact.add(self._digest(text))
        for key in self._band_keys(text):
            self.bands[key] = self.n_docs
        self.n_docs += 1

    def check_and_add(self, text: str) -> str | None:
        reason = self.check(text)
        if reason is None:
            self.add(text)
        return reason


# Codepoint ranges per language, for the script sanity check. Only languages whose script is disjoint
# from Latin are listed: for de/fr/es there is no dependency-free signal that separates them from
# English, and pretending otherwise would be worse than admitting the gap.
_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "zh": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF)),
    "ja": ((0x3040, 0x30FF), (0x4E00, 0x9FFF), (0x31F0, 0x31FF)),
    "ko": ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)),
    "ru": ((0x0400, 0x04FF), (0x0500, 0x052F)),
    "uk": ((0x0400, 0x04FF),),
    "bg": ((0x0400, 0x04FF),),
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "fa": ((0x0600, 0x06FF), (0xFB50, 0xFDFF)),
    "ur": ((0x0600, 0x06FF), (0xFB50, 0xFDFF)),
    "he": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "hi": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "mr": ((0x0900, 0x097F),),
    "ne": ((0x0900, 0x097F),),
    "bn": ((0x0980, 0x09FF),),
    "ta": ((0x0B80, 0x0BFF),),
    "te": ((0x0C00, 0x0C7F),),
    "th": ((0x0E00, 0x0E7F),),
    "el": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
}

# A translated page or a code block inside a Chinese article legitimately contains long ASCII spans,
# so the bar is "some of the expected script is present", not "most of it is".
MIN_SCRIPT_FRACTION = 0.10


def looks_like_language(text: str, lang: str, *, min_fraction: float = MIN_SCRIPT_FRACTION) -> bool:
    """Heuristic script check for one language. ``True`` when we have no signal.

    **This is a heuristic, not language identification.** It answers one narrow question: does a
    document labelled with a non-Latin-script language actually contain that script? A CulturaX ``zh``
    record that is 100% ASCII is a mislabelled English page, and it lands in the single condition T9.4
    cares most about -- the non-Latin multilingual share is the whole test of the token-ID hypothesis,
    because that is where the panel's tokenizers diverge most (OLMoE fragments toward bytes at 50k
    vocab, Gemma 4 does not at 262k). Mislabelled English there does not add noise; it *reduces* the
    measured vocabulary shift and therefore biases the result toward "token-ID predictability
    survives", which is one of the answers the paper is trying to distinguish between.

    Deliberately dependency-free (no fastText, no langdetect): those are model downloads on a session
    that must not need any, and the question above does not need a classifier. The cost is that it
    cannot tell German from English, cannot tell Hindi from Marathi (shared Devanagari), and will
    accept a Japanese document labelled ``zh`` (shared Han). Languages with no entry in
    :data:`_SCRIPT_RANGES` always pass, so this filter never rejects for a reason it cannot support.
    """
    ranges = _SCRIPT_RANGES.get(lang)
    if not ranges:
        return True
    body = [ch for ch in text if not ch.isspace()]
    if not body:
        return False
    hits = sum(1 for ch in body if any(lo <= ord(ch) <= hi for lo, hi in ranges))
    return hits / len(body) >= min_fraction


# --------------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------------


@dataclass
class SourceReport:
    """Requested vs delivered vs dropped for one source. Every field is meant to be read."""

    name: str
    domain: str
    role: str
    requested_tokens: int
    """Budget from ``spec.source_token_targets()`` plus any :attr:`reallocated_tokens`."""
    base_tokens: int = 0
    """Budget before reallocation, i.e. straight from the spec."""
    reallocated_tokens: int = 0
    """Extra budget inherited from a parallel control in the same domain that ran dry."""
    delivered_tokens: int = 0
    delivered_docs: int = 0
    n_records_read: int = 0
    n_truncated: int = 0
    dropped: dict[str, int] = field(default_factory=lambda: {r: 0 for r in DROP_REASONS})
    ran_dry: bool = False
    """The source's iterator was exhausted before the budget was met."""
    by_lang_docs: dict[str, int] = field(default_factory=dict)
    by_lang_tokens: dict[str, int] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return sum(self.dropped.values())

    @property
    def shortfall_tokens(self) -> int:
        return max(0, self.requested_tokens - self.delivered_tokens)

    @property
    def drop_rate(self) -> float:
        """Dropped records over records read. The number that must be looked at before collecting."""
        return self.n_dropped / self.n_records_read if self.n_records_read else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "role": self.role,
            "requested_tokens": self.requested_tokens,
            "base_tokens": self.base_tokens,
            "reallocated_tokens": self.reallocated_tokens,
            "delivered_tokens": self.delivered_tokens,
            "delivered_docs": self.delivered_docs,
            "n_records_read": self.n_records_read,
            "n_truncated": self.n_truncated,
            "dropped": {r: self.dropped.get(r, 0) for r in DROP_REASONS},
            "n_dropped": self.n_dropped,
            "drop_rate": self.drop_rate,
            "ran_dry": self.ran_dry,
            "shortfall_tokens": self.shortfall_tokens,
            "by_lang_docs": dict(sorted(self.by_lang_docs.items())),
            "by_lang_tokens": dict(sorted(self.by_lang_tokens.items())),
        }


@dataclass
class FetchResult:
    """Documents plus the full accounting of how they were obtained."""

    spec: CorpusSpec
    docs: list[Document]
    seed: int
    reports: dict[str, SourceReport]
    realized: dict[str, Any]
    problems: list[str] = field(default_factory=list)
    """Human-readable statements of everything that makes this corpus not the one the spec asked for.
    Non-empty means the composition in the paper needs a caveat, so it is surfaced rather than logged
    -- a volume source running dry is exactly the failure that otherwise reads as a clean run."""
    notes: list[str] = field(default_factory=list)
    """Expected, benign deviations: a parallel control running dry and its reallocation."""

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def total_tokens_ref(self) -> int:
        return sum(d.n_tokens_ref for d in self.docs)

    @property
    def shortfalls(self) -> dict[str, int]:
        return {n: r.shortfall_tokens for n, r in sorted(self.reports.items()) if r.shortfall_tokens}

    @property
    def volume_shortfalls(self) -> dict[str, int]:
        """The ones that matter: a ``volume`` source that could not fill its budget."""
        return {
            n: r.shortfall_tokens
            for n, r in sorted(self.reports.items())
            if r.role == "volume" and r.shortfall_tokens
        }

    @property
    def dropped_by_reason(self) -> dict[str, int]:
        out = {r: 0 for r in DROP_REASONS}
        for report in self.reports.values():
            for reason, n in report.dropped.items():
                out[reason] = out.get(reason, 0) + n
        return out

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.name,
            "seed": self.seed,
            "target_tokens": self.spec.target_tokens,
            "n_docs": len(self.docs),
            "total_tokens_ref": self.total_tokens_ref,
            "sources": {n: r.to_dict() for n, r in sorted(self.reports.items())},
            "shortfalls": self.shortfalls,
            "volume_shortfalls": self.volume_shortfalls,
            "dropped_by_reason": self.dropped_by_reason,
            "realized": self.realized,
            "target_shares": dict(sorted(self.spec.shares.items())),
            "problems": list(self.problems),
            "notes": list(self.notes),
            "ok": self.ok,
        }


# --------------------------------------------------------------------------------------------------
# The fetcher
# --------------------------------------------------------------------------------------------------


def _resolve_sources(
    spec: CorpusSpec, sources: Mapping[str, DocumentSource] | DocumentSource
) -> dict[str, DocumentSource]:
    if isinstance(sources, Mapping):
        missing = [s.name for s in spec.sources if s.name not in sources]
        if missing:
            raise FetchError(
                f"no DocumentSource for {missing}; every source in the spec must have a reader, "
                "because a silently skipped source is a composition change (T4.1)"
            )
        return {s.name: sources[s.name] for s in spec.sources}
    return {s.name: sources for s in spec.sources}


def _lang_order(source: SourceSpec, seed: int) -> tuple[str | None, ...]:
    """Deterministic language visitation order for one source.

    Hashed on (seed, source, lang) rather than shuffled with an RNG so the order is a pure function of
    identity: adding a language to the spec must not renumber the documents of the others, or two
    corpora built from the same spec at different times would not be comparable.
    """
    if not source.langs:
        return (None,)
    return tuple(
        sorted(source.langs, key=lambda l: (_hash64(f"lang|{seed}|{source.name}|{l}"), l))
    )


def _make_doc(
    doc_id: int, text: str, source: SourceSpec, lang: str | None, n_tokens: int
) -> Document:
    # lang is "" for code, where a natural-language label would create an empty stratum, and "en" for
    # English prose/math -- Document.lang's own contract (build.py).
    if lang is not None:
        label = lang
    elif source.domain == "code":
        label = ""
    else:
        label = "en"
    return Document(
        doc_id=doc_id,
        text=text,
        domain=source.domain,
        lang=label,
        source=source.name,
        n_tokens_ref=n_tokens,
    )


def fetch_corpus(
    spec: CorpusSpec,
    sources: Mapping[str, DocumentSource] | DocumentSource,
    *,
    counter: TokenCounter | None = None,
    seed: int = 0,
    min_doc_tokens: int = DEFAULT_MIN_DOC_TOKENS,
    dedup: NearDuplicateFilter | None = None,
    warn: bool = True,
) -> FetchResult:
    """Fill ``spec.source_token_targets()`` from ``sources`` and return buildable documents.

    Deterministic given ``(spec, sources, seed)``: source visitation order comes from the spec, the
    language order from :func:`_lang_order`, and ``doc_id`` is assigned sequentially in that fetch
    order. No RNG is instantiated anywhere in this module -- ``build()`` re-derives split and shard
    assignment from ``(doc_id, spec.seed)`` alone, so the fetcher only has to be reproducible, not
    random. ``seed`` is threaded separately from ``spec.seed`` so that mixed-v1-scale can draw
    different text from the same spec shape (its own ``seed=1``).

    Within each domain, **parallel controls are read first**. They are capped by their own size, not
    by the budget: FLORES-200 devtest is ~1012 sentences per language, so running dry is the expected
    outcome and is recorded in :attr:`FetchResult.notes`, with the unspent remainder added to the
    domain's volume sources in proportion to their weights. Reading them first is what makes that
    reallocation possible in one pass. A ``volume`` source running dry is the opposite: it silently
    changes the composition the paper claims, so it goes to :attr:`FetchResult.problems`, warns, and
    makes :attr:`FetchResult.ok` false.

    **FLORES alignment is guaranteed, by row.** A parallel control is read *index-major*: index 0 of
    every language, then index 1, and so on, and a row is committed only if every language's copy
    survives the whole pipeline. If any one fails (script check, dedup, length) the entire row is
    dropped as ``parallel_row_incomplete``. Emitting a partial row would leave sentence *i* present in
    six languages and absent in two, and the T9.4 cross-language comparison would then be comparing
    different content between those languages -- which is the exact confound a parallel control exists
    to remove. The cost is a slightly smaller control; the alternative is a control that does not
    control. Alignment holds as far as the reader's own ordering does: it assumes ``iter_texts``
    yields devtest in its canonical order per language, which is true of a streamed HF split.
    """
    counter = counter or CharRatioCounter()
    readers = _resolve_sources(spec, sources)
    by_name = {s.name: s for s in spec.sources}
    base_targets = spec.source_token_targets()
    dedup = dedup if dedup is not None else NearDuplicateFilter()

    reports: dict[str, SourceReport] = {
        s.name: SourceReport(
            name=s.name,
            domain=s.domain,
            role=s.role,
            requested_tokens=base_targets.get(s.name, 0),
            base_tokens=base_targets.get(s.name, 0),
        )
        for s in spec.sources
    }
    docs: list[Document] = []
    problems: list[str] = []
    notes: list[str] = []
    next_id = 0

    def emit(text: str, source: SourceSpec, lang: str | None, n_tokens: int) -> Document:
        nonlocal next_id
        doc = _make_doc(next_id, text, source, lang, n_tokens)
        next_id += 1
        docs.append(doc)
        report = reports[source.name]
        report.delivered_docs += 1
        report.delivered_tokens += n_tokens
        key = doc.lang or "-"
        report.by_lang_docs[key] = report.by_lang_docs.get(key, 0) + 1
        report.by_lang_tokens[key] = report.by_lang_tokens.get(key, 0) + n_tokens
        return doc

    def clean(
        text: str, source: SourceSpec, lang: str | None, *, commit: bool = True
    ) -> tuple[str | None, int, str | None]:
        """Run the pipeline on one record: ``(text, n_tokens, drop_reason)``.

        Order is load-bearing. Normalise first, so dedup compares canonical text and a
        whitespace-only difference is not a "new" document. Truncate second, so the length rule and
        the dedup sketch see the text that will actually be written -- deduping the full record and
        then truncating would let two documents that differ only past the cut point both survive as
        identical corpus lines. Script check before dedup because it is cheaper than sketching.

        ``commit=False`` checks against the dedup index without inserting, for a parallel row that may
        still be rejected as a whole; the caller inserts on commit.
        """
        report = reports[source.name]
        text = normalise_whitespace(text)
        if not text:
            report.dropped["empty_after_normalise"] += 1
            return None, 0, "empty_after_normalise"
        text, was_truncated = truncate_to_tokens(text, counter, spec.max_doc_tokens)
        reason = within_length_bounds(
            text, counter, min_tokens=min_doc_tokens, max_tokens=spec.max_doc_tokens
        )
        if reason:
            report.dropped[reason] += 1
            return None, 0, reason
        if lang is not None and not looks_like_language(text, lang):
            report.dropped["lang_mismatch"] += 1
            return None, 0, "lang_mismatch"
        reason = dedup.check_and_add(text) if commit else dedup.check(text)
        if reason:
            report.dropped[reason] += 1
            return None, 0, reason
        if was_truncated:
            report.n_truncated += 1
        return text, counter.count(text), None

    # -- parallel controls first, index-major, so their shortfall can be reallocated in one pass ----
    for source in spec.sources:
        if source.role != "parallel_control":
            continue
        report = reports[source.name]
        budget = report.requested_tokens
        langs = [l for l in _lang_order(source, seed) if l is not None]
        iterators = {l: readers[source.name].iter_texts(source, lang=l) for l in langs}
        exhausted = False
        while not exhausted and report.delivered_tokens < budget:
            row: list[tuple[str, str, int]] = []
            for lang in langs:
                try:
                    raw, _meta = next(iterators[lang])
                except StopIteration:
                    exhausted = True
                    break
                report.n_records_read += 1
                text, n_tokens, _reason = clean(raw, source, lang, commit=False)
                if text is None:
                    continue
                row.append((lang, text, n_tokens))
            if exhausted or len(row) != len(langs):
                # A row that lost a language, and any partial trailing row, is discarded whole: see
                # the docstring on row-level alignment.
                report.dropped["parallel_row_incomplete"] += len(row)
                if exhausted:
                    break
                continue
            for lang, text, n_tokens in row:
                dedup.add(text)
                emit(text, source, lang, n_tokens)
        report.ran_dry = exhausted and report.delivered_tokens < budget
        if report.ran_dry:
            notes.append(
                f"parallel control {source.name!r} ran dry at {report.delivered_tokens} of "
                f"{budget} reference tokens ({report.shortfall_tokens} short). Expected: a parallel "
                f"control is capped by its own size (FLORES-200 devtest is ~1012 sentences per "
                f"language), not by the budget. Remainder reallocated to the {source.domain!r} "
                f"volume sources."
            )

    # -- reallocate every parallel shortfall to the domain's volume sources, by weight --------------
    for domain in DOMAINS:
        shortfall = sum(
            reports[s.name].shortfall_tokens
            for s in spec.sources
            if s.domain == domain and s.role == "parallel_control"
        )
        volume = [s for s in spec.sources if s.domain == domain and s.role == "volume"]
        if not shortfall or not volume:
            if shortfall and not volume:
                problems.append(
                    f"domain {domain!r} is {shortfall} reference tokens short and has no volume "
                    "source to absorb it; the domain share will come in under target"
                )
            continue
        denom = sum(s.weight for s in volume)
        handed = 0
        for i, source in enumerate(volume):
            take = shortfall - handed if i == len(volume) - 1 else int(round(shortfall * source.weight / denom))
            reports[source.name].reallocated_tokens += take
            reports[source.name].requested_tokens += take
            handed += take

    # -- volume sources ----------------------------------------------------------------------------
    for source in spec.sources:
        if source.role == "parallel_control":
            continue
        report = reports[source.name]
        budget = report.requested_tokens
        langs = _lang_order(source, seed)
        # Per-language budgets are equal within a source: the spec weights sources, not languages, and
        # an unequal multilingual mix would make the per-language cells of T9.4 different sample sizes
        # for no stated reason. A language that runs dry leaves its remainder to the languages after
        # it, because `remaining` is recomputed from what has actually been delivered.
        for i, lang in enumerate(langs):
            # Cumulative, so a language that runs dry leaves its remainder to the ones after it rather
            # than to nobody -- the source only counts as having run dry if the *whole* budget is
            # unmet after every language has been read.
            cumulative = budget * (i + 1) // len(langs)
            records = readers[source.name].iter_texts(source, lang=lang)
            while report.delivered_tokens < cumulative:
                try:
                    raw, _meta = next(records)
                except StopIteration:
                    break
                report.n_records_read += 1
                text, n_tokens, _reason = clean(raw, source, lang)
                if text is None:
                    continue
                emit(text, source, lang, n_tokens)
        report.ran_dry = report.delivered_tokens < budget
        if report.ran_dry:
            problems.append(
                f"volume source {source.name!r} ({source.domain}) ran dry at "
                f"{report.delivered_tokens} of {budget} reference tokens "
                f"({report.shortfall_tokens} short, "
                f"{report.n_dropped} of {report.n_records_read} records dropped). A volume source "
                f"failing to fill its budget changes the realized composition away from the "
                f"{spec.shares.get(source.domain, 0):.0%} share T4.1 claims for {source.domain!r}, "
                f"and nothing downstream can detect that. Inspect "
                f"FetchResult.reports[{source.name!r}].dropped before collecting."
            )

    realized = realized_shares(docs, counter) if docs else {"total_tokens": 0}
    # Share drift is reported next to the shortfalls that cause it, so the caveat the paper needs is
    # in the artifact rather than reconstructed later from the manifest.
    for domain, target in sorted(spec.shares.items()):
        if target <= 0:
            continue
        got = realized.get("by_domain_share", {}).get(domain, 0.0)
        if abs(got - target) > 0.02:
            problems.append(
                f"domain {domain!r} realized share {got:.1%} vs target {target:.0%} "
                "(T4.3 acceptance is +/-1% per domain)"
            )

    result = FetchResult(
        spec=spec,
        docs=docs,
        seed=seed,
        reports=reports,
        realized=realized,
        problems=problems,
        notes=notes,
    )
    if warn and problems:
        warnings.warn(
            "corpus fetch has "
            + str(len(problems))
            + " problem(s): "
            + " | ".join(problems),
            FetchWarning,
            stacklevel=2,
        )
    return result


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

SPECS: dict[str, CorpusSpec] = {MIXED_V1.name: MIXED_V1, MIXED_V1_SCALE.name: MIXED_V1_SCALE}


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch, build and write a corpus — the CPU-session entry point for T4.1/T4.2.

    The only place :class:`HFDatasetSource` is constructed, which is what keeps the network out of
    every other code path in the package. Exits non-zero when :attr:`FetchResult.ok` is false unless
    ``--allow-shortfall`` is passed: writing a corpus whose composition differs from the spec has to
    be a decision someone made on the command line, not a warning scrolled past in a session log.
    """
    parser = argparse.ArgumentParser(description="Fetch and write a corpus (plan T4.1/T4.2)")
    parser.add_argument("--spec", default=MIXED_V1.name, choices=sorted(SPECS))
    parser.add_argument("--out", default=None, help="default corpora/<spec>.jsonl")
    parser.add_argument("--target-tokens", type=int, default=None, help="override, e.g. 500000 if Q1 fired")
    parser.add_argument("--seed", type=int, default=None, help="fetch seed; defaults to spec.seed")
    parser.add_argument("--revision", default=None, help="pin dataset revision (recorded in the report)")
    parser.add_argument("--min-doc-tokens", type=int, default=DEFAULT_MIN_DOC_TOKENS)
    parser.add_argument("--report", default=None, help="default <out>.fetch.json")
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="write the corpus even when a volume source ran dry or a share drifted",
    )
    args = parser.parse_args(argv)

    spec = SPECS[args.spec]
    if args.target_tokens:
        from dataclasses import replace  # noqa: PLC0415 -- only the CLI overrides the spec

        spec = replace(spec, target_tokens=args.target_tokens)
    out = Path(args.out) if args.out else Path("corpora") / f"{spec.name}.jsonl"
    report_path = Path(args.report) if args.report else out.with_name(out.name + ".fetch.json")

    result = fetch_corpus(
        spec,
        HFDatasetSource(revision=args.revision),
        seed=args.seed if args.seed is not None else spec.seed,
        min_doc_tokens=args.min_doc_tokens,
        warn=False,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.to_json(), encoding="utf-8")

    for note in result.notes:
        print(f"note: {note}")
    for problem in result.problems:
        print(f"PROBLEM: {problem}")
    if not result.ok and not args.allow_shortfall:
        print(f"refusing to write {out} ({len(result.problems)} problem(s)); see {report_path}")
        print("pass --allow-shortfall to write anyway, and record the deviation in T10.1")
        return 1

    built = build(spec, result.docs)
    write_corpus(out, built.docs, spec)
    print(
        f"wrote {out}: {len(built.docs)} docs, {built.total_tokens_ref} reference tokens, "
        f"{built.n_shards} shards; report {report_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
