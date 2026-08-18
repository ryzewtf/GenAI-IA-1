"""Corpus fetch-stage tests — plan T4.1.

Offline by construction: every test drives the production path through
:class:`~src.corpus.fetch.InMemorySource`, and the one test that touches
:class:`~src.corpus.fetch.HFDatasetSource` does it against a fake ``datasets`` module in
``sys.modules``. Nothing here imports ``datasets``, opens a socket, or takes more than milliseconds.

Each test names an invariant that, if broken, would produce a corpus that *looks* correct -- right
share table, right document count -- and is quietly the wrong experiment: flattened code indentation,
a repeated document inflating F1, an English page filed under ``zh``, a FLORES row present in six
languages out of eight, or a volume source that came in 40% short with nothing saying so.
"""

from __future__ import annotations

import random
import string
import sys
import types

import pytest

from src.corpus.build import CharRatioCounter, build, load_corpus, write_corpus
from src.corpus.fetch import (
    DROP_REASONS,
    FetchError,
    FetchResult,
    FetchWarning,
    HFDatasetSource,
    InMemorySource,
    NearDuplicateFilter,
    fetch_corpus,
    looks_like_language,
    main,
    normalise_whitespace,
    shingle_signature,
    truncate_to_tokens,
)
from src.corpus.spec import CorpusSpec, SourceSpec

COUNTER = CharRatioCounter()
MAX_DOC_TOKENS = 128
LANGS = ("zh", "ru", "hi", "ar")

PROSE = SourceSpec(name="prose-vol", domain="prose", hf_dataset="fake/prose")
CODE = SourceSpec(name="code-vol", domain="code", hf_dataset="fake/code")
MATH = SourceSpec(name="math-vol", domain="math", hf_dataset="fake/math")
ML_VOL = SourceSpec(
    name="ml-vol", domain="multilingual", hf_dataset="fake/culturax", langs=LANGS, weight=4.0
)
ML_PAR = SourceSpec(
    name="ml-par",
    domain="multilingual",
    hf_dataset="fake/flores",
    role="parallel_control",
    hf_split="devtest",
    langs=LANGS,
    weight=1.0,
)
ALL_SOURCES = (PROSE, CODE, MATH, ML_VOL, ML_PAR)


# -- synthetic snapshots ---------------------------------------------------------------------------
# Word-salad text, not prose: the point is that documents are mutually dissimilar so that the dedup
# stage does not fire except where a test asks it to, and that the multilingual ones carry their own
# script so the language heuristic has something true to accept.

_POOL = tuple(
    "".join(random.Random(f"pool-{i}").choices(string.ascii_lowercase, k=5)) for i in range(400)
)

_SCRIPT_BASE = {"zh": (0x4E00, 0x2000), "ru": (0x0410, 0x0040), "hi": (0x0915, 0x0020), "ar": (0x0630, 0x0020)}


def latin_doc(prefix: str, i: int, n_words: int = 38) -> str:
    rng = random.Random(f"{prefix}-{i}")
    return f"{prefix} {i} " + " ".join(rng.choice(_POOL) for _ in range(n_words))


def script_doc(prefix: str, lang: str, i: int, n_chars: int = 190) -> str:
    base, span = _SCRIPT_BASE[lang]
    rng = random.Random(f"{prefix}-{lang}-{i}")
    chars = [chr(base + rng.randrange(span)) for _ in range(n_chars)]
    if lang == "zh":
        return "".join(chars)  # unspaced on purpose: exercises the character-shingle path
    out, cursor = [], 0
    while cursor < len(chars):
        step = rng.randrange(3, 8)
        out.append("".join(chars[cursor : cursor + step]))
        cursor += step
    return " ".join(out)


def code_doc(i: int) -> str:
    rng = random.Random(f"code-{i}")
    body = "\n".join(
        f"        {rng.choice(_POOL)}_{j} = {rng.choice(_POOL)}({i} + {j})" for j in range(6)
    )
    return (
        f"def fn_{i}(a, b):\n"
        f"    total = 0\n"
        f"    for x in range(a):\n"
        f"        if x % 2 == 0:\n"
        f"{body}\n"
        f"            total += x * {i} + b\n"
        f"    return total\n"
    )


def snapshot(*, n: int = 300, par_n: int = 200) -> dict:
    return {
        PROSE.name: [latin_doc("prose", i) for i in range(n)],
        CODE.name: [code_doc(i) for i in range(n)],
        MATH.name: [latin_doc("math", i) for i in range(n)],
        ML_VOL.name: {l: [script_doc("vol", l, i) for i in range(n)] for l in LANGS},
        ML_PAR.name: {l: [script_doc("par", l, i) for i in range(par_n)] for l in LANGS},
    }


def make_spec(
    *,
    sources=ALL_SOURCES,
    target_tokens: int = 40_000,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    **kwargs,
) -> CorpusSpec:
    return CorpusSpec(
        name="fetch-test",
        target_tokens=target_tokens,
        max_doc_tokens=max_doc_tokens,
        sources=tuple(sources),
        **kwargs,
    )


def ml_only_spec(*, sources, target_tokens: int) -> CorpusSpec:
    """A spec whose only domain is multilingual, for the per-language and parallel-row tests."""
    return make_spec(
        sources=sources, target_tokens=target_tokens, shares={"multilingual": 1.0}
    )


def run(spec: CorpusSpec, texts: dict, **kwargs) -> FetchResult:
    return fetch_corpus(spec, InMemorySource(texts), counter=COUNTER, **kwargs)


# -- budgets and composition -----------------------------------------------------------------------


def test_every_source_delivers_at_least_its_token_budget_and_overshoots_by_at_most_one_document():
    spec = make_spec()
    result = run(spec, snapshot())
    for name, report in result.reports.items():
        assert report.delivered_tokens >= report.requested_tokens, (name, report.to_dict())
        n_streams = max(1, len(dict(zip(report.by_lang_tokens, report.by_lang_tokens))))
        assert report.delivered_tokens <= report.requested_tokens + n_streams * MAX_DOC_TOKENS
    assert not result.shortfalls


def test_per_domain_delivered_tokens_meet_the_domain_budget_derived_from_the_spec():
    spec = make_spec()
    result = run(spec, snapshot())
    delivered: dict[str, int] = {}
    for doc in result.docs:
        delivered[doc.domain] = delivered.get(doc.domain, 0) + doc.n_tokens_ref
    for domain, target in spec.domain_token_targets().items():
        assert delivered[domain] >= target, (domain, delivered[domain], target)


def test_the_realized_composition_lands_within_two_points_of_the_target_shares():
    spec = make_spec()
    result = run(spec, snapshot())
    shares = result.realized["by_domain_share"]
    for domain, target in spec.shares.items():
        assert abs(shares[domain] - target) <= 0.02, (domain, shares[domain], target)
    # The share check is also self-reported, so a drift that a future change introduces surfaces in
    # the artifact and not only in this test.
    assert result.ok, result.problems


# -- determinism -----------------------------------------------------------------------------------


def test_the_same_spec_sources_and_seed_produce_identical_doc_ids_order_and_text():
    spec = make_spec()
    first = run(spec, snapshot(), seed=7)
    second = run(spec, snapshot(), seed=7)
    assert [(d.doc_id, d.domain, d.lang, d.source, d.text) for d in first.docs] == [
        (d.doc_id, d.domain, d.lang, d.source, d.text) for d in second.docs
    ]
    assert first.to_json() == second.to_json()


def test_a_different_seed_produces_a_different_draw():
    spec = make_spec()
    baseline = run(spec, snapshot(), seed=0)
    base_texts = [d.text for d in baseline.docs]
    # Only the multilingual sources have a seed-dependent language order, so a seed that happens to
    # give the same order would legitimately give the same draw; find one that does not rather than
    # asserting on an arbitrary number.
    for seed in range(1, 64):
        other = run(spec, snapshot(), seed=seed)
        if [d.text for d in other.docs] != base_texts:
            return
    pytest.fail("no seed in 1..63 changed the draw; the seed is not reaching the fetcher")


def test_doc_ids_are_a_contiguous_range_assigned_in_fetch_order():
    result = run(make_spec(), snapshot())
    assert [d.doc_id for d in result.docs] == list(range(len(result.docs)))


# -- the fetcher's output must be buildable --------------------------------------------------------


def test_every_emitted_document_is_within_max_doc_tokens_so_build_accepts_the_fetchers_own_output():
    spec = make_spec()
    result = run(spec, snapshot())
    for doc in result.docs:
        assert 0 < doc.n_tokens_ref <= spec.max_doc_tokens
        assert COUNTER.count(doc.text) == doc.n_tokens_ref
    built = build(spec, result.docs)  # would raise CorpusSpecError on an over-cap document
    assert len(built.docs) == len(result.docs)
    assert all(d.split and d.shard_id is not None for d in built.docs)


def test_a_document_over_the_cap_is_truncated_rather_than_dropped():
    long_doc = latin_doc("prose", 0, n_words=4000)
    assert COUNTER.count(long_doc) > MAX_DOC_TOKENS
    spec = make_spec(sources=(PROSE,), target_tokens=MAX_DOC_TOKENS, shares={"prose": 1.0})
    result = run(spec, {PROSE.name: [long_doc]})
    assert len(result.docs) == 1, "dropping over-cap documents would select for short web pages"
    assert result.docs[0].n_tokens_ref <= MAX_DOC_TOKENS
    assert result.reports[PROSE.name].n_truncated == 1
    assert result.reports[PROSE.name].n_dropped == 0
    assert long_doc.startswith(result.docs[0].text[:80])


def test_truncation_never_exceeds_the_cap_for_any_length_and_prefers_a_line_boundary():
    text = "\n".join(f"    line_{i} = {i} * value" for i in range(400))
    cut, truncated = truncate_to_tokens(text, COUNTER, MAX_DOC_TOKENS)
    assert truncated
    assert COUNTER.count(cut) <= MAX_DOC_TOKENS
    assert "\n" in cut and not cut.endswith(" ")
    assert text.startswith(cut)


# -- cleaning --------------------------------------------------------------------------------------


def test_code_indentation_survives_normalisation_and_the_fetch_pipeline():
    source = "def f(x):\r\n    if x:   \r\n        return x + 1   \r\n\r\n\r\n    return 0\r\n"
    cleaned = normalise_whitespace(source)
    assert "\n    if x:\n        return x + 1\n" in cleaned
    assert "\n\n\n" not in cleaned  # blank runs squeezed
    assert cleaned != " ".join(source.split()), "the naive normaliser would destroy the code domain"

    padded = source + "\n".join(f"    step_{i} = {i}" for i in range(20))
    spec = make_spec(sources=(CODE,), target_tokens=64, shares={"code": 1.0})
    result = run(spec, {CODE.name: [padded]})
    (doc,) = result.docs
    assert "\n        return x + 1" in doc.text
    assert doc.lang == "", "code has no natural-language label; a lang there would create a stratum"


def test_a_document_below_the_minimum_length_is_dropped_and_counted_as_too_short():
    spec = make_spec(sources=(PROSE,), target_tokens=200, shares={"prose": 1.0})
    result = run(spec, {PROSE.name: ["tiny", latin_doc("prose", 1), latin_doc("prose", 2)]})
    assert result.reports[PROSE.name].dropped["too_short"] == 1
    assert all(d.text != "tiny" for d in result.docs)


def test_every_record_read_is_either_delivered_or_dropped_for_exactly_one_named_reason():
    result = run(make_spec(), snapshot(par_n=6))
    for name, report in result.reports.items():
        assert report.n_records_read == report.delivered_docs + report.n_dropped, name
        assert set(report.dropped) == set(DROP_REASONS), name
        assert set(report.to_dict()["dropped"]) == set(DROP_REASONS)


# -- deduplication ---------------------------------------------------------------------------------


def test_an_exact_duplicate_is_dropped_and_attributed_to_the_exact_hash_stage():
    text = latin_doc("prose", 3)
    spec = make_spec(sources=(PROSE,), target_tokens=200, shares={"prose": 1.0})
    result = run(spec, {PROSE.name: [text, text, latin_doc("prose", 4)]})
    dropped = result.reports[PROSE.name].dropped
    assert dropped["duplicate_exact"] == 1
    assert dropped["duplicate_near"] == 0
    assert sum(1 for d in result.docs if d.text == text) == 1


def test_a_near_duplicate_differing_only_by_a_boilerplate_footer_is_dropped_by_the_dedup_stage():
    base = latin_doc("prose", 5, n_words=150)
    footer = "\n\nCopyright 2026 example.com -- all rights reserved. Subscribe to our newsletter."
    # The cap has to clear both documents, and that is not incidental to what is being tested.
    # ``clean()`` truncates before it dedups -- deliberately, so that two records differing only past
    # the cut point cannot both be written as identical corpus lines. Under this module's default
    # 128-token cap a 150-word page is cut at 511 characters, the footer is gone before the sketch is
    # taken, and the pair arrives at the dedup stage byte-identical: the fetcher still drops it, but
    # as ``duplicate_exact``, which is the *previous* test's invariant. Sizing the cap so no
    # truncation happens is what makes this test exercise the near-duplicate path it names -- and
    # 150 words against an 80-character footer is also the realistic shape of the case (J ~ 0.9,
    # comfortably inside the banded sketch's detection range, unlike a 38-word stub).
    spec = make_spec(
        sources=(PROSE,), target_tokens=4000, max_doc_tokens=512, shares={"prose": 1.0}
    )
    result = run(spec, {PROSE.name: [base, base + footer, latin_doc("prose", 6, n_words=150)]})
    dropped = result.reports[PROSE.name].dropped
    assert result.reports[PROSE.name].n_truncated == 0, "the footer must reach the dedup stage"
    assert dropped["duplicate_near"] == 1, result.reports[PROSE.name].to_dict()
    assert dropped["duplicate_exact"] == 0, "a footer makes it byte-different; exact hashing misses it"
    assert result.dropped_by_reason["duplicate_near"] == 1
    assert len(result.docs) == 2


def test_unspaced_script_text_gets_a_nondegenerate_signature_so_dedup_is_not_exact_only():
    zh = script_doc("vol", "zh", 0)
    assert " " not in zh
    assert len(shingle_signature(zh)) > 1, (
        "a word-shingle sketch collapses to one hash on Chinese, which would turn near-duplicate "
        "detection into exact matching over 30% of the corpus"
    )
    near = zh[:-4] + script_doc("vol", "zh", 1)[:4]
    dedup = NearDuplicateFilter()
    assert dedup.check_and_add(zh) is None
    assert dedup.check_and_add(near) == "duplicate_near"


def test_the_dedup_index_is_not_poisoned_by_a_check_that_did_not_commit():
    dedup = NearDuplicateFilter()
    text = latin_doc("prose", 9, n_words=80)
    assert dedup.check(text) is None
    assert dedup.check(text) is None, "check() must be side-effect free"
    assert dedup.check_and_add(text) is None
    assert dedup.check(text) == "duplicate_exact"


# -- shortfalls ------------------------------------------------------------------------------------


def test_a_parallel_control_running_dry_is_a_recorded_note_and_its_budget_moves_to_volume():
    spec = make_spec(par_n=None) if False else make_spec()
    result = run(spec, snapshot(par_n=4))
    par, vol = result.reports[ML_PAR.name], result.reports[ML_VOL.name]
    assert par.ran_dry and par.shortfall_tokens > 0
    assert any(ML_PAR.name in note and "ran dry" in note for note in result.notes)
    assert not any(ML_PAR.name in problem for problem in result.problems), (
        "a capped parallel control running dry is the expected outcome, not a problem"
    )
    assert vol.reallocated_tokens == par.shortfall_tokens
    assert vol.requested_tokens == vol.base_tokens + par.shortfall_tokens
    assert vol.delivered_tokens >= vol.requested_tokens
    # The point of the reallocation: the domain still gets its share.
    got = result.realized["by_domain_share"]["multilingual"]
    assert abs(got - spec.shares["multilingual"]) <= 0.02, got
    assert result.ok, result.problems


def test_a_volume_source_running_dry_is_reported_loudly_and_not_silently_absorbed():
    thin = snapshot()
    thin[PROSE.name] = thin[PROSE.name][:3]
    spec = make_spec()
    with pytest.warns(FetchWarning):
        result = run(spec, thin)
    assert not result.ok
    assert result.volume_shortfalls[PROSE.name] > 0
    assert result.reports[PROSE.name].ran_dry
    problems = " | ".join(result.problems)
    assert PROSE.name in problems and "ran dry" in problems
    assert "prose" in problems and "25%" in problems, "the message must name the share it breaks"
    # Inspectable, not just narrated.
    assert result.to_dict()["volume_shortfalls"][PROSE.name] > 0


def test_the_cli_refuses_to_write_a_corpus_whose_volume_source_ran_dry(tmp_path, monkeypatch):
    thin = snapshot()
    thin[PROSE.name] = thin[PROSE.name][:3]
    spec = make_spec()
    monkeypatch.setattr("src.corpus.fetch.SPECS", {spec.name: spec})
    monkeypatch.setattr("src.corpus.fetch.HFDatasetSource", lambda **kw: InMemorySource(thin))
    out = tmp_path / "c.jsonl"
    assert main(["--spec", spec.name, "--out", str(out)]) == 1
    assert not out.exists()
    report = out.with_name(out.name + ".fetch.json")
    assert report.exists(), "the report is written even when the corpus is refused"
    assert main(["--spec", spec.name, "--out", str(out), "--allow-shortfall"]) == 0
    assert out.exists() and (tmp_path / "c.jsonl.meta.json").exists()


# -- language sanity check -------------------------------------------------------------------------


def test_an_all_ascii_document_tagged_as_a_non_latin_language_is_rejected():
    assert not looks_like_language("The quick brown fox jumps over the lazy dog.", "zh")
    assert looks_like_language(script_doc("vol", "zh", 0), "zh")
    # No signal for a Latin-script language, so the filter must not pretend to have one.
    assert looks_like_language("The quick brown fox.", "de")

    source = SourceSpec(name="ml-vol", domain="multilingual", hf_dataset="fake/x", langs=("zh",))
    spec = ml_only_spec(sources=(source,), target_tokens=200)
    ascii_page = latin_doc("mislabelled-english", 0)
    result = run(spec, {source.name: {"zh": [ascii_page, script_doc("vol", "zh", 1), script_doc("vol", "zh", 2)]}})
    assert result.reports[source.name].dropped["lang_mismatch"] == 1
    assert all(d.text != ascii_page for d in result.docs)
    assert all(d.lang == "zh" for d in result.docs)


# -- parallel-control alignment --------------------------------------------------------------------


def test_a_parallel_row_that_loses_one_language_is_dropped_whole_so_the_control_stays_aligned():
    par_texts = {l: [script_doc("par", l, i) for i in range(4)] for l in LANGS}
    par_texts["ru"][1] = latin_doc("mislabelled", 1)  # fails the script check for ru only
    spec = ml_only_spec(sources=(ML_VOL, ML_PAR), target_tokens=2_000)
    texts = {ML_VOL.name: {l: [script_doc("vol", l, i) for i in range(200)] for l in LANGS},
             ML_PAR.name: par_texts}
    result = run(spec, texts)

    par = result.reports[ML_PAR.name]
    assert par.dropped["lang_mismatch"] == 1
    assert par.dropped["parallel_row_incomplete"] == len(LANGS) - 1, (
        "the other languages' copies of that sentence must go too, or sentence 1 exists in 3 "
        "languages out of 4 and the parallel control stops controlling"
    )
    counts = {lang: par.by_lang_docs.get(lang, 0) for lang in LANGS}
    assert len(set(counts.values())) == 1, counts
    assert all(d.text != par_texts["zh"][1] for d in result.docs if d.source == ML_PAR.name)


def test_a_parallel_control_delivers_the_same_number_of_documents_in_every_language():
    result = run(make_spec(), snapshot(par_n=40))
    par = result.reports[ML_PAR.name]
    assert set(par.by_lang_docs) == set(LANGS)
    assert len(set(par.by_lang_docs.values())) == 1, par.by_lang_docs


# -- non-BMP round trip ----------------------------------------------------------------------------


def test_non_bmp_text_survives_fetch_and_round_trips_through_write_corpus(tmp_path):
    emoji_doc = script_doc("vol", "zh", 0) + " 🌍🚀 " + script_doc("vol", "zh", 1)
    source = SourceSpec(name="ml-vol", domain="multilingual", hf_dataset="fake/x", langs=("zh",))
    spec = ml_only_spec(sources=(source,), target_tokens=32)
    result = run(spec, {source.name: {"zh": [emoji_doc]}})
    (doc,) = result.docs
    assert "🌍🚀" in doc.text, "fetch must not strip non-BMP text; the C++ parser handles pairs now"

    built = build(spec, result.docs)
    path = write_corpus(tmp_path / "corpus.jsonl", built.docs, spec)
    raw = path.read_text(encoding="ascii")
    assert "\\ud83c" in raw.lower() or "\\ud83d" in raw.lower(), "written as a surrogate pair"
    assert load_corpus(path)[0].text == doc.text


# -- the real reader -------------------------------------------------------------------------------


def test_hf_dataset_source_raises_an_actionable_fetch_error_when_datasets_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(FetchError) as excinfo:
        list(HFDatasetSource().iter_texts(PROSE))
    message = str(excinfo.value)
    assert "datasets" in message
    assert "pip install datasets" in message, "the error must say how to fix it"
    assert "InMemorySource" in message, "and name the offline alternative"


def test_hf_dataset_source_always_streams_and_resolves_the_per_language_config(monkeypatch):
    seen: dict[str, object] = {}

    def load_dataset(path, name=None, **kwargs):
        seen.update({"path": path, "name": name, **kwargs})
        return [{"text": "alpha"}, {"text": "beta"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    reader = HFDatasetSource(revision="abc123")
    out = list(reader.iter_texts(ML_PAR, lang="zh", limit=1))

    assert seen["streaming"] is True, (
        "a non-streaming read of a volume dataset does not fit in the session doing the reading"
    )
    assert seen["path"] == ML_PAR.hf_dataset
    assert seen["name"] == "zho_Hans", "FLORES configs are ISO 639-3 plus script, not the spec's code"
    assert seen["split"] == "devtest"
    assert seen["revision"] == "abc123"
    assert [text for text, _meta in out] == ["alpha"]
    assert out[0][1]["hf_config"] == "zho_Hans"


def test_hf_dataset_source_reports_a_missing_text_field_by_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *a, **k: [{"body": "x"}]),
    )
    with pytest.raises(FetchError, match="no text field"):
        list(HFDatasetSource().iter_texts(PROSE))


def test_importing_the_fetch_module_does_not_require_the_datasets_package():
    assert "datasets" not in sys.modules or sys.modules["datasets"] is None


# -- seam contract ---------------------------------------------------------------------------------


def test_a_missing_reader_for_a_declared_source_is_a_fatal_error_not_a_skipped_source():
    with pytest.raises(FetchError, match="no DocumentSource"):
        fetch_corpus(make_spec(), {PROSE.name: InMemorySource({})}, counter=COUNTER)


def test_a_per_language_source_backed_by_a_flat_snapshot_is_refused():
    spec = ml_only_spec(sources=(ML_VOL,), target_tokens=200)
    with pytest.raises(FetchError, match="flat sequence"):
        run(spec, {ML_VOL.name: [script_doc("vol", "zh", 0)]})


def test_the_json_report_carries_the_numbers_a_reviewer_needs_before_collection():
    import json

    result = run(make_spec(), snapshot(par_n=6))
    payload = json.loads(result.to_json())
    assert payload["spec"] == "fetch-test"
    assert set(payload["sources"]) == {s.name for s in ALL_SOURCES}
    assert payload["total_tokens_ref"] == result.total_tokens_ref
    assert payload["target_shares"] == dict(sorted(make_spec().shares.items()))
    row = payload["sources"][ML_PAR.name]
    assert row["role"] == "parallel_control"
    assert 0.0 <= row["drop_rate"] <= 1.0
    assert row["requested_tokens"] == row["base_tokens"] + row["reallocated_tokens"]
