"""Corpus builder tests — plan T4.2 / T4.3 / T4.4.

The corpus file is written once and then seven models are traced against it, so every property tested
here is one that would be undetectable after the fact: a split that depended on fetch order, a shard
that straddles a document, a prefix of shards that is all one domain. None of this needs a tokenizer
or a network, which is the point of the reference counter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from src.corpus.build import (
    FALLBACK_LADDER,
    BuildResult,
    CharRatioCounter,
    Document,
    assign_shards,
    assign_splits,
    build,
    hidden_budget_report,
    interleave_documents,
    load_corpus,
    load_doc_splits,
    realized_shares,
    shard_table,
    write_corpus,
)
from src.corpus.spec import CorpusSpec, CorpusSpecError

COUNTER = CharRatioCounter()

# (domain, lang, n_docs, chars_per_doc) -- multilingual is split over languages so there are small
# strata (hi with 4 docs) as well as large ones, which is where the rounding rule shows.
LAYOUT = (
    ("prose", "en", 40, 400),
    ("code", "", 40, 200),
    ("math", "en", 25, 320),
    ("multilingual", "de", 20, 400),
    ("multilingual", "zh", 12, 400),
    ("multilingual", "hi", 4, 400),
)


def make_docs(layout=LAYOUT) -> list[Document]:
    docs: list[Document] = []
    doc_id = 0
    for domain, lang, n, chars in layout:
        for i in range(n):
            text = f"{domain}-{lang}-{i} " + "x" * chars
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=text,
                    domain=domain,
                    lang=lang,
                    source=f"src-{domain}",
                    n_tokens_ref=COUNTER.count(text),
                )
            )
            doc_id += 1
    return docs


def spec(**overrides) -> CorpusSpec:
    base = dict(name="unit", target_tokens=10_000, max_doc_tokens=2048, shard_tokens=2048, seed=0)
    base.update(overrides)
    return CorpusSpec(**base)


@pytest.fixture
def built() -> BuildResult:
    return build(spec(), make_docs())


# -- splits (T4.3) ---------------------------------------------------------------------------------


def test_every_document_gets_exactly_one_split_and_no_doc_id_appears_twice(built):
    assert all(d.split in {"train", "val", "test"} for d in built.docs)
    ids = [d.doc_id for d in built.docs]
    assert len(set(ids)) == len(ids)


def test_split_assignment_is_byte_identical_when_the_build_is_repeated(built):
    again = build(spec(), make_docs())
    assert {d.doc_id: d.split for d in built.docs} == {d.doc_id: d.split for d in again.docs}


def test_split_assignment_is_independent_of_the_order_documents_are_handed_in(built):
    shuffled = make_docs()[::-1]
    other = build(spec(), shuffled)
    assert {d.doc_id: d.split for d in built.docs} == {d.doc_id: d.split for d in other.docs}


def test_a_different_seed_produces_a_different_split_assignment(built):
    other = build(spec(seed=7), make_docs())
    assert {d.doc_id: d.split for d in built.docs} != {d.doc_id: d.split for d in other.docs}


def test_overall_split_ratios_are_close_to_eighty_ten_ten(built):
    n = len(built.docs)
    counts = built.split_counts["docs"]
    assert counts["train"] / n == pytest.approx(0.8, abs=0.03)
    assert counts["val"] / n == pytest.approx(0.1, abs=0.03)
    assert counts["test"] / n == pytest.approx(0.1, abs=0.03)


def test_every_domain_lang_stratum_is_represented_in_train(built):
    strata = {(d.domain, d.lang) for d in built.docs}
    in_train = {(d.domain, d.lang) for d in built.docs if d.split == "train"}
    assert strata == in_train


def test_a_tiny_stratum_still_reaches_val_and_test_instead_of_being_rounded_away():
    # Plain largest-remainder would give this stratum train=3/val=0/test=0 and the (math, en) cell
    # would silently vanish from the T9.4 per-stratum test table.
    docs = make_docs((("math", "en", 3, 100),))
    assign_splits(docs, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)
    assert sorted(d.split for d in docs) == ["test", "train", "val"]


def test_a_stratum_too_small_to_cover_every_split_still_assigns_a_split_to_each_doc():
    docs = make_docs((("code", "", 2, 100),))
    assign_splits(docs, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)
    assert all(d.split in {"train", "val", "test"} for d in docs)


# -- shards (T4.2) ---------------------------------------------------------------------------------


def test_no_document_is_split_across_two_shards(built):
    per_doc = {}
    for doc in built.docs:
        per_doc.setdefault(doc.doc_id, set()).add(doc.shard_id)
    assert all(len(s) == 1 for s in per_doc.values())
    assert all(d.shard_id is not None for d in built.docs)


def test_shard_ids_are_contiguous_from_zero_and_monotone_in_write_order(built):
    ids = [d.shard_id for d in built.docs]
    assert ids == sorted(ids)
    assert set(ids) == set(range(max(ids) + 1))
    assert [row["shard_id"] for row in built.shards] == list(range(len(built.shards)))


def test_every_full_shard_is_within_twenty_percent_of_the_target_size(built):
    # T4.2 acceptance. The last shard is whatever is left over, so it is exempt.
    target = spec().shard_tokens
    for row in built.shards[:-1]:
        assert row["n_tokens_ref"] <= target
        assert row["n_tokens_ref"] >= 0.8 * target, row


def test_a_document_larger_than_a_shard_gets_a_shard_to_itself():
    docs = make_docs((("prose", "en", 2, 40),))
    big = Document(
        doc_id=99, text="y" * 4000, domain="math", lang="en", source="s", n_tokens_ref=1000
    )
    ordered = [docs[0], big, docs[1]]
    assign_shards(ordered, shard_tokens=100)
    assert big.shard_id is not None
    alone = [d for d in ordered if d.shard_id == big.shard_id]
    assert alone == [big]


def test_shard_table_refuses_documents_that_were_never_sharded():
    docs = make_docs((("prose", "en", 1, 40),))
    with pytest.raises(CorpusSpecError, match="no shard_id"):
        shard_table(docs)


# -- interleaving ----------------------------------------------------------------------------------


def test_every_prefix_of_shards_contains_more_than_one_domain(built):
    # T5.1 can stop mid-model at the 12 h Kaggle cap; a domain-ordered corpus would make the
    # surviving shards silently unrepresentative.
    for cut in range(1, len(built.shards) + 1):
        prefix = [d for d in built.docs if d.shard_id < cut]
        assert len({d.domain for d in prefix}) > 1, f"prefix of {cut} shard(s) is single-domain"


def test_the_first_shard_already_contains_every_domain_in_the_corpus(built):
    first = {d.domain for d in built.docs if d.shard_id == 0}
    assert first == {d.domain for d in built.docs}


def test_write_order_is_deterministic_and_independent_of_input_order():
    a = interleave_documents(make_docs(), seed=0)
    b = interleave_documents(make_docs()[::-1], seed=0)
    assert [d.doc_id for d in a] == [d.doc_id for d in b]


def test_interleaving_is_token_weighted_so_a_prefix_tracks_the_corpus_composition(built):
    total = built.total_tokens_ref
    overall = {
        d: t / total for d, t in built.realized["by_domain"].items()
    }
    half = [d for d in built.docs if d.shard_id < max(1, len(built.shards) // 2)]
    half_total = sum(d.n_tokens_ref for d in half)
    for domain, share in overall.items():
        got = sum(d.n_tokens_ref for d in half if d.domain == domain) / half_total
        assert got == pytest.approx(share, abs=0.05)


# -- validation ------------------------------------------------------------------------------------


def test_a_document_over_max_doc_tokens_is_refused_by_name():
    docs = make_docs((("prose", "en", 3, 40),))
    docs[1] = replace(docs[1], doc_id=4242, n_tokens_ref=5000, source="oversized-src")
    with pytest.raises(CorpusSpecError) as exc:
        build(spec(max_doc_tokens=2048, shard_tokens=4096), docs)
    assert "4242" in str(exc.value) and "oversized-src" in str(exc.value)


def test_duplicate_doc_ids_are_fatal():
    docs = make_docs((("prose", "en", 2, 40),))
    docs[1] = replace(docs[1], doc_id=docs[0].doc_id)
    with pytest.raises(CorpusSpecError, match="duplicate doc_id"):
        build(spec(), docs)


def test_a_zero_token_document_is_fatal_because_capture_treats_it_as_an_error():
    docs = make_docs((("prose", "en", 2, 40),))
    docs[0] = replace(docs[0], n_tokens_ref=0)
    with pytest.raises(CorpusSpecError, match="zero-token"):
        build(spec(), docs)


def test_an_empty_corpus_is_fatal():
    with pytest.raises(CorpusSpecError, match="zero documents"):
        build(spec(), [])


# -- file format -----------------------------------------------------------------------------------


def test_the_corpus_file_round_trips_through_load_corpus(tmp_path, built):
    path = write_corpus(tmp_path / "unit.jsonl", built.docs, built.spec)
    back = load_corpus(path)
    assert [d.__dict__ for d in back] == [d.__dict__ for d in built.docs]


def test_load_doc_splits_returns_exactly_the_doc_id_to_split_mapping(tmp_path, built):
    path = write_corpus(tmp_path / "unit.jsonl", built.docs, built.spec)
    assert load_doc_splits(path) == {d.doc_id: d.split for d in built.docs}


def test_doc_id_and_text_are_the_first_two_keys_so_the_cpp_parser_cannot_be_fooled(tmp_path, built):
    # parse_jsonl_line searches the raw line for "doc_id" and "text"; a field written earlier whose
    # value contained either substring would be picked up instead of the real key.
    path = write_corpus(tmp_path / "unit.jsonl", built.docs, built.spec)
    first = json.loads(path.read_text(encoding="ascii").splitlines()[0])
    assert list(first)[:2] == ["doc_id", "text"]


def test_the_written_jsonl_is_ascii_only_even_for_non_latin_text(tmp_path):
    docs = make_docs((("prose", "en", 2, 40),))
    docs[0] = replace(docs[0], text="मुझे नमस्ते 中文 مرحبا Ω")
    docs[0].n_tokens_ref = COUNTER.count(docs[0].text)
    result = build(spec(), docs)
    path = write_corpus(tmp_path / "unit.jsonl", result.docs, result.spec)
    raw = path.read_bytes()
    assert max(raw) < 128, "non-ASCII byte in a file the byte-oriented C++ parser will read"
    assert any(d.text == "मुझे नमस्ते 中文 مرحبا Ω" for d in load_corpus(path))


def test_the_sidecar_records_the_spec_the_shard_table_and_the_realized_composition(tmp_path, built):
    path = write_corpus(tmp_path / "unit.jsonl", built.docs, built.spec)
    meta = json.loads(path.with_name(path.name + ".meta.json").read_text(encoding="utf-8"))
    assert meta["spec"]["name"] == built.spec.name
    assert len(meta["shards"]) == len(built.shards)
    assert set(meta["realized"]["by_domain"]) == {d.domain for d in built.docs}
    assert meta["split_counts"]["docs"]["train"] > 0


def test_a_corpus_file_missing_a_field_is_refused(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"doc_id": 0, "text": "hi"}) + "\n", encoding="ascii")
    with pytest.raises(CorpusSpecError, match="missing field"):
        load_corpus(path)


# -- realized shares (T4.1) ------------------------------------------------------------------------


@dataclass(frozen=True)
class ByteCounter:
    """Stand-in for a byte-level tokenizer: non-Latin text costs several tokens per character, which
    is exactly how OLMoE's 50k vocab differs from Gemma 4's 262k on the multilingual share."""

    def count(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")))


def test_realized_shares_change_with_the_counter_so_the_table_is_not_hardcoded(built):
    ref = realized_shares(built.docs, CharRatioCounter())
    byte = realized_shares(built.docs, ByteCounter())
    assert byte["total_tokens"] != ref["total_tokens"]
    assert byte["by_domain"] != ref["by_domain"]


def test_realized_shares_of_a_non_latin_domain_grow_under_a_byte_level_counter():
    docs = make_docs((("prose", "en", 4, 40), ("multilingual", "zh", 4, 0)))
    for doc in docs:
        if doc.domain == "multilingual":
            doc.text = "中文" * 20
            doc.n_tokens_ref = COUNTER.count(doc.text)
    ref = realized_shares(docs, CharRatioCounter())
    byte = realized_shares(docs, ByteCounter())
    assert byte["by_domain_share"]["multilingual"] > ref["by_domain_share"]["multilingual"]


def test_realized_shares_are_reported_per_domain_lang_stratum_too(built):
    out = realized_shares(built.docs, CharRatioCounter())
    assert "multilingual/hi" in out["by_stratum"]
    assert sum(out["by_stratum"].values()) == out["total_tokens"]


# -- hidden budget (T4.4 / O2) ---------------------------------------------------------------------


def test_no_rung_of_the_fallback_ladder_ever_drops_topk():
    assert all(rung["keep_topk"] for rung in FALLBACK_LADDER)


def test_the_budget_report_never_recommends_dropping_topk_even_when_nothing_fits(built):
    report = hidden_budget_report(
        built.docs,
        n_experts=128,
        hidden_dim=2048,
        subsample_n=50_000,
        n_moe_layers=48,
        max_bytes_per_shard=1,  # nothing can clear this
    )
    assert all(rung["keep_topk"] for rung in report["ladder"])
    assert report["topk_never_dropped"] is True
    assert report["recommended"] is None or report["recommended"]["keep_topk"]


def test_the_budget_report_flags_the_shards_that_exceed_the_byte_cap(built):
    report = hidden_budget_report(
        built.docs,
        n_experts=128,
        hidden_dim=2880,
        subsample_n=10_000,
        n_moe_layers=48,
        max_bytes_per_shard=1024,
    )
    assert report["within_cap"] is False
    assert report["shards_over_cap"] == [row["shard_id"] for row in report["shards"]]


def test_the_ladder_steps_down_the_hidden_subsample_before_dropping_logits(built):
    report = hidden_budget_report(
        built.docs, n_experts=64, hidden_dim=2048, subsample_n=50_000, n_moe_layers=16
    )
    subs = [rung["hidden_subsample"] for rung in report["ladder"]]
    assert subs == sorted(subs, reverse=True)
    first_logit_drop = next(i for i, r in enumerate(report["ladder"]) if not r["keep_logits"])
    assert subs[first_logit_drop - 1] > 0, "logits are dropped only after the subsample is reduced"


def test_a_generous_cap_recommends_the_untouched_first_rung(built):
    report = hidden_budget_report(
        built.docs,
        n_experts=64,
        hidden_dim=2048,
        subsample_n=1_000,
        n_moe_layers=16,
        max_bytes_per_shard=500 * 1024**2,
    )
    assert report["within_cap"] is True
    assert report["recommended"]["step"] == 0
    assert report["recommended"]["hidden_subsample"] == 1_000


def test_captured_token_counts_sum_to_the_subsample_and_follow_the_global_stride(built):
    report = hidden_budget_report(
        built.docs, n_experts=64, hidden_dim=16, subsample_n=100, n_moe_layers=2
    )
    captured = sum(row["n_captured"] for row in report["shards"])
    assert captured == pytest.approx(100, rel=0.15)
    assert report["bytes_per_token"]["hidden"] == 2 * 16 * 2


def test_the_budget_report_refuses_a_nonpositive_subsample(built):
    with pytest.raises(CorpusSpecError, match="subsample_n"):
        hidden_budget_report(
            built.docs, n_experts=8, hidden_dim=8, subsample_n=0, n_moe_layers=2
        )


# -- reference counter -----------------------------------------------------------------------------


def test_the_reference_counter_never_returns_zero_for_a_nonempty_document():
    assert CharRatioCounter().count("a") == 1
    assert CharRatioCounter(chars_per_token=4.0).count("a" * 9) == 3


def test_the_reference_counter_refuses_a_nonpositive_ratio():
    with pytest.raises(CorpusSpecError):
        CharRatioCounter(chars_per_token=0.0)
