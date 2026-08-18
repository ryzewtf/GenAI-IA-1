"""Trace format and reader tests — plan T6.1 acceptance.

The reader is validated against synthetic traces with known contents, so every assertion is
equality rather than plausibility. The tests that matter most are the ones covering invariants
that fail *silently* in production: I1 (labels come from topk.bin), shard-merge refusal, and
truncation detection.
"""

from __future__ import annotations

import numpy as np
import json

import pytest

from src.runtime.config import IncompatibleShardError
from src.traces.format import (
    FLAG_HIDDEN_CAPTURED,
    FormatError,
    TraceSpec,
    check_file_sizes,
    emit_c_header,
    expected_file_sizes,
    read_manifest,
)
from src.traces.reader import TraceReader
from src.traces.synth import make_synthetic_trace

SPEC = TraceSpec(n_moe_layers=4, n_experts=16, top_k=3, hidden_dim=8)


@pytest.fixture
def trace(tmp_path):
    return make_synthetic_trace(tmp_path, spec=SPEC, shard_sizes=(40, 25, 35), seed=7)


@pytest.fixture
def reader(trace):
    with TraceReader(trace.root, trace.model, trace.corpus, doc_splits=trace.doc_splits) as r:
        yield r


# -- format -----------------------------------------------------------------------------------


def test_strides_are_layer_major_within_token():
    assert SPEC.topk_stride == 4 * 3 * 4
    assert SPEC.logit_stride == 4 * 16 * 2
    assert SPEC.hidden_stride == 4 * 8 * 2
    assert SPEC.token_stride == 16


def test_spec_rejects_impossible_shapes():
    with pytest.raises(FormatError):
        TraceSpec(n_moe_layers=4, n_experts=8, top_k=16, hidden_dim=8)
    with pytest.raises(FormatError):
        TraceSpec(n_moe_layers=0, n_experts=8, top_k=2, hidden_dim=8)


def test_expected_sizes_match_what_synth_wrote(trace):
    shard_dir = trace.root / trace.model / trace.corpus / "shard_00000"
    manifest = read_manifest(shard_dir)
    # This SHARD's row count. `hidden_subsample_n` is the collection-wide budget and is only
    # equal to it by coincidence — reading it here failed a healthy multi-shard trace.
    n_captured = manifest["n_captured"]
    sizes = expected_file_sizes(SPEC, manifest["n_tokens"], n_captured)
    for name, want in sizes.items():
        assert (shard_dir / name).stat().st_size == want
    check_file_sizes(shard_dir, SPEC, manifest["n_tokens"], n_captured)


def test_truncated_stream_is_detected(trace):
    """A truncated upload that silently succeeds is the main way to lose a session (T5.3)."""
    path = trace.root / trace.model / trace.corpus / "shard_00001" / "topk.bin"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 4])

    with pytest.raises(FormatError, match="size"):
        TraceReader(trace.root, trace.model, trace.corpus)


def test_c_header_carries_the_same_constants():
    header = emit_c_header()
    assert "MOE_TOPK_ELEM_BYTES   4" in header
    assert "MOE_LOGIT_ELEM_BYTES  2" in header
    assert "MOE_TOKEN_RECORD_BYTES   16" in header
    assert "LAYER-MAJOR WITHIN TOKEN" in header


# -- reader round-trips -------------------------------------------------------------------------


def test_reader_discovers_and_orders_shards(reader, trace):
    assert reader.shard_ids == [0, 1, 2]
    assert reader.n_tokens == sum(trace.shard_sizes) == 100
    assert reader.n_moe_layers == 4
    assert reader.top_k == 3


@pytest.mark.parametrize("layer", range(SPEC.n_moe_layers))
def test_topk_roundtrips_exactly_across_shard_boundaries(reader, trace, layer):
    got = reader.topk_sets(layer)
    np.testing.assert_array_equal(got, trace.topk[:, layer, :].astype(np.int16))


def test_slices_spanning_shard_boundaries_are_correct(reader, trace):
    # 30..70 straddles all three shards (sizes 40/25/35).
    window = slice(30, 70)
    np.testing.assert_array_equal(
        reader.topk_sets(2, window), trace.topk[30:70, 2, :].astype(np.int16)
    )
    np.testing.assert_array_equal(
        reader.logits(2, window), trace.logits[30:70, 2, :].astype(np.float32)
    )
    np.testing.assert_array_equal(reader.tokens(window), trace.tokens[30:70])


def test_logits_roundtrip(reader, trace):
    np.testing.assert_array_equal(
        reader.logits(1), trace.logits[:, 1, :].astype(np.float32)
    )


def test_tokens_roundtrip_and_hidden_flag(reader, trace):
    """The flagged tokens and the index stream must name the same tokens.

    They are no longer the same *numbers*: an index is `doc_id * n_ctx + pos_in_doc`, a reserved
    block per document, not a running token count. So the flag positions are mapped through that
    rule before comparing — which is the real invariant, and the one that lets shards concatenate
    without rewriting (T2.3).
    """
    got = reader.tokens()
    np.testing.assert_array_equal(got, trace.tokens)

    manifest = read_manifest(trace.root / trace.model / trace.corpus / "shard_00000")
    span = manifest["index_doc_span"]
    flagged = np.flatnonzero(got["flags"] & FLAG_HIDDEN_CAPTURED)
    expected = got["doc_id"][flagged].astype(np.int64) * span + got["pos_in_doc"][flagged]
    np.testing.assert_array_equal(expected, trace.hidden_index)
    assert np.all(np.diff(trace.hidden_index) > 0), "must ascend across the whole trace"


def test_margins_are_kth_minus_k_plus_first(reader, trace):
    layer, k = 3, SPEC.top_k
    values = trace.logits[:, layer, :].astype(np.float32)
    ordered = np.sort(values, axis=1)[:, ::-1]
    expected = ordered[:, k - 1] - ordered[:, k]
    np.testing.assert_allclose(reader.margins(layer), expected, rtol=0, atol=1e-6)


def test_margins_are_non_negative(reader):
    for layer in range(SPEC.n_moe_layers):
        assert np.all(reader.margins(layer) >= 0)


# -- invariant I1: labels are never derived from logits ----------------------------------------


def test_topk_is_unaffected_by_corrupting_logits(trace):
    """The strongest available test of I1.

    If any code path derived expert sets from ``logits.bin``, zeroing that file would change
    the labels. It must not. GPT-OSS selects on *biased* logits, so a recomputation would be
    wrong for half of the headline pair — and wrong silently.
    """
    with TraceReader(trace.root, trace.model, trace.corpus) as reader:
        before = reader.topk_sets(0)

    path = trace.root / trace.model / trace.corpus / "shard_00000" / "logits.bin"
    path.write_bytes(b"\0" * path.stat().st_size)

    with TraceReader(trace.root, trace.model, trace.corpus) as reader:
        after = reader.topk_sets(0)

    np.testing.assert_array_equal(before, after)


def test_labels_satisfy_the_t5_3_range_and_distinctness_checks(reader):
    for layer in range(reader.n_moe_layers):
        sets = reader.topk_sets(layer)
        assert sets.min() >= 0
        assert sets.max() < reader.n_experts
        # every row holds top_k distinct experts
        assert all(len(np.unique(row)) == reader.top_k for row in sets)


# -- hidden states -------------------------------------------------------------------------------


def test_hidden_lookup_is_by_token_row_not_by_stored_index(reader, trace):
    """`hidden()` takes row positions, which is what every consumer actually holds.

    The stored index is `doc_id * n_ctx + pos_in_doc`, a sparse per-document block. Passing
    those to a consumer that indexes with `topk_sets(layer)[rows]` produced no error, just an
    empty intersection — the FV probe reported "no usable rows after exclusions" on a healthy
    trace.
    """
    rows = reader.captured_rows()
    assert not np.array_equal(rows, trace.hidden_index), "the two spaces must not be conflated"

    got = reader.hidden(2, rows[[0, 3, 7]])
    np.testing.assert_array_equal(got, trace.hidden[[0, 3, 7], 2, :].astype(np.float32))


def test_hidden_lookup_spans_shards(reader, trace):
    got = reader.hidden(0, reader.captured_rows())
    np.testing.assert_array_equal(got, trace.hidden[:, 0, :].astype(np.float32))


def test_captured_rows_agree_with_the_flag_bits_in_tokens_bin(reader):
    """The flags and hidden_index.bin are written by different code; a count mismatch is fatal."""
    rows = reader.captured_rows()
    flagged = np.flatnonzero(reader.tokens()["flags"] & FLAG_HIDDEN_CAPTURED)
    np.testing.assert_array_equal(rows, flagged)


def test_uncaptured_token_raises_rather_than_silently_dropping(reader, trace):
    uncaptured = int(reader.captured_rows()[0]) + 1  # subsample is every 4th token
    with pytest.raises(KeyError, match="no captured hidden state"):
        reader.hidden(0, [uncaptured])


def test_layer_bounds_are_checked(reader):
    with pytest.raises(IndexError):
        reader.topk_sets(reader.n_moe_layers)
    with pytest.raises(IndexError):
        reader.logits(-1)


# -- splits ------------------------------------------------------------------------------------------


def test_split_mask_is_document_level(reader, trace):
    mask = reader.split_mask("test")
    doc_ids = trace.tokens["doc_id"]
    expected = np.array([trace.doc_splits[int(d)] == "test" for d in doc_ids])
    np.testing.assert_array_equal(mask, expected)

    # No document may appear in two splits (plan T4.3).
    masks = {s: reader.split_mask(s) for s in ("train", "val", "test")}
    stacked = np.vstack(list(masks.values()))
    assert np.all(stacked.sum(axis=0) == 1)


def test_split_mask_requires_the_mapping(trace):
    with TraceReader(trace.root, trace.model, trace.corpus) as reader:
        with pytest.raises(ValueError, match="doc_splits"):
            reader.split_mask("train")


# -- shard-merge refusal (plan S.3) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("run_config_sha256", "1" * 64),
        ("logit_tensor_used", "ffn_moe_logits_biased"),
        ("gguf_sha256", "a" * 64),
        ("n_experts", 32),
        ("quant", "Q8_0"),
    ],
)
def test_shards_from_different_experiments_refuse_to_merge(tmp_path, key, value):
    import json

    trace = make_synthetic_trace(tmp_path, spec=SPEC, shard_sizes=(20, 20))
    path = trace.root / trace.model / trace.corpus / "shard_00001" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(IncompatibleShardError, match="different experiments"):
        TraceReader(trace.root, trace.model, trace.corpus, validate_sizes=False)


def test_layer_index_map_is_used_not_assumed(tmp_path):
    """DeepSeek's first_k_dense_replace makes trace layer 0 == model layer 1 (plan T3.5)."""
    trace = make_synthetic_trace(
        tmp_path,
        spec=SPEC,
        shard_sizes=(20,),
        manifest_overrides={
            "layer_index_map": {f"trace_{i}": f"model_layer_{i + 1}" for i in range(4)}
        },
    )
    with TraceReader(trace.root, trace.model, trace.corpus) as reader:
        assert reader.model_layer(0) == 1
        assert reader.model_layer(3) == 4


def test_iter_chunks_tiles_the_corpus_exactly(reader):
    seen = np.zeros(reader.n_tokens, dtype=int)
    for window in reader.iter_chunks(chunk_tokens=16):
        seen[window] += 1
    assert np.all(seen == 1)


def test_layer_index_map_reads_both_the_list_and_the_keyed_form(trace, tmp_path):
    """The runner writes a list; the synthetic fixtures wrote a dict. Both are real manifests.

    Reading only the keyed form meant a genuine collection's manifest raised `FormatError` in
    the first probe that asked which model layer a trace layer came from — found by pointing
    the FV probe at a real OLMoE trace. An off-by-one here corrupts every per-depth result
    (T3.5), so the mapping stays explicit rather than re-derived.
    """
    shard_dir = trace.root / trace.model / trace.corpus / "shard_00000"
    manifest = read_manifest(shard_dir)

    manifest["layer_index_map"] = list(range(1, SPEC.n_moe_layers + 1))
    (shard_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    reader = TraceReader(trace.root, trace.model, trace.corpus)
    assert [reader.model_layer(i) for i in range(SPEC.n_moe_layers)] == list(
        range(1, SPEC.n_moe_layers + 1)
    )

    manifest["layer_index_map"] = {
        f"trace_{i}": f"model_layer_{i + 1}" for i in range(SPEC.n_moe_layers)
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    reader = TraceReader(trace.root, trace.model, trace.corpus)
    assert [reader.model_layer(i) for i in range(SPEC.n_moe_layers)] == list(
        range(1, SPEC.n_moe_layers + 1)
    )

    manifest["layer_index_map"] = [0, 1]
    (shard_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    reader = TraceReader(trace.root, trace.model, trace.corpus)
    with pytest.raises(FormatError, match="no trace layer"):
        reader.model_layer(SPEC.n_moe_layers - 1)
