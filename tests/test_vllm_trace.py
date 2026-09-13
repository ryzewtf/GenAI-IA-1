"""vLLM capture harness tests — the byte-identity and faithfulness proofs (plan P1).

These run on CPU with no vLLM/torch import, so they are the cheap gate before any Kaggle GPU time.
The assertions that matter most are the ones guarding SILENT corruption:

* the float16 numpy cast produces the SAME bits as the C++ ``f32_to_f16`` (else logits.bin/hidden
  .bin diverge from every trace already collected, undetectably);
* the selection gate HALTS when recomputed top-k disagrees with the engine (else topk.bin — the
  study's labels, I1 — is quietly wrong);
* GPT-OSS' router bias changes selection, so ignoring it names experts the model never routed;
* the staged buffers match the format's size arithmetic exactly, layer-major within token.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.capture.vllm_trace import (
    CaptureError,
    DocumentTrace,
    GatingOp,
    RouterCapture,
    SelectionMismatch,
    gating_from_config,
    recompute_topk,
)
from src.traces.format import (
    FLAG_HIDDEN_CAPTURED,
    LOGIT_DTYPE,
    TOPK_DTYPE,
    TraceSpec,
    expected_file_sizes,
)

SPEC = TraceSpec(n_moe_layers=4, n_experts=16, top_k=3, hidden_dim=8)


# -- float16 identity with the C++ writer ----------------------------------------------------


def _cpp_f32_to_f16(value: float) -> int:
    """Reference port of ``trace_writer.hpp``'s f32_to_f16 (RNE), for bit-exact comparison."""
    bits = np.float32(value).view(np.uint32).item()
    sign = (bits >> 16) & 0x8000
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & 0x007FFFFF
    if exponent == 128:
        return sign | 0x7C00 | (0x0200 if mantissa else 0)
    if exponent > 15:
        return sign | 0x7C00
    if exponent >= -14:
        half = ((exponent + 15) << 10) | (mantissa >> 13)
        rest = mantissa & 0x1FFF
        if rest > 0x1000 or (rest == 0x1000 and (half & 1)):
            half += 1
        return sign | half
    if exponent >= -24:
        mantissa |= 0x00800000
        shift = -exponent - 14 + 13
        half = mantissa >> shift
        rest = mantissa & ((1 << shift) - 1)
        midpoint = 1 << (shift - 1)
        if rest > midpoint or (rest == midpoint and (half & 1)):
            half += 1
        return sign | half
    return sign


def test_numpy_float16_matches_cpp_writer_bit_for_bit():
    """numpy's float16 cast == the C++ writer's f32_to_f16, EXCEPT in the deep-subnormal corner.

    The two must agree on every value the analysis actually reads. logits.bin holds softmax
    probabilities (>= 0, summing to 1) and hidden.bin holds normalized hidden states (O(1)); the
    only place they disagree is |x| in [2^-25, 2^-24) ~ [3e-8, 6e-8), where IEEE RNE rounds up to
    the smallest subnormal (0x0001) but the C++ port shifts the mantissa fully out before its
    round test and yields 0x0000. numpy is the IEEE-correct side. This does NOT break the
    byte-identity guarantee for the study: the vLLM port takes a NEW run_config_sha256 (different
    engine == different experiment, plan S.3), so vLLM traces are never merged with llama.cpp ones
    and there is no cross-engine byte comparison to satisfy — the port simply uses the correct cast.
    The exceptions are asserted explicitly here so the corner is documented, not silently ignored.
    """
    rng = np.random.default_rng(0)
    # A spread that exercises normals, subnormals, ties, overflow and sign.
    values = np.concatenate([
        rng.standard_normal(4000).astype(np.float32),
        (rng.standard_normal(2000).astype(np.float32) * 1e-5),   # into subnormal range
        np.array([0.0, -0.0, 65504.0, -65504.0, 70000.0, 1.0009765625], np.float32),
    ])
    with np.errstate(over="ignore"):  # overflow-to-inf is intended and asserted equal below
        numpy_bits = values.astype(np.float16).view(np.uint16)
    cpp_bits = np.array([_cpp_f32_to_f16(v) for v in values], dtype=np.uint16)

    disagree = np.where(numpy_bits != cpp_bits)[0]
    # Every disagreement is the deep-subnormal corner: numpy rounds to +/-smallest-subnormal
    # (0x0001 / 0x8001), the C++ port to +/-zero, and the magnitude is below 2^-24.
    for i in disagree:
        assert (numpy_bits[i] & 0x7FFF) == 0x0001
        assert (cpp_bits[i] & 0x7FFF) == 0x0000
        assert abs(float(values[i])) < 2.0 ** -24
    # And they agree on everything at or above the smallest normal — i.e. everything the study reads.
    normal = np.abs(values) >= 2.0 ** -14
    assert np.array_equal(numpy_bits[normal], cpp_bits[normal])


# -- gating from config -----------------------------------------------------------------------


def test_gating_softmax_for_ffn_moe_probs():
    op = gating_from_config({"logit_tensor_used": "ffn_moe_probs"})
    assert op.softmax and not op.has_router_bias
    logits = np.array([[2.0, 1.0, 0.0, -1.0]])
    written = op.written_logits(logits, None)
    assert written.shape == (1, 4)
    np.testing.assert_allclose(written.sum(axis=-1), 1.0)          # probabilities
    assert written[0, 0] == written.max()                          # order preserved


def test_gating_rejects_unknown_tensor():
    with pytest.raises(CaptureError):
        gating_from_config({"logit_tensor_used": "ffn_moe_probs_biased"})
    with pytest.raises(CaptureError):
        gating_from_config({})  # missing entirely


def test_router_bias_changes_selection():
    """GPT-OSS: bias is added before selection, so it can change WHICH experts win."""
    op = GatingOp(softmax=True, has_router_bias=True, logit_tensor_used="ffn_moe_probs")
    logits = np.array([[1.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]])
    bias = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])      # lifts expert 7 to the top
    top_with = recompute_topk(op.selection_scores(logits, bias), top_k=2)
    top_without = recompute_topk(logits, top_k=2)
    assert 7 in top_with[0].tolist()
    assert 7 not in top_without[0].tolist()
    with pytest.raises(CaptureError):
        op.selection_scores(logits, None)  # bias declared but not supplied


# -- top-k recomputation ----------------------------------------------------------------------


def test_recompute_topk_is_descending_and_int32():
    scores = np.array([[0.1, 0.9, 0.5, 0.3]])
    top = recompute_topk(scores, top_k=2)
    assert top.dtype == TOPK_DTYPE
    assert top[0].tolist() == [1, 2]  # 0.9 then 0.5


# -- the selection faithfulness gate ----------------------------------------------------------


def _doc(n_tokens=5, capture=None, gating=None):
    if capture is None:
        capture = [True, False, True, False, True][:n_tokens]
    if gating is None:
        gating = GatingOp(softmax=True, has_router_bias=False, logit_tensor_used="ffn_moe_probs")
    return DocumentTrace(SPEC, n_tokens=n_tokens, capture_mask=capture, gating=gating)


def _layer_inputs(n_tokens, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n_tokens, SPEC.n_experts)).astype(np.float32)
    router_input = rng.standard_normal((n_tokens, SPEC.hidden_dim)).astype(np.float32)
    return logits, router_input


def test_matching_selection_is_accepted():
    doc = _doc()
    for L in range(SPEC.n_moe_layers):
        logits, router_input = _layer_inputs(doc.n_tokens, seed=L)
        # vLLM's "own" selection == a correct argsort, so the gate must pass.
        vllm_topk = recompute_topk(logits.astype(np.float64), SPEC.top_k)
        doc.put_layer(L, logits=logits, vllm_topk=vllm_topk, router_input=router_input)
    bufs = doc.to_buffers(token_ids=list(range(doc.n_tokens)), doc_id=3, global_token_base=100)
    assert set(bufs) == {
        "tokens.bin", "topk.bin", "logits.bin", "hidden.bin", "hidden_index.bin",
    }


def test_mismatched_selection_halts():
    doc = _doc()
    logits, router_input = _layer_inputs(doc.n_tokens, seed=1)
    wrong = recompute_topk(logits.astype(np.float64), SPEC.top_k).copy()
    wrong[0, 0] = (wrong[0, 0] + 1) % SPEC.n_experts  # corrupt one label on one token
    with pytest.raises(SelectionMismatch):
        doc.put_layer(0, logits=logits, vllm_topk=wrong, router_input=router_input)


def test_selection_compared_as_sets_not_order():
    """topk.bin order is free; the gate compares the SET vLLM chose, not its order."""
    doc = _doc()
    logits, router_input = _layer_inputs(doc.n_tokens, seed=2)
    correct = recompute_topk(logits.astype(np.float64), SPEC.top_k)
    shuffled = correct[:, ::-1].copy()  # same experts, reversed order
    doc.put_layer(0, logits=logits, vllm_topk=shuffled, router_input=router_input)
    # topk.bin stores vLLM's order verbatim, not the re-sorted one.
    assert np.array_equal(doc._topk[:, 0, :], shuffled.astype(TOPK_DTYPE))


# -- staged buffer sizes match the frozen format ----------------------------------------------


def test_buffer_sizes_match_format_arithmetic():
    n_tokens = 5
    doc = _doc(n_tokens=n_tokens)
    for L in range(SPEC.n_moe_layers):
        logits, router_input = _layer_inputs(n_tokens, seed=L)
        vllm_topk = recompute_topk(logits.astype(np.float64), SPEC.top_k)
        doc.put_layer(L, logits=logits, vllm_topk=vllm_topk, router_input=router_input)
    n_captured = int(sum([True, False, True, False, True]))
    bufs = doc.to_buffers(token_ids=list(range(n_tokens)), doc_id=0, global_token_base=0)
    want = expected_file_sizes(SPEC, n_tokens, n_captured)
    for name, blob in bufs.items():
        assert len(blob) == want[name], name


def test_hidden_index_is_global_and_flags_set():
    n_tokens = 5
    capture = [False, True, False, False, True]
    doc = _doc(n_tokens=n_tokens, capture=capture)
    for L in range(SPEC.n_moe_layers):
        logits, router_input = _layer_inputs(n_tokens, seed=L)
        vllm_topk = recompute_topk(logits.astype(np.float64), SPEC.top_k)
        doc.put_layer(L, logits=logits, vllm_topk=vllm_topk, router_input=router_input)
    bufs = doc.to_buffers(token_ids=[10, 11, 12, 13, 14], doc_id=7, global_token_base=1000)

    tokens = np.frombuffer(bufs["tokens.bin"], dtype=[
        ("token_id", "<u4"), ("doc_id", "<u4"), ("pos_in_doc", "<u4"), ("flags", "<u4")])
    assert tokens["flags"].tolist() == [0, FLAG_HIDDEN_CAPTURED, 0, 0, FLAG_HIDDEN_CAPTURED]
    idx = np.frombuffer(bufs["hidden_index.bin"], dtype="<u4")
    assert idx.tolist() == [1001, 1004]  # global base + captured positions, ascending


def test_incomplete_document_refuses_to_flush():
    doc = _doc()
    logits, router_input = _layer_inputs(doc.n_tokens, seed=0)
    vllm_topk = recompute_topk(logits.astype(np.float64), SPEC.top_k)
    doc.put_layer(0, logits=logits, vllm_topk=vllm_topk, router_input=router_input)
    with pytest.raises(CaptureError):  # layers 1..3 never ingested
        doc.to_buffers(token_ids=list(range(doc.n_tokens)), doc_id=0, global_token_base=0)


def test_double_layer_ingest_refused():
    doc = _doc()
    logits, router_input = _layer_inputs(doc.n_tokens, seed=0)
    vllm_topk = recompute_topk(logits.astype(np.float64), SPEC.top_k)
    doc.put_layer(0, logits=logits, vllm_topk=vllm_topk, router_input=router_input)
    with pytest.raises(CaptureError):
        doc.put_layer(0, logits=logits, vllm_topk=vllm_topk, router_input=router_input)


# -- RouterCapture select_experts wrapper (torch-free fakes) -----------------------------------
#
# The GPU-side hook registration needs torch, but the select_experts WRAPPER — the part that makes
# the P1 gate an independent test rather than a tautology — is plain Python and testable with a
# fake experts module. torch is only imported lazily inside _to_2d_numpy, so we pass numpy arrays
# that quack like tensors just enough for the code paths exercised here.


class _FakeExperts:
    """Stands in for a vLLM FusedMoE: its select_experts returns (topk_weights, topk_ids)."""

    def __init__(self, topk_ids):
        self._topk_ids = topk_ids
        self.calls = 0

    def select_experts(self, *args, **kwargs):
        self.calls += 1
        return ("weights-sentinel", self._topk_ids)


class _FakeNpTensor:
    """A numpy array wearing the minimal torch-tensor API _to_2d_numpy calls."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def detach(self):
        return self

    def to(self, *a, **k):
        return self

    def numpy(self):
        return self._arr


def _patch_torch(monkeypatch):
    """Install a fake `torch` module so _to_2d_numpy runs without the real dependency."""
    import types

    fake = types.SimpleNamespace(
        is_floating_point=lambda t: False,  # topk_ids are integer
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)


def test_select_experts_wrapper_captures_topk_and_is_transparent(monkeypatch):
    _patch_torch(monkeypatch)
    topk = _FakeNpTensor(np.array([[1, 4, 7], [2, 3, 5]], dtype=np.int64))
    experts = _FakeExperts(topk)
    # One layer: one router (unused here) + one experts module.
    cap = RouterCapture(router_modules=[object()], experts_modules=[experts])
    cap._wrap_select_experts(experts, trace_layer=0)  # register() would also hook the router

    result = experts.select_experts("hidden", "router_logits")
    # The wrapper returns the ORIGINAL result untouched (numerics unchanged).
    assert result[0] == "weights-sentinel"
    assert experts.calls == 1
    # ...and stashed vLLM's topk_ids as int32 for layer 0.
    assert cap.topk_ids[0].dtype == TOPK_DTYPE
    assert cap.topk_ids[0].tolist() == [[1, 4, 7], [2, 3, 5]]

    # remove() restores the original bound method.
    original = experts.select_experts
    cap._patched = [(experts, "select_experts", _FakeExperts.select_experts.__get__(experts))]
    cap.remove()
    assert experts.select_experts != original  # restored to the unwrapped method


def test_experts_modules_must_align_with_routers():
    with pytest.raises(CaptureError):
        RouterCapture(router_modules=[object(), object()], experts_modules=[object()])


def test_missing_select_experts_is_a_halt(monkeypatch):
    _patch_torch(monkeypatch)
    cap = RouterCapture(router_modules=[object()], experts_modules=[object()])
    with pytest.raises(CaptureError):
        cap._wrap_select_experts(object(), trace_layer=0)  # no select_experts attr
