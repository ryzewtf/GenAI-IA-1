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
    discover_router_and_experts,
    gating_from_config,
    recompute_topk,
    worker_drain_capture,
    worker_install_capture,
    worker_remove_capture,
    worker_reset_capture,
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
#
# The wrapper patches select_experts on the CLASS as a staticmethod, because that is exactly how
# vLLM calls it: `FusedMoE.select_experts(...)`, via the class, never `self.select_experts(...)`.
# The fakes model that faithfully — a staticmethod invoked through the class — so the test exercises
# the real dispatch path (an instance-attribute patch, the original bug, would silently capture
# nothing here just as it did on the GPU).


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


def test_select_experts_wrapper_captures_topk_in_order_and_is_transparent(monkeypatch):
    _patch_torch(monkeypatch)

    class _FakeMoE:
        """Stands in for vLLM FusedMoE: select_experts is a STATICMETHOD called via the class."""

        @staticmethod
        def select_experts(*args, **kwargs):
            return ("weights-sentinel", kwargs["topk_ids"])

    original = _FakeMoE.__dict__["select_experts"]  # the staticmethod descriptor, to compare later
    e0, e1 = _FakeMoE(), _FakeMoE()                  # two layers sharing one class
    cap = RouterCapture(router_modules=[object(), object()], experts_modules=[e0, e1])
    cap._wrap_select_experts([e0, e1])               # patches the class once

    # vLLM invokes it through the CLASS (staticmethod) — the case an instance patch would miss.
    r0 = _FakeMoE.select_experts(topk_ids=_FakeNpTensor(np.array([[1, 4, 7], [2, 3, 5]], np.int64)))
    r1 = _FakeMoE.select_experts(topk_ids=_FakeNpTensor(np.array([[0, 2, 6]], np.int64)))

    # The wrapper returns the ORIGINAL result untouched (numerics unchanged).
    assert r0[0] == "weights-sentinel" and r1[0] == "weights-sentinel"
    # ...and stashed each call's topk_ids as int32, keyed by CALL ORDER == layer.
    assert cap.topk_ids[0].dtype == TOPK_DTYPE
    assert cap.topk_ids[0].tolist() == [[1, 4, 7], [2, 3, 5]]
    assert cap.topk_ids[1].tolist() == [[0, 2, 6]]

    # reset() clears the per-document call log; remove() restores the original staticmethod.
    cap.reset()
    assert cap.topk_ids == {}
    cap.remove()
    assert _FakeMoE.__dict__["select_experts"] is original  # exact descriptor restored


def test_experts_modules_must_align_with_routers():
    with pytest.raises(CaptureError):
        RouterCapture(router_modules=[object(), object()], experts_modules=[object()])


def test_missing_select_experts_is_a_halt(monkeypatch):
    _patch_torch(monkeypatch)
    cap = RouterCapture(router_modules=[object()], experts_modules=[object()])
    with pytest.raises(CaptureError):
        cap._wrap_select_experts([object()])  # object() has no select_experts anywhere in its MRO


# -- module discovery + TP>1 worker-injection glue (torch-free fakes) --------------------------
#
# The TP=2 path can only be proven on a paid multi-GPU session, but its plumbing — discovering the
# router/experts modules by suffix, and the collective_rpc entrypoints that stash/drain a capture on
# a worker — is plain Python and testable here. The fakes model the shapes the real code touches:
# a module tree with `.named_modules()`, hookable router modules, and a FusedMoE-like experts CLASS
# whose select_experts is a staticmethod (so the class-level patch is what fires, as on the GPU).


class _FakeHandle:
    def remove(self):
        pass


class _FakeGate:
    def register_forward_pre_hook(self, fn):
        return _FakeHandle()

    def register_forward_hook(self, fn):
        return _FakeHandle()


def _fake_moe_model(n_layers):
    """A stand-in vLLM model: n_layers of `model.layers.{i}.mlp.{gate,experts}`.

    Returns (model, ExpertsClass). All experts instances share ExpertsClass, so the class-level
    select_experts patch is installed once — exactly the real dispatch.
    """
    class _FakeExpertsCls:
        @staticmethod
        def select_experts(*args, **kwargs):
            return ("weights-sentinel", kwargs["topk_ids"])

    mods = {}
    for i in range(n_layers):
        mods[f"model.layers.{i}.mlp.gate"] = _FakeGate()
        mods[f"model.layers.{i}.mlp.experts"] = _FakeExpertsCls()

    class _FakeModel:
        def named_modules(self):
            return list(mods.items())

    return _FakeModel(), _FakeExpertsCls


def test_discover_orders_by_layer_number_not_lexically():
    model, _ = _fake_moe_model(11)  # layers 0..10 — a lexical sort would put '10' before '2'
    routers, experts, gnames, enames = discover_router_and_experts(model, n_expected=11)
    assert len(routers) == 11 and len(experts) == 11
    assert gnames[0] == "model.layers.0.mlp.gate"
    assert gnames[-1] == "model.layers.10.mlp.gate"    # numeric ordering, not "model.layers.9..."
    assert enames[-1] == "model.layers.10.mlp.experts"


def test_discover_raises_on_bad_suffix_and_wrong_count():
    model, _ = _fake_moe_model(4)
    with pytest.raises(CaptureError, match="router_suffix"):
        discover_router_and_experts(model, router_suffix=".mlp.nope")
    with pytest.raises(CaptureError, match="expected 99"):
        discover_router_and_experts(model, n_expected=99)


def test_worker_install_reset_drain_remove_roundtrip(monkeypatch):
    _patch_torch(monkeypatch)
    import types

    model, ExpertsCls = _fake_moe_model(3)
    worker = types.SimpleNamespace(rank=0, model_runner=types.SimpleNamespace(model=model))

    ack = worker_install_capture(worker, n_moe_layers=3)
    assert ack == {"rank": 0, "n_gates": 3, "first_gate": "model.layers.0.mlp.gate"}
    assert worker._moe_capture is not None

    # vLLM calls select_experts via the CLASS; two calls -> layers 0,1 in order.
    ExpertsCls.select_experts(topk_ids=_FakeNpTensor(np.array([[1, 4, 7]], np.int64)))
    ExpertsCls.select_experts(topk_ids=_FakeNpTensor(np.array([[2, 3, 5]], np.int64)))
    drained = worker_drain_capture(worker)
    assert drained["rank"] == 0
    assert drained["topk_ids"][0].tolist() == [[1, 4, 7]]
    assert drained["topk_ids"][1].tolist() == [[2, 3, 5]]

    worker_reset_capture(worker)
    assert worker_drain_capture(worker)["topk_ids"] == {}      # per-document buffers cleared

    worker_remove_capture(worker)
    assert worker._moe_capture is None
    assert isinstance(ExpertsCls.__dict__["select_experts"], staticmethod)  # class restored


def test_worker_drain_is_none_before_install():
    import types

    assert worker_drain_capture(types.SimpleNamespace()) is None
    # reset/remove on a worker with no capture must be no-ops, not errors.
    worker_reset_capture(types.SimpleNamespace())
    worker_remove_capture(types.SimpleNamespace())
