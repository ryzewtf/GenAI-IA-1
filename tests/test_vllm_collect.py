"""vLLM collection loop tests — CPU-only, no torch/vllm.

The `VLLMCaptureEngine` GPU load is proven by the P3 notebook, not here. Everything else — the
hidden subsample/index-scheme replication, shard concatenation, stats validation, the engine-neutral
manifest, and the full S.3 loop (capture -> validate -> manifest -> upload+verify -> ledger) — is
plain Python and is exercised here against a fake engine and a `LocalDirBackend`, the same
production round-trip code the real run uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.capture.vllm_collect import (
    CollectError,
    build_vllm_manifest,
    collect_shard,
    run_vllm_collection,
    spec_and_gating_for,
    validate_vllm_stats,
    _capture_mask,
)
from src.capture.vllm_trace import gating_from_config, recompute_topk
from src.runtime.config import RunConfig
from src.runtime.runner import ShardPlan, plan_shards
from src.runtime.state import ShardState
from src.runtime.upload import LocalDirBackend
from src.traces.format import (
    HIDDEN_INDEX_DTYPE,
    REQUIRED_MANIFEST_KEYS,
    SHARD_INVARIANT_KEYS,
    STREAM_FILES,
    TraceSpec,
    read_manifest,
)

RUN_VLLM = "configs/run_vllm.yaml"
SPEC = TraceSpec(n_moe_layers=4, n_experts=16, top_k=3, hidden_dim=8)
GATING = gating_from_config({"logit_tensor_used": "ffn_moe_probs", "has_router_bias": False})

MODEL_META = {
    "checkpoint_status": "base",
    "router_dtype": "F32",
    "logit_tensor_used": "ffn_moe_probs",
    "has_router_bias": False,
    "n_moe_layers": SPEC.n_moe_layers,
    "n_experts": SPEC.n_experts,
    "top_k": SPEC.top_k,
    "hidden_dim": SPEC.hidden_dim,
    "moe_layer_offset": 0,
    "vllm": {
        "model_id": "fake/model",
        "quant": "fp16",
        "tensor_parallel": 1,
        "model_sha256": "a" * 64,
    },
}


class _FakeEngine:
    """Stands in for VLLMCaptureEngine: tokenize by whitespace, emit gate-consistent captures.

    `capture_document` returns, per layer, (logits, topk, router_input) where topk is the argsort of
    the logits — so `DocumentTrace.put_layer`'s faithfulness gate passes, which is the state a real
    faithful capture is in. A test that wants the gate to FAIL corrupts topk itself.
    """

    def __init__(self, spec: TraceSpec, *, tensor_parallel_size: int = 1, seed: int = 0):
        self.spec = spec
        self.tensor_parallel_size = tensor_parallel_size
        self.vllm_version = "0.10.2"
        self._rng = np.random.default_rng(seed)

    def tokenize(self, text: str) -> list[int]:
        ids = [(abs(hash(w)) % 5000) for w in text.split()]
        return ids or [1]

    def capture_document(self, ids):
        n, s = len(ids), self.spec
        out = {}
        for L in range(s.n_moe_layers):
            logits = self._rng.standard_normal((n, s.n_experts)).astype(np.float32)
            topk = recompute_topk(logits.astype(np.float64), s.top_k)
            router_input = self._rng.standard_normal((n, s.hidden_dim)).astype(np.float32)
            out[L] = (logits, topk, router_input)
        return out


def _write_shard_jsonl(path: Path, docs) -> Path:
    """docs: list of (doc_id, text). Writes the CORPUS_FIELDS a shard JSONL needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for did, text in docs:
            fh.write(json.dumps({
                "doc_id": did, "text": text, "domain": "prose", "lang": "en",
                "source": "t", "n_tokens_ref": len(text.split()), "split": "train", "shard_id": 0,
            }) + "\n")
    return path


def _plan(path: Path, docs, *, hidden_stride: int) -> ShardPlan:
    return ShardPlan(
        shard_id=0,
        doc_ids=tuple(d for d, _ in docs),
        n_docs=len(docs),
        ref_token_offset=0,
        corpus_path=path,
        n_tokens_ref=sum(len(t.split()) for _, t in docs),
        hidden_stride=hidden_stride,
    )


# -- hidden subsample + index scheme -----------------------------------------------------------


def test_capture_mask_matches_moe_trace_rule():
    # (doc_id*n_ctx + pos) % stride == 0
    assert _capture_mask(doc_id=0, n_tokens=6, n_ctx=64, hidden_stride=2) == [
        True, False, True, False, True, False
    ]
    # doc 1 with n_ctx=64, stride=3: base=64, 64%3==1 so pos where (64+p)%3==0 -> p=2,5
    assert _capture_mask(doc_id=1, n_tokens=6, n_ctx=64, hidden_stride=3) == [
        False, False, True, False, False, True
    ]
    # stride 0 disables capture entirely (runner._check_n_captured contract)
    assert _capture_mask(doc_id=0, n_tokens=4, n_ctx=64, hidden_stride=0) == [False] * 4


# -- collect_shard -----------------------------------------------------------------------------


def test_collect_shard_concatenates_streams_and_indexes_globally(tmp_path):
    docs = [(0, "alpha beta gamma delta"), (1, "one two three"), (2, "x y z w v")]
    jsonl = _write_shard_jsonl(tmp_path / "shard.jsonl", docs)
    plan = _plan(jsonl, docs, hidden_stride=2)
    out = tmp_path / "shard_00000"

    stats = collect_shard(_FakeEngine(SPEC), plan, spec=SPEC, gating=GATING, n_ctx=64, out_dir=out)

    # collect_shard runs check_file_sizes internally; reaching here means every stream is exact.
    assert stats["n_docs"] == 3 and stats["n_docs_in_shard"] == 3
    assert stats["n_docs_truncated"] == 0
    assert stats["index_scheme"] == "doc_id*n_ctx+pos_in_doc"
    assert stats["index_doc_span"] == 64
    n_tokens = 4 + 3 + 5
    assert stats["n_tokens"] == n_tokens

    # hidden_index is global (doc_id*n_ctx + pos) and strictly ascending across docs.
    idx = np.frombuffer((out / STREAM_FILES["hidden_index"]).read_bytes(), dtype=HIDDEN_INDEX_DTYPE)
    assert stats["n_captured"] == idx.size
    assert np.all(np.diff(idx.astype(np.int64)) > 0)
    expected = []
    for did, n in ((0, 4), (1, 3), (2, 5)):
        expected += [did * 64 + p for p in range(n) if (did * 64 + p) % 2 == 0]
    assert idx.tolist() == expected


def test_collect_shard_flags_truncation_and_does_not_write_the_doc(tmp_path):
    docs = [(0, "a b c"), (1, "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10")]  # doc 1 has 10 tokens
    jsonl = _write_shard_jsonl(tmp_path / "shard.jsonl", docs)
    plan = _plan(jsonl, docs, hidden_stride=0)
    stats = collect_shard(_FakeEngine(SPEC), plan, spec=SPEC, gating=GATING, n_ctx=8, out_dir=tmp_path / "o")
    assert stats["n_docs_truncated"] == 1
    assert stats["first_truncated_doc"] == 1
    assert stats["n_tokens_dropped"] == 2
    assert stats["n_docs"] == 1  # only the doc that fit was written


def test_collect_shard_halts_when_capture_is_unfaithful(tmp_path):
    docs = [(0, "alpha beta gamma")]
    jsonl = _write_shard_jsonl(tmp_path / "shard.jsonl", docs)
    plan = _plan(jsonl, docs, hidden_stride=0)

    class _LyingEngine(_FakeEngine):
        def capture_document(self, ids):
            out = super().capture_document(ids)
            logits, topk, router_input = out[0]
            topk = topk.copy()
            topk[0, 0] = (topk[0, 0] + 1) % self.spec.n_experts  # corrupt one label
            out[0] = (logits, topk, router_input)
            return out

    with pytest.raises(Exception):  # SelectionMismatch (a CaptureError) bubbles out of put_layer
        collect_shard(_LyingEngine(SPEC), plan, spec=SPEC, gating=GATING, n_ctx=64, out_dir=tmp_path / "o")


# -- validate_vllm_stats -----------------------------------------------------------------------


def _good_stats(plan: ShardPlan, *, n_ctx=64):
    return {
        "shard_id": plan.shard_id, "n_docs": plan.n_docs, "n_docs_in_shard": plan.n_docs,
        "n_tokens": 12, "n_captured": 0, "n_moe_layers": SPEC.n_moe_layers,
        "n_experts": SPEC.n_experts, "top_k": SPEC.top_k, "hidden_dim": SPEC.hidden_dim,
        "hidden_stride": plan.hidden_stride, "index_scheme": "doc_id*n_ctx+pos_in_doc",
        "index_doc_span": n_ctx, "n_docs_truncated": 0, "n_tokens_dropped": 0,
        "first_truncated_doc": None,
    }


def test_validate_vllm_stats_accepts_a_good_shard():
    plan = _plan(Path("x"), [(0, "a"), (1, "b")], hidden_stride=0)
    validate_vllm_stats(_good_stats(plan), plan, spec=SPEC, n_ctx=64)  # must not raise


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(shard_id=7),
    lambda s: s.update(n_docs=1),
    lambda s: s.update(index_scheme="running_counter"),
    lambda s: s.update(index_doc_span=2048),
    lambda s: s.update(n_docs_truncated=1, first_truncated_doc=0),
    lambda s: s.update(n_experts=32),
])
def test_validate_vllm_stats_rejects_the_bad_ones(mutate):
    plan = _plan(Path("x"), [(0, "a"), (1, "b")], hidden_stride=0)
    stats = _good_stats(plan)
    mutate(stats)
    with pytest.raises(CollectError):
        validate_vllm_stats(stats, plan, spec=SPEC, n_ctx=64)


# -- build_vllm_manifest -----------------------------------------------------------------------


def test_build_vllm_manifest_has_engine_neutral_keys(tmp_path):
    config = RunConfig.load(RUN_VLLM)
    docs = [(0, "a b c d")]
    plan = _plan(tmp_path / "s.jsonl", docs, hidden_stride=0)
    shard_dir = tmp_path / "shard_00000"
    shard_dir.mkdir()
    stats = {"n_docs": 1, "n_tokens": 4, "n_captured": 0, "tensor_parallel_size": 1}

    manifest = build_vllm_manifest(
        shard_dir, plan, stats, config=config, spec=SPEC, model="olmoe-0125",
        corpus="unit", model_meta=MODEL_META, n_ctx=2048,
    )
    assert manifest["model_sha256"] == "a" * 64
    assert manifest["engine_build"] == "vllm@0.10.2"
    assert "gguf_sha256" not in manifest and "llama_cpp_commit" not in manifest
    # every required + invariant key present and non-null, so the standard readers accept it.
    got = read_manifest(shard_dir)
    for key in (*REQUIRED_MANIFEST_KEYS, *SHARD_INVARIANT_KEYS):
        assert got.get(key) is not None, key


# -- the full loop, through the real ledger + LocalDirBackend round-trip -----------------------


def _corpus(path: Path, n_docs: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for did in range(n_docs):
            fh.write(json.dumps({
                "doc_id": did, "text": f"doc {did} alpha beta gamma", "domain": "prose",
                "lang": "en", "source": "t", "n_tokens_ref": 5, "split": "train", "shard_id": did // 2,
            }) + "\n")
    return path


def test_run_vllm_collection_local_roundtrip_and_resumes(tmp_path):
    config = RunConfig.load(RUN_VLLM)
    corpus = _corpus(tmp_path / "corpus.jsonl", n_docs=6)  # 3 shards of 2 docs
    scratch = tmp_path / "scratch"
    plans = plan_shards(corpus, out_root=scratch / "shards", subsample_n=4)
    backend = LocalDirBackend(tmp_path / "traces")
    ledger = ShardState.load_or_create(scratch / "state.json", "olmoe-0125", "unit", config.sha256)

    outcome = run_vllm_collection(
        plans, engine=_FakeEngine(SPEC), config=config, backend=backend, ledger=ledger,
        spec=SPEC, gating=GATING, model="olmoe-0125", corpus="unit", model_meta=MODEL_META,
        scratch_root=scratch, remote_root="traces/olmoe-0125/unit",
        log_path=tmp_path / "log.csv", verbose=False,
    )
    assert outcome.ok
    assert sorted(outcome.completed) == [p.shard_id for p in plans]
    # uploads verified into the local backend, and the ledger recorded every shard.
    assert ledger.completed_ids() == {p.shard_id for p in plans}
    # scratch shard dirs were deleted after the verified upload (I9 / T4.4).
    assert not (scratch / "shard_00000").exists()
    # a remote manifest is readable back with the engine-neutral keys.
    man = json.loads((tmp_path / "traces" / "traces" / "olmoe-0125" / "unit" / "shard_00000"
                      / "manifest.json").read_text())
    assert man["engine_build"] == "vllm@0.10.2"

    # Resume: a second run skips every shard without re-capturing (T3.6).
    ledger2 = ShardState.load_or_create(scratch / "state.json", "olmoe-0125", "unit", config.sha256)
    again = run_vllm_collection(
        plans, engine=_FakeEngine(SPEC), config=config, backend=backend, ledger=ledger2,
        spec=SPEC, gating=GATING, model="olmoe-0125", corpus="unit", model_meta=MODEL_META,
        scratch_root=scratch, remote_root="traces/olmoe-0125/unit", verbose=False,
    )
    assert again.ok and again.completed == [] and len(again.results) == len(plans)
    assert all(r.status == "skipped" for r in again.results)


def test_spec_and_gating_for_reads_the_card():
    spec, gating = spec_and_gating_for(MODEL_META)
    assert spec == SPEC
    assert gating.softmax and not gating.has_router_bias
