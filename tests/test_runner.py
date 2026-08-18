"""Shard-loop tests — plan S.3 step 2, T5.2.

Everything here runs offline in milliseconds. `moe_trace` needs a GPU and a multi-GB GGUF, so the
capture subprocess is driven through the :class:`CaptureInvoker` seam by :class:`FakeInvoker`,
which writes stream files of exactly the right sizes and returns a ``capture_stats.json`` payload
shaped like the real binary's. Uploads go through ``LocalDirBackend``, which is shipped code
rather than a mock, so the round-trip and delete-on-success path under test is the production one.

What these tests are for: the loop's failure mode is not crashing, it is *recording a bad shard as
good*. A shard in the ledger is a claim that its bytes are durable, checksummed, and legal to
concatenate with every other shard of the model. So the tests below are mostly about exit code 0
with something wrong underneath — a ``mixed`` topk layout (I12), silent per-model truncation
(I15), a stats file left over from another shard — and about the two properties resumption needs:
a failed shard leaves *no* trace in the ledger, and a completed one is never re-invoked.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from src.capture.nodespec import NodeSpec
from src.corpus.build import Document, load_corpus, write_corpus
from src.corpus.spec import CorpusSpec
from src.runtime.config import ConfigError, RunConfig
from src.runtime.runner import (
    BASELINE_VARIANT,
    INDEX_SCHEME,
    CaptureVariant,
    RunnerError,
    ShardPlan,
    SubprocessInvoker,
    append_log_row,
    build_argv,
    hidden_stride_for,
    plan_shards,
    run_collection,
    run_shard,
)
from src.runtime.session import SessionBudget
from src.runtime.state import ShardState
from src.runtime.upload import LocalDirBackend
from src.traces.format import MANIFEST_NAME, STREAM_FILES, TraceSpec, expected_file_sizes

TOKENS_PER_DOC = 7
SUBSAMPLE_N = 20  # 60 reference tokens / 20 -> stride 3
DOCS_PER_SHARD = 2
N_SHARDS = 3

SPEC = NodeSpec(
    model="unit-moe",
    n_moe_layers=2,
    n_experts=8,
    top_k=2,
    hidden_dim=4,
    layer_map=[0, 1],
    node_topk="ffn_moe_topk-%d",
    node_logits="ffn_moe_probs-%d",
    node_router_input="ffn_norm-%d",
    verified=True,
)
TRACE_SPEC = TraceSpec(
    n_moe_layers=SPEC.n_moe_layers,
    n_experts=SPEC.n_experts,
    top_k=SPEC.top_k,
    hidden_dim=SPEC.hidden_dim,
)

#: Fields T1.1/T1.2 fill in for real. `write_manifest` refuses a null required key, so a shard
#: cannot be collected without them — these stand in for a filled-out models.yaml entry.
MODEL_META = {
    "checkpoint_status": "base",
    "quant": "Q4_K_M",
    "router_dtype": "f32",
    "gguf": {"sha256": "a" * 64},
}


# -- fixtures ---------------------------------------------------------------------------------


def make_docs(n_shards: int = N_SHARDS, per_shard: int = DOCS_PER_SHARD) -> list[Document]:
    """Documents already carrying the ``shard_id`` and ``split`` that T4.2/T4.3 assign."""
    docs: list[Document] = []
    doc_id = 0
    for shard in range(n_shards):
        for _ in range(per_shard):
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=f"document {doc_id} é \U0001f600 with a quoted \"text\" inside",
                    domain="prose",
                    lang="en",
                    source="unit",
                    n_tokens_ref=10,
                    split="train",
                    shard_id=shard,
                )
            )
            doc_id += 1
    return docs


@pytest.fixture
def corpus_path(tmp_path) -> Path:
    docs = make_docs()
    path = tmp_path / "corpus" / "unit.jsonl"
    write_corpus(path, docs, CorpusSpec(name="unit", target_tokens=1000, max_doc_tokens=20, shard_tokens=20))
    return path


@pytest.fixture
def spec_path(tmp_path) -> Path:
    return SPEC.write(tmp_path / "unit-moe.spec")


def write_run_config(tmp_path, **mutations) -> RunConfig:
    """A copy of the real ``configs/run.yaml``, optionally broken in one specific way.

    Mutations are ``<block>__<key>=value`` against the ``hashed`` block. Loading the shipped file
    rather than a hand-written fixture is deliberate: the argv assertions below are checking that
    the *pinned* values reach the binary, and a fixture with its own ctx_size would pass happily
    while the real config's value never got read.
    """
    raw = yaml.safe_load(Path("configs/run.yaml").read_text(encoding="utf-8"))
    for dotted, value in mutations.items():
        block, key = dotted.split("__", 1)
        raw["hashed"][block][key] = value
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return RunConfig.load(path)


@pytest.fixture
def config(tmp_path) -> RunConfig:
    return write_run_config(tmp_path)


@pytest.fixture
def plans(corpus_path, tmp_path) -> list[ShardPlan]:
    return plan_shards(corpus_path, out_root=tmp_path / "shards", subsample_n=SUBSAMPLE_N)


@pytest.fixture
def ledger(tmp_path, config) -> ShardState:
    return ShardState.load_or_create(tmp_path / "state.json", "unit-moe", "unit", config.sha256)


@pytest.fixture
def backend(tmp_path) -> LocalDirBackend:
    return LocalDirBackend(tmp_path / "remote")


class FakeInvoker:
    """Stands in for `moe_trace`: writes correctly-sized streams, reports plausible stats.

    ``overrides`` mutates the stats *after* they are computed, which is how each test injects one
    specific lie (a mixed layout, a truncation count, a wrong document count) into an otherwise
    self-consistent capture.
    """

    def __init__(self, *, exit_code: int = 0, overrides: dict | None = None, write_files: bool = True):
        self.exit_code = exit_code
        self.overrides = dict(overrides or {})
        self.write_files = write_files
        self.calls: list[int] = []

    def capture(self, plan, *, config, spec_path, model_path, out_dir):
        self.calls.append(plan.shard_id)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_docs = len([ln for ln in plan.corpus_path.read_text(encoding="ascii").splitlines() if ln.strip()])
        n_tokens = n_docs * TOKENS_PER_DOC
        stride = plan.hidden_stride
        n_ctx = int(config.inference["ctx_size"])
        # Mirror moe_trace: a document's rows are the multiples of the stride inside its own
        # reserved block [doc_id*n_ctx, doc_id*n_ctx + n_d).
        n_captured = 0
        if stride:
            for doc_id in plan.doc_ids:
                base = doc_id * n_ctx
                n_captured += (
                    math.ceil((base + TOKENS_PER_DOC) / stride) - math.ceil(base / stride)
                )
        ubatch = int(config.inference["ubatch_size"])
        # One callback per (stream, trace layer, ubatch), exactly as moe_cb counts them.
        n_ubatches = n_docs * math.ceil(TOKENS_PER_DOC / ubatch)

        stats = {
            "shard_id": plan.shard_id,
            "model_spec": SPEC.model,
            "n_docs": n_docs,
            "n_docs_in_shard": n_docs,
            "n_tokens": n_tokens,
            "n_captured": n_captured,
            "n_docs_truncated": 0,
            "n_tokens_dropped": 0,
            "first_truncated_doc": -1,
            "n_moe_layers": SPEC.n_moe_layers,
            "n_experts": SPEC.n_experts,
            "top_k": SPEC.top_k,
            "hidden_dim": SPEC.hidden_dim,
            "hidden_stride": stride,
            "index_scheme": INDEX_SCHEME,
            "index_doc_span": n_ctx,
            "nodes_captured": 3 * SPEC.n_moe_layers * n_ubatches,
            "topk_layout": "strided_view",
            "node_topk": SPEC.node_topk,
            "node_logits": SPEC.node_logits,
            "node_router_input": SPEC.node_router_input,
            "exit_code": self.exit_code,
        }
        stats.update(self.overrides)

        if self.write_files:
            sizes = expected_file_sizes(TRACE_SPEC, int(stats["n_tokens"]), int(stats["n_captured"]))
            for name, size in sizes.items():
                (out_dir / name).write_bytes(b"\0" * size)
        return self.exit_code, stats


def collect(plans, *, config, ledger, backend, invoker, spec_path, tmp_path, **kwargs):
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("scratch_root", tmp_path / "scratch")
    kwargs.setdefault("remote_root", "traces/unit-moe/unit")
    return run_collection(
        plans,
        config=config,
        invoker=invoker,
        backend=backend,
        ledger=ledger,
        spec_path=spec_path,
        model_path=tmp_path / "model.gguf",
        model_meta=MODEL_META,
        **kwargs,
    )


def one_shard(plan, *, config, ledger, backend, invoker, spec_path, tmp_path):
    return run_shard(
        plan,
        config=config,
        invoker=invoker,
        backend=backend,
        ledger=ledger,
        spec_path=spec_path,
        model_path=tmp_path / "model.gguf",
        out_dir=tmp_path / "scratch" / f"shard_{plan.shard_id:05d}",
        remote_prefix=f"traces/unit-moe/unit/shard_{plan.shard_id:05d}",
        model_meta=MODEL_META,
    )


# -- shard planning ----------------------------------------------------------------------------


def test_every_document_lands_in_exactly_one_shard_and_none_is_duplicated_or_lost(plans, corpus_path):
    original = [d.doc_id for d in load_corpus(corpus_path)]
    planned = [doc_id for plan in plans for doc_id in plan.doc_ids]

    assert planned == original, "shards must partition the corpus in corpus order"
    assert len(set(planned)) == len(planned)
    assert sum(p.n_docs for p in plans) == len(original)
    for plan in plans:
        on_disk = [d.doc_id for d in load_corpus(plan.corpus_path)]
        assert on_disk == list(plan.doc_ids)


def test_ref_token_offsets_are_monotone_non_overlapping_and_cumulative_in_reference_tokens(plans):
    """Sizing information only: T4.4's hidden budget is computed from exactly these offsets.

    Deliberately no longer a token index — see `plan_shards` — but it still has to be the
    cumulative reference-token offset, or the budget is sized for the wrong shard.
    """
    assert [p.ref_token_offset for p in plans] == sorted(p.ref_token_offset for p in plans)
    running = 0
    for plan in plans:
        assert plan.ref_token_offset == running
        running += plan.n_tokens_ref
    spans = [(p.ref_token_offset, p.ref_token_offset + p.n_tokens_ref) for p in plans]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start


def test_the_hidden_stride_matches_the_arithmetic_the_corpus_budget_report_used():
    assert hidden_stride_for(60, 20) == 3
    assert hidden_stride_for(60, 0) == 0
    assert hidden_stride_for(10, 1000) == 1, "the stride never drops below 1"


def test_per_shard_jsonl_puts_doc_id_and_text_before_every_other_key(plans):
    """The C++ parser finds fields by substring search over the raw line, so order is load-bearing.

    ``parse_jsonl_line`` does ``line.find("\\"doc_id\\"")`` and ``line.find("\\"text\\"")``. If any
    field whose *value* could contain those substrings came first, the parser would read a doc_id
    out of somebody else's text and mis-attribute every row of the shard.
    """
    for plan in plans:
        for line in plan.corpus_path.read_text(encoding="ascii").splitlines():
            assert line.isascii(), "the C++ parser is byte-oriented; the corpus must be ensure_ascii"
            id_at = line.find('"doc_id"')
            text_at = line.find('"text"')
            assert 0 <= id_at < text_at
            for other in ("domain", "lang", "source", "n_tokens_ref", "split", "shard_id"):
                assert text_at < line.find(f'"{other}"')
            # And the value the parser would actually extract is the one we wrote.
            assert int(line[line.find(":", id_at) + 1 :].split(",", 1)[0]) in plan.doc_ids


def test_a_shard_whose_documents_are_not_contiguous_in_corpus_order_is_refused(tmp_path):
    docs = make_docs()
    docs[1].shard_id = 1  # interleave: shard 0, 1, 0, ...
    docs[2].shard_id = 0
    path = tmp_path / "interleaved.jsonl"
    write_corpus(path, docs, CorpusSpec(name="unit", target_tokens=1000, max_doc_tokens=20, shard_tokens=20))

    with pytest.raises(RunnerError, match="not contiguous"):
        plan_shards(path, out_root=tmp_path / "out", subsample_n=SUBSAMPLE_N)


# -- argv construction -------------------------------------------------------------------------


def test_the_argv_takes_ctx_and_ubatch_from_the_hashed_config_and_never_hardcodes_them(
    plans, tmp_path, config
):
    argv = build_argv(
        plans[0],
        config=config,
        binary="moe_trace",
        spec_path="unit.spec",
        model_path="m.gguf",
        out_dir=tmp_path / "out",
    )
    assert argv[argv.index("--ctx") + 1] == str(config.inference["ctx_size"])
    assert argv[argv.index("--ubatch") + 1] == str(config.inference["ubatch_size"])
    assert "--global-token-base" not in argv, (
        "the flag was removed on both sides: a binary that still accepts it indexes hidden.bin "
        "by a cumulative token count, and its shards silently overlap their neighbours'"
    )

    # Change the pinned value and the command line must follow it, not a constant in runner.py.
    other = write_run_config(tmp_path, inference__ctx_size=1024)
    argv2 = build_argv(
        plans[0],
        config=other,
        binary="moe_trace",
        spec_path="unit.spec",
        model_path="m.gguf",
        out_dir=tmp_path / "out",
    )
    assert argv2[argv2.index("--ctx") + 1] == "1024"


def test_the_subprocess_invoker_refuses_to_run_a_binary_that_was_never_built(plans, tmp_path, config):
    invoker = SubprocessInvoker(tmp_path / "does-not-exist.exe")
    with pytest.raises(RunnerError, match="not found"):
        invoker.capture(
            plans[0],
            config=config,
            spec_path=tmp_path / "unit.spec",
            model_path=tmp_path / "m.gguf",
            out_dir=tmp_path / "out",
        )
    assert invoker.calls == [], "nothing was invoked, so nothing should be recorded as invoked"


# -- per-shard validation ----------------------------------------------------------------------


def test_a_clean_shard_is_verified_recorded_and_removed_from_scratch(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker()
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )

    assert result.status == "complete" and result.upload_verified
    assert ledger.is_complete(0)
    record = ledger.record(0)
    assert record.upload_verified and record.n_tokens == DOCS_PER_SHARD * TOKENS_PER_DOC
    assert set(record.file_sha256) == set(STREAM_FILES.values()), "manifest.json is not a stream"

    remote = backend.list_files("traces/unit-moe/unit/shard_00000")
    assert {Path(r).name for r in remote} == set(STREAM_FILES.values()) | {MANIFEST_NAME}
    assert not (tmp_path / "scratch" / "shard_00000").exists(), "scratch holds one shard at a time (I9/T4.4)"


def test_a_nonzero_exit_code_does_not_mark_the_shard_complete_and_does_not_delete_local_files(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker(exit_code=7)  # 7 = lockstep failure (T3.1)
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )

    assert result.status == "failed" and result.exit_code == 7
    assert "LOCKSTEP" in result.error
    assert ledger.completed_ids() == set()
    assert backend.list_files("traces/unit-moe/unit/shard_00000") == []
    shard_dir = tmp_path / "scratch" / "shard_00000"
    assert (shard_dir / "topk.bin").is_file(), "the only copy of the capture must survive a failure"


def test_a_mixed_topk_layout_fails_the_shard_even_though_the_binary_exited_zero(
    plans, config, ledger, backend, spec_path, tmp_path
):
    """The silent-corruption case, and the reason validate_stats exists.

    ``mixed`` means ``ffn_moe_topk`` arrived packed for some ubatches and strided for others, so
    the de-striding read one of the two wrong (I12). The result is expert ids that are in range,
    ``top_k``-distinct and false — every downstream check in T5.3 passes and every number in
    Phase 6 onwards is quietly wrong. Exit code 0 must not be enough here.
    """
    invoker = FakeInvoker(overrides={"topk_layout": "mixed"})
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )

    assert result.status == "failed" and result.exit_code == 0
    assert "mixed" in result.error and "I12" in result.error
    assert ledger.completed_ids() == set()
    assert backend.list_files("traces/unit-moe/unit/shard_00000") == []
    assert (tmp_path / "scratch" / "shard_00000" / "topk.bin").is_file()


def test_a_topk_layout_of_none_fails_the_shard_because_the_spec_named_a_node_that_was_never_emitted(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker(overrides={"topk_layout": "none"})
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )
    assert result.status == "failed" and ledger.completed_ids() == set()


def test_a_truncated_document_fails_the_shard_because_the_panel_was_not_shown_the_same_text(
    plans, config, ledger, backend, spec_path, tmp_path
):
    """Invariant I15. T4.2's cap is enforced under one reference tokenizer, truncation happens per
    model, and a truncated document means this checkpoint saw less text than a smaller-vocab one
    did — which is the assumption every cross-model comparison in Phase 9 rests on."""
    invoker = FakeInvoker(
        overrides={"n_docs_truncated": 1, "n_tokens_dropped": 31, "first_truncated_doc": 1}
    )
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )

    assert result.status == "failed"
    assert "I15" in result.error and "truncated" in result.error
    assert ledger.completed_ids() == set()


def test_a_stats_plan_mismatch_on_the_document_count_fails_the_shard(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker(overrides={"n_docs": DOCS_PER_SHARD - 1})
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )
    assert result.status == "failed"
    assert "documents" in result.error and ledger.completed_ids() == set()


def test_a_nodes_captured_count_that_is_not_three_streams_times_layers_fails_the_shard(
    plans, config, ledger, backend, spec_path, tmp_path
):
    """The derived invariant: nodes_captured == 3 * n_moe_layers * total ubatches.

    Dropping one stream for one layer keeps per-document lockstep satisfied for the layers that
    were emitted, so this divisibility is the only thing that notices.
    """
    clean = FakeInvoker()
    _, stats = clean.capture(
        plans[0],
        config=config,
        spec_path=spec_path,
        model_path=tmp_path / "m.gguf",
        out_dir=tmp_path / "probe",
    )
    invoker = FakeInvoker(overrides={"nodes_captured": stats["nodes_captured"] - 1})
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )
    assert result.status == "failed" and "nodes_captured" in result.error


def test_a_hidden_row_count_that_the_global_stride_cannot_produce_fails_the_shard(
    plans, config, ledger, backend, spec_path, tmp_path
):
    """The per-document index blocks make the exact count unknowable here, the bound is not.

    A document contributes floor(n_d/stride) or ceil(n_d/stride) rows depending on where its
    reserved block starts, and only the tokenizer knows n_d. The bound still catches what
    matters: a stride that is not the one this run asked for is wrong by a factor.
    """
    invoker = FakeInvoker(overrides={"n_captured": 1})
    result = one_shard(
        plans[1], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )
    assert result.status == "failed" and "must subsample between" in result.error


def test_a_hidden_row_count_inside_the_bound_is_accepted(
    plans, config, ledger, backend, spec_path, tmp_path
):
    """Guards the other side: the bound must not be so tight that a legal count fails.

    Two 7-token documents at stride 3 give 4, 5 or 6 rows depending on their block offsets, and
    all three are healthy captures.
    """
    for n_captured in (4, 5, 6):
        led = ShardState.load_or_create(
            tmp_path / f"state_{n_captured}.json", "unit-moe", "unit", config.sha256
        )
        result = one_shard(
            plans[1],
            config=config,
            ledger=led,
            backend=LocalDirBackend(tmp_path / f"remote_{n_captured}"),
            invoker=FakeInvoker(overrides={"n_captured": n_captured}),
            spec_path=spec_path,
            tmp_path=tmp_path / f"scratch_{n_captured}",
        )
        assert result.status == "complete", (n_captured, result.error)


def test_a_stats_file_left_behind_by_another_shard_is_not_accepted_as_this_shards_result(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker(overrides={"shard_id": 99, "index_doc_span": 4242})
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )
    assert result.status == "failed" and "stale" in result.error


def test_a_stats_exit_code_that_disagrees_with_the_process_exit_code_fails_the_shard(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker(overrides={"exit_code": 9})
    result = one_shard(
        plans[0], config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )
    assert result.status == "failed" and "different run" in result.error


# -- I9 ----------------------------------------------------------------------------------------


def test_the_runner_refuses_an_out_dir_under_kaggle_working(
    plans, config, ledger, backend, spec_path, tmp_path
):
    """Invariant I9: /kaggle/working is 20 GB and ~500 files and persists per notebook version, so
    a trace written there does not merely overflow — it wedges the notebook."""
    invoker = FakeInvoker()
    with pytest.raises(RunnerError, match="I9"):
        run_shard(
            plans[0],
            config=config,
            invoker=invoker,
            backend=backend,
            ledger=ledger,
            spec_path=spec_path,
            model_path=tmp_path / "model.gguf",
            out_dir=tmp_path / "kaggle" / "working" / "traces" / "shard_00000",
            remote_prefix="traces/unit-moe/unit/shard_00000",
            model_meta=MODEL_META,
        )
    assert invoker.calls == [], "the refusal must happen before any capture is started"

    with pytest.raises(RunnerError, match="I9"):
        collect(
            plans,
            config=config,
            ledger=ledger,
            backend=backend,
            invoker=invoker,
            spec_path=spec_path,
            tmp_path=tmp_path,
            scratch_root=tmp_path / "kaggle" / "working" / "scratch",
        )


# -- readiness gate ----------------------------------------------------------------------------


def test_the_loop_refuses_to_start_when_the_build_commit_is_not_pinned(
    plans, tmp_path, backend, spec_path
):
    """`assert_collection_ready` is the gate; the loop's job is to consult it *before* shard one.

    An unpinned build commit cannot be recovered after the fact: the shards are simply of unknown
    provenance, and I2 makes them unmergeable with everything collected before or after.
    """
    broken = write_run_config(tmp_path, build__llama_cpp_commit=None)
    ledger = ShardState.load_or_create(tmp_path / "state.json", "unit-moe", "unit", broken.sha256)
    invoker = FakeInvoker()

    with pytest.raises(ConfigError, match="llama_cpp_commit"):
        collect(
            plans,
            config=broken,
            ledger=ledger,
            backend=backend,
            invoker=invoker,
            spec_path=spec_path,
            tmp_path=tmp_path,
        )
    assert invoker.calls == [] and ledger.completed_ids() == set()


def test_the_loop_refuses_a_ledger_written_under_a_different_run_config(
    plans, config, tmp_path, backend, spec_path
):
    ledger = ShardState.load_or_create(tmp_path / "state.json", "unit-moe", "unit", "0" * 64)
    invoker = FakeInvoker()
    with pytest.raises(RunnerError, match="I2"):
        collect(
            plans,
            config=config,
            ledger=ledger,
            backend=backend,
            invoker=invoker,
            spec_path=spec_path,
            tmp_path=tmp_path,
        )
    assert invoker.calls == []


# -- the loop ----------------------------------------------------------------------------------


def test_a_full_pass_collects_every_shard_exactly_once(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker()
    outcome = collect(
        plans, config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )

    assert outcome.ok and not outcome.stopped_early
    assert outcome.completed == [0, 1, 2] and invoker.calls == [0, 1, 2]
    assert ledger.completed_ids() == {0, 1, 2}
    assert outcome.n_tokens == N_SHARDS * DOCS_PER_SHARD * TOKENS_PER_DOC


def test_the_loop_stops_at_the_failing_shard_instead_of_burning_quota_on_the_rest(
    plans, config, ledger, backend, spec_path, tmp_path
):
    invoker = FakeInvoker(overrides={"topk_layout": "mixed"})
    outcome = collect(
        plans, config=config, ledger=ledger, backend=backend, invoker=invoker, spec_path=spec_path, tmp_path=tmp_path
    )
    assert not outcome.ok and outcome.failed == [0]
    assert invoker.calls == [0], "a bad layout is a property of the run, not of one shard's luck"


def test_a_second_run_after_a_budget_stop_completes_only_the_remaining_shards(
    plans, config, tmp_path, backend, spec_path
):
    """Resumption, the whole point of S.3: the pending set comes off disk, not out of memory."""
    state_path = tmp_path / "state.json"
    first_ledger = ShardState.load_or_create(state_path, "unit-moe", "unit", config.sha256)
    first_invoker = FakeInvoker()

    # A budget already inside its reserve window: the loop must still finish shard 0 (a shard is
    # atomic) and only then notice it should stop.
    exhausted = SessionBudget(wall_limit_s=0.002, reserve_s=0.001)
    first = collect(
        plans,
        config=config,
        ledger=first_ledger,
        backend=backend,
        invoker=first_invoker,
        spec_path=spec_path,
        tmp_path=tmp_path,
        budget=exhausted,
        log_path=tmp_path / "collection_log.csv",
    )
    assert first.stopped_early and first.completed == [0] and first_invoker.calls == [0]

    second_ledger = ShardState.load_or_create(state_path, "unit-moe", "unit", config.sha256)
    assert second_ledger.completed_ids() == {0}
    assert second_ledger.pending([p.shard_id for p in plans]) == [1, 2]

    second_invoker = FakeInvoker()
    second = collect(
        plans,
        config=config,
        ledger=second_ledger,
        backend=backend,
        invoker=second_invoker,
        spec_path=spec_path,
        tmp_path=tmp_path,
        log_path=tmp_path / "collection_log.csv",
    )
    assert second_invoker.calls == [1, 2], "a completed shard must never be re-invoked"
    assert second.completed == [1, 2] and second.skipped == [0]
    assert second_ledger.completed_ids() == {0, 1, 2}


def test_an_interrupted_shard_is_absent_from_the_ledger_and_is_redone_on_the_next_run(
    plans, config, tmp_path, backend, spec_path
):
    state_path = tmp_path / "state.json"
    ledger = ShardState.load_or_create(state_path, "unit-moe", "unit", config.sha256)
    killed = FakeInvoker(exit_code=9)  # writer verify failure: a short file, i.e. a partial shard
    collect(
        plans, config=config, ledger=ledger, backend=backend, invoker=killed, spec_path=spec_path, tmp_path=tmp_path
    )
    assert ledger.completed_ids() == set()

    retry_ledger = ShardState.load_or_create(state_path, "unit-moe", "unit", config.sha256)
    retry = FakeInvoker()
    collect(
        plans, config=config, ledger=retry_ledger, backend=backend, invoker=retry, spec_path=spec_path, tmp_path=tmp_path
    )
    assert retry.calls == [0, 1, 2] and retry_ledger.completed_ids() == {0, 1, 2}


# -- collection_log.csv (T5.2) -----------------------------------------------------------------


def test_the_collection_log_gets_one_row_per_attempted_shard_and_appending_never_repeats_the_header(
    plans, config, tmp_path, backend, spec_path
):
    log = tmp_path / "results" / "collection_log.csv"
    state_path = tmp_path / "state.json"

    first_ledger = ShardState.load_or_create(state_path, "unit-moe", "unit", config.sha256)
    collect(
        plans,
        config=config,
        ledger=first_ledger,
        backend=backend,
        invoker=FakeInvoker(),
        spec_path=spec_path,
        tmp_path=tmp_path,
        budget=SessionBudget(wall_limit_s=0.002, reserve_s=0.001),
        log_path=log,
    )
    second_ledger = ShardState.load_or_create(state_path, "unit-moe", "unit", config.sha256)
    collect(
        plans,
        config=config,
        ledger=second_ledger,
        backend=backend,
        invoker=FakeInvoker(),
        spec_path=spec_path,
        tmp_path=tmp_path,
        log_path=log,
    )

    rows = list(csv.reader(log.read_text(encoding="utf-8").splitlines()))
    header, body = rows[0], rows[1:]
    assert header == [
        "model", "shard_id", "n_docs", "n_tokens", "wall_s", "tokens_per_s",
        "exit_code", "upload_verified", "run_config_sha256",
    ]
    assert [r for r in body if r == header] == [], "a repeated header makes the log unparseable"
    assert [int(r[1]) for r in body] == [0, 1, 2], "one row per attempt; a skipped shard is not one"
    assert {r[0] for r in body} == {"unit-moe"}
    assert {r[8] for r in body} == {config.sha256}, "every row carries the config it was collected under (I2)"
    assert all(int(r[7]) == 1 for r in body)


def test_a_failed_shard_is_still_logged_so_the_session_leaves_a_record_of_what_it_tried(
    plans, config, ledger, backend, spec_path, tmp_path
):
    log = tmp_path / "collection_log.csv"
    collect(
        plans,
        config=config,
        ledger=ledger,
        backend=backend,
        invoker=FakeInvoker(exit_code=3),  # model load failure
        spec_path=spec_path,
        tmp_path=tmp_path,
        log_path=log,
    )
    rows = list(csv.DictReader(log.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert rows[0]["exit_code"] == "3" and rows[0]["upload_verified"] == "0"


def test_the_log_header_is_written_once_even_when_the_file_already_exists_but_is_empty(
    tmp_path, config
):
    from src.runtime.runner import ShardResult

    log = tmp_path / "collection_log.csv"
    log.touch()
    append_log_row(log, ShardResult(shard_id=0, status="complete", exit_code=0), model="m", config=config)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("model,shard_id") and len(lines) == 2


# -- dry run -----------------------------------------------------------------------------------


def test_dry_run_invokes_nothing_and_still_prints_the_pinned_ctx_and_ubatch(
    plans, config, ledger, backend, spec_path, tmp_path, capsys
):
    invoker = FakeInvoker()
    log = tmp_path / "collection_log.csv"
    outcome = run_collection(
        plans,
        config=config,
        invoker=invoker,
        backend=backend,
        ledger=ledger,
        spec_path=spec_path,
        model_path=tmp_path / "model.gguf",
        scratch_root=tmp_path / "scratch",
        remote_root="traces/unit-moe/unit",
        model_meta=MODEL_META,
        log_path=log,
        dry_run=True,
    )
    printed = capsys.readouterr().out

    assert invoker.calls == [], "--dry-run must not invoke the capture binary"
    assert len(outcome.dry_run_argv) == len(plans)
    assert f"--ctx {config.inference['ctx_size']}" in printed
    assert f"--ubatch {config.inference['ubatch_size']}" in printed
    assert ledger.completed_ids() == set() and not log.exists()
    assert backend.list_files("traces") == []
    assert not (tmp_path / "scratch").exists(), "a dry run writes no trace scratch"


# -- manifest ----------------------------------------------------------------------------------


def test_the_manifest_written_beside_the_streams_carries_the_run_config_hash_and_the_shard_range(
    plans, config, ledger, backend, spec_path, tmp_path
):
    one_shard(
        plans[1], config=config, ledger=ledger, backend=backend, invoker=FakeInvoker(),
        spec_path=spec_path, tmp_path=tmp_path,
    )
    scratch = tmp_path / "remote" / "traces" / "unit-moe" / "unit" / "shard_00001" / MANIFEST_NAME
    manifest = json.loads(scratch.read_text(encoding="utf-8"))

    assert manifest["run_config_sha256"] == config.sha256
    assert manifest["shard_id"] == 1
    assert manifest["shard_doc_range"] == [2, 4]
    assert manifest["ref_token_offset"] == plans[1].ref_token_offset
    assert manifest["index_scheme"] == INDEX_SCHEME
    assert manifest["logit_tensor_used"] == "ffn_moe_probs", "I13: names the node top-k consumed"
    assert manifest["llama_cpp_commit"] == config.build["llama_cpp_commit"]


def test_a_shard_cannot_be_collected_while_the_gguf_hash_is_still_unknown(
    plans, config, ledger, backend, spec_path, tmp_path
):
    """T1.1 fills `gguf.sha256`. Until then a trace would be unattributable to a checkpoint, which
    makes every number derived from it unciteable — so the manifest writer refuses it."""
    result = run_shard(
        plans[0],
        config=config,
        invoker=FakeInvoker(),
        backend=backend,
        ledger=ledger,
        spec_path=spec_path,
        model_path=tmp_path / "model.gguf",
        out_dir=tmp_path / "scratch" / "shard_00000",
        remote_prefix="traces/unit-moe/unit/shard_00000",
        model_meta={**MODEL_META, "gguf": {"sha256": None}},
    )
    assert result.status == "failed" and "gguf_sha256" in result.error
    assert ledger.completed_ids() == set()


# -- capture variants (T3.7 / T3.8) --------------------------------------------------------------


def test_a_baseline_run_adds_no_flags(plans, tmp_path, config):
    argv = build_argv(plans[0], config=config, binary="moe_trace", spec_path="unit.spec",
                      model_path="m.gguf", out_dir=tmp_path / "out")
    assert "--override-tensor" not in argv and "--decode-mode" not in argv
    assert BASELINE_VARIANT.is_baseline


def test_a_variant_appends_after_every_pinned_flag(plans, tmp_path, config):
    """So a variant always reads as an ADDITION to the collection command line rather than an edit
    to it — the pinned flags are what run_config_sha256 claims, and a variant must not disturb
    them."""
    variant = CaptureVariant(name="router-cpu",
                             override_tensor=r"blk\.\d+\.ffn_gate_inp\.weight=CPU")
    argv = build_argv(plans[0], config=config, binary="moe_trace", spec_path="unit.spec",
                      model_path="m.gguf", out_dir=tmp_path / "out", variant=variant)
    assert argv[-2:] == ["--override-tensor", r"blk\.\d+\.ffn_gate_inp\.weight=CPU"]
    assert argv[argv.index("--ctx") + 1] == str(config.inference["ctx_size"])


def test_the_decode_variants_spell_out_their_flags(plans, tmp_path, config):
    full = CaptureVariant(name="decode-full", decode_mode="full")
    assert full.argv() == ["--decode-mode", "full"]

    tail = CaptureVariant(name="decode-tail", decode_mode="tail", decode_prefix=128)
    assert tail.argv() == ["--decode-mode", "tail", "--decode-prefix", "128"]

    argv = build_argv(plans[0], config=config, binary="moe_trace", spec_path="unit.spec",
                      model_path="m.gguf", out_dir=tmp_path / "out", variant=tail)
    assert argv[argv.index("--decode-mode") + 1] == "tail"


def test_an_impossible_variant_is_refused_at_construction():
    with pytest.raises(RunnerError, match="off, full or tail"):
        CaptureVariant(decode_mode="turbo")
    with pytest.raises(RunnerError, match="decode_prefix"):
        CaptureVariant(decode_mode="tail", decode_prefix=0)


def test_the_variant_serializes_for_the_manifest():
    """It has to reach the manifest: run_config_sha256 does NOT cover these flags — they are not
    in run.yaml — so without this field a T3.8 decode leg and a real collection shard would look
    like the same experiment, and check_equivalence could not tell that anything varied."""
    payload = CaptureVariant(name="decode-tail", decode_mode="tail", decode_prefix=64).to_json()
    assert payload == {"name": "decode-tail", "override_tensor": None,
                       "decode_mode": "tail", "decode_prefix": 64}
    assert BASELINE_VARIANT.to_json()["decode_prefix"] == 0, (
        "decode_prefix is meaningless unless the mode is 'tail'; reporting 512 there would look "
        "like a pinned value that did something"
    )


def test_the_invoker_carries_the_variant_into_every_shard(tmp_path, config):
    """Held on the invoker, not passed per shard: a run where some shards carried the variant and
    some did not would be an unanalysable mixture that every downstream check reads as one
    experiment."""
    from src.runtime.runner import SubprocessInvoker

    variant = CaptureVariant(name="decode-full", decode_mode="full")
    invoker = SubprocessInvoker(tmp_path / "moe_trace", variant=variant)
    assert invoker.variant is variant
    assert SubprocessInvoker(tmp_path / "moe_trace").variant is BASELINE_VARIANT


def test_a_dry_run_shows_the_command_line_that_would_actually_run(plans, spec_path, tmp_path, config):
    """The dry run's whole value is fidelity. Printing a baseline invocation for a gate leg would
    hide the one thing you dry-run in order to check."""
    from src.runtime.runner import SubprocessInvoker, run_collection
    from src.runtime.state import ShardState
    from src.runtime.upload import LocalDirBackend

    variant = CaptureVariant(name="router-cpu", override_tensor=r"blk\.\d+\.ffn_gate_inp\.weight=CPU")
    outcome = run_collection(
        plans,
        config=config,
        invoker=SubprocessInvoker(tmp_path / "moe_trace", variant=variant),
        backend=LocalDirBackend(tmp_path / "out"),
        ledger=ShardState.load_or_create(tmp_path / "state.json", "m", "c", config.sha256),
        spec_path=spec_path,
        model_path=tmp_path / "m.gguf",
        scratch_root=tmp_path / "scratch",
        remote_root="traces/m/c",
        dry_run=True,
        verbose=False,
    )
    assert outcome.dry_run_argv
    for argv in outcome.dry_run_argv:
        assert "--override-tensor" in argv
        assert argv[argv.index("--override-tensor") + 1].endswith("=CPU")
