"""The shard loop — plan S.3 step 2, T5.2.

This is the piece that turns `moe_trace.exe` into a collection run: it derives shards from the
corpus, invokes the capture binary once per shard, validates what came back, uploads it, and only
then records it in the ledger. Everything it does is composition — :mod:`src.runtime.config`,
:mod:`src.runtime.state`, :mod:`src.runtime.upload`, :mod:`src.runtime.session` and
:mod:`src.runtime.preflight` already own the individual guarantees — so the value here is entirely
in the *ordering* and in the checks between the steps.

The order is fixed (plan S.3 step 2 a-e), and each arrow is a place a session can die::

    preflight -> plan shards -> [per shard] capture -> validate stats -> write manifest
              -> upload + round-trip verify -> mark complete -> delete scratch -> budget check

A shard is the atomic unit. :func:`run_collection` consults :meth:`SessionBudget.should_stop`
*between* shards and never inside one, because a shard killed mid-capture leaves a truncated
`logits.bin` that is indistinguishable from a good one to everything except the size arithmetic —
and the whole point of S.3 is that such a shard is simply absent from the ledger and gets
recollected, at the cost of one shard rather than one session.

Two things this module deliberately does *not* trust:

* **The exit code alone.** `moe_trace` exits 0 on a run whose `topk` arrived in a layout the
  de-striding logic could not identify consistently, or whose documents were silently truncated
  under this model's tokenizer (I15). Both produce plausible, in-range, wrong labels. So the
  stats file is validated against the plan before the shard is allowed anywhere near the ledger —
  see :func:`validate_stats`, which is the reason this file exists.
* **`capture_stats.json` being fresh.** A binary that dies before writing it leaves the previous
  attempt's file in place. The invoker unlinks it first, and the stats are cross-checked against
  the invocation (`shard_id`, `index_scheme`, `exit_code`) so a stale file cannot be read as
  a successful run.

The subprocess call is behind :class:`CaptureInvoker` so the tests drive the whole loop offline
against a fake. `moe_trace` needs a GPU and a 4 GB GGUF; the loop's failure modes must be testable
without either.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..capture.nodespec import NodeSpec, parse_spec
from ..corpus.build import CORPUS_FIELDS, Document, load_corpus
from ..traces.format import STREAM_FILES, FormatError, write_manifest
from .config import RunConfig
from .preflight import run_preflight
from .session import SessionBudget
from .state import ShardState
from .upload import StorageBackend, UploadError, sha256_file, upload_shard

__all__ = [
    "RunnerError",
    "CaptureError",
    "CaptureInvoker",
    "SubprocessInvoker",
    "ShardPlan",
    "ShardResult",
    "CollectionResult",
    "STATS_NAME",
    "build_argv",
    "plan_shards",
    "hidden_stride_for",
    "validate_stats",
    "build_shard_manifest",
    "run_shard",
    "run_collection",
    "load_model_meta",
    "main",
]

#: `moe_trace` writes its stats here by default (`--stats` defaults to `<out>/capture_stats.json`).
#: The runner always passes `--stats` explicitly anyway, so the two sides cannot disagree.
STATS_NAME = "capture_stats.json"

# How `moe_trace` indexes rows of hidden.bin. Pinned here and cross-checked against every
# capture_stats.json, because a binary using a different scheme produces a shard that is valid on
# its own and unconcatenable with its neighbours -- see plan_shards' docstring.
INDEX_SCHEME = "doc_id*n_ctx+pos_in_doc"

#: Trailing stderr kept in an error message. Enough to see a llama.cpp assertion or a CUDA OOM,
#: short enough that a Kaggle log is still readable when twenty shards fail the same way.
STDERR_TAIL_BYTES = 4000

LOG_COLUMNS = (
    "model",
    "shard_id",
    "n_docs",
    "n_tokens",
    "wall_s",
    "tokens_per_s",
    "exit_code",
    "upload_verified",
    "run_config_sha256",
)

#: Layouts `moe_trace` may legitimately report. "mixed" means some ubatches came back packed and
#: some strided, i.e. the de-striding logic has misidentified a tensor's layout (I12) — the
#: labels are then in-range, distinct and wrong. "none" means no topk node was ever seen.
_OK_TOPK_LAYOUTS = ("strided_view", "contiguous")


class RunnerError(RuntimeError):
    """The collection run cannot proceed, or a shard failed validation."""


class CaptureError(RunnerError):
    """The capture binary could not be run, or did not leave usable stats behind."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------------------------------
# I9 — traces never go under /kaggle/working
# --------------------------------------------------------------------------------------


def assert_not_kaggle_working(path: Path | str, *, what: str = "trace output") -> Path:
    """Refuse a path under ``/kaggle/working`` — invariant I9.

    Working is capped at 20 GB and ~500 files and *persists per notebook version*, so a trace
    written there does not merely overflow: it wedges the notebook so it can no longer be saved.
    Matched on the ``kaggle/working`` component pair rather than on an absolute prefix, so the
    check also fires for a relative path or a drive-qualified one — a false positive costs a
    renamed directory, a false negative costs the session.
    """
    resolved = Path(path).absolute()
    parts = [p.lower() for p in PurePosixPath(resolved.as_posix()).parts]
    for a, b in zip(parts, parts[1:]):
        if a == "kaggle" and b == "working":
            raise RunnerError(
                f"refusing to write {what} to {resolved} (plan invariant I9): /kaggle/working is "
                "capped at 20 GB and ~500 files and persists per notebook version. Traces go to "
                "/kaggle/temp and are uploaded per shard; only results/ and manifests belong in "
                "working."
            )
    return resolved


# --------------------------------------------------------------------------------------
# shard planning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardPlan:
    """One unit of work: the documents of one ``shard_id`` and their corpus position.

    ``hidden_stride`` and ``n_tokens_ref`` are corpus-level quantities carried per shard on
    purpose. Both are derived from the *whole* corpus, so keeping them on the plan means the
    argv for a shard can never be assembled from one corpus's totals and another's documents.
    """

    shard_id: int
    doc_ids: tuple[int, ...]
    n_docs: int
    ref_token_offset: int
    corpus_path: Path
    n_tokens_ref: int = 0
    hidden_stride: int = 0

    @property
    def doc_range(self) -> tuple[int, int]:
        """Half-open ``doc_id`` range, the form ``shard_doc_range`` takes in the manifest."""
        return (self.doc_ids[0], self.doc_ids[-1] + 1)


def hidden_stride_for(total_tokens_ref: int, subsample_n: int | None) -> int:
    """Global token stride that yields ``subsample_n`` captured tokens corpus-wide (T4.4/O2).

    Identical arithmetic to ``src.corpus.build.hidden_budget_report``, which is what sized the
    per-shard hidden-state budget; if the two ever disagree the budget report is describing a
    capture that is not the one being run. 0 disables the subsample entirely.
    """
    if not subsample_n:
        return 0
    if total_tokens_ref <= 0:
        raise RunnerError("cannot derive a hidden stride for an empty corpus")
    return max(1, total_tokens_ref // int(subsample_n))


def plan_shards(
    corpus_path: Path | str,
    *,
    out_root: Path | str,
    subsample_n: int | None = None,
) -> list[ShardPlan]:
    """Split the corpus file into one JSONL per ``shard_id`` and describe each shard.

    ``ref_token_offset`` is the cumulative **reference**-token offset of the shard's first
    document, i.e. the sum of ``n_tokens_ref`` over every document before it in corpus order.
    ``src.corpus.build.hidden_budget_report`` sizes each shard's hidden output from exactly these
    offsets. It is *sizing information only*.

    It is deliberately **not** handed to `moe_trace` as a token index, and it used to be. The old
    scheme seeded a running token counter with it, so ``hidden_index`` held ``ref_offset + i``.
    That is stable across sessions — the property T3.6's resume acceptance needs — but it is not
    a token count, so consecutive shards overlap or leave gaps and the concatenated
    ``hidden_index`` stops ascending. ``src.traces.reader`` refuses such a set, and a real
    three-shard OLMoE collection produced exactly that: shard 0 ended at index 816 while shard 1
    began at 720.

    `moe_trace` now derives the index itself as ``doc_id * n_ctx + pos_in_doc``, which needs
    nothing from the runner: documents are capped at ``n_ctx`` tokens so the per-document blocks
    cannot collide, and the index is a pure function of the corpus, so it is identical in every
    session and under any re-sharding.

    A shard's documents must be contiguous in corpus order (``assign_shards`` fills greedily, so
    they are); a non-contiguous shard is refused rather than given an offset that silently means
    nothing.
    """
    corpus_path = Path(corpus_path)
    docs = load_corpus(corpus_path)
    if not docs:
        raise RunnerError(f"{corpus_path}: no documents")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    order: list[int] = []
    groups: dict[int, list[Document]] = {}
    offsets: dict[int, int] = {}
    running = 0

    for doc in docs:
        if doc.shard_id is None:
            raise RunnerError(
                f"{corpus_path}: doc {doc.doc_id} has no shard_id — run "
                "src.corpus.build.assign_shards before collecting (T4.2)"
            )
        if doc.doc_id in seen_ids:
            raise RunnerError(f"{corpus_path}: duplicate doc_id {doc.doc_id}")
        seen_ids.add(doc.doc_id)

        shard = int(doc.shard_id)
        if shard not in groups:
            groups[shard] = []
            offsets[shard] = running
            order.append(shard)
        elif order[-1] != shard:
            raise RunnerError(
                f"{corpus_path}: shard {shard} is not contiguous in corpus order (doc "
                f"{doc.doc_id} reopens it after shard {order[-1]}). ref_token_offset is a "
                "cumulative offset and is meaningless for an interleaved shard."
            )
        groups[shard].append(doc)
        running += int(doc.n_tokens_ref)

    plans: list[ShardPlan] = []
    stride = hidden_stride_for(running, subsample_n)

    for shard in sorted(groups):
        members = groups[shard]
        shard_file = out_root / f"shard_{shard:05d}.jsonl"
        _write_shard_jsonl(shard_file, members)
        plans.append(
            ShardPlan(
                shard_id=shard,
                doc_ids=tuple(d.doc_id for d in members),
                n_docs=len(members),
                ref_token_offset=offsets[shard],
                corpus_path=shard_file,
                n_tokens_ref=sum(int(d.n_tokens_ref) for d in members),
                hidden_stride=stride,
            )
        )
    return plans


def _write_shard_jsonl(path: Path, docs: Sequence[Document]) -> Path:
    """Write one shard's documents in the byte format `moe_trace` can parse.

    Key order comes from ``CORPUS_FIELDS`` rather than being spelled out here, because it is
    load-bearing and must not be able to drift from what ``write_corpus`` produces: the C++
    ``parse_jsonl_line`` locates fields by searching the raw line for ``"doc_id"`` and ``"text"``,
    so both must precede any field whose *value* could contain those substrings. ``ensure_ascii``
    for the same reason — the parser decodes ``\\uXXXX`` (recombining surrogate pairs) and treats
    anything it does not understand as a hard error, not a skipped line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        for doc in docs:
            row = {k: getattr(doc, k) for k in CORPUS_FIELDS}
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")
    return path


# --------------------------------------------------------------------------------------
# invoking the capture binary
# --------------------------------------------------------------------------------------


def build_argv(
    plan: ShardPlan,
    *,
    config: RunConfig,
    binary: Path | str,
    spec_path: Path | str,
    model_path: Path | str,
    out_dir: Path | str,
    stats_path: Path | str | None = None,
) -> list[str]:
    """The exact `moe_trace` command line for one shard.

    Every numerics-visible flag is read out of the hashed config block, never defaulted here.
    That is the point: `run_config_sha256` is written into the manifest as the claim that the
    trace was collected under these values, and a flag hardcoded in this function would be a
    value the hash does not cover — the class of silent cross-session confound that S.3 exists
    to eliminate.
    """
    inf = config.inference
    stats_path = Path(stats_path) if stats_path else Path(out_dir) / STATS_NAME

    argv = [
        str(binary),
        "--model", str(model_path),
        "--spec", str(spec_path),
        "--corpus", str(plan.corpus_path),
        "--out", str(out_dir),
        "--stats", str(stats_path),
        "--ctx", str(int(inf["ctx_size"])),
        "--batch", str(int(inf["batch_size"])),
        "--ubatch", str(int(inf["ubatch_size"])),
        "--ngl", str(int(inf["n_gpu_layers"])),
        "--threads", str(int(inf["n_threads"])),
        "--hidden-stride", str(int(plan.hidden_stride)),
        "--shard-id", str(int(plan.shard_id)),
    ]

    split = inf.get("tensor_split")
    if split:
        argv += ["--tensor-split", ",".join(str(float(x)) for x in split)]
    if str(inf.get("split_mode", "layer")).lower() == "row":
        argv.append("--split-mode-row")
    # Flags the binary treats as opt-*out*: their absence means "on", so only the false case is
    # expressible on the command line. `assert_collection_ready` has already refused a config
    # that leaves flash_attn unpinned, so False here is a decision and not a default.
    if not inf.get("flash_attn", True):
        argv.append("--no-flash-attn")
    return argv


def capture_env(config: RunConfig, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the capture subprocess.

    ``GGML_CUDA_DISABLE_FUSION`` is in the *hashed* block, so it is part of the experiment's
    identity: the CUDA backend otherwise fuses softmax+argsort+top-k into one kernel and the
    capture depends on `ffn_moe_topk` being a materialised tensor. Setting it in the child's
    environment rather than relying on the notebook having exported it means the value the
    manifest claims is the value the process actually ran with.
    """
    env = dict(os.environ if base is None else base)
    if config.inference.get("disable_cuda_fusion", False):
        env["GGML_CUDA_DISABLE_FUSION"] = "1"
    return env


@runtime_checkable
class CaptureInvoker(Protocol):
    """Runs the capture for one shard. The seam that keeps this module testable offline.

    Returns ``(exit_code, stats)`` where ``stats`` is the parsed `capture_stats.json`. Raising
    :class:`CaptureError` and returning a nonzero exit code mean different things: the former is
    "the binary never ran or left nothing to read", the latter is "it ran and reported a
    failure", and only the second has usable stats.
    """

    def capture(
        self,
        plan: ShardPlan,
        *,
        config: RunConfig,
        spec_path: Path | str,
        model_path: Path | str,
        out_dir: Path | str,
    ) -> tuple[int, dict[str, Any]]:
        ...


class SubprocessInvoker:
    """The real invoker: runs `moe_trace` as a child process.

    ``shell=False`` always (it is not even expressible here) — a corpus path or a model name with
    a space in it is not an opportunity for the shell to re-tokenize the command line. stderr is
    captured rather than inherited because the binary's lockstep and layout diagnostics go there,
    and on a batch commit an exception message in the notebook output is what a human will
    actually read.
    """

    def __init__(self, binary: Path | str, *, timeout_s: float | None = None) -> None:
        self.binary = Path(binary)
        self.timeout_s = timeout_s
        self.calls: list[list[str]] = []

    def capture(
        self,
        plan: ShardPlan,
        *,
        config: RunConfig,
        spec_path: Path | str,
        model_path: Path | str,
        out_dir: Path | str,
    ) -> tuple[int, dict[str, Any]]:
        if not self.binary.is_file():
            raise CaptureError(
                f"{self.binary}: capture binary not found. Build it first (T0.2): "
                "cmake -B build/cpu -S src/capture ... && ninja -C build/cpu"
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / STATS_NAME

        # A binary that dies before writing stats leaves the previous attempt's file behind, and
        # reading that would report the previous shard's numbers as this one's.
        stats_path.unlink(missing_ok=True)

        argv = build_argv(
            plan,
            config=config,
            binary=self.binary,
            spec_path=spec_path,
            model_path=model_path,
            out_dir=out_dir,
            stats_path=stats_path,
        )
        self.calls.append(list(argv))

        try:
            proc = subprocess.run(  # noqa: S603 - argv list, never a shell string
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=capture_env(config),
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CaptureError(
                f"shard {plan.shard_id}: {self.binary.name} exceeded {self.timeout_s}s and was "
                f"killed; the shard is not complete\n{_tail(exc.stderr)}"
            ) from exc
        except OSError as exc:
            raise CaptureError(f"shard {plan.shard_id}: cannot execute {self.binary} — {exc}") from exc

        if not stats_path.is_file():
            raise CaptureError(
                f"shard {plan.shard_id}: {self.binary.name} exited {proc.returncode} without "
                f"writing {stats_path.name}, so nothing about the run can be validated\n"
                f"{_tail(proc.stderr)}"
            )
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CaptureError(
                f"shard {plan.shard_id}: {stats_path} is unreadable — {exc}\n{_tail(proc.stderr)}"
            ) from exc

        stats["stderr_tail"] = _tail(proc.stderr, prefix="")
        return int(proc.returncode), stats

    def __repr__(self) -> str:
        return f"SubprocessInvoker(binary={str(self.binary)!r})"


def _tail(raw: bytes | str | None, *, prefix: str = "stderr tail:\n") -> str:
    if not raw:
        return ""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if len(text) > STDERR_TAIL_BYTES:
        text = "...\n" + text[-STDERR_TAIL_BYTES:]
    return prefix + text.strip()


# --------------------------------------------------------------------------------------
# stats validation — the silent-corruption gate
# --------------------------------------------------------------------------------------


def validate_stats(
    stats: Mapping[str, Any],
    plan: ShardPlan,
    *,
    config: RunConfig,
    spec: NodeSpec,
) -> None:
    """Raise :class:`RunnerError` unless the stats describe *this* plan, run cleanly.

    Exit code 0 is necessary and nowhere near sufficient. Each check below corresponds to a way
    the capture can succeed loudly and be wrong quietly:

    ``shard_id`` / ``index_scheme`` / ``index_doc_span``
        The stats file belongs to this invocation and is not a leftover from another shard, and
        the binary indexed hidden rows the way this runner expects. A binary predating the
        ``doc_id * n_ctx + pos_in_doc`` scheme writes indices that overlap between shards, which
        only shows up once the whole set is concatenated.
    ``n_docs`` / ``n_docs_in_shard``
        Every document in the plan was processed and none was invented. A short shard concatenates
        cleanly with its neighbours and silently drops corpus.
    ``topk_layout``
        "mixed" means the strided-view detection (I12) disagreed with itself between ubatches, so
        one of the two readings was wrong — and a wrong reading yields in-range, top_k-distinct,
        plausible expert ids. Nothing downstream can detect it. "none" means no topk node was ever
        matched, i.e. the spec names a tensor this build does not emit.
    ``n_docs_truncated`` / ``n_tokens_dropped``
        Invariant I15: the corpus cap was enforced under one reference tokenizer, so a document
        at the cap for a 50k-vocab model can exceed it for a 262k-vocab one. A truncated document
        means this model was not shown the same text as the rest of the panel, which every
        cross-model comparison assumes. T5.3 requires the count to be 0.
    ``nodes_captured``
        See below — it is the only end-to-end evidence that all three streams were captured for
        every ubatch of every layer.
    ``n_captured``
        The hidden subsample is a closed-form function of the global token range, so the row count
        of `hidden.bin` is predictable exactly and disagreement means the stride the binary used
        was not the stride this run asked for.
    """
    problems: list[str] = []

    got_shard = int(stats.get("shard_id", -1))
    if got_shard != plan.shard_id:
        problems.append(
            f"stats are for shard {got_shard}, this is shard {plan.shard_id} — a stale "
            f"{STATS_NAME} from an earlier run"
        )
    n_ctx = int(config.inference["ctx_size"])
    scheme = str(stats.get("index_scheme", ""))
    if scheme != INDEX_SCHEME:
        problems.append(
            f"index_scheme is {scheme!r}, expected {INDEX_SCHEME!r}; this binary indexes "
            "hidden.bin differently and its shards will not concatenate with the rest"
        )
    span = int(stats.get("index_doc_span", -1))
    if span != n_ctx:
        problems.append(
            f"index_doc_span is {span}, but the run pins n_ctx={n_ctx}; the per-document index "
            "blocks would not be the ones every other shard reserved"
        )

    n_docs = int(stats.get("n_docs", -1))
    n_in_shard = int(stats.get("n_docs_in_shard", -1))
    if n_docs != plan.n_docs:
        problems.append(f"processed {n_docs} documents, the shard has {plan.n_docs}")
    if n_in_shard != plan.n_docs:
        problems.append(
            f"the binary read {n_in_shard} documents from {plan.corpus_path.name}, the plan "
            f"wrote {plan.n_docs}"
        )

    layout = str(stats.get("topk_layout", ""))
    if layout not in _OK_TOPK_LAYOUTS:
        problems.append(
            f"topk_layout={layout!r}: refusing the shard. 'mixed' means ffn_moe_topk arrived both "
            "packed and strided so the de-striding read one of them wrong (I12) — the labels are "
            "in-range, distinct and false; 'none' means the spec's topk node was never emitted"
        )

    n_trunc = int(stats.get("n_docs_truncated", 0))
    if n_trunc:
        problems.append(
            f"{n_trunc} document(s) exceeded ctx_size and were truncated "
            f"({int(stats.get('n_tokens_dropped', 0))} tokens dropped, first doc_id "
            f"{stats.get('first_truncated_doc')}) — invariant I15: this model was not shown the "
            "same text as the rest of the panel. Fix the corpus cap, do not keep the shard"
        )

    for key, want in (
        ("n_moe_layers", spec.n_moe_layers),
        ("n_experts", spec.n_experts),
        ("top_k", spec.top_k),
        ("hidden_dim", spec.hidden_dim),
    ):
        got = int(stats.get(key, -1))
        if got != want:
            problems.append(f"{key}={got} but the node spec says {want}")

    n_tokens = int(stats.get("n_tokens", 0))
    if n_tokens < plan.n_docs:
        problems.append(
            f"n_tokens={n_tokens} for {plan.n_docs} documents; every document tokenizes to at "
            "least one token (the binary exits 4 on a zero-token document)"
        )

    problems += _check_nodes_captured(stats, plan, config=config, spec=spec, n_tokens=n_tokens)
    problems += _check_n_captured(stats, plan, n_tokens=n_tokens)

    if problems:
        raise RunnerError(
            f"shard {plan.shard_id} failed capture validation (exit code was "
            f"{stats.get('exit_code')}); NOT marking it complete:\n  - " + "\n  - ".join(problems)
        )


def _check_nodes_captured(
    stats: Mapping[str, Any],
    plan: ShardPlan,
    *,
    config: RunConfig,
    spec: NodeSpec,
    n_tokens: int,
) -> list[str]:
    """Bracket ``nodes_captured``, derived from what the callback actually counts.

    ``st.n_nodes_captured`` is incremented once per *successful node capture*: once for each of
    the three streams, for each of the ``n_moe_layers`` trace layers, for each ubatch of each
    document (moe_trace.cpp, end of ``moe_cb``). So::

        nodes_captured == 3 * n_moe_layers * sum_over_docs(ceil(doc_tokens / ubatch_size))

    The multiple of ``3 * n_moe_layers`` is exact and is the load-bearing half: a stream that
    stopped being emitted, a layer whose node name went stale, or a fused kernel that swallowed
    one of the three tensors all break the divisibility even when per-document lockstep (T3.1,
    which only compares the three cursors to each other) is satisfied for the layers that *were*
    emitted.

    The ubatch total cannot be pinned exactly from here without the per-document token counts
    under this model's tokenizer, which only the model has. It is bracketed instead, tightly:
    every document is at least one ubatch, the shard is at least ``ceil(n_tokens/ubatch)``
    ubatches in total, and no document contributes more than ``ceil(t/ubatch)``. A count outside
    the bracket means the callback fired a different number of times than the batching can
    explain.
    """
    nodes = int(stats.get("nodes_captured", -1))
    per_ubatch = 3 * spec.n_moe_layers
    problems: list[str] = []

    if nodes <= 0:
        return [f"nodes_captured={nodes}: the eval callback never captured a node"]
    if nodes % per_ubatch:
        return [
            f"nodes_captured={nodes} is not a multiple of 3 streams x {spec.n_moe_layers} MoE "
            f"layers ({per_ubatch}): some (stream, layer) pair was captured for a different "
            "number of ubatches than the others, so at least one stream is short"
        ]

    ubatch = int(config.inference["ubatch_size"])
    n_ubatches = nodes // per_ubatch
    lo = max(plan.n_docs, _ceil_div(n_tokens, ubatch))
    hi = (n_tokens + plan.n_docs * (ubatch - 1)) // ubatch
    if not lo <= n_ubatches <= hi:
        problems.append(
            f"nodes_captured={nodes} implies {n_ubatches} ubatches, but {plan.n_docs} documents "
            f"of {n_tokens} tokens at ubatch_size={ubatch} must be between {lo} and {hi}"
        )
    return problems


def _check_n_captured(
    stats: Mapping[str, Any], plan: ShardPlan, *, n_tokens: int
) -> list[str]:
    """Bound ``n_captured`` from the shard's token and document totals.

    `moe_trace` sets ``capture[i] = (doc_id * n_ctx + i) % hidden_stride == 0``, so each document
    contributes either ``floor(n_d / stride)`` or ``ceil(n_d / stride)`` rows depending on where
    its reserved block starts. The runner knows the shard's *total* token count but not the
    per-document split — only the tokenizer produces that — so the exact count is not available
    here and a two-sided bound is the honest check. It still catches the failure that matters:
    a stride that is not the stride this run asked for is off by a factor, not by one row per
    document.

    The exactness is not lost, it moved. ``src.traces.validate.check_hidden_stride`` has
    ``tokens.bin``, so it can verify every index against ``doc_id * n_ctx + pos_in_doc``
    individually — a stronger check than the closed form this function used to run, which only
    ever compared a total.
    """
    got = int(stats.get("n_captured", -1))
    stride = int(stats.get("hidden_stride", plan.hidden_stride))
    if stride != plan.hidden_stride:
        return [f"hidden_stride={stride} but the plan asked for {plan.hidden_stride}"]
    if stride <= 0:
        return [] if got == 0 else [f"hidden_stride is 0 but n_captured={got}"]

    slack = plan.n_docs * (stride - 1)
    lo = max(0, _ceil_div(n_tokens - slack, stride))
    hi = (n_tokens + slack) // stride
    if not lo <= got <= hi:
        return [
            f"n_captured={got}, but {plan.n_docs} document(s) totalling {n_tokens} tokens at "
            f"hidden_stride={stride} must subsample between {lo} and {hi} tokens"
        ]
    return []


# --------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------


def build_shard_manifest(
    shard_dir: Path,
    plan: ShardPlan,
    stats: Mapping[str, Any],
    *,
    config: RunConfig,
    spec: NodeSpec,
    model: str,
    corpus: str,
    model_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `manifest.json` beside the streams and return it.

    The C++ writer deliberately does not do this (see ``trace_writer.hpp``: "for the stats report
    and the manifest Python writes") — the manifest carries provenance the binary has no access
    to, notably ``run_config_sha256`` and the GGUF hash. :func:`write_manifest` refuses a manifest
    with a null required key, which is what stops a shard being collected before T1.1/T1.2 have
    filled in ``gguf.sha256`` and ``router_dtype``: an untraceable trace is worse than no trace.

    ``logit_tensor_used`` is the node-name *template* with its layer placeholder stripped, because
    it must name the tensor top-k actually consumed (I13, §1.6) and that is per-layer the same
    node; the reader compares it across shards for equality.
    """
    meta = dict(model_meta or {})
    gguf = dict(meta.get("gguf") or {})

    manifest: dict[str, Any] = {
        "model": model,
        "corpus": corpus,
        "checkpoint_status": meta.get("checkpoint_status"),
        "gguf_sha256": gguf.get("sha256"),
        "quant": meta.get("quant") or gguf.get("quant"),
        "router_dtype": meta.get("router_dtype"),
        "logit_tensor_used": spec.node_logits.replace("-%d", "").replace("%d", ""),
        "shard_id": plan.shard_id,
        "shard_doc_range": list(plan.doc_range),
        "n_docs": plan.n_docs,
        "n_tokens": int(stats["n_tokens"]),
        "n_captured": int(stats.get("n_captured", 0)),
        "n_moe_layers": spec.n_moe_layers,
        "n_experts": spec.n_experts,
        "top_k": spec.top_k,
        "hidden_dim": spec.hidden_dim,
        "layer_index_map": list(spec.layer_map),
        "index_scheme": INDEX_SCHEME,
        "index_doc_span": int(config.inference["ctx_size"]),
        "ref_token_offset": plan.ref_token_offset,
        "hidden_stride": plan.hidden_stride,
        "node_spec_verified": bool(spec.verified),
        "collected_utc": _utc_now(),
        # Kaggle sets this in a batch commit; absent locally, and the field is not merge-invariant
        # (it is excluded from file_sha256 for exactly that reason — see ShardUploadResult).
        "kaggle_session_id": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local"),
        "capture_stats": {
            k: v for k, v in stats.items() if k not in ("stderr_tail",)
        },
        "file_sha256": {
            name: sha256_file(shard_dir / name)
            for name in STREAM_FILES.values()
            if (shard_dir / name).is_file()
        },
    }
    manifest.update(config.manifest_fields())
    write_manifest(shard_dir, manifest)
    return manifest


# --------------------------------------------------------------------------------------
# one shard
# --------------------------------------------------------------------------------------


@dataclass
class ShardResult:
    """Outcome of one shard. ``status`` is the only thing callers should branch on."""

    shard_id: int
    status: str  # "complete" | "skipped" | "failed"
    exit_code: int | None = None
    n_docs: int = 0
    n_tokens: int = 0
    n_captured: int = 0
    wall_s: float = 0.0
    upload_verified: bool = False
    bytes_uploaded: int = 0
    error: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("complete", "skipped")

    @property
    def tokens_per_s(self) -> float:
        return self.n_tokens / self.wall_s if self.wall_s > 0 else 0.0


def run_shard(
    plan: ShardPlan,
    *,
    config: RunConfig,
    invoker: CaptureInvoker,
    backend: StorageBackend,
    ledger: ShardState,
    spec_path: Path | str,
    model_path: Path | str,
    out_dir: Path | str,
    remote_prefix: str,
    spec: NodeSpec | None = None,
    model: str | None = None,
    corpus: str | None = None,
    model_meta: Mapping[str, Any] | None = None,
    preflight_probe_bytes: int | None = None,
    preflight_min_mbps: float = 50.0,
) -> ShardResult:
    """Capture, validate, upload and record one shard. Never raises for a shard-level failure.

    Failure returns ``status="failed"`` and leaves the world exactly as it was: nothing in the
    ledger, every local file still on disk. That asymmetry is the contract — the next session
    finds the shard absent and recollects it, which costs one shard, whereas a shard recorded on
    the strength of an exit code costs the analysis its trustworthiness.

    A failure is *returned* rather than raised so that the caller can log the attempt to
    ``collection_log.csv`` before stopping; :func:`run_collection` stops the run on the first one.
    """
    spec = spec or parse_spec(Path(spec_path).read_text(encoding="utf-8"))
    model = model or ledger.model
    corpus = corpus or ledger.corpus
    out_dir = assert_not_kaggle_working(out_dir, what=f"shard {plan.shard_id}")

    if ledger.is_complete(plan.shard_id):
        # Resumption. The ledger only ever holds shards whose upload round-trip verified, so
        # "present" already means "durable and checksummed"; re-running would burn quota to
        # produce bytes that must be identical anyway (T3.6).
        record = ledger.record(plan.shard_id)
        return ShardResult(
            shard_id=plan.shard_id,
            status="skipped",
            exit_code=0,
            n_docs=plan.n_docs,
            n_tokens=record.n_tokens if record else 0,
            n_captured=record.n_captured if record else 0,
            upload_verified=True,
        )

    started = time.monotonic()
    exit_code: int | None = None
    stats: dict[str, Any] = {}
    try:
        if preflight_probe_bytes:
            # T0.6. os.statvfs on Kaggle reports the read-only Docker layer, so the only evidence
            # that scratch can take a shard is writing to it.
            run_preflight(
                out_dir.parent,
                probe_bytes=int(preflight_probe_bytes),
                min_write_mbps=preflight_min_mbps,
            )

        raw_exit, raw_stats = invoker.capture(
            plan,
            config=config,
            spec_path=spec_path,
            model_path=model_path,
            out_dir=out_dir,
        )
        exit_code, stats = int(raw_exit), dict(raw_stats or {})
        if exit_code != 0:
            raise RunnerError(
                f"shard {plan.shard_id}: {_exit_code_meaning(exit_code)} (exit {exit_code}); "
                f"leaving the local files in place and the ledger untouched\n"
                f"{stats.get('stderr_tail', '')}".rstrip()
            )
        reported = int(stats.get("exit_code", 0))
        if reported != exit_code:
            raise RunnerError(
                f"shard {plan.shard_id}: process exited {exit_code} but {STATS_NAME} records "
                f"exit_code={reported} — the stats file is from a different run"
            )

        validate_stats(stats, plan, config=config, spec=spec)

        build_shard_manifest(
            out_dir,
            plan,
            stats,
            config=config,
            spec=spec,
            model=model,
            corpus=corpus,
            model_meta=model_meta,
        )

        # verify=True and delete_local_on_success=True together are the S.3 step-d ordering:
        # bytes read back off the wire must hash to the local value before the ledger records the
        # shard, and only then may scratch be freed for the next one (I9/T4.4).
        upload = upload_shard(
            out_dir,
            backend,
            remote_prefix=remote_prefix,
            verify=True,
            delete_local_on_success=True,
            state=ledger,
        )
    # FormatError is in here because `write_manifest` raises it for a null required key (an
    # unfilled gguf_sha256 from T1.1, say). That is a shard-level failure with the same remedy as
    # any other — nothing recorded, files kept — not a reason to lose the loop's log row.
    except (RunnerError, UploadError, FormatError, OSError, ValueError, KeyError) as exc:
        return ShardResult(
            shard_id=plan.shard_id,
            status="failed",
            exit_code=exit_code,
            n_docs=plan.n_docs,
            n_tokens=int(stats.get("n_tokens", 0) or 0),
            wall_s=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
            stats=stats,
        )

    return ShardResult(
        shard_id=plan.shard_id,
        status="complete",
        exit_code=exit_code,
        n_docs=plan.n_docs,
        n_tokens=int(stats["n_tokens"]),
        n_captured=int(stats.get("n_captured", 0)),
        wall_s=time.monotonic() - started,
        upload_verified=upload.verified,
        bytes_uploaded=upload.bytes_uploaded,
        stats=dict(stats),
    )


def _exit_code_meaning(code: int) -> str:
    """Name what `moe_trace`'s exit codes mean, so a Kaggle log says what to go fix."""
    return {
        2: "configuration or usage error (bad flag, unreadable spec, unparseable corpus)",
        3: "model load failure (GGUF missing, or VRAM/context allocation failed)",
        4: "a document tokenized to zero tokens — the corpus has an empty or unusable document",
        5: "llama_decode failed",
        6: "the eval callback failed (type, shape or spec mismatch on a captured node)",
        7: "LOCKSTEP FAILURE (T3.1): a stream did not receive one row per token per layer",
        8: "the trace writer failed to close a document",
        9: "writer verify failure: an output file's size does not match its arithmetic",
        10: "mixed topk layout: ffn_moe_topk arrived both packed and strided (I12)",
    }.get(code, "capture failed")


# --------------------------------------------------------------------------------------
# the collection loop
# --------------------------------------------------------------------------------------


@dataclass
class CollectionResult:
    """What one session did. ``stopped_early`` distinguishes a budget stop from being finished."""

    model: str
    corpus: str
    run_config_sha256: str
    results: list[ShardResult] = field(default_factory=list)
    planned: list[int] = field(default_factory=list)
    stopped_early: bool = False
    log_path: Path | None = None
    dry_run_argv: list[list[str]] = field(default_factory=list)

    @property
    def completed(self) -> list[int]:
        return [r.shard_id for r in self.results if r.status == "complete"]

    @property
    def skipped(self) -> list[int]:
        return [r.shard_id for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[int]:
        return [r.shard_id for r in self.results if r.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def n_tokens(self) -> int:
        return sum(r.n_tokens for r in self.results if r.status == "complete")

    def summary(self) -> str:
        state = "stopped on session budget" if self.stopped_early else "finished the shard list"
        return (
            f"{self.model}/{self.corpus}: {len(self.completed)} collected, "
            f"{len(self.skipped)} already complete, {len(self.failed)} failed, "
            f"{self.n_tokens} tokens — {state}"
        )


def run_collection(
    plans: Sequence[ShardPlan],
    *,
    config: RunConfig,
    invoker: CaptureInvoker,
    backend: StorageBackend,
    ledger: ShardState,
    spec_path: Path | str,
    model_path: Path | str,
    scratch_root: Path | str,
    remote_root: str,
    model_meta: Mapping[str, Any] | None = None,
    budget: SessionBudget | None = None,
    log_path: Path | str | None = None,
    preflight_probe_bytes: int | None = None,
    preflight_min_mbps: float = 50.0,
    dry_run: bool = False,
    binary: Path | str | None = None,
    verbose: bool = True,
) -> CollectionResult:
    """The shard loop — plan S.3 step 2, logging to ``results/collection_log.csv`` (T5.2).

    Resumable by construction: the pending set is recomputed from the ledger every session, so a
    kill mid-shard means that shard is absent and is redone, and completed shards are skipped
    without re-invoking the binary.

    :meth:`SessionBudget.should_stop` is consulted only *between* shards. Stopping inside one
    would leave a partial trace in scratch, and a partial trace that gets marked complete is the
    plan's highest-rated silent failure mode (risk table: "session killed mid-collection, partial
    shard merged").

    The first failure stops the run rather than continuing to the next shard. Every failure this
    loop can see is a property of the *run* (a stale spec, a layout the reader misidentifies, a
    truncating corpus cap), not of one shard's luck, so continuing would spend GPU quota
    manufacturing more shards with the same defect.
    """
    # Before anything: an under-specified config makes every shard in the session unusable, and
    # the cheapest moment to discover that is now (an unpinned build commit cannot be recovered
    # after the fact — the shards are simply of unknown provenance).
    config.assert_collection_ready()
    if config.sha256 != ledger.run_config_sha256:
        raise RunnerError(
            f"run config {config.short} does not match the ledger's "
            f"{ledger.run_config_sha256[:12]} (invariant I2)"
        )

    scratch_root = assert_not_kaggle_working(scratch_root, what="shard scratch")
    budget = budget or SessionBudget.from_config(config)
    spec = parse_spec(Path(spec_path).read_text(encoding="utf-8"))

    result = CollectionResult(
        model=ledger.model,
        corpus=ledger.corpus,
        run_config_sha256=config.sha256,
        planned=[p.shard_id for p in plans],
        log_path=Path(log_path) if log_path else None,
    )

    if dry_run:
        # Prints the argv without touching the binary, the backend or the ledger, so a notebook
        # can show what a 12-hour commit is about to do before it spends the quota.
        show = binary or getattr(invoker, "binary", "moe_trace")
        for plan in plans:
            argv = build_argv(
                plan,
                config=config,
                binary=show,
                spec_path=spec_path,
                model_path=model_path,
                out_dir=Path(scratch_root) / f"shard_{plan.shard_id:05d}",
            )
            result.dry_run_argv.append(argv)
            if verbose:
                done = " [already complete]" if ledger.is_complete(plan.shard_id) else ""
                print(f"# shard {plan.shard_id}: {plan.n_docs} docs, ~{plan.n_tokens_ref} ref tokens{done}")
                print(subprocess.list2cmdline(argv))
        return result

    if preflight_probe_bytes:
        # Once per session, not per shard: scratch throughput cannot change between shards, and a
        # 2 GB probe per shard would spend GPU-quota minutes re-measuring it. A shard that no
        # longer fits is caught by the writer's size arithmetic, which runs per shard anyway.
        run_preflight(
            scratch_root,
            probe_bytes=int(preflight_probe_bytes),
            min_write_mbps=preflight_min_mbps,
        )

    for index, plan in enumerate(plans):
        if index and budget.should_stop():
            result.stopped_early = True
            if verbose:
                print(f"# session budget reached ({budget}); stopping between shards")
            break

        shard_result = run_shard(
            plan,
            config=config,
            invoker=invoker,
            backend=backend,
            ledger=ledger,
            spec_path=spec_path,
            model_path=model_path,
            out_dir=Path(scratch_root) / f"shard_{plan.shard_id:05d}",
            remote_prefix=f"{str(remote_root).strip('/')}/shard_{plan.shard_id:05d}",
            spec=spec,
            model_meta=model_meta,
        )
        result.results.append(shard_result)

        if shard_result.status != "skipped" and result.log_path is not None:
            append_log_row(result.log_path, shard_result, model=ledger.model, config=config)

        if verbose:
            print(f"# shard {plan.shard_id}: {shard_result.status} {shard_result.error}".rstrip())
        if shard_result.status == "failed":
            break

    return result


def append_log_row(
    log_path: Path | str, shard: ShardResult, *, model: str, config: RunConfig
) -> Path:
    """Append one row to ``results/collection_log.csv`` — T5.2.

    The header is written only when the file does not yet exist or is empty. A collection spans a
    dozen sessions and appends to the same file, and a header repeated mid-file turns the log into
    something pandas reads as a string column without complaining.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not log_path.exists() or log_path.stat().st_size == 0

    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if fresh:
            writer.writerow(LOG_COLUMNS)
        writer.writerow(
            [
                model,
                shard.shard_id,
                shard.n_docs,
                shard.n_tokens,
                round(shard.wall_s, 3),
                round(shard.tokens_per_s, 2),
                "" if shard.exit_code is None else shard.exit_code,
                int(bool(shard.upload_verified)),
                config.sha256,
            ]
        )
    return log_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def load_model_meta(models_path: Path | str, model_key: str) -> dict[str, Any]:
    """One model's block from ``configs/models.yaml``, with the panel defaults merged under it."""
    import yaml  # noqa: PLC0415 - only the CLI needs it; the loop takes the dict

    raw = yaml.safe_load(Path(models_path).read_text(encoding="utf-8")) or {}
    models = raw.get("models") or {}
    if model_key not in models:
        raise RunnerError(f"{models_path}: no model {model_key!r}; have {sorted(models)}")
    meta = dict(raw.get("defaults") or {})
    meta.update(models[model_key] or {})
    return meta


def _parse_shard_filter(text: str | None) -> set[int] | None:
    """``"0-19"`` or ``"3,7,11"`` -> the set of shard ids to consider this session."""
    if not text:
        return None
    out: set[int] = set()
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(piece))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the resumable capture shard loop (plan S.3, T5.2)"
    )
    parser.add_argument("--model", required=True, help="model key in configs/models.yaml")
    parser.add_argument("--model-path", required=True, type=Path, help="GGUF path")
    parser.add_argument("--corpus", required=True, type=Path, help="corpus JSONL (T4.2 output)")
    parser.add_argument("--corpus-name", default=None, help="defaults to the corpus file stem")
    parser.add_argument("--spec", required=True, type=Path, help="node spec (src.capture.nodespec)")
    parser.add_argument("--binary", type=Path, default=Path("build/cpu/moe_trace.exe"))
    parser.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="per-shard trace scratch; defaults to unhashed.paths.scratch (never /kaggle/working, I9)",
    )
    parser.add_argument("--shard-jsonl-dir", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None, help="shard ledger path")
    parser.add_argument("--backend", choices=("local", "hf"), default="local")
    parser.add_argument("--local-root", type=Path, default=None, help="--backend local target")
    parser.add_argument("--repo-id", default=None, help="--backend hf dataset repo")
    parser.add_argument("--remote-root", default=None, help="defaults to traces/<model>/<corpus>")
    parser.add_argument("--log", type=Path, default=Path("results/collection_log.csv"))
    parser.add_argument("--shards", default=None, help="restrict to '0-19' or '3,7,11'")
    parser.add_argument("--preflight-gb", type=float, default=0.0, help="0 disables (T0.6)")
    parser.add_argument("--timeout-s", type=float, default=None, help="per-shard subprocess cap")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan the shards and print the argv without invoking anything",
    )
    args = parser.parse_args(argv)

    config = RunConfig.load(args.run_config)
    model_meta = load_model_meta(args.models, args.model)
    corpus_name = args.corpus_name or args.corpus.stem

    scratch = args.scratch or Path((config.unhashed.get("paths") or {}).get("scratch", "."))
    scratch_root = assert_not_kaggle_working(Path(scratch) / args.model / corpus_name)
    jsonl_dir = args.shard_jsonl_dir or (scratch_root / "shards")

    plans = plan_shards(
        args.corpus,
        out_root=jsonl_dir,
        subsample_n=config.capture.get("hidden_subsample_n"),
    )
    wanted = _parse_shard_filter(args.shards)
    if wanted is not None:
        plans = [p for p in plans if p.shard_id in wanted]
        if not plans:
            print(f"no shards match --shards {args.shards}", file=sys.stderr)
            return 2

    state_path = args.state or (scratch_root / "state.json")
    ledger = ShardState.load_or_create(state_path, args.model, corpus_name, config.sha256)

    if args.backend == "hf":
        from .upload import HFBackend  # noqa: PLC0415 - lazy: huggingface_hub is optional

        if not args.repo_id:
            print("--backend hf requires --repo-id", file=sys.stderr)
            return 2
        backend: StorageBackend = HFBackend(args.repo_id, create=True)
    else:
        if not args.local_root:
            print("--backend local requires --local-root", file=sys.stderr)
            return 2
        from .upload import LocalDirBackend  # noqa: PLC0415 - symmetry with the branch above

        backend = LocalDirBackend(args.local_root)

    outcome = run_collection(
        plans,
        config=config,
        invoker=SubprocessInvoker(args.binary, timeout_s=args.timeout_s),
        backend=backend,
        ledger=ledger,
        spec_path=args.spec,
        model_path=args.model_path,
        scratch_root=scratch_root,
        remote_root=args.remote_root or f"traces/{args.model}/{corpus_name}",
        model_meta=model_meta,
        log_path=args.log,
        preflight_probe_bytes=int(args.preflight_gb * 1024**3) or None,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"# {len(outcome.dry_run_argv)} shard(s) planned, nothing invoked")
        return 0

    print(outcome.summary())
    for failure in (r for r in outcome.results if r.status == "failed"):
        print(f"shard {failure.shard_id}: {failure.error}", file=sys.stderr)
    return 0 if outcome.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
