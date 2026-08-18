"""T0.5 — throughput calibration, capture overhead, and Gate Q1 (resolves O4).

Three separable jobs, deliberately kept apart because they fail differently:

1. **Prefill throughput per model**, from ``llama-bench -n 0 -o json``. This is the number the
   plan calls O4 and refuses to let anyone guess.
2. **Capture overhead**, which ``llama-bench`` structurally cannot measure because it does not
   install ``cb_eval``. Measured by running ``moe_trace`` over the same corpus in each of its
   three ``--capture-mode`` legs.
3. **The Phase 5 projection and Gate Q1**, which turn (1) and (2) into a decision about the token
   budget *before* anything is collected.

**Why three capture legs and not the plan's two.** The plan says to compare capture against "the
callback filter returning false for everything". That isolates the cost of reading tensors back,
but ``cb_eval`` is still installed, and installing it changes how ``ggml_backend_sched`` executes
a graph no matter what the callback answers: instead of dispatching a whole backend split it
computes node ranges bounded by each requested tensor. Separating ``no-callback`` from
``filter-off`` splits the overhead into *the cost of being observable at all* and *the cost of
actually observing*, and those have different remedies. If the first term dominates, no amount of
capturing fewer tensors helps.

**Repetition is not optional here.** A first local run on the workstation measured 220 / 204 / 256
tok/s for no-callback / filter-off / full — i.e. full capture came out *fastest*, which cannot be
true. That is run-to-run noise on a shared machine, and it is the same order as the effect being
measured. So :func:`measure_capture_overhead` takes repeats, reports the **median**, and refuses
to state a ratio whose confidence interval spans 1.0 -- an overhead number that is really noise
would propagate straight into the Phase 5 projection and from there into Gate Q1, which is a
decision about whether the study collects 1M or 500k tokens per model.

Everything here is pure stdlib plus numpy, and the parsing half runs anywhere: the actual
measurement is `KGPU`, but the arithmetic that consumes it is testable on a workstation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ThroughputError",
    "BenchRow",
    "ModeTiming",
    "CaptureOverhead",
    "Projection",
    "GateQ1",
    "parse_llama_bench_json",
    "measure_capture_overhead",
    "overhead_from_stats",
    "project_phase5",
    "evaluate_gate_q1",
    "write_calibration_csv",
    "append_quota_log",
    "main",
]

#: Plan S.4. A rolling weekly allowance, not a per-session cap.
WEEKLY_GPU_HOURS = 30.0

#: Plan Gate Q1: above this fraction of the weekly allowance, the v1 token budget is cut BEFORE
#: collecting anything rather than discovered to be unaffordable half way through.
GATE_Q1_FRACTION = 0.60

#: Plan Gate Q1's drop order, in order. Never reorder this to fit a result.
DROP_ORDER = (
    "OLMoE-0924 (Pair A)",
    "OLMoE-Instruct",
    "the 4M scale run, reduced to 2M",
)

CAPTURE_MODES = ("no-callback", "filter-off", "full")


class ThroughputError(RuntimeError):
    """A calibration input is missing, malformed, or measuring the wrong thing."""


# -- llama-bench ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchRow:
    model_filename: str
    n_prompt: int
    n_gen: int
    avg_ts: float
    stddev_ts: float
    n_gpu_layers: int
    n_threads: int
    n_batch: int
    n_ubatch: int
    split_mode: str
    tensor_split: str
    backends: str
    build_commit: str
    gpu_info: str

    @property
    def rel_stddev(self) -> float:
        return self.stddev_ts / self.avg_ts if self.avg_ts else float("inf")


def _num(value: Any, name: str, cast) -> Any:
    """llama-bench declares avg_ts/stddev_ts as FLOAT but the JSON writer is string-oriented and
    the exact quoting has changed upstream before. Accept either rather than depending on it."""
    if isinstance(value, str):
        value = value.strip()
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ThroughputError(f"llama-bench field {name!r} is not numeric: {value!r}") from exc


def parse_llama_bench_json(
    text: str | bytes, *, expect_commit: str | None = None
) -> list[BenchRow]:
    """Parse ``llama-bench -o json`` output into rows, refusing the ones that mislead.

    Two rejections are deliberate and both are silent-corruption guards rather than hygiene:

    * **``n_gen != 0``.** The plan pins ``-n 0`` because this study is prefill-only. A generation
      row reports token-generation throughput, which on an MoE model is a completely different
      number (one token per forward, memory-bound, no expert parallelism). Averaging one into the
      projection would produce a plausible figure that is wrong by an order of magnitude.
    * **A build commit other than the pinned one**, when ``expect_commit`` is given. Throughput
      measured on a different build does not describe the binary that will do the collecting, and
      ``build.llama_cpp_commit`` is inside ``run_config_sha256`` for exactly this reason.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ThroughputError(f"llama-bench output is not JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ThroughputError(f"expected a JSON array of runs, got {type(payload).__name__}")
    if not payload:
        raise ThroughputError("llama-bench returned no rows")

    rows: list[BenchRow] = []
    skipped_gen = 0
    for entry in payload:
        if not isinstance(entry, Mapping):
            raise ThroughputError(f"expected objects in the array, got {type(entry).__name__}")
        n_gen = _num(entry.get("n_gen", 0), "n_gen", int)
        if n_gen != 0:
            skipped_gen += 1
            continue
        commit = str(entry.get("build_commit", ""))
        if expect_commit and commit and not expect_commit.startswith(commit) \
                and not commit.startswith(expect_commit):
            raise ThroughputError(
                f"llama-bench reports build_commit {commit!r} but the pinned build is "
                f"{expect_commit!r}. This measures a different binary than the one that will "
                "collect, and the commit is inside run_config_sha256."
            )
        rows.append(BenchRow(
            model_filename=str(entry.get("model_filename", "")),
            n_prompt=_num(entry.get("n_prompt", 0), "n_prompt", int),
            n_gen=n_gen,
            avg_ts=_num(entry.get("avg_ts", 0), "avg_ts", float),
            stddev_ts=_num(entry.get("stddev_ts", 0), "stddev_ts", float),
            n_gpu_layers=_num(entry.get("n_gpu_layers", 0), "n_gpu_layers", int),
            n_threads=_num(entry.get("n_threads", 0), "n_threads", int),
            n_batch=_num(entry.get("n_batch", 0), "n_batch", int),
            n_ubatch=_num(entry.get("n_ubatch", 0), "n_ubatch", int),
            split_mode=str(entry.get("split_mode", "")),
            tensor_split=str(entry.get("tensor_split", "")),
            backends=str(entry.get("backends", "")),
            build_commit=commit,
            gpu_info=str(entry.get("gpu_info", "")),
        ))

    if not rows:
        raise ThroughputError(
            f"every row had n_gen != 0 ({skipped_gen} skipped). Re-run llama-bench with -n 0; "
            "this study is prefill-only and generation throughput does not describe it."
        )
    return rows


# -- capture overhead ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModeTiming:
    mode: str
    tok_per_s: tuple[float, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.tok_per_s)

    @property
    def spread(self) -> float:
        """Half the min-max range as a fraction of the median — a crude but assumption-free
        interval. With three repeats a standard deviation is not worth the pretence of rigour."""
        if len(self.tok_per_s) < 2 or self.median == 0:
            return 0.0
        return (max(self.tok_per_s) - min(self.tok_per_s)) / 2.0 / self.median


@dataclass
class CaptureOverhead:
    """Ratios of *time*, so a ratio above 1.0 always means "capture costs more"."""

    timings: dict[str, ModeTiming] = field(default_factory=dict)
    n_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    def _ratio(self, numerator: str, denominator: str) -> float:
        a, b = self.timings.get(numerator), self.timings.get(denominator)
        if a is None or b is None or a.median <= 0:
            return float("nan")
        return b.median / a.median

    @property
    def observability_ratio(self) -> float:
        """Cost of installing ``cb_eval`` at all: filter-off vs no-callback."""
        return self._ratio("filter-off", "no-callback")

    @property
    def readback_ratio(self) -> float:
        """Cost of actually requesting and copying tensors: full vs filter-off."""
        return self._ratio("full", "filter-off")

    @property
    def total_ratio(self) -> float:
        """What the Phase 5 projection multiplies by: full vs no-callback."""
        return self._ratio("full", "no-callback")

    @property
    def worst_spread(self) -> float:
        return max((t.spread for t in self.timings.values()), default=0.0)

    @property
    def conclusive(self) -> bool:
        """True when the measured effect is larger than the run-to-run noise.

        Not a formality. The first workstation run produced a *negative* overhead — full capture
        apparently faster than no callback at all — because the spread was larger than the effect.
        A tool that reported 0.86 there would have fed a fabricated speedup into Gate Q1.
        """
        ratio = self.total_ratio
        if ratio != ratio:  # NaN
            return False
        return abs(ratio - 1.0) > 2.0 * self.worst_spread

    def to_json(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "modes": {k: {"tok_per_s": list(v.tok_per_s), "median": v.median,
                          "spread": v.spread} for k, v in self.timings.items()},
            "observability_ratio": self.observability_ratio,
            "readback_ratio": self.readback_ratio,
            "total_ratio": self.total_ratio,
            "worst_spread": self.worst_spread,
            "conclusive": self.conclusive,
            "notes": self.notes,
        }


def overhead_from_stats(stats_by_mode: Mapping[str, Sequence[Mapping[str, Any]]]) -> CaptureOverhead:
    """Build a :class:`CaptureOverhead` from parsed ``capture_stats.json`` payloads.

    Kept separate from the subprocess runner so the arithmetic — including the inconclusive
    branch, which is the one that matters — is testable without a model.
    """
    result = CaptureOverhead()
    token_counts: set[int] = set()

    for mode, runs in stats_by_mode.items():
        if mode not in CAPTURE_MODES:
            raise ThroughputError(f"unknown capture mode {mode!r}; expected {CAPTURE_MODES}")
        if not runs:
            raise ThroughputError(f"no runs recorded for mode {mode!r}")
        rates = []
        for run in runs:
            if int(run.get("exit_code", -1)) != 0:
                raise ThroughputError(
                    f"mode {mode!r} has a run with exit_code={run.get('exit_code')}; a failed "
                    "capture's timing describes how long it took to fail"
                )
            declared = str(run.get("capture_mode", ""))
            if declared and declared != mode:
                raise ThroughputError(
                    f"stats file says capture_mode={declared!r} but it was filed under {mode!r}"
                )
            rate = float(run.get("prefill_tok_per_s", 0.0))
            if rate <= 0:
                raise ThroughputError(f"mode {mode!r}: prefill_tok_per_s is {rate}")
            rates.append(rate)
            token_counts.add(int(run.get("n_tokens_decoded", 0)))
        result.timings[mode] = ModeTiming(mode, tuple(rates))

    if len(token_counts) > 1:
        raise ThroughputError(
            f"the modes decoded different token counts {sorted(token_counts)}; a throughput ratio "
            "between different amounts of work is not an overhead"
        )
    result.n_tokens = token_counts.pop() if token_counts else 0

    missing = [m for m in CAPTURE_MODES if m not in result.timings]
    if missing:
        result.notes.append(f"modes not measured: {missing}")
    if not result.conclusive:
        result.notes.append(
            f"INCONCLUSIVE: total_ratio {result.total_ratio:.3f} is within run-to-run noise "
            f"(worst spread {result.worst_spread:.1%}). Add repeats or a quieter machine before "
            "using this number in a projection."
        )
    return result


def measure_capture_overhead(
    binary: Path | str,
    *,
    model: Path | str,
    spec: Path | str,
    corpus: Path | str,
    scratch: Path | str,
    repeats: int = 3,
    modes: Sequence[str] = CAPTURE_MODES,
    extra_args: Sequence[str] = (),
    timeout: float = 3600.0,
) -> CaptureOverhead:
    """Run ``moe_trace`` in each mode ``repeats`` times and collect the timings.

    Modes are interleaved rather than run in blocks (all no-callback, then all filter-off, ...)
    because a machine that gets busier over the course of the measurement would otherwise assign
    the whole drift to whichever mode ran last, and that is exactly the shape of the artefact we
    are trying not to report.
    """
    binary = Path(binary).resolve()
    if not binary.exists():
        raise ThroughputError(f"moe_trace binary not found: {binary}")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    for rep in range(repeats):
        for mode in modes:
            stats_path = scratch / f"{mode}.rep{rep}.json"
            argv = [
                str(binary),
                "--model", str(Path(model).resolve()),
                "--spec", str(Path(spec).resolve()),
                "--corpus", str(Path(corpus).resolve()),
                "--capture-mode", mode,
                "--stats", str(stats_path),
                *extra_args,
            ]
            if mode == "full":
                out_dir = scratch / f"trace.rep{rep}"
                out_dir.mkdir(parents=True, exist_ok=True)
                argv += ["--out", str(out_dir)]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                  check=False)
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or "").strip().splitlines()[-20:])
                raise ThroughputError(
                    f"moe_trace --capture-mode {mode} exited {proc.returncode}\n{tail}"
                )
            try:
                collected[mode].append(json.loads(stats_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                raise ThroughputError(f"cannot read {stats_path}: {exc}") from exc

    return overhead_from_stats(collected)


# -- projection and Gate Q1 ----------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """Projected Phase 5 GPU-hours, broken out so a cut can be costed without a re-run."""

    primary_hours: float
    additions_hours: float
    scale_hours: float
    overhead_ratio: float
    tokens_per_model: int
    per_model_hours: dict[str, float] = field(default_factory=dict)

    @property
    def total_hours(self) -> float:
        return self.primary_hours + self.additions_hours + self.scale_hours

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "total_hours": self.total_hours}


def project_phase5(
    tok_per_s: Mapping[str, float],
    *,
    primary_models: Sequence[str],
    addition_models: Sequence[str] = (),
    scale_model: str | None = None,
    tokens_per_model: int = 1_000_000,
    scale_tokens: int = 4_000_000,
    overhead_ratio: float = 1.0,
) -> Projection:
    """Project Phase 5 GPU-hours from measured per-model prefill throughput.

    Computed from each model's own measured rate rather than from the plan's S.4 multipliers
    (``O4 x ~0.4``, ``O4 x 4``). Those multipliers assume the additions and the scale run happen
    on the fastest model in the panel, which is true — but a projection that bakes in the
    assumption cannot notice when it stops being true, and OLMoE being fastest is a measurement,
    not a definition.
    """
    if overhead_ratio <= 0:
        raise ThroughputError(f"overhead_ratio must be positive, got {overhead_ratio}")
    if tokens_per_model <= 0:
        raise ThroughputError("tokens_per_model must be positive")

    def hours(model: str, tokens: int) -> float:
        rate = tok_per_s.get(model)
        if rate is None:
            raise ThroughputError(
                f"no measured throughput for {model!r}; have {sorted(tok_per_s)}. The plan is "
                "explicit that this term is measured, not guessed."
            )
        if rate <= 0:
            raise ThroughputError(f"{model!r} has non-positive throughput {rate}")
        return tokens / rate * overhead_ratio / 3600.0

    per_model = {m: hours(m, tokens_per_model) for m in primary_models}
    primary_hours = sum(per_model.values())

    additions_hours = 0.0
    for model in addition_models:
        h = hours(model, tokens_per_model)
        per_model[model] = h
        additions_hours += h

    scale_hours = hours(scale_model, scale_tokens) if scale_model else 0.0

    return Projection(
        primary_hours=primary_hours,
        additions_hours=additions_hours,
        scale_hours=scale_hours,
        overhead_ratio=overhead_ratio,
        tokens_per_model=tokens_per_model,
        per_model_hours=per_model,
    )


@dataclass(frozen=True)
class GateQ1:
    projection: Projection
    weekly_hours: float
    fraction: float
    passed: bool
    reduced: Projection | None
    drop_order: tuple[str, ...]
    verdict: str

    @property
    def budget_hours(self) -> float:
        return self.weekly_hours * self.fraction

    def to_json(self) -> dict[str, Any]:
        return {
            "gate": "Q1",
            "projection": self.projection.to_json(),
            "weekly_gpu_hours": self.weekly_hours,
            "fraction": self.fraction,
            "budget_hours": self.budget_hours,
            "passed": self.passed,
            "reduced_projection": self.reduced.to_json() if self.reduced else None,
            "drop_order": list(self.drop_order),
            "verdict": self.verdict,
        }


def evaluate_gate_q1(
    tok_per_s: Mapping[str, float],
    *,
    primary_models: Sequence[str],
    addition_models: Sequence[str] = (),
    scale_model: str | None = None,
    overhead_ratio: float = 1.0,
    weekly_hours: float = WEEKLY_GPU_HOURS,
    fraction: float = GATE_Q1_FRACTION,
    tokens_per_model: int = 1_000_000,
    reduced_tokens_per_model: int = 500_000,
    scale_tokens: int = 4_000_000,
) -> GateQ1:
    """Evaluate Gate Q1 and, if it fails, cost the plan's prescribed reduction.

    The gate does not choose between cuts — the plan already fixed the drop order and it is not
    this function's business to reorder it to fit a number. What it does is state whether the
    prescribed 1M→500k reduction is *sufficient*, because if it is not, the drop list is the next
    lever and someone has to decide that deliberately.
    """
    projection = project_phase5(
        tok_per_s,
        primary_models=primary_models,
        addition_models=addition_models,
        scale_model=scale_model,
        tokens_per_model=tokens_per_model,
        scale_tokens=scale_tokens,
        overhead_ratio=overhead_ratio,
    )
    budget = weekly_hours * fraction
    if projection.total_hours <= budget:
        return GateQ1(
            projection=projection, weekly_hours=weekly_hours, fraction=fraction, passed=True,
            reduced=None, drop_order=(),
            verdict=(f"PASS: {projection.total_hours:.1f} GPU-h projected against a "
                     f"{budget:.1f} h budget ({fraction:.0%} of {weekly_hours:.0f} h/week). "
                     f"Collect {tokens_per_model:,} tokens per model as planned."),
        )

    reduced = project_phase5(
        tok_per_s,
        primary_models=primary_models,
        addition_models=addition_models,
        scale_model=scale_model,
        tokens_per_model=reduced_tokens_per_model,
        scale_tokens=scale_tokens // 2,
        overhead_ratio=overhead_ratio,
    )
    enough = reduced.total_hours <= budget
    verdict = (
        f"FAIL: {projection.total_hours:.1f} GPU-h projected against a {budget:.1f} h budget. "
        f"Plan Gate Q1: cut the v1 budget to {reduced_tokens_per_model:,} tokens per model "
        f"BEFORE collecting anything, and record the reduction plus the T9.4 sample-size "
        f"sensitivity curve as the justification. Reduced projection: "
        f"{reduced.total_hours:.1f} GPU-h"
    )
    verdict += (
        ", which fits." if enough else
        f", which still exceeds the budget. Drop order applies next: {'; '.join(DROP_ORDER)}. "
        "Do not cut the panel's five primary models."
    )
    return GateQ1(
        projection=projection, weekly_hours=weekly_hours, fraction=fraction, passed=False,
        reduced=reduced, drop_order=DROP_ORDER, verdict=verdict,
    )


# -- artifacts -----------------------------------------------------------------------------

CALIBRATION_COLUMNS = (
    "model", "n_prompt", "avg_ts", "stddev_ts", "rel_stddev", "n_gpu_layers", "n_threads",
    "n_batch", "n_ubatch", "split_mode", "tensor_split", "backends", "build_commit", "gpu_info",
)


def write_calibration_csv(
    path: Path | str, rows: Iterable[BenchRow], overhead: CaptureOverhead | None = None
) -> Path:
    """Write ``results/throughput_calibration.csv``.

    The capture-overhead ratios are appended as ``#`` comment lines rather than as extra columns:
    they are properties of the harness, not of a (model, n_prompt) row, and repeating them on
    every row would invite someone to average them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if overhead is not None:
            handle.write(
                f"# capture overhead over {overhead.n_tokens} tokens: "
                f"observability {overhead.observability_ratio:.3f}x, "
                f"readback {overhead.readback_ratio:.3f}x, "
                f"total {overhead.total_ratio:.3f}x, "
                f"worst spread {overhead.worst_spread:.1%}, "
                f"conclusive={overhead.conclusive}\n"
            )
            for note in overhead.notes:
                handle.write(f"# {note}\n")
        writer = csv.DictWriter(handle, fieldnames=list(CALIBRATION_COLUMNS))
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record.pop("n_gen", None)
            record["model"] = record.pop("model_filename")
            record["rel_stddev"] = round(row.rel_stddev, 5)
            writer.writerow({k: record.get(k, "") for k in CALIBRATION_COLUMNS})
    return path


QUOTA_COLUMNS = ("session_id", "date", "notebook", "gpu_hours", "tokens_processed", "purpose")


def append_quota_log(
    path: Path | str, *, session_id: str, notebook: str, gpu_hours: float,
    tokens_processed: int, purpose: str, date: str | None = None,
) -> Path:
    """Append one row to ``results/quota_log.csv`` (plan S.4), creating it with a header.

    Append-only by construction: the weekly allowance is a rolling budget, so a rewritten log is
    a lost budget.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QUOTA_COLUMNS))
        if not exists:
            writer.writeheader()
        writer.writerow({
            "session_id": session_id,
            "date": date or time.strftime("%Y-%m-%d", time.gmtime()),
            "notebook": notebook,
            "gpu_hours": round(float(gpu_hours), 3),
            "tokens_processed": int(tokens_processed),
            "purpose": purpose,
        })
    return path


# -- CLI -----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T0.5 throughput calibration and Gate Q1")
    parser.add_argument("--bench-json", type=Path, action="append", default=[],
                        help="llama-bench -o json output; repeatable")
    parser.add_argument("--expect-commit", default=None, help="pinned build.llama_cpp_commit")
    parser.add_argument("--overhead-json", type=Path, default=None,
                        help="a CaptureOverhead.to_json() written by --measure-overhead")
    parser.add_argument("--measure-overhead", action="store_true")
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-arg", action="append", default=[],
                        help="extra argv passed through to moe_trace, e.g. --trace-arg=--ngl")
    parser.add_argument("--primary", nargs="*", default=[])
    parser.add_argument("--additions", nargs="*", default=[])
    parser.add_argument("--scale-model", default=None)
    parser.add_argument("--tokens-per-model", type=int, default=1_000_000)
    parser.add_argument("--calibration-csv", type=Path, default=None)
    parser.add_argument("--gate-json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        overhead: CaptureOverhead | None = None
        if args.measure_overhead:
            missing = [n for n in ("binary", "model", "spec", "corpus", "scratch")
                       if getattr(args, n) is None]
            if missing:
                raise ThroughputError(f"--measure-overhead needs {missing}")
            overhead = measure_capture_overhead(
                args.binary, model=args.model, spec=args.spec, corpus=args.corpus,
                scratch=args.scratch, repeats=args.repeats, extra_args=args.trace_arg,
            )
        elif args.overhead_json:
            payload = json.loads(args.overhead_json.read_text(encoding="utf-8"))
            overhead = overhead_from_stats({
                mode: [{"capture_mode": mode, "exit_code": 0, "prefill_tok_per_s": rate,
                        "n_tokens_decoded": payload.get("n_tokens", 0)}
                       for rate in data["tok_per_s"]]
                for mode, data in payload["modes"].items()
            })

        rows: list[BenchRow] = []
        for path in args.bench_json:
            rows.extend(parse_llama_bench_json(
                path.read_text(encoding="utf-8"), expect_commit=args.expect_commit
            ))

        if args.calibration_csv and rows:
            write_calibration_csv(args.calibration_csv, rows, overhead)
            print(f"wrote {args.calibration_csv}")

        if overhead is not None:
            print(json.dumps(overhead.to_json(), indent=2))
            if not overhead.conclusive:
                print("T0.5: capture overhead is INCONCLUSIVE - see the note above.")

        if args.primary:
            # The plan pins prefill at 2048 (I4), so that is the row the projection reads.
            by_model: dict[str, float] = {}
            for row in rows:
                if row.n_prompt >= 2048 or row.model_filename not in by_model:
                    by_model[row.model_filename] = row.avg_ts
            gate = evaluate_gate_q1(
                by_model, primary_models=args.primary, addition_models=args.additions,
                scale_model=args.scale_model, tokens_per_model=args.tokens_per_model,
                overhead_ratio=(overhead.total_ratio if overhead and overhead.conclusive else 1.0),
            )
            print("\nGate Q1: " + gate.verdict)
            if args.gate_json:
                args.gate_json.parent.mkdir(parents=True, exist_ok=True)
                args.gate_json.write_text(json.dumps(gate.to_json(), indent=2), encoding="utf-8")
            return 0 if gate.passed else 1
    except ThroughputError as exc:
        print(f"T0.5: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
