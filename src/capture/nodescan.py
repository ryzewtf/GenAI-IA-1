"""T1.4 -- node-name discovery and confirmation. **HALT GATE.**

`nodespec.DEFAULT_NODE_TEMPLATES` is a *hypothesis* about what llama.cpp names the three tensors
this study reads. This module turns it into a fact, per model, by reading what the graph actually
contained.

The gate it enforces (TASKS.md T1.4): **exactly one logit node, one top-k node, and one
router-input node per MoE layer.** Each half of that matters for a different reason.

*At least one* -- if a name is absent, `moe_trace` captures nothing for that layer and the failure
is loud. That case is not the danger.

*At most one* -- this is the case worth building a tool for. If a name appears twice per layer
(two graph nodes cb'd identically, or a model with `moe_layer_freq` interleaving that makes the
layer numbering non-contiguous), the capture callback fires twice per layer and the writer lays
down rows in an order nothing downstream can reconstruct. The files are exactly the right size.
Every index is in range and distinct. T5.3 passes. The traces are wrong, and the first symptom is
a predictability number that disagrees with the literature -- which is precisely the finding this
study is trying to make, so it would be believed.

So this module refuses to be reassuring: an off-count is an error, never a warning.

Two entry points, deliberately separated:

  * `parse_eval_callback(text)` -- pure text -> observations. No model, no subprocess. This is what
    the tests drive, and it is why the gate is testable on a machine with no checkpoint.
  * `run_eval_callback(...)` -- invokes `llama-eval-callback -ngl 0 -c 512`, which **loads a
    model**. Behind an explicit flag, never the default.

The line format is pinned to `common/debug.cpp:172` at commit 7077abbe:

    common_debug_cb_eval: %24s = (%s) %10s(%s{%s}, %s}) = {%s}
                          name     type    op   src0  ne   src1     ne_of_t

(The stray `}` after the src1 field is upstream's, not a typo here -- the parser tolerates it
rather than depending on it.)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.capture.nodespec import (
    SELECTION_CHAIN,
    NodeSpecError,
    layer_index_map,
    selection_chain,
)

__all__ = [
    "NodeObservation",
    "LayerInventory",
    "ConfirmationReport",
    "NodeScanError",
    "parse_eval_callback",
    "inventory_by_layer",
    "confirm_nodes",
    "run_eval_callback",
    "write_back",
    "set_model_fields",
    "main",
]

# `common_debug_cb_eval:  <name> = (<type>) <op>(<src0>{<ne>}, <src1>{<ne>}}) = {<ne>}`
# Anchored on the function-name prefix so that ordinary log chatter, a progress bar, or a tensor
# value dump (which follows each line) can never be mistaken for a node.
_LINE = re.compile(
    r"^common_debug_cb_eval:\s+"
    r"(?P<name>\S+)\s+=\s+"
    r"\((?P<dtype>[^)]*)\)\s+"
    r"(?P<op>\S+)\("
    r"(?P<srcs>.*)"
    r"\)\s+=\s+\{(?P<ne>[^}]*)\}\s*$"
)

# `ffn_moe_topk-11` -> ("ffn_moe_topk", 11). The suffix is the llama.cpp `il`, not the trace layer.
_SUFFIXED = re.compile(r"^(?P<base>.+?)-(?P<il>\d+)$")

_ROUTER_INPUT_FALLBACKS: tuple[str, ...] = ("ffn_norm", "ffn_inp")
"""Last-resort guesses, used only when the graph does not report the logits node's sources.

Deliberately short, and deliberately excluding `attn_norm`: that name is present in every layer of
every transformer, so including it would make the fallback *always* succeed and frequently name
the attention norm -- the wrong tensor, silently. A guess that cannot fail is not a check.

The primary resolution is structural (`_resolve_router_input` below) and does not guess at all.
"""


class NodeScanError(RuntimeError):
    """T1.4 did not confirm. A halt gate failing is not a warning."""


def _parse_ne(text: str) -> tuple[int, ...]:
    """`"64, 5, 1, 1"` -> `(64, 5, 1, 1)`, trailing 1s kept (ggml is always 4-D)."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise NodeScanError(f"unparseable shape {text!r}: {exc}") from None


@dataclass(frozen=True)
class NodeObservation:
    """One node as the graph actually contained it."""

    name: str
    dtype: str
    op: str
    ne: tuple[int, ...]
    src_names: tuple[str, ...] = ()

    @property
    def base(self) -> str:
        """The name with its `-<il>` layer suffix removed."""
        m = _SUFFIXED.match(self.name)
        return m.group("base") if m else self.name

    @property
    def il(self) -> int | None:
        """The llama.cpp layer index, or None for a graph-level node like `inp_embd`."""
        m = _SUFFIXED.match(self.name)
        return int(m.group("il")) if m else None


def parse_eval_callback(text: str) -> list[NodeObservation]:
    """Parse `llama-eval-callback` output into node observations, in graph order.

    Non-matching lines are ignored: the tool interleaves each node with a dump of its first few
    values, plus the usual load-time logging.
    """
    out: list[NodeObservation] = []
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        srcs = tuple(
            s.split("{", 1)[0].strip()
            for s in m.group("srcs").split(",")
            # The src list is `name{ne}, name{ne}}` -- shape fragments and the stray brace are not
            # sources. Splitting on "," alone would yield "5", " 1", " 1}" as phantom src names.
            if s.strip() and "{" in s
        )
        out.append(
            NodeObservation(
                name=m.group("name"),
                dtype=m.group("dtype").strip(),
                op=m.group("op").strip(),
                ne=_parse_ne(m.group("ne")),
                src_names=srcs,
            )
        )
    return out


@dataclass
class LayerInventory:
    """What one llama.cpp layer `il` contained, restricted to names we care about."""

    il: int
    counts: dict[str, int] = field(default_factory=dict)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dtypes: dict[str, str] = field(default_factory=dict)

    def present(self, base: str) -> bool:
        return self.counts.get(base, 0) > 0


def inventory_by_layer(
    observations: Iterable[NodeObservation],
    *,
    bases: Sequence[str] | None = None,
) -> dict[int, LayerInventory]:
    """Group observations by `il`, counting occurrences of each base name.

    Counting rather than collecting is the point: the duplicate case is the one that corrupts data
    silently, so the count must survive into the report.
    """
    wanted = set(bases) if bases is not None else None
    inv: dict[int, LayerInventory] = {}
    for o in observations:
        il = o.il
        if il is None:
            continue
        if wanted is not None and o.base not in wanted:
            continue
        entry = inv.setdefault(il, LayerInventory(il=il))
        entry.counts[o.base] = entry.counts.get(o.base, 0) + 1
        # Keep the FIRST shape seen. If a duplicate has a different shape that is itself a finding,
        # but the count check below fires first regardless.
        entry.shapes.setdefault(o.base, o.ne)
        entry.dtypes.setdefault(o.base, o.dtype)
    return inv


def _resolve_router_input(
    observations: Sequence[NodeObservation],
    layers: Sequence[int],
    inv: Mapping[int, LayerInventory],
    *,
    forced: str | None,
    errors: list[str],
) -> tuple[str, str]:
    """Work out which node carries the hidden state the router read. Returns (base, how).

    Structural first, guessing last. `ffn_moe_logits` is `mul_mat(ffn_gate_inp, cur)`, so the
    logits node's own source list *names* the router input -- that is a fact recovered from the
    graph, not a hypothesis about naming conventions, and it is right by construction for any
    architecture including ones nobody has looked at yet.

    Only if the dump carries no source information do we fall back to a name guess, and then the
    guess must hold in **every** MoE layer or it is an error. A router input that is right for
    some layers and wrong for others is the worst outcome available: the F4/F5/FV probes would be
    conditioned on a mixture, underperform, and the underperformance is indistinguishable from
    the study's actual finding about predictability.
    """
    if forced:
        if not all(inv.get(il, LayerInventory(il=il)).counts.get(forced, 0) == 1 for il in layers):
            errors.append(
                f"forced router input {forced!r} does not appear exactly once in every MoE layer"
            )
        return forced, "by explicit --router-input"

    # Structural: the source of the logits node, per layer.
    by_name = {o.name: o for o in observations}
    resolved: dict[int, str] = {}
    for il in layers:
        for chain_base in SELECTION_CHAIN[:1]:  # ffn_moe_logits is where `cur` enters
            node = by_name.get(f"{chain_base}-{il}")
            if node is None or not node.src_names:
                continue
            # src[0] is the router weight `ffn_gate_inp`; the activation is the other operand.
            acts = [s for s in node.src_names if "ffn_gate_inp" not in s]
            if len(acts) == 1:
                obs = by_name.get(acts[0])
                resolved[il] = obs.base if obs is not None else acts[0].rsplit("-", 1)[0]
    if len(resolved) == len(layers):
        bases = set(resolved.values())
        if len(bases) == 1:
            return bases.pop(), "structurally, from the source of ffn_moe_logits"
        errors.append(
            f"the router input is not the same node in every MoE layer: {sorted(bases)}. The "
            "probes would be conditioned on a mixture of tensors, and the resulting "
            "underperformance is indistinguishable from the study's actual finding."
        )
        return sorted(bases)[0], "structurally, INCONSISTENTLY"

    # Fallback: name guess, required to hold in every layer.
    for cand in _ROUTER_INPUT_FALLBACKS:
        if all(inv.get(il, LayerInventory(il=il)).counts.get(cand, 0) == 1 for il in layers):
            return cand, f"by name fallback (no source info in the dump); verify with the FV probe"
    errors.append(
        f"the router input could not be resolved: ffn_moe_logits reports no usable sources, and "
        f"no fallback name {list(_ROUTER_INPUT_FALLBACKS)} appears exactly once in every MoE "
        "layer. Read build_moe_ffn for this architecture rather than adding another guess."
    )
    return _ROUTER_INPUT_FALLBACKS[0], "UNRESOLVED"


@dataclass
class ConfirmationReport:
    """The T1.4 verdict for one model."""

    model: str
    moe_layers: list[int]
    node_topk: str
    node_logits: str
    node_router_input: str
    selection_node: str
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [
            f"T1.4 node scan: {self.model}",
            f"  MoE layers (il): {self.moe_layers[0]}..{self.moe_layers[-1]} "
            f"({len(self.moe_layers)} layers)",
            f"  logits       : {self.node_logits}",
            f"  topk         : {self.node_topk}",
            f"  router_input : {self.node_router_input}",
            f"  selection    : {self.selection_node}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        if self.errors:
            lines.append("  NOT CONFIRMED:")
            lines += [f"    - {e}" for e in self.errors]
        else:
            lines.append("  CONFIRMED")
        return "\n".join(lines)


def _check_count(
    inv: Mapping[int, LayerInventory],
    base: str,
    layers: Sequence[int],
    errors: list[str],
    *,
    role: str,
) -> None:
    for il in layers:
        n = inv.get(il, LayerInventory(il=il)).counts.get(base, 0)
        if n == 1:
            continue
        if n == 0:
            errors.append(
                f"layer {il}: no node named {base!r} ({role}). The name hypothesis is wrong for "
                "this architecture; read build_moe_ffn for it rather than guessing another suffix."
            )
        else:
            errors.append(
                f"layer {il}: {n} nodes named {base!r} ({role}), expected exactly 1. A duplicate "
                "fires the capture callback twice per layer and the writer emits rows in an order "
                "nothing downstream can reconstruct -- the trace would be the right SIZE and the "
                "wrong CONTENT, and T5.3 would pass it."
            )


def confirm_nodes(
    config: Mapping[str, object],
    observations: Iterable[NodeObservation],
    *,
    model: str | None = None,
    router_input_base: str | None = None,
) -> ConfirmationReport:
    """Confirm the three node names against an observed graph. Raises nothing; reports errors.

    `config` is one model's block from `configs/models.yaml`.
    """
    observations = list(observations)
    model_name = str(model or config.get("key") or config.get("hf_repo") or "<model>")
    layers = layer_index_map(config)  # raises NodeSpecError on a non-contiguous layer map
    present_chain, selection = selection_chain(config)

    n_experts = int(config["n_experts"])
    top_k = int(config["top_k"])
    hidden_dim = int(config["hidden_dim"])

    errors: list[str] = []
    notes: list[str] = []

    all_inv = inventory_by_layer(observations)
    ri_base, how = _resolve_router_input(
        observations, layers, all_inv, forced=router_input_base, errors=errors
    )
    notes.append(f"router input {ri_base!r} resolved {how}")

    bases = {"ffn_moe_topk", selection, ri_base}
    inv = inventory_by_layer(observations, bases=sorted(bases))

    _check_count(inv, "ffn_moe_topk", layers, errors, role="expert indices")
    _check_count(inv, selection, layers, errors, role="selection logits")
    _check_count(inv, ri_base, layers, errors, role="router input")

    # The chain prediction is confirmed against what is actually there: a node LATER in the chain
    # than the predicted selection node means top-k consumed something else, and `logit_tensor_used`
    # would name a tensor the router never saw.
    later = SELECTION_CHAIN[SELECTION_CHAIN.index(selection) + 1:] if selection in SELECTION_CHAIN else ()
    for base in later:
        n = sum(1 for il in layers if all_inv.get(il, LayerInventory(il=il)).counts.get(base, 0))
        if n:
            errors.append(
                f"{base!r} is present in {n} layer(s) but is LATER in the selection chain than the "
                f"predicted selection node {selection!r}. top-k consumed {base!r}, so the margins "
                "captured from the predicted node are not the margins that decided the routing."
            )

    # Shapes. These catch a name that exists but denotes a different tensor -- the failure mode a
    # name-only check cannot see.
    for il in layers:
        e = inv.get(il)
        if e is None:
            continue
        shp = e.shapes.get("ffn_moe_topk")
        if shp and shp[0] != top_k:
            errors.append(
                f"layer {il}: {'ffn_moe_topk'} has ne[0]={shp[0]}, expected top_k={top_k}"
            )
        shp = e.shapes.get(selection)
        if shp and shp[0] != n_experts:
            errors.append(
                f"layer {il}: {selection!r} has ne[0]={shp[0]}, expected n_experts={n_experts}"
            )
        shp = e.shapes.get(ri_base)
        if shp and shp[0] != hidden_dim:
            errors.append(
                f"layer {il}: {ri_base!r} has ne[0]={shp[0]}, expected hidden_dim={hidden_dim}"
            )
        dt = e.dtypes.get("ffn_moe_topk")
        if dt and dt.lower() not in ("i32", "int32"):
            errors.append(f"layer {il}: ffn_moe_topk has dtype {dt}, expected i32")

    if not observations:
        errors.append("no eval-callback node lines were parsed at all; the log is empty or its "
                      "format has changed from common/debug.cpp at the pinned commit")

    notes.append(f"selection chain present: {' -> '.join(present_chain)}")

    return ConfirmationReport(
        model=model_name,
        moe_layers=layers,
        node_topk="ffn_moe_topk-%d",
        node_logits=f"{selection}-%d",
        node_router_input=f"{ri_base}-%d",
        selection_node=selection,
        errors=errors,
        notes=notes,
    )


def run_eval_callback(
    binary: Path | str,
    model_path: Path | str,
    *,
    n_ctx: int = 512,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    timeout: float = 900.0,
) -> str:
    """Invoke `llama-eval-callback -ngl 0`. **This loads a model.** Returns combined output.

    `-ngl 0` keeps every layer on the CPU: the point is the graph's node names, which are
    identical either way, and offloading would need a GPU that the audit machine may not have free.

    No `-n` and no `--no-warmup`: the tool registers under `LLAMA_EXAMPLE_COMMON`, which does not
    accept either flag (an unknown flag is a hard parse error, not a warning), and it decodes the
    prompt exactly once and sets `params.warmup = false` in its own `main` regardless.
    """
    # Absolute, and with native separators: Windows CreateProcess rejects a *relative* path
    # written with forward slashes (WinError 2) even though the file is right there, and the
    # POSIX-flavoured path is exactly what a shell on this workstation hands us.
    exe = Path(binary).resolve()
    if not exe.exists():
        raise NodeScanError(f"eval-callback binary not found: {exe}")
    argv = [
        str(exe), "-m", str(Path(model_path).resolve()), "-ngl", "0", "-c", str(n_ctx),
        "-p", prompt,
    ]
    proc = subprocess.run(  # noqa: S603 -- shell=False, argv is a list, nothing is interpolated
        argv, capture_output=True, text=True, errors="replace", timeout=timeout,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "common_debug_cb_eval:" not in combined:
        tail = "\n".join(combined.splitlines()[-40:])
        raise NodeScanError(
            f"llama-eval-callback exited {proc.returncode} with no node output:\n{tail}"
        )
    return combined


def _load_model_config(models_path: Path | str, key: str) -> dict[str, object]:
    import yaml

    with open(models_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    models = doc.get("models", doc)
    if key not in models:
        raise NodeScanError(f"{key!r} is not in {models_path}; have: {sorted(models)}")
    cfg = dict(models[key])
    cfg.setdefault("key", key)
    return cfg


def write_back(models_path: Path | str, model_key: str, report: "NodeScanReport",
               *, force: bool = False) -> list[str]:
    """Record a confirmed T1.4 result into ``configs/models.yaml``, in place.

    A thin wrapper over :func:`set_model_fields`, which does the editing. The split exists because
    T1.1 has the same problem for a different set of fields (``gguf``, ``quant``), and a second
    hand-rolled YAML editor is a second place for the multi-line-flow-mapping bug below to be
    reintroduced without its regression test noticing.
    """
    return set_model_fields(
        models_path,
        model_key,
        {
            "node_names": (
                f"{{logits: {report.node_logits}, logits_biased: "
                f"{getattr(report, 'node_logits_biased', None) or 'null'}, "
                f"topk: {report.node_topk}, router_input: {report.node_router_input}}}"
            ),
            "logit_tensor_used": report.selection_node,
        },
        force=force,
    )


def set_model_fields(models_path: Path | str, model_key: str,
                     wanted: "Mapping[str, str]", *, force: bool = False) -> list[str]:
    """Set top-level scalar/flow fields on one model block in ``configs/models.yaml``, in place.

    Line-based rather than ``yaml.safe_load`` + ``safe_dump``: models.yaml is more comment than
    data — the T1.2 gate result, the §1.5 pair structure, the GPT-OSS requantization warning — and
    a round trip through PyYAML would silently delete all of it.

    **An existing value that disagrees is an error, not something to overwrite.** A node name that
    changed between two scans of the same checkpoint means either the llama.cpp build moved or the
    artifact did, and both invalidate every trace already collected under the old name. Re-running
    discovery is cheap; discovering afterwards that half the panel used a different selection node
    (invariant I13) is not. ``force`` exists for the deliberate case and says so in the diff.
    """
    path = Path(models_path)
    raw = path.read_text(encoding="utf-8", newline="")
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.replace("\r\n", "\n").split("\n")

    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == f"{model_key}:")
    except StopIteration as exc:
        raise NodeScanError(f"{path}: no block for model {model_key!r}") from exc

    indent = len(lines[start]) - len(lines[start].lstrip())
    stop = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if (len(lines[i]) - len(lines[i].lstrip())) <= indent:
            stop = i
            break

    changes: list[str] = []
    # Rebuilt after each replacement, because a multi-line value collapses to one line and shifts
    # every index after it.
    for key, value in wanted.items():
        target = None
        for i in range(start + 1, stop):
            if lines[i].lstrip().startswith(f"{key}:"):
                target = i
                break
        if target is None:
            raise NodeScanError(f"{path}: {model_key} has no {key}: line to update")

        # A flow mapping may be wrapped across lines -- models.yaml already writes OLMoE's
        # node_names over two. Replacing only the first line leaves the continuation orphaned and
        # the file stops being valid YAML, which is a corruption that no test of THIS function
        # would notice: it parses fine right up until someone loads the config.
        last = target
        current = lines[target].split(":", 1)[1].split("#")[0].strip()
        if current.startswith("{") and current.count("{") > current.count("}"):
            depth = current.count("{") - current.count("}")
            while depth > 0 and last + 1 < stop:
                last += 1
                piece = lines[last].split("#")[0].strip()
                current += " " + piece
                depth += piece.count("{") - piece.count("}")
            if depth > 0:
                raise NodeScanError(
                    f"{path}: {model_key}.{key} opens a flow mapping that never closes"
                )
        if current == value:
            continue
        # "Unset" is a bare null, or a node_names mapping whose every value is null. Both are what
        # a model looks like before T1.4 has run, and neither is a prior claim worth protecting.
        if current.startswith("{"):
            unset = all(
                piece.split(":", 1)[-1].strip() in ("null", "")
                for piece in current.strip("{}").split(",")
                if piece.strip()
            )
        else:
            unset = current in ("", "null")
        if not unset and not force:
            raise NodeScanError(
                f"{path}: {model_key}.{key} already records {current!r} but this scan found "
                f"{value!r}. A node name that changed between two scans of the same checkpoint "
                "means the build or the artifact moved, and every trace collected under the old "
                "name is a different experiment (invariant I13). Investigate, then pass --force "
                "if the change is intended."
            )
        pad = " " * (len(lines[target]) - len(lines[target].lstrip()))
        lines[target : last + 1] = [f"{pad}{key}: {value}"]
        stop -= last - target
        changes.append(f"{key}: {current or '(empty)'} -> {value}")

    if changes:
        path.write_text(newline.join(lines), encoding="utf-8", newline="")
    return changes


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="nodescan",
        description="T1.4 node-name discovery/confirmation (HALT GATE).",
    )
    ap.add_argument("model_key", help="key in configs/models.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-log", type=Path,
                     help="parse a saved llama-eval-callback dump (loads nothing)")
    src.add_argument("--run", action="store_true",
                     help="invoke llama-eval-callback -- THIS LOADS THE MODEL")
    ap.add_argument("--binary", default="build/cpu/bin/llama-eval-callback")
    ap.add_argument("--gguf", type=Path, help="path to the GGUF (required with --run)")
    ap.add_argument("--save-log", type=Path, help="write the raw dump here for re-parsing")
    ap.add_argument("--router-input", help="force the router-input base name")
    ap.add_argument("--write", action="store_true",
                    help="record the confirmed node names back into configs/models.yaml")
    ap.add_argument("--force", action="store_true",
                    help="with --write, allow overwriting a DIFFERENT recorded value")
    args = ap.parse_args(argv)

    try:
        cfg = _load_model_config(args.models, args.model_key)
        if args.run:
            if not args.gguf:
                ap.error("--run requires --gguf")
            text = run_eval_callback(args.binary, args.gguf)
        else:
            text = args.from_log.read_text(encoding="utf-8", errors="replace")
        if args.save_log:
            args.save_log.parent.mkdir(parents=True, exist_ok=True)
            args.save_log.write_text(text, encoding="utf-8")
        obs = parse_eval_callback(text)
        report = confirm_nodes(cfg, obs, model=args.model_key,
                               router_input_base=args.router_input)
    except (NodeScanError, NodeSpecError, OSError) as exc:
        print(f"T1.4 FAILED: {exc}", file=sys.stderr)
        return 1

    print(report.render())
    if not report.confirmed:
        print("\nT1.4 is a HALT GATE. Fix the node filter before writing a byte of trace.",
              file=sys.stderr)
        return 1
    if args.write:
        try:
            changes = write_back(args.models, args.model_key, report, force=args.force)
        except NodeScanError as exc:
            print(f"T1.4 could not update {args.models}: {exc}", file=sys.stderr)
            return 1
        if changes:
            print(f"\nUpdated {args.models} for {args.model_key}:")
            for line in changes:
                print(f"  {line}")
        else:
            print(f"\n{args.models} already matches this scan for {args.model_key}.")
        return 0

    print(f"\nRecord in models.yaml under {args.model_key}:")
    print(f"  node_names: {{logits: {report.node_logits}, logits_biased: null, "
          f"topk: {report.node_topk}, router_input: {report.node_router_input}}}")
    print(f"  logit_tensor_used: {report.selection_node}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
