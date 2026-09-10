"""T3.3 -- prove ``hidden.bin`` is the router input by recomputing the routing from it.

The plan specifies T3.3 as an *FV probe*: train a linear classifier on the router-input tensor at
layer l and require ``set_agreement >= 0.99``. The reasoning is sound -- the router is a linear
map followed by top-k, so a linear probe on its own input must recover it, and a low score means
the trace rows are misaligned. But a probe is an indirect instrument. It has to be fitted, so it
can fail for reasons that have nothing to do with alignment: too few rows for the width, a
learning rate, an unlucky split. On a small shard it reports 0.49 on a *perfectly aligned* trace,
which is not evidence of anything.

The router weight is right there in the GGUF, in F32 -- T1.2 gates exactly that. So this module
computes the router instead of approximating it::

    argsort(-(hidden @ ffn_gate_inp.weight.T))[:top_k]  ==  topk.bin

No fitting, no split, no hyperparameter, no sample-size floor: it is exact on a single token.
Agreement of 1.0 establishes four things at once that are otherwise checked only indirectly, or
not at all:

  * ``hidden.bin`` holds the *router input* -- the normalized hidden state passed to
    ``build_moe_ffn`` -- and not the attention norm, the residual, or the previous layer's output;
  * hidden rows line up with the tokens ``hidden_index`` and ``tokens.flags`` claim (a relabelled
    feature is silent everywhere else);
  * ``topk.bin`` was de-strided correctly (I12 -- a contiguous read of the strided view yields
    in-range, distinct, wrong indices that no label check can catch);
  * the trace-layer to model-layer mapping is right (T3.5).

Disagreement is expected only at ties: ``hidden.bin`` is fp16, so a k-th/(k+1)-th logit gap below
its resolution can reorder. Those are counted separately from unexplained mismatches, because a
floor of benign noise under this check would hide the thing it exists to find.

Runs on CPU with numpy, and reads only the router weights out of the GGUF -- a few MB, not the
model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..capture.router_audit import ACCEPTABLE_ROUTER_TYPES, GGUFError, read_header
from .format import FLAG_HIDDEN_CAPTURED, TOKEN_DTYPE, read_manifest

__all__ = [
    "RouterCheckError",
    "LayerAgreement",
    "RouterCheckReport",
    "read_router_weight",
    "check_shard",
    "main",
]

#: Unit roundoff of float16 under round-to-nearest: 2^-11.
FP16_EPS = 2.0 ** -11


class RouterCheckError(RuntimeError):
    """The check could not be run. Never raised for *disagreement* -- that is a result."""


@dataclass(frozen=True)
class LayerAgreement:
    """One trace layer's agreement between the recomputed routing and ``topk.bin``."""

    layer: int
    model_layer: int
    n_rows: int
    exact_match: float
    set_agreement: float
    n_mismatched: int
    n_mismatched_at_tie: int

    @property
    def n_unexplained(self) -> int:
        return self.n_mismatched - self.n_mismatched_at_tie

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "model_layer": self.model_layer,
            "n_rows": self.n_rows,
            "exact_match": self.exact_match,
            "set_agreement": self.set_agreement,
            "n_mismatched": self.n_mismatched,
            "n_mismatched_at_tie": self.n_mismatched_at_tie,
            "n_unexplained": self.n_unexplained,
        }


@dataclass
class RouterCheckReport:
    shard_dir: Path
    model: str
    threshold: float
    layers: list[LayerAgreement] = field(default_factory=list)

    @property
    def worst(self) -> LayerAgreement | None:
        return min(self.layers, key=lambda a: a.set_agreement) if self.layers else None

    @property
    def ok(self) -> bool:
        """Passing needs the threshold *and* no unexplained mismatch on any layer.

        The two are not redundant. A handful of genuinely wrong rows out of a million still
        rounds to 1.0000, and "a few tokens route differently than the weights say" has no benign
        explanation once ties are accounted for.
        """
        if not self.layers:
            return False
        return all(
            a.set_agreement >= self.threshold and a.n_unexplained == 0 for a in self.layers
        )

    def to_dict(self) -> dict:
        return {
            "shard_dir": str(self.shard_dir),
            "model": self.model,
            "threshold": self.threshold,
            "ok": self.ok,
            "layers": [a.to_dict() for a in self.layers],
        }


def read_router_weight(gguf_path: Path | str, model_layer: int) -> np.ndarray:
    """``blk.<il>.ffn_gate_inp.weight`` as float32, shaped ``(n_experts, hidden_dim)``.

    Refuses a quantized router rather than dequantizing one. T1.2 gates the dtype precisely so
    that this can be a plain read; a checkpoint failing that gate must be requantized, and coping
    silently here would let the gate be bypassed by the one tool that depends on it.
    """
    header = read_header(gguf_path)
    name = f"blk.{model_layer}.ffn_gate_inp.weight"
    info = header.by_name(name)
    if info.type_name not in ACCEPTABLE_ROUTER_TYPES:
        raise RouterCheckError(
            f"{name} is {info.type_name}; the router must be F32/F16/BF16 (T1.2 gate). "
            "Requantize with --tensor-type rather than dequantizing here."
        )
    if len(info.dims) != 2:
        raise RouterCheckError(f"{name} has dims {info.dims}, expected 2")

    # GGUF dims are ggml's ne[], fastest-varying first: ne[0] is the input width.
    hidden_dim, n_experts = int(info.dims[0]), int(info.dims[1])
    dtype = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[info.type_name]
    itemsize = np.dtype(dtype).itemsize
    count = hidden_dim * n_experts

    with open(gguf_path, "rb") as f:
        f.seek(header.file_offset(info))
        raw = f.read(count * itemsize)
    if len(raw) != count * itemsize:
        raise RouterCheckError(f"{name}: truncated tensor data (wanted {count * itemsize} B)")

    flat = np.frombuffer(raw, dtype=dtype)
    if info.type_name == "BF16":
        # bf16 is the top 16 bits of an fp32; numpy has no native dtype for it.
        flat = (flat.astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(flat.astype(np.float32).reshape(n_experts, hidden_dim))


def check_shard(
    shard_dir: Path | str,
    gguf_path: Path | str,
    *,
    layers: Sequence[int] | None = None,
    threshold: float = 0.99,
    max_rows: int = 20_000,
) -> RouterCheckReport:
    """Recompute the routing of one shard's captured tokens and compare against ``topk.bin``."""
    shard_dir = Path(shard_dir)
    manifest = read_manifest(shard_dir)
    n_tokens = int(manifest["n_tokens"])
    n_captured = int(manifest["n_captured"])
    n_layers = int(manifest["n_moe_layers"])
    n_experts = int(manifest["n_experts"])
    top_k = int(manifest["top_k"])
    hidden_dim = int(manifest["hidden_dim"])
    layer_map = manifest["layer_index_map"]

    if n_captured == 0:
        raise RouterCheckError(
            f"{shard_dir} captured no hidden states; T3.3 needs the router input and this "
            "shard's subsample was dropped (T4.4 fallback ladder)"
        )

    hidden = np.memmap(shard_dir / "hidden.bin", dtype=np.float16, mode="r").reshape(
        n_captured, n_layers, hidden_dim
    )
    topk = np.memmap(shard_dir / "topk.bin", dtype=np.int32, mode="r").reshape(
        n_tokens, n_layers, top_k
    )
    tokens = np.memmap(shard_dir / "tokens.bin", dtype=TOKEN_DTYPE, mode="r")

    captured_rows = np.flatnonzero(np.asarray(tokens["flags"]) & FLAG_HIDDEN_CAPTURED)
    if captured_rows.size != n_captured:
        raise RouterCheckError(
            f"{shard_dir}: {captured_rows.size} token(s) flagged but hidden.bin holds "
            f"{n_captured} rows"
        )

    # Evenly spaced, never a prefix: a misalignment that only begins partway through the shard --
    # a re-shard, a resumed session -- would sit entirely outside a prefix sample.
    if captured_rows.size > max_rows:
        take = np.linspace(0, captured_rows.size - 1, max_rows).astype(np.int64)
    else:
        take = np.arange(captured_rows.size, dtype=np.int64)
    rows = captured_rows[take]

    report = RouterCheckReport(
        shard_dir=shard_dir, model=str(manifest["model"]), threshold=threshold
    )
    arange = np.arange(rows.size)[:, None]
    for layer in range(n_layers) if layers is None else layers:
        if not 0 <= layer < n_layers:
            raise RouterCheckError(f"layer {layer} is outside the trace's {n_layers} MoE layers")
        model_layer = int(layer_map[layer]) if isinstance(layer_map, (list, tuple)) else layer
        weight = read_router_weight(gguf_path, model_layer)
        if weight.shape != (n_experts, hidden_dim):
            raise RouterCheckError(
                f"router at model layer {model_layer} is {weight.shape}, the trace says "
                f"{(n_experts, hidden_dim)}"
            )

        logits = np.asarray(hidden[take, layer, :], dtype=np.float32) @ weight.T
        pred = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]
        truth = np.asarray(topk[rows, layer, :], dtype=np.int64)
        agree_rows = (np.sort(pred, axis=1) == np.sort(truth, axis=1)).all(axis=1)

        # Set agreement without materialising a Python set per row.
        hit = np.zeros((rows.size, n_experts), dtype=bool)
        hit[arange, truth] = True
        overlap = hit[arange, pred].sum(axis=1)

        n_bad = int((~agree_rows).sum())
        n_tie = 0
        if n_bad:
            # Is the k/(k+1) gap inside this recomputation's own numerical resolution?
            #
            # Not a fixed relative tolerance on the logit -- that model is wrong here. The logit
            # is a 2048-term dot product whose inputs were rounded to fp16 before storage, and
            # cancellation makes the result's relative error unbounded in terms of its own
            # magnitude: layer 4 of the first OLMoE shard has a logit of 0.042 assembled from
            # terms far larger than that. The standard forward bound is what applies:
            #
            #     |d(x . w)| <= eps * sum_i |x_i w_i|
            #
            # so the resolution of a *difference* between two experts is the sum of their
            # bounds. Computed per row, from the actual magnitudes, with no constant to tune.
            x_bad = np.abs(np.asarray(hidden[take, layer, :], dtype=np.float32)[~agree_rows])
            bound = FP16_EPS * (x_bad @ np.abs(weight).T)
            order = np.argsort(-logits[~agree_rows], axis=1, kind="stable")
            kth, nxt = order[:, top_k - 1], order[:, top_k]
            idx = np.arange(n_bad)
            srt = np.sort(logits[~agree_rows], axis=1)[:, ::-1]
            gap = srt[:, top_k - 1] - srt[:, top_k]
            n_tie = int((gap <= bound[idx, kth] + bound[idx, nxt]).sum())

        report.layers.append(
            LayerAgreement(
                layer=int(layer),
                model_layer=model_layer,
                n_rows=int(rows.size),
                exact_match=float(agree_rows.mean()),
                set_agreement=float(overlap.mean() / top_k),
                n_mismatched=n_bad,
                n_mismatched_at_tie=n_tie,
            )
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.traces.router_check",
        description="T3.3 -- recompute routing from the GGUF router weight and hidden.bin",
    )
    ap.add_argument("shard_dir", type=Path)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", help="trace layers; default all")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--max-rows", type=int, default=20_000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    try:
        report = check_shard(
            args.shard_dir,
            args.gguf,
            layers=args.layers,
            threshold=args.threshold,
            max_rows=args.max_rows,
        )
    except (RouterCheckError, GGUFError) as exc:
        print(f"T3.3 could not run: {exc}")
        return 2

    for a in report.layers:
        flag = "" if a.set_agreement >= report.threshold and a.n_unexplained == 0 else "  <-- FAIL"
        print(
            f"  layer {a.layer:3d} (model {a.model_layer:3d}): "
            f"set_agreement={a.set_agreement:.6f} exact={a.exact_match:.4f} "
            f"mismatched={a.n_mismatched} (ties {a.n_mismatched_at_tie}, "
            f"unexplained {a.n_unexplained}) over {a.n_rows} rows{flag}"
        )
    worst = report.worst
    print(
        f"T3.3 {'PASSED' if report.ok else 'FAILED'} for {report.model}: worst layer "
        f"{worst.layer} at set_agreement {worst.set_agreement:.6f} "
        f"(threshold {report.threshold})"
    )
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
