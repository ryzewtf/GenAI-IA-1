"""T1.2 -- router dtype audit. **GATE.**

Every routing decision in this study is `argsort(top_k(router_logits))`. The router is a single
`[n_embd, n_experts]` matrix per MoE layer -- `ffn_gate_inp` -- and it is minuscule: for OLMoE it
is 2048x64 = 128K parameters against 6.9B total, about 0.002% of the checkpoint. Quantizing it
saves nothing measurable and costs the one thing this study measures.

Why that matters more here than it would elsewhere. The expert *weights* being Q4_K is fine: we
never claim anything about the FFN outputs, only about which experts were chosen. But if
`ffn_gate_inp` is quantized, the logits are computed from a perturbed matrix, and near-ties in the
top-k boundary flip. A flipped tie is not noise that averages out -- it changes the discrete label
the probe is trained to predict, and it does so *systematically* (quantization error is a
deterministic function of the weights, not a random draw). The measured predictability would then
be the predictability of a checkpoint nobody else has, and T3.2's PyTorch cross-validation --
which reads the unquantized safetensors -- would fail against our own trace for a reason that
looks like a harness bug.

So this is a gate, not a report. The remedy when it fails is in TASKS.md: requantize with the
router excluded (`llama-quantize --tensor-type ffn_gate_inp=f32`).

Reads GGUF **headers only**. The tensor data section is never touched, so auditing a 4 GB
checkpoint reads a few hundred KB and needs no GPU, no llama.cpp build, and no RAM budget --
deliberately, so this gate can run on the workstation while something else is using the machine.

Pure stdlib: no numpy, no gguf-py, no torch. The format is pinned by
`llama_cpp_pull/ggml/include/gguf.h` at the commit in `configs/run.yaml`, and it is stable enough
that a 120-line parser is a smaller liability than a dependency that must exist inside a Kaggle
session.

Usage:
    python -m src.capture.router_audit models/*.gguf -o results/router_dtype_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

__all__ = [
    "GGUFError",
    "RouterAuditError",
    "TensorInfo",
    "RouterFinding",
    "ModelAudit",
    "GGML_TYPE_NAMES",
    "ROUTER_SUFFIXES",
    "ACCEPTABLE_ROUTER_TYPES",
    "read_tensor_infos",
    "audit_gguf",
    "write_csv",
    "main",
]

GGUF_MAGIC = b"GGUF"

# ggml/include/ggml.h at 7077abbe. Gaps are removed types -- an id absent from this table is
# reported by number rather than guessed at, because a wrong name on a FAIL row would send someone
# to requantize with the wrong flag.
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}

# ggml/include/gguf.h at 7077abbe.
_GGUF_TYPE_UINT8, _GGUF_TYPE_INT8 = 0, 1
_GGUF_TYPE_UINT16, _GGUF_TYPE_INT16 = 2, 3
_GGUF_TYPE_UINT32, _GGUF_TYPE_INT32 = 4, 5
_GGUF_TYPE_FLOAT32, _GGUF_TYPE_BOOL, _GGUF_TYPE_STRING = 6, 7, 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64, _GGUF_TYPE_INT64, _GGUF_TYPE_FLOAT64 = 10, 11, 12

# Fixed-width metadata scalars, by GGUF type id.
_SCALAR_FMT: dict[int, str] = {
    _GGUF_TYPE_UINT8: "<B", _GGUF_TYPE_INT8: "<b",
    _GGUF_TYPE_UINT16: "<H", _GGUF_TYPE_INT16: "<h",
    _GGUF_TYPE_UINT32: "<I", _GGUF_TYPE_INT32: "<i",
    _GGUF_TYPE_FLOAT32: "<f", _GGUF_TYPE_BOOL: "<?",
    _GGUF_TYPE_UINT64: "<Q", _GGUF_TYPE_INT64: "<q",
    _GGUF_TYPE_FLOAT64: "<d",
}

#: Metadata types `general.alignment` may legitimately take.
_UINT_TYPES: frozenset[int] = frozenset(
    {_GGUF_TYPE_UINT8, _GGUF_TYPE_UINT16, _GGUF_TYPE_UINT32, _GGUF_TYPE_UINT64}
)

#: GGUF's default when `general.alignment` is absent, per gguf.h at the pinned commit.
_DEFAULT_ALIGNMENT = 32


def _read_uint_value(f, vtype: int, key: str) -> int:
    fmt = _SCALAR_FMT[vtype]
    return int(struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt), f"value for {key!r}"))[0])


# llama.cpp names the router `ffn_gate_inp`; `ffn_gate_inp_shexp` is the *shared*-expert gate,
# which is a different tensor and is audited separately -- I14 keeps shared experts out of the
# routed-expert accounting, but a quantized shared gate still perturbs the residual the router
# reads at the next layer, so it is reported rather than ignored.
ROUTER_SUFFIXES: tuple[str, ...] = ("ffn_gate_inp", "ffn_gate_inp_shexp")

# F32 is what an unquantized router is. F16/BF16 are accepted because several publishers ship the
# whole checkpoint in half precision and the router with it: that is the *native* dtype, not a
# quantization of it, so it is what the reference PyTorch forward in T3.2 computes with too, and
# the two sides still agree. Anything else is a lossy re-encoding of a matrix we cannot afford to
# re-encode.
ACCEPTABLE_ROUTER_TYPES: frozenset[str] = frozenset({"F32", "F16", "BF16"})

_MAX_SANE_NAME_LEN = 1 << 16
_MAX_SANE_DIMS = 4


class GGUFError(RuntimeError):
    """The file is not a GGUF we can parse."""


class RouterAuditError(RuntimeError):
    """The file parsed, but says nothing about a router."""


def _read_exact(f: BinaryIO, n: int, what: str) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise GGUFError(f"unexpected end of file reading {what}: wanted {n} bytes, got {len(buf)}")
    return buf


def _read_scalar(f: BinaryIO, fmt: str, what: str) -> int | float | bool:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(f, size, what))[0]


def _read_u32(f: BinaryIO, what: str) -> int:
    return int(_read_scalar(f, "<I", what))


def _read_u64(f: BinaryIO, what: str) -> int:
    return int(_read_scalar(f, "<Q", what))


def _read_string(f: BinaryIO, what: str) -> str:
    n = _read_u64(f, f"{what} length")
    if n > _MAX_SANE_NAME_LEN:
        # A corrupt length would otherwise turn into a multi-gigabyte read that looks like a hang.
        raise GGUFError(f"{what} claims an implausible length of {n} bytes; file is corrupt")
    raw = _read_exact(f, n, what)
    # Tensor names are ASCII in practice; metadata strings are not always. Never fail the audit
    # over an undecodable byte in a field we do not use.
    return raw.decode("utf-8", errors="replace")


def _skip_metadata_value(f: BinaryIO, vtype: int, what: str) -> None:
    """Advance past one metadata value without materializing it.

    The audit needs no metadata at all -- only the tensor table, which sits after it. But the
    metadata section is variable-length, so it must be *walked* to be skipped. Arrays are walked
    element-by-element rather than by computing a byte width, because an array of strings has no
    fixed stride.
    """
    if vtype == _GGUF_TYPE_STRING:
        _read_string(f, what)
        return
    if vtype == _GGUF_TYPE_ARRAY:
        etype = _read_u32(f, f"{what} array element type")
        n = _read_u64(f, f"{what} array length")
        if etype == _GGUF_TYPE_ARRAY:
            raise GGUFError(f"{what}: nested arrays are not valid GGUF")
        if etype == _GGUF_TYPE_STRING:
            for i in range(n):
                _read_string(f, f"{what}[{i}]")
            return
        fmt = _SCALAR_FMT.get(etype)
        if fmt is None:
            raise GGUFError(f"{what}: unknown array element type {etype}")
        # Fixed stride, so one seek. Arrays here are things like a 262K-entry tokenizer vocab;
        # reading them element-wise would dominate the runtime of the whole audit.
        f.seek(struct.calcsize(fmt) * n, 1)
        return
    fmt = _SCALAR_FMT.get(vtype)
    if fmt is None:
        raise GGUFError(f"{what}: unknown metadata value type {vtype}")
    f.seek(struct.calcsize(fmt), 1)


@dataclass(frozen=True)
class TensorInfo:
    """One row of the GGUF tensor table."""

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"UNKNOWN({self.ggml_type})")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass(frozen=True)
class GGUFHeader:
    """A parsed GGUF v3 header: the tensor table plus where the data section begins.

    ``data_offset`` and ``alignment`` are what turn a ``TensorInfo.offset`` (which is relative to
    the start of the data section, not the file) into a file position. The T1.2 gate never needs
    them — it reads no tensor data at all, which is why a 4.2 GB checkpoint audits in 0.2 s — but
    T3.3 does: recomputing the routing from the router weight is the one check that proves
    hidden.bin holds the tensor the manifest claims, and it has to actually read the weight.
    """

    tensors: tuple[TensorInfo, ...]
    data_offset: int
    alignment: int

    def by_name(self, name: str) -> TensorInfo:
        for t in self.tensors:
            if t.name == name:
                return t
        raise GGUFError(f"no tensor named {name!r}")

    def file_offset(self, tensor: TensorInfo) -> int:
        return self.data_offset + tensor.offset


def read_header(path: Path | str) -> GGUFHeader:
    """Parse a GGUF v3 header. Reads no tensor data."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = _read_exact(f, 4, "magic")
        if magic != GGUF_MAGIC:
            raise GGUFError(
                f"{path.name} is not a GGUF file: magic is {magic!r}, expected {GGUF_MAGIC!r}. "
                "A safetensors or PyTorch checkpoint cannot be audited here -- this gate is about "
                "the quantized artifact that llama.cpp will actually load."
            )
        version = _read_u32(f, "version")
        if version != 3:
            # v1/v2 differ in the width of the count fields, so parsing on would silently
            # misinterpret every offset rather than fail.
            raise GGUFError(
                f"{path.name} is GGUF v{version}; this parser implements v3 (GGUF_VERSION at the "
                "pinned llama.cpp commit). Re-convert the checkpoint."
            )
        n_tensors = _read_u64(f, "tensor count")
        n_kv = _read_u64(f, "metadata kv count")

        alignment = _DEFAULT_ALIGNMENT
        for i in range(n_kv):
            key = _read_string(f, f"metadata key {i}")
            vtype = _read_u32(f, f"metadata value type for {key!r}")
            if key == "general.alignment" and vtype in _UINT_TYPES:
                alignment = _read_uint_value(f, vtype, key)
            else:
                _skip_metadata_value(f, vtype, f"metadata value for {key!r}")

        infos: list[TensorInfo] = []
        for i in range(n_tensors):
            name = _read_string(f, f"tensor {i} name")
            n_dims = _read_u32(f, f"tensor {name!r} n_dims")
            if n_dims > _MAX_SANE_DIMS:
                raise GGUFError(f"tensor {name!r} claims {n_dims} dimensions; ggml allows 4")
            dims = tuple(_read_u64(f, f"tensor {name!r} dim {d}") for d in range(n_dims))
            ggml_type = _read_u32(f, f"tensor {name!r} type")
            offset = _read_u64(f, f"tensor {name!r} offset")
            infos.append(TensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset))

        # The data section starts at the first `alignment` boundary at or after the header.
        end = f.tell()
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError(f"general.alignment={alignment} is not a positive power of two")
        data_offset = (end + alignment - 1) // alignment * alignment

    return GGUFHeader(tensors=tuple(infos), data_offset=data_offset, alignment=alignment)


def read_tensor_infos(path: Path | str) -> list[TensorInfo]:
    """Parse a GGUF header and return its tensor table. Never reads the tensor data section."""
    return list(read_header(path).tensors)


def _is_router(name: str) -> bool:
    # Names look like `blk.11.ffn_gate_inp.weight`. Match on the dot-delimited component so that a
    # hypothetical `ffn_gate_inp_v2` is not swept in silently.
    parts = name.split(".")
    return any(p in ROUTER_SUFFIXES for p in parts)


@dataclass(frozen=True)
class RouterFinding:
    """One router tensor and its verdict."""

    tensor: TensorInfo
    ok: bool

    @property
    def is_shared_expert_gate(self) -> bool:
        return "ffn_gate_inp_shexp" in self.tensor.name.split(".")


@dataclass(frozen=True)
class ModelAudit:
    """The verdict for one checkpoint."""

    path: Path
    model_key: str
    findings: tuple[RouterFinding, ...]
    n_tensors_total: int

    @property
    def n_routers(self) -> int:
        return len(self.findings)

    @property
    def passed(self) -> bool:
        return all(fi.ok for fi in self.findings)

    @property
    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fi in self.findings:
            counts[fi.tensor.type_name] = counts.get(fi.tensor.type_name, 0) + 1
        return counts

    @property
    def offending(self) -> tuple[RouterFinding, ...]:
        return tuple(fi for fi in self.findings if not fi.ok)

    def summary(self) -> str:
        types = ", ".join(f"{t}x{n}" for t, n in sorted(self.type_counts.items()))
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict}  {self.model_key:<32} {self.n_routers:>3} router tensors  [{types}]"


def audit_gguf(path: Path | str, *, model_key: str | None = None) -> ModelAudit:
    """Audit one GGUF. Raises `RouterAuditError` if it contains no router tensors at all."""
    path = Path(path)
    infos = read_tensor_infos(path)
    routers = [t for t in infos if _is_router(t.name)]
    if not routers:
        # Not a PASS. A dense checkpoint, a wrong file, or a converter that renamed the tensor all
        # land here, and all three mean the trace this gate is protecting cannot be collected.
        raise RouterAuditError(
            f"{path.name} contains no {' or '.join(ROUTER_SUFFIXES)} tensor among its "
            f"{len(infos)} tensors. Either this is not an MoE checkpoint, or the converter used a "
            "name this audit does not know -- check with `llama-gguf` before assuming it passed."
        )
    findings = tuple(
        RouterFinding(tensor=t, ok=t.type_name in ACCEPTABLE_ROUTER_TYPES) for t in routers
    )
    return ModelAudit(
        path=path,
        model_key=model_key or path.stem,
        findings=findings,
        n_tensors_total=len(infos),
    )


CSV_COLUMNS = (
    "model_key", "gguf_file", "tensor", "dims", "ggml_type", "n_elements",
    "is_shared_expert_gate", "verdict",
)


def write_csv(audits: list[ModelAudit], out_path: Path | str) -> Path:
    """Write the per-tensor audit table. One row per router tensor, not per model.

    Per-tensor because a mixed checkpoint is a real outcome: `llama-quantize` applies per-tensor
    overrides, so a partially-corrected requantization shows up as some layers F32 and some Q4_K,
    and a per-model summary row would average that into a single misleading verdict.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for a in audits:
            for fi in a.findings:
                t = fi.tensor
                w.writerow([
                    a.model_key, a.path.name, t.name,
                    "x".join(str(d) for d in t.dims), t.type_name, t.n_elements,
                    "true" if fi.is_shared_expert_gate else "false",
                    "PASS" if fi.ok else "FAIL",
                ])
    return out_path


def _iter_paths(patterns: list[str]) -> Iterator[Path]:
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            yield from sorted(p.glob("*.gguf"))
        elif any(ch in pat for ch in "*?["):
            # The shell does not expand globs for us on Windows.
            yield from sorted(Path().glob(pat))
        else:
            yield p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="router_audit",
        description="T1.2 router dtype audit (GATE). Reads GGUF headers only.",
    )
    ap.add_argument("paths", nargs="+", help="GGUF files, directories, or globs")
    ap.add_argument("-o", "--out", default="results/router_dtype_audit.csv",
                    help="CSV output path (default: %(default)s)")
    ap.add_argument("--no-csv", action="store_true", help="print only, write nothing")
    args = ap.parse_args(argv)

    paths = list(_iter_paths(args.paths))
    if not paths:
        print("no GGUF files matched", file=sys.stderr)
        return 2

    audits: list[ModelAudit] = []
    errors: list[str] = []
    for p in paths:
        try:
            audits.append(audit_gguf(p))
        except (GGUFError, RouterAuditError, OSError) as exc:
            errors.append(f"{p}: {exc}")

    for a in audits:
        print(a.summary())
        for fi in a.offending:
            print(f"        {fi.tensor.name}  {fi.tensor.type_name}  "
                  f"({'x'.join(str(d) for d in fi.tensor.dims)})")

    if audits and not args.no_csv:
        out = write_csv(audits, args.out)
        print(f"\nwrote {out}")

    for e in errors:
        print(f"ERROR  {e}", file=sys.stderr)

    failed = [a for a in audits if not a.passed]
    if failed or errors:
        print(
            f"\nGATE FAILED: {len(failed)} of {len(audits)} checkpoint(s) have a quantized router"
            f"{f', {len(errors)} unreadable' if errors else ''}.\n"
            "Remedy (TASKS.md T1.2): requantize with the router excluded, e.g.\n"
            "  llama-quantize --tensor-type ffn_gate_inp=f32 <in.gguf> <out.gguf> Q4_K_M\n"
            "Do not collect traces from a checkpoint that failed here: near-ties at the top-k "
            "boundary flip systematically, and T3.2 will fail against our own trace.",
            file=sys.stderr,
        )
        return 1

    print(f"\nGATE PASSED: {len(audits)} checkpoint(s), all routers in "
          f"{sorted(ACCEPTABLE_ROUTER_TYPES)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
