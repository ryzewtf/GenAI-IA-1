"""Tests for T1.2, the router dtype gate (`src/capture/router_audit.py`).

The gate's whole value is that it FAILS on a quantized router. So the tests build GGUF headers
byte by byte -- including deliberately quantized ones -- rather than depending on a checkpoint
being present, which would make the suite unrunnable on Kaggle and on any machine that has not
downloaded 4 GB.

`_gguf` below is a minimal GGUF v3 writer matching `llama_cpp_pull/ggml/include/gguf.h`. It
writes headers only, exactly as the parser reads them.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from src.capture.router_audit import (
    ACCEPTABLE_ROUTER_TYPES,
    GGUFError,
    ModelAudit,
    RouterAuditError,
    audit_gguf,
    main,
    read_tensor_infos,
    write_csv,
)

# ggml_type ids, from ggml.h.
F32, F16, Q4_K, BF16, Q6_K = 0, 1, 12, 30, 14

# GGUF metadata value type ids, from gguf.h.
T_UINT32, T_STRING, T_ARRAY = 4, 8, 9


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(
    path: Path,
    tensors: list[tuple[str, tuple[int, ...], int]],
    *,
    kv: list[tuple[str, int, bytes]] | None = None,
    version: int = 3,
    magic: bytes = b"GGUF",
) -> Path:
    """Write a GGUF header. `tensors` is (name, dims, ggml_type); `kv` is (key, type, payload)."""
    kv = kv or []
    out = bytearray()
    out += magic
    out += struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(kv))
    for key, vtype, payload in kv:
        out += _s(key) + struct.pack("<I", vtype) + payload
    offset = 0
    for name, dims, ttype in tensors:
        out += _s(name)
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ttype)
        out += struct.pack("<Q", offset)
        offset += 4096
    path.write_bytes(bytes(out))
    return path


def _moe(tmp: Path, router_type: int, *, n_layers: int = 4, name: str = "m.gguf") -> Path:
    """A plausible MoE checkpoint: quantized experts, router dtype under test."""
    tensors: list[tuple[str, tuple[int, ...], int]] = [("token_embd.weight", (2048, 50304), Q4_K)]
    for i in range(n_layers):
        tensors.append((f"blk.{i}.ffn_gate_inp.weight", (2048, 64), router_type))
        tensors.append((f"blk.{i}.ffn_gate_exps.weight", (2048, 1024, 64), Q4_K))
        tensors.append((f"blk.{i}.attn_q.weight", (2048, 2048), Q6_K))
    tensors.append(("output.weight", (2048, 50304), Q6_K))
    return _gguf(tmp / name, tensors)


# --- header parsing -------------------------------------------------------------------------


def test_the_tensor_table_survives_a_metadata_section_it_does_not_understand(tmp_path):
    """Metadata is skipped, not parsed. A 262K-entry tokenizer array must not derail the walk."""
    vocab = struct.pack("<I", T_STRING) + struct.pack("<Q", 3) + _s("a") + _s("bb") + _s("ccc")
    ints = struct.pack("<I", T_UINT32) + struct.pack("<Q", 5) + struct.pack("<5I", 1, 2, 3, 4, 5)
    kv = [
        ("general.architecture", T_STRING, _s("olmoe")),
        ("olmoe.expert_count", T_UINT32, struct.pack("<I", 64)),
        ("tokenizer.ggml.tokens", T_ARRAY, vocab),
        ("tokenizer.ggml.token_type", T_ARRAY, ints),
    ]
    p = _gguf(tmp_path / "kv.gguf", [("blk.0.ffn_gate_inp.weight", (2048, 64), F32)], kv=kv)
    infos = read_tensor_infos(p)
    # If any metadata value were mis-skipped, the reader would land mid-stream and either raise or
    # produce a garbage name -- so an exact-name assertion is the real check here.
    assert [t.name for t in infos] == ["blk.0.ffn_gate_inp.weight"]
    assert infos[0].dims == (2048, 64)
    assert infos[0].type_name == "F32"


def test_a_non_gguf_file_is_rejected_by_magic_rather_than_misparsed(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b'{"__metadata__":{}}' + b"\x00" * 64)
    with pytest.raises(GGUFError, match="not a GGUF"):
        read_tensor_infos(p)


def test_an_older_gguf_version_is_refused_because_its_count_fields_are_narrower(tmp_path):
    p = _gguf(tmp_path / "v2.gguf", [("blk.0.ffn_gate_inp.weight", (8, 8), F32)], version=2)
    with pytest.raises(GGUFError, match="v2"):
        read_tensor_infos(p)


def test_a_truncated_header_raises_instead_of_returning_a_short_tensor_table(tmp_path):
    """Silently returning the tensors that happened to fit would PASS a corrupt file."""
    p = _moe(tmp_path, F32, n_layers=4)
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(GGUFError, match="unexpected end of file"):
        read_tensor_infos(p)


def test_an_implausible_string_length_fails_fast_rather_than_attempting_a_huge_read(tmp_path):
    p = tmp_path / "corrupt.gguf"
    p.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 0)
        + struct.pack("<Q", 1 << 40)  # tensor name claims a terabyte
    )
    with pytest.raises(GGUFError, match="implausible length"):
        read_tensor_infos(p)


# --- the gate itself ------------------------------------------------------------------------


@pytest.mark.parametrize("ttype,name", [(F32, "F32"), (F16, "F16"), (BF16, "BF16")])
def test_an_unquantized_or_natively_half_precision_router_passes(tmp_path, ttype, name):
    a = audit_gguf(_moe(tmp_path, ttype))
    assert a.passed
    assert a.n_routers == 4
    assert a.type_counts == {name: 4}
    assert name in ACCEPTABLE_ROUTER_TYPES


def test_a_quantized_router_fails_the_gate_and_names_the_offending_tensors(tmp_path):
    """The reason this module exists: Q4_K routing weights flip top-k near-ties systematically."""
    a = audit_gguf(_moe(tmp_path, Q4_K))
    assert not a.passed
    assert len(a.offending) == 4
    assert {fi.tensor.type_name for fi in a.offending} == {"Q4_K"}
    assert "blk.0.ffn_gate_inp.weight" in {fi.tensor.name for fi in a.offending}


def test_a_partially_requantized_checkpoint_fails_on_the_layers_that_are_still_quantized(tmp_path):
    """`llama-quantize --tensor-type` is applied per tensor, so a half-done fix is a real outcome.

    A per-model summary verdict would average this into something reassuring; the gate must key
    on the worst tensor, not the majority.
    """
    tensors = [
        ("blk.0.ffn_gate_inp.weight", (2048, 64), F32),
        ("blk.1.ffn_gate_inp.weight", (2048, 64), F32),
        ("blk.2.ffn_gate_inp.weight", (2048, 64), Q4_K),  # missed by the override
        ("blk.3.ffn_gate_inp.weight", (2048, 64), F32),
    ]
    a = audit_gguf(_gguf(tmp_path / "half.gguf", tensors))
    assert not a.passed
    assert [fi.tensor.name for fi in a.offending] == ["blk.2.ffn_gate_inp.weight"]
    assert a.type_counts == {"F32": 3, "Q4_K": 1}


def test_a_checkpoint_with_no_router_tensor_is_an_error_and_never_a_pass(tmp_path):
    """A dense model, a wrong file, or a renaming converter must not read as 'nothing to fix'."""
    dense = [("token_embd.weight", (2048, 50304), Q4_K), ("blk.0.ffn_gate.weight", (2048, 8192), Q4_K)]
    with pytest.raises(RouterAuditError, match="no ffn_gate_inp"):
        audit_gguf(_gguf(tmp_path / "dense.gguf", dense))


def test_the_shared_expert_gate_is_audited_and_flagged_as_distinct_from_the_routed_router(tmp_path):
    """I14 keeps shared experts out of the routed accounting, but a quantized shared gate still
    perturbs the residual the next layer's router reads -- so it is audited, and labelled."""
    tensors = [
        ("blk.0.ffn_gate_inp.weight", (2048, 64), F32),
        ("blk.0.ffn_gate_inp_shexp.weight", (2048, 1), Q4_K),
    ]
    a = audit_gguf(_gguf(tmp_path / "shexp.gguf", tensors))
    assert a.n_routers == 2
    assert not a.passed
    shared = [fi for fi in a.findings if fi.is_shared_expert_gate]
    routed = [fi for fi in a.findings if not fi.is_shared_expert_gate]
    assert len(shared) == 1 and len(routed) == 1
    assert routed[0].ok and not shared[0].ok


def test_a_tensor_merely_named_like_the_router_is_not_swept_in(tmp_path):
    """Substring matching on 'ffn_gate_inp' would silently audit unrelated tensors."""
    tensors = [
        ("blk.0.ffn_gate_inp.weight", (2048, 64), F32),
        ("blk.0.ffn_gate_inp_v2_experimental.weight", (2048, 64), Q4_K),
    ]
    a = audit_gguf(_gguf(tmp_path / "lookalike.gguf", tensors))
    assert a.n_routers == 1
    assert a.passed


def test_an_unknown_ggml_type_is_reported_by_number_and_still_fails(tmp_path):
    """A future quantization we have never seen must not default to acceptable."""
    a = audit_gguf(_gguf(tmp_path / "future.gguf", [("blk.0.ffn_gate_inp.weight", (8, 8), 99)]))
    assert not a.passed
    assert a.findings[0].tensor.type_name == "UNKNOWN(99)"


# --- reporting ------------------------------------------------------------------------------


def test_the_csv_has_one_row_per_router_tensor_with_its_verdict(tmp_path):
    audits = [
        audit_gguf(_moe(tmp_path, F32, n_layers=2, name="good.gguf"), model_key="good"),
        audit_gguf(_moe(tmp_path, Q4_K, n_layers=2, name="bad.gguf"), model_key="bad"),
    ]
    out = write_csv(audits, tmp_path / "results" / "router_dtype_audit.csv")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("model_key,gguf_file,tensor,dims,ggml_type")
    assert len(lines) == 1 + 4  # header + 2 routers per model
    body = "\n".join(lines[1:])
    assert body.count("PASS") == 2 and body.count("FAIL") == 2
    assert "2048x64" in body  # dims are rendered, so a wrong-shaped router is visible


def test_n_elements_is_the_product_of_dims(tmp_path):
    a = audit_gguf(_gguf(tmp_path / "d.gguf", [("blk.0.ffn_gate_inp.weight", (2048, 64), F32)]))
    assert a.findings[0].tensor.n_elements == 2048 * 64


def test_the_cli_exits_nonzero_when_any_checkpoint_fails(tmp_path, capsys):
    """The gate is meant to be usable as `&&` in a setup script, so the exit code is the contract."""
    _moe(tmp_path, F32, n_layers=1, name="ok.gguf")
    assert main([str(tmp_path / "ok.gguf"), "--no-csv"]) == 0

    _moe(tmp_path, Q4_K, n_layers=1, name="nope.gguf")
    assert main([str(tmp_path / "nope.gguf"), "--no-csv"]) == 1
    err = capsys.readouterr().err
    assert "GATE FAILED" in err
    assert "llama-quantize" in err  # the remedy travels with the failure


def test_the_cli_reports_an_unreadable_file_as_a_failure_rather_than_skipping_it(tmp_path, capsys):
    _moe(tmp_path, F32, n_layers=1, name="ok.gguf")
    (tmp_path / "junk.gguf").write_bytes(b"not a gguf at all")
    rc = main([str(tmp_path), "--no-csv"])
    assert rc == 1, "one good and one unreadable file must not report success"
    assert "ERROR" in capsys.readouterr().err


def test_the_cli_accepts_a_directory_and_audits_every_gguf_in_it(tmp_path, capsys):
    _moe(tmp_path, F32, n_layers=1, name="a.gguf")
    _moe(tmp_path, F32, n_layers=1, name="b.gguf")
    assert main([str(tmp_path), "--no-csv"]) == 0
    assert "2 checkpoint(s)" in capsys.readouterr().out


def test_summary_line_is_greppable_for_the_verdict(tmp_path):
    a: ModelAudit = audit_gguf(_moe(tmp_path, F32, n_layers=3), model_key="olmoe-0125-instruct")
    line = a.summary()
    assert line.startswith("PASS")
    assert "olmoe-0125-instruct" in line
    assert "F32x3" in line
