"""T3.3 tests — recomputing the routing from the GGUF router weight.

The module's value is that it is *exact*: no fitting, no split, no sample-size floor. So the
tests are mostly about what it must refuse to excuse. A check that quietly forgives disagreement
is worse than no check, because the four things it establishes at once (hidden.bin is the router
input, rows are aligned, topk.bin was de-strided, layers are mapped right) are each silent
elsewhere.

The fixtures build a GGUF header by hand and a matching shard, so a "correct" trace here is one
constructed to satisfy the router equation, not one blessed by the code under test.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.capture.router_audit import GGUFError, read_header
from src.traces.format import FLAG_HIDDEN_CAPTURED, TOKEN_DTYPE, write_manifest
from src.traces.router_check import (
    RouterCheckError,
    check_shard,
    main,
    read_router_weight,
)

N_LAYERS, N_EXPERTS, TOP_K, HIDDEN = 3, 8, 2, 16
N_TOKENS, STRIDE = 40, 4

_GGUF_TYPE_UINT32 = 4
_TYPE_F32, _TYPE_F16 = 0, 1


# -- a minimal GGUF v3 writer -------------------------------------------------------------------


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return _u64(len(raw)) + raw


def write_gguf(path: Path, weights: dict[str, np.ndarray], *, ggml_type: int = _TYPE_F32,
               alignment: int = 32) -> Path:
    """Write a GGUF v3 file holding exactly ``weights``, in insertion order.

    Real enough to exercise the offset arithmetic the check depends on: the data section starts
    at the first `alignment` boundary after the header, and each tensor's recorded offset is
    relative to that, not to the file.
    """
    header = bytearray(b"GGUF" + _u32(3) + _u64(len(weights)) + _u64(1))
    header += _string("general.alignment") + _u32(_GGUF_TYPE_UINT32) + _u32(alignment)

    blobs: list[bytes] = []
    offset = 0
    for name, array in weights.items():
        stored = array.astype(np.float32 if ggml_type == _TYPE_F32 else np.float16)
        # GGUF dims are ne[], fastest-varying first: (hidden_dim, n_experts) for a router.
        header += _string(name) + _u32(2)
        header += _u64(stored.shape[1]) + _u64(stored.shape[0])
        header += _u32(ggml_type) + _u64(offset)
        blob = stored.tobytes()
        pad = (-len(blob)) % alignment
        blobs.append(blob + b"\0" * pad)
        offset += len(blob) + pad

    pad = (-len(header)) % alignment
    path.write_bytes(bytes(header) + b"\0" * pad + b"".join(blobs))
    return path


# -- fixtures -----------------------------------------------------------------------------------


@pytest.fixture
def routers() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(3)
    return {
        layer: rng.normal(0, 1, size=(N_EXPERTS, HIDDEN)).astype(np.float32)
        for layer in range(N_LAYERS)
    }


@pytest.fixture
def gguf(tmp_path, routers) -> Path:
    return write_gguf(
        tmp_path / "unit.gguf",
        {f"blk.{layer}.ffn_gate_inp.weight": w for layer, w in routers.items()},
    )


def build_shard(
    shard_dir: Path,
    routers: dict[int, np.ndarray],
    *,
    layer_map: list[int] | None = None,
    seed: int = 5,
) -> dict:
    """A shard whose ``topk.bin`` is, by construction, what the routers imply for ``hidden.bin``.

    Built forwards from the weights, so passing the check means the check agrees with arithmetic
    done independently of it — not that both sides share a bug.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    tokens = np.zeros(N_TOKENS, dtype=TOKEN_DTYPE)
    tokens["token_id"] = rng.integers(0, 1000, size=N_TOKENS, dtype=np.uint32)
    tokens["doc_id"] = np.arange(N_TOKENS, dtype=np.uint32) // 10
    tokens["pos_in_doc"] = np.arange(N_TOKENS, dtype=np.uint32) % 10
    captured = np.arange(0, N_TOKENS, STRIDE, dtype=np.int64)
    tokens["flags"][captured] |= FLAG_HIDDEN_CAPTURED

    hidden = rng.normal(0, 1, size=(captured.size, N_LAYERS, HIDDEN)).astype(np.float16)

    topk = rng.integers(0, N_EXPERTS, size=(N_TOKENS, N_LAYERS, TOP_K)).astype(np.int32)
    for trace_layer in range(N_LAYERS):
        model_layer = layer_map[trace_layer] if layer_map else trace_layer
        logits = np.asarray(hidden[:, trace_layer, :], dtype=np.float32) @ routers[model_layer].T
        topk[captured, trace_layer, :] = np.argsort(-logits, axis=1, kind="stable")[:, :TOP_K]

    (shard_dir / "tokens.bin").write_bytes(tokens.tobytes())
    (shard_dir / "topk.bin").write_bytes(topk.tobytes())
    (shard_dir / "hidden.bin").write_bytes(hidden.tobytes())
    (shard_dir / "hidden_index.bin").write_bytes(
        (tokens["doc_id"][captured].astype(np.uint32) * 64 + tokens["pos_in_doc"][captured])
        .astype(np.uint32)
        .tobytes()
    )
    logits_stream = rng.normal(0, 1, size=(N_TOKENS, N_LAYERS, N_EXPERTS)).astype(np.float16)
    (shard_dir / "logits.bin").write_bytes(logits_stream.tobytes())

    manifest = {
        "model": "unit-moe",
        "checkpoint_status": "base",
        "gguf_sha256": "a" * 64,
        "llama_cpp_commit": "abc123",
        "run_config_sha256": "b" * 64,
        "quant": "Q4_K_M",
        "router_dtype": "F32",
        "logit_tensor_used": "ffn_moe_probs",
        "corpus": "unit",
        "shard_id": 0,
        "shard_doc_range": [0, 4],
        "n_tokens": N_TOKENS,
        "n_moe_layers": N_LAYERS,
        "n_experts": N_EXPERTS,
        "top_k": TOP_K,
        "hidden_dim": HIDDEN,
        "hidden_subsample_n": int(captured.size),
        "n_captured": int(captured.size),
        "hidden_stride": STRIDE,
        "index_scheme": "doc_id*n_ctx+pos_in_doc",
        "index_doc_span": 64,
        "capture_flags": {"pre_topk": True, "pre_norm": True, "topk_captured": True},
        "layer_index_map": layer_map or list(range(N_LAYERS)),
        "device_plan": {"n_gpu": 1, "split_mode": "layer", "tensor_split": None},
        "file_sha256": {},
        "collected_utc": "2026-08-18T00:00:00+00:00",
    }
    write_manifest(shard_dir, manifest)
    return manifest


@pytest.fixture
def shard(tmp_path, routers) -> Path:
    sd = tmp_path / "shard_00000"
    build_shard(sd, routers)
    return sd


# -- reading the weight ---------------------------------------------------------------------------


def test_the_weight_is_read_at_the_right_offset_and_orientation(gguf, routers):
    """Offsets are relative to the data section, and GGUF dims are ne[] — both easy to invert."""
    for layer, want in routers.items():
        got = read_router_weight(gguf, layer)
        assert got.shape == (N_EXPERTS, HIDDEN)
        np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_a_second_tensor_does_not_shift_the_first(tmp_path, routers):
    """Every tensor after the first is where its recorded offset says, or nothing matches."""
    path = write_gguf(
        tmp_path / "two.gguf",
        {
            "blk.0.ffn_gate_inp.weight": routers[0],
            "blk.1.ffn_gate_inp.weight": routers[1],
        },
    )
    np.testing.assert_allclose(read_router_weight(path, 1), routers[1], rtol=0, atol=0)


def test_an_f16_router_is_read_and_widened(tmp_path, routers):
    path = write_gguf(tmp_path / "f16.gguf", {"blk.0.ffn_gate_inp.weight": routers[0]},
                      ggml_type=_TYPE_F16)
    got = read_router_weight(path, 0)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, routers[0].astype(np.float16), rtol=0, atol=0)


def test_a_missing_router_is_an_error_not_a_zero_matrix(gguf):
    with pytest.raises(GGUFError, match="no tensor named"):
        read_router_weight(gguf, 99)


def test_a_non_default_alignment_is_honoured(tmp_path, routers):
    """`general.alignment` is metadata, not a constant; assuming 32 corrupts every offset."""
    path = write_gguf(tmp_path / "aligned.gguf", {"blk.0.ffn_gate_inp.weight": routers[0]},
                      alignment=64)
    assert read_header(path).alignment == 64
    np.testing.assert_allclose(read_router_weight(path, 0), routers[0], rtol=0, atol=0)


# -- the check itself -----------------------------------------------------------------------------


def test_a_correctly_aligned_shard_reproduces_the_routing_exactly(shard, gguf):
    report = check_shard(shard, gguf)
    assert report.ok, [a.to_dict() for a in report.layers]
    assert len(report.layers) == N_LAYERS
    for a in report.layers:
        assert a.set_agreement == 1.0 and a.exact_match == 1.0
        assert a.n_unexplained == 0


def test_hidden_rows_shifted_by_one_token_are_caught(shard, gguf):
    """The failure this exists for: a relabelled feature, silent in every other check.

    The file is the right size, every index is in range and distinct, the stride rule holds —
    only the *contents* belong to a neighbouring token.
    """
    n_captured = json.loads((shard / "manifest.json").read_text())["n_captured"]
    hidden = np.fromfile(shard / "hidden.bin", dtype=np.float16).reshape(
        n_captured, N_LAYERS, HIDDEN
    )
    np.roll(hidden, 1, axis=0).tofile(shard / "hidden.bin")
    report = check_shard(shard, gguf)
    assert not report.ok
    assert report.worst.set_agreement < 0.5


def test_a_contiguous_read_of_the_strided_topk_view_is_caught(shard, gguf):
    """I12 — in-range, distinct, wrong. No label check can see it; this one can."""
    topk = np.fromfile(shard / "topk.bin", dtype=np.int32).reshape(N_TOKENS, N_LAYERS, TOP_K)
    topk[:, 0, :] = np.roll(topk[:, 0, :], 1, axis=0)
    topk.tofile(shard / "topk.bin")
    report = check_shard(shard, gguf)
    assert not report.ok
    assert min(a.set_agreement for a in report.layers) < 1.0


def test_an_off_by_one_layer_map_is_caught(tmp_path, routers, gguf):
    """T3.5: DeepSeek's dense first layer makes trace layer 0 model layer 1. Getting it backwards
    produces a complete, plausible trace with every per-depth result shifted."""
    sd = tmp_path / "shifted"
    build_shard(sd, routers, layer_map=[0, 1, 2])
    manifest = json.loads((sd / "manifest.json").read_text())
    manifest["layer_index_map"] = [1, 2, 0]  # claim a mapping the bytes do not have
    (sd / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = check_shard(sd, gguf)
    assert not report.ok


def test_a_declared_layer_map_that_is_honoured_passes(tmp_path, routers, gguf):
    """The mirror of the previous test: a non-identity map is fine when the bytes match it."""
    sd = tmp_path / "mapped"
    build_shard(sd, routers, layer_map=[2, 0, 1])
    assert check_shard(sd, gguf).ok


def test_ties_within_the_fp16_bound_are_forgiven_but_counted(tmp_path, routers, gguf):
    """A k/(k+1) swap the stored precision cannot resolve is not evidence about the tensor.

    Constructed rather than hoped for: two experts are given identical weight rows, so their
    logits are equal to the bit and the order between them is arbitrary.
    """
    tied = {layer: w.copy() for layer, w in routers.items()}
    for w in tied.values():
        w[1] = w[0]
    path = write_gguf(
        tmp_path / "tied.gguf",
        {f"blk.{i}.ffn_gate_inp.weight": w for i, w in tied.items()},
    )
    sd = tmp_path / "tied_shard"
    build_shard(sd, tied, seed=9)

    topk = np.fromfile(sd / "topk.bin", dtype=np.int32).reshape(N_TOKENS, N_LAYERS, TOP_K)
    swapped = np.where(topk == 0, 1, np.where(topk == 1, 0, topk))
    swapped.tofile(sd / "topk.bin")

    # A low threshold on purpose: the fixture has 10 captured rows, so a single forgiven tie
    # moves set_agreement to 0.95 — arithmetic of the sample size, not of the tie handling,
    # which is what this test is about.
    report = check_shard(sd, path, threshold=0.9)
    assert report.ok, [a.to_dict() for a in report.layers]
    assert any(a.n_mismatched_at_tie > 0 for a in report.layers), "the tie must be counted"
    assert all(a.n_unexplained == 0 for a in report.layers)


def test_a_handful_of_wrong_rows_fails_even_though_agreement_rounds_to_one(tmp_path, routers, gguf):
    """Set agreement alone is not enough at scale: 3 bad rows in 10k still reads 0.9999."""
    sd = tmp_path / "mostly_right"
    build_shard(sd, routers)
    topk = np.fromfile(sd / "topk.bin", dtype=np.int32).reshape(N_TOKENS, N_LAYERS, TOP_K)
    topk[0, 0, 0] = (topk[0, 0, 0] + 3) % N_EXPERTS  # one token, one layer, far from any tie
    topk.tofile(sd / "topk.bin")

    report = check_shard(sd, gguf, threshold=0.5)
    assert report.worst.set_agreement > 0.5, "the threshold alone would pass this"
    assert not report.ok, "an unexplained mismatch must fail regardless of the average"


def test_a_quantized_router_is_refused_rather_than_dequantized(tmp_path, routers):
    """T1.2 gates the dtype so this can be a plain read; coping here would bypass the gate."""
    path = tmp_path / "quant.gguf"
    write_gguf(path, {"blk.0.ffn_gate_inp.weight": routers[0]})
    raw = bytearray(path.read_bytes())
    at = raw.find(b"blk.0.ffn_gate_inp.weight")
    # name + n_dims + 2 dims, then the type field.
    type_at = at + len("blk.0.ffn_gate_inp.weight") + 4 + 16
    raw[type_at : type_at + 4] = struct.pack("<I", 12)  # Q4_K
    path.write_bytes(bytes(raw))
    with pytest.raises(RouterCheckError, match="T1.2 gate"):
        read_router_weight(path, 0)


def test_a_shard_with_no_captured_rows_cannot_be_checked_and_says_so(tmp_path, routers, gguf):
    sd = tmp_path / "empty"
    build_shard(sd, routers)
    manifest = json.loads((sd / "manifest.json").read_text())
    manifest["n_captured"] = 0
    (sd / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RouterCheckError, match="captured no hidden states"):
        check_shard(sd, gguf)


def test_a_flag_count_disagreeing_with_hidden_rows_is_refused(shard, gguf):
    tokens = np.fromfile(shard / "tokens.bin", dtype=TOKEN_DTYPE)
    tokens["flags"][1] |= FLAG_HIDDEN_CAPTURED
    tokens.tofile(shard / "tokens.bin")
    with pytest.raises(RouterCheckError, match="flagged but hidden.bin"):
        check_shard(shard, gguf)


def test_only_the_requested_layers_are_checked(shard, gguf):
    report = check_shard(shard, gguf, layers=[1])
    assert [a.layer for a in report.layers] == [1]


def test_a_layer_outside_the_trace_is_an_error(shard, gguf):
    with pytest.raises(RouterCheckError, match="outside the trace"):
        check_shard(shard, gguf, layers=[N_LAYERS])


def test_the_sample_spans_the_shard_rather_than_taking_a_prefix(shard, gguf):
    """A misalignment starting halfway through would hide entirely inside a prefix."""
    report = check_shard(shard, gguf, max_rows=4)
    assert all(a.n_rows == 4 for a in report.layers)

    n_captured = json.loads((shard / "manifest.json").read_text())["n_captured"]
    hidden = np.fromfile(shard / "hidden.bin", dtype=np.float16).reshape(
        n_captured, N_LAYERS, HIDDEN
    )
    hidden[n_captured - 1] = hidden[0]  # corrupt only the last captured row
    hidden.tofile(shard / "hidden.bin")
    assert not check_shard(shard, gguf, max_rows=4).ok


# -- CLI ------------------------------------------------------------------------------------------


def test_the_cli_exits_zero_and_writes_json_on_a_good_shard(shard, gguf, tmp_path, capsys):
    out = tmp_path / "t33.json"
    assert main([str(shard), "--gguf", str(gguf), "--json", str(out)]) == 0
    assert "T3.3 PASSED" in capsys.readouterr().out
    payload = json.loads(out.read_text())
    assert payload["ok"] is True and len(payload["layers"]) == N_LAYERS


def test_the_cli_exits_one_so_a_bad_shard_never_passes_silently(shard, gguf, capsys):
    topk = np.fromfile(shard / "topk.bin", dtype=np.int32).reshape(N_TOKENS, N_LAYERS, TOP_K)
    topk[:, 1, :] = (topk[:, 1, :] + 4) % N_EXPERTS
    topk.tofile(shard / "topk.bin")
    assert main([str(shard), "--gguf", str(gguf)]) == 1
    assert "T3.3 FAILED" in capsys.readouterr().out


def test_the_cli_exits_two_when_the_check_cannot_run_at_all(shard, tmp_path, capsys):
    """"Could not run" and "ran and disagreed" are different outcomes and get different codes."""
    missing = tmp_path / "nope.gguf"
    missing.write_bytes(b"not a gguf at all")
    assert main([str(shard), "--gguf", str(missing)]) == 2
    assert "could not run" in capsys.readouterr().out
