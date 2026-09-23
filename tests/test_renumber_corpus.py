"""Tests for scripts/renumber_corpus.py — the mixed-v2 doc_id renumber transform."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.renumber_corpus import renumber

# One line per corpus doc, written exactly as src.corpus.build.write_corpus does
# (json.dumps(row, ensure_ascii=True, sort_keys=False), doc_id first). doc_ids are scattered across
# shards, the shape the fetch-order numbering produces and the bug this transform fixes.
_ROWS = [
    {"doc_id": 437, "text": "alpha é", "domain": "prose", "lang": "en", "source": "s",
     "n_tokens_ref": 3, "split": "train", "shard_id": 0},
    {"doc_id": 12, "text": "beta \"q\"", "domain": "code", "lang": "en", "source": "s",
     "n_tokens_ref": 2, "split": "val", "shard_id": 0},
    {"doc_id": 900, "text": "gamma", "domain": "prose", "lang": "en", "source": "s",
     "n_tokens_ref": 1, "split": "test", "shard_id": 1},
]


def _write(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True, sort_keys=False) + "\n" for r in rows),
        encoding="utf-8", newline="",
    )
    return path


def test_renumber_sets_doc_id_to_line_index_and_keeps_everything_else(tmp_path):
    inp = _write(tmp_path / "v1.jsonl", _ROWS)
    out = tmp_path / "v2.jsonl"
    st = renumber(inp, out)
    assert st["n_docs"] == 3

    got = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["doc_id"] for r in got] == [0, 1, 2]  # write-order line index
    for original, new in zip(_ROWS, got):
        a, b = dict(original), dict(new)
        a.pop("doc_id"), b.pop("doc_id")
        assert a == b  # every field but doc_id is unchanged


def test_renumber_is_byte_identical_except_doc_id(tmp_path):
    inp = _write(tmp_path / "v1.jsonl", _ROWS)
    out = renumber(inp, tmp_path / "v2.jsonl")  # noqa: F841
    in_lines = (tmp_path / "v1.jsonl").read_text(encoding="utf-8").splitlines()
    out_lines = (tmp_path / "v2.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(in_lines) == len(out_lines)
    for i, (a, b) in enumerate(zip(in_lines, out_lines)):
        # Only the doc_id value differs; splice the original doc_id back and the bytes must match.
        spliced = b.replace(f'"doc_id": {i}', f'"doc_id": {_ROWS[i]["doc_id"]}', 1)
        assert spliced == a


def test_renumber_refuses_wrong_input_sha(tmp_path):
    inp = _write(tmp_path / "v1.jsonl", _ROWS)
    with pytest.raises(SystemExit, match="refusing to transform the wrong corpus"):
        renumber(inp, tmp_path / "v2.jsonl", expect_input_sha="0" * 64)


def test_renumber_aborts_if_source_not_canonical_json(tmp_path):
    # A line reformatted (extra spaces) is not what write_corpus emits; the transform must refuse
    # rather than silently reserialize and change other fields.
    p = tmp_path / "v1.jsonl"
    p.write_text('{"doc_id": 5,  "text": "x", "domain": "prose", "lang": "en", "source": "s", '
                 '"n_tokens_ref": 1, "split": "train", "shard_id": 0}\n', encoding="utf-8",
                 newline="")
    with pytest.raises(SystemExit, match="does not match the source bytes"):
        renumber(p, tmp_path / "v2.jsonl")
