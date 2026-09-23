"""Renumber a published corpus's ``doc_id`` to its write-order line index — the mixed-v2 transform.

WHY. The trace index is ``hidden_index = doc_id * n_ctx + pos_in_doc`` and every reader
(``TraceReader.captured_rows`` → ``_hidden_token_ids``, plan T2.3) requires it to ascend across the
concatenated shards. But ``doc_id`` is assigned in *fetch* order (``fetch.py`` ``next_id``) while the
corpus is written and sharded in *domain-interleaved* order (``build.interleave_documents`` +
``assign_shards``, deliberate for T5.1 early-stop balance). So ``doc_id`` never ascends with shard
order, and the whole trace set is unreadable. Setting ``doc_id`` = the line's position in the file
makes file order == doc_id order, so plan_shards / collector / moe_trace / reader all work unchanged.

THIS IS A TRANSFORM, NOT A REBUILD. Re-running ``build()`` would reshuffle the write order (splits and
interleave key on ``_doc_key(doc_id, seed)``) and undo the fix. We touch ONLY the ``doc_id`` value of
each line and prove every other byte is unchanged:

  * our serializer must reproduce the input line exactly (same ensure_ascii / separators / key order),
    which pins the format; then
  * after setting ``doc_id = i`` the re-parsed line must equal the input line in every field but
    ``doc_id``.

If either check fails on any line, nothing is written. Splits are stored per line and read from the
file (``load_doc_splits``), so renumbering keeps each line's split; nothing re-derives split from
``doc_id`` at read time (verified). Use a NEW corpus name (mixed-v2) — corpus identity is the name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def renumber(inp: Path, out: Path, *, expect_input_sha: str | None = None) -> dict:
    raw = inp.read_bytes()
    in_sha = hashlib.sha256(raw).hexdigest()
    if expect_input_sha and in_sha != expect_input_sha:
        raise SystemExit(
            f"input sha256 {in_sha} != expected {expect_input_sha}; refusing to transform the "
            "wrong corpus file"
        )

    text = raw.decode("utf-8")
    # Preserve the exact line structure. The corpus writer emits one object per line + a trailing
    # newline; splitlines(keepends=False) + explicit '\n' join below reproduces it, which we assert.
    lines = text.split("\n")
    trailing_newline = lines and lines[-1] == ""
    if trailing_newline:
        lines = lines[:-1]

    out_lines: list[str] = []
    for i, line in enumerate(lines):
        row = json.loads(line)
        # 1) pin the format: our serialization must reproduce the corpus writer's bytes exactly.
        if json.dumps(row, ensure_ascii=True, sort_keys=False) != line:
            raise SystemExit(
                f"line {i}: re-serialization does not match the source bytes; the corpus was not "
                "written with json.dumps(ensure_ascii=True, sort_keys=False) — aborting so the "
                "transform cannot silently reformat other fields"
            )
        if "doc_id" not in row:
            raise SystemExit(f"line {i}: no doc_id field")
        original = dict(row)
        row["doc_id"] = i  # dict preserves order, doc_id stays the first key
        new_line = json.dumps(row, ensure_ascii=True, sort_keys=False)

        # 2) prove ONLY doc_id changed.
        back = json.loads(new_line)
        if int(back["doc_id"]) != i:
            raise SystemExit(f"line {i}: doc_id not set")
        b, o = dict(back), dict(original)
        b.pop("doc_id"), o.pop("doc_id")
        if b != o:
            diff = {k: (o.get(k), b.get(k)) for k in set(o) | set(b) if o.get(k) != b.get(k)}
            raise SystemExit(f"line {i}: fields other than doc_id changed: {diff}")
        out_lines.append(new_line)

    blob = "\n".join(out_lines) + ("\n" if trailing_newline else "")
    out.write_text(blob, encoding="utf-8", newline="")
    out_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    return {
        "in_sha": in_sha,
        "out_sha": out_sha,
        "n_docs": len(out_lines),
        "doc_id_ascending": [json.loads(l)["doc_id"] for l in out_lines[:3]],
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Renumber corpus doc_id to line index (mixed-v2).")
    ap.add_argument("input", type=Path, help="published mixed-v1.jsonl")
    ap.add_argument("output", type=Path, help="mixed-v2.jsonl to write")
    ap.add_argument("--expect-input-sha", default=None,
                    help="refuse unless the input file has this sha256 (identity guard)")
    args = ap.parse_args(argv)

    st = renumber(args.input, args.output, expect_input_sha=args.expect_input_sha)
    print(f"input  sha256: {st['in_sha']}")
    print(f"output sha256: {st['out_sha']}")
    print(f"docs: {st['n_docs']}, first doc_ids now: {st['doc_id_ascending']} (0,1,2,...)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
