#!/usr/bin/env python3
"""Restate v0.1 side results under the attainable normalisation Î/(H − log2 k) (audit L1/L2).

The word-level control (``results/phase9/t9_2_word_f1.json``) stored Î/H and per-domain cross-entropy
but not H. H is recovered exactly as CE / (1 − Î/H), with CE the slot-weighted mean of the per-domain
CEs (the domains partition the test rows; checked). Within a model the word/token ratio does not
change, because both are scored on the same rows; across models it does, because the ceiling
(H − log2 k)/H differs with k.

Writes ``results/audit/word_control_attainable.{json,md}``.

Usage: python -m scripts.audit_restate
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

WORD = Path("results/phase9/t9_2_word_f1.json")
TOP_K = {"gpt-oss-20b": 4, "deepseek-v2-lite": 6}  # every other panel model routes top-8
OUT = Path("results/audit")


def attainable(rec: dict, k: int) -> float:
    strata = rec["strata"].values()
    n_slots = sum(s["n_slots"] for s in strata)
    if n_slots != rec["n_test_rows"] * k:
        raise ValueError(f"domains do not partition the rows: {n_slots} != {rec['n_test_rows']} * {k}")
    ce = sum(s["cross_entropy_bits"] * s["n_slots"] for s in strata) / n_slots
    h = ce / (1.0 - rec["mi_ratio"])
    return (h - ce) / (h - math.log2(k))


def main() -> int:
    data = json.loads(WORD.read_text(encoding="utf-8"))
    out = {}
    for model, md in data.items():
        k = TOP_K.get(model, 8)
        per_depth = {str(rec["depth"]): {"word": attainable(rec["word"], k),
                                         "token_same_rows": attainable(rec["token_same_rows"], k)}
                     for rec in md["layers"].values()}
        w = st.mean(v["word"] for v in per_depth.values())
        t = st.mean(v["token_same_rows"] for v in per_depth.values())
        out[model] = {"word_attainable": w, "token_same_rows_attainable": t, "word_over_token": w / t,
                      "frac_word_initial": md["frac_word_initial"], "per_depth": per_depth}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "word_control_attainable.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    depths = sorted({d for v in out.values() for d in v["per_depth"]}, key=float)
    md = ["# Word-level control, attainable normalisation Î/(H − log2 k)", "",
          "Word-initial tokens only; token predictor refit on the same rows. Mean over five relative depths.", "",
          "| model | word | token (same rows) | word / token | word-initial share | "
          + " | ".join(f"word @ {d}" for d in depths) + " |",
          "|---|---:|---:|---:|---:|" + "---:|" * len(depths)]
    for m, v in out.items():
        md.append(f"| {m} | {v['word_attainable']:.3f} | {v['token_same_rows_attainable']:.3f} | "
                  f"{v['word_over_token']:.2f} | {v['frac_word_initial']:.2f} | "
                  + " | ".join(f"{v['per_depth'][d]['word']:.3f}" for d in depths) + " |")
    (OUT / "word_control_attainable.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
