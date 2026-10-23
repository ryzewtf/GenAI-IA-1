#!/usr/bin/env python3
"""Two draft [TODO]s that the existing traces can answer — top-1 repetition and the frequent decile.

* **Stored expert order.** Is ``topk.bin`` written score-descending on both engines? Checked against
  the stored routing tensor on the first 20k tokens at three depths. Only if it is does
  ``topk[:, 0]`` mean "first choice", which a Mixtral-comparable statistic needs.
* **Top-1 consecutive repetition** (Mixtral report, Table 5): share of consecutive same-document
  tokens whose first-choice expert is the same, per layer, with its chance level Σ p_e² from the
  first-choice marginal. Mixtral reports 23.6–28.4% at layer 15 of 32 (chance 12.5%).
* **Frequent decile composition**: of the test slots in the most frequent train-frequency decile of
  token types (the bucket F1 uses), the share that are punctuation/whitespace/digits, and the most
  frequent pieces. This tests the draft's conjecture that the decile is dominated by function words
  and punctuation.

    python -m scripts.audit_token_checks [--models ...]   # -> results/audit/token_checks.{json,md}
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from scripts.phase6_metrics import ENGINE_BY_MODEL
from scripts.phase9_word_f1 import TOKENIZER_REPO, piece_table
from src.probes.counts import frequency_buckets
from src.probes.features import CorpusIndex
from src.traces.reader import TraceReader

MIXTRAL = {"layer": 15, "n_layers": 32, "top1_rate": [0.236, 0.284], "chance": 0.125}
# A small closed-class list (English); multilingual function words are not covered, so the share
# below is a lower bound on "function words" outside English.
STOPWORDS = set("""the of and to a in is that for it as was with be by on not he i this are or his
from at which but have an they you were her she there their one all we can has been if more when
will would who so no its also into than them other some what my do only these two may out any
our""".split())
PUNCT = set(string.punctuation) | set("“”‘’«»—–…、。，！？：；（）「」《》")


def piece_class(piece: str) -> str:
    s = piece.strip()
    if not s:
        return "whitespace"
    if all(c in PUNCT or c.isdigit() for c in s):
        return "punct_digit"
    if s.lower() in STOPWORDS:
        return "stopword_en"
    return "other"


def order_check(model: str, full_root: Path, splits) -> dict:
    root = full_root / ENGINE_BY_MODEL[model] / "traces"
    with TraceReader(root, model, "mixed-v2", doc_splits=splits, validate_sizes=False) as r:
        L = r.n_moe_layers
        out = {}
        for layer in sorted({0, L // 2, L - 1}):
            n = min(20000, r.n_tokens)
            sets = r.topk_sets(layer, slice(0, n)).astype(np.int64)
            scores = r.logits(layer, slice(0, n))
            picked = np.take_along_axis(scores, sets, axis=1)
            out[str(layer)] = {
                "first_is_argmax": float(np.mean(sets[:, 0] == scores.argmax(axis=1))),
                "score_descending": float(np.mean(np.all(np.diff(picked, axis=1) <= 0, axis=1))),
            }
        return out


def top1_and_decile(model: str, fast_root: Path, splits, tok_root: Path) -> dict:
    root = fast_root / ENGINE_BY_MODEL[model] / "traces"
    with TraceReader(root, model, "mixed-v2", doc_splits=splits, validate_sizes=False) as r:
        index = CorpusIndex.from_reader(r)
        same_doc = np.r_[False, index.doc_ids[1:] == index.doc_ids[:-1]]
        per_layer = []
        for layer in range(r.n_moe_layers):
            top1 = r.topk_sets(layer)[:, 0].astype(np.int64)
            rate = float(np.mean(top1[1:][same_doc[1:]] == top1[:-1][same_doc[1:]]))
            p = np.bincount(top1, minlength=r.n_experts) / top1.size
            per_layer.append({"layer": layer, "rel_depth": layer / max(1, r.n_moe_layers - 1),
                              "top1_repeat": rate, "chance": float((p ** 2).sum())})
        train = index.token_ids[index.rows("train")]
        test = index.token_ids[index.rows("test")]
        buckets, meta = frequency_buckets(train, test, vocab_size=index.vocab_size)
        top = buckets == buckets.max()
    tok = Tokenizer.from_file(str(tok_root / TOKENIZER_REPO[model] / "tokenizer.json"))
    pieces = piece_table(tok, max(index.vocab_size, tok.get_vocab_size()))
    ids, counts = np.unique(test[top], return_counts=True)
    classes = {}
    for i, c in zip(ids, counts):
        k = piece_class(pieces[int(i)] if int(i) < len(pieces) else "")
        classes[k] = classes.get(k, 0) + int(c)
    total = int(counts.sum())
    order = np.argsort(-counts)[:15]
    mid = min(per_layer, key=lambda d: abs(d["rel_depth"] - MIXTRAL["layer"] / (MIXTRAL["n_layers"] - 1)))
    return {
        "top1_per_layer": per_layer, "top1_at_mixtral_depth": mid,
        "top1_max": max(per_layer, key=lambda d: d["top1_repeat"]),
        "top_decile": {"test_token_share": float(top.mean()), "n_types": int(ids.size),
                       "class_share": {k: v / total for k, v in sorted(classes.items())},
                       "top_pieces": [[pieces[int(ids[j])], int(counts[j]) / total] for j in order]},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(m for m in ENGINE_BY_MODEL if m != "olmoe-0924"))
    ap.add_argument("--full-root", default="FULL RESULTS")
    ap.add_argument("--fast-root", default="E:/temp/traces")
    ap.add_argument("--tokenizers", default="E:/temp/tok")
    ap.add_argument("--splits", default="corpora/mixed-v2.splits.json")
    ap.add_argument("--out", default="results/audit")
    args = ap.parse_args(argv)
    splits = {int(k): str(v) for k, v in json.loads(Path(args.splits).read_text(encoding="utf-8")).items()}
    res = {"mixtral_reference": MIXTRAL, "models": {}}
    for model in args.models.split(","):
        res["models"][model] = {"order": order_check(model, Path(args.full_root), splits),
                                **top1_and_decile(model, Path(args.fast_root), splits, Path(args.tokenizers))}
        print(model, json.dumps(res["models"][model]["order"]), flush=True)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "token_checks.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Stored expert order, top-1 repetition, frequent-decile composition", "",
          "| model | first = argmax | score-descending | top-1 repeat at depth ≈0.48 | chance | max (depth) | "
          "top decile: share of test tokens | punct/digit | whitespace | EN stopword |",
          "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for m, d in res["models"].items():
        o = list(d["order"].values())
        a, mx, td = d["top1_at_mixtral_depth"], d["top1_max"], d["top_decile"]
        cs = td["class_share"]
        md.append(f"| {m} | {min(x['first_is_argmax'] for x in o):.4f} | {min(x['score_descending'] for x in o):.4f} | "
                  f"{a['top1_repeat']:.3f} (L{a['layer']}) | {a['chance']:.3f} | {mx['top1_repeat']:.3f} ({mx['rel_depth']:.2f}) | "
                  f"{td['test_token_share']:.3f} | {cs.get('punct_digit', 0):.3f} | {cs.get('whitespace', 0):.3f} | "
                  f"{cs.get('stopword_en', 0):.3f} |")
    md += ["", f"Mixtral 8x7B (report Table 5): {MIXTRAL['top1_rate'][0]:.1%}–{MIXTRAL['top1_rate'][1]:.1%} at layer "
           f"{MIXTRAL['layer']} of {MIXTRAL['n_layers']}, chance {MIXTRAL['chance']:.1%}.", "", "## Most frequent pieces in the top decile", ""]
    for m, d in res["models"].items():
        md.append(f"- **{m}**: " + ", ".join(f"`{p!r}` {s:.1%}" for p, s in d["top_decile"]["top_pieces"][:10]))
    (dst / "token_checks.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
