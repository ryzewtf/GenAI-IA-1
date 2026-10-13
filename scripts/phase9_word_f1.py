#!/usr/bin/env python3
"""T9.2 word-level control for F1: condition on whitespace-word identity, word-initial tokens only.

Token-ID F1 does not condition on the same variable across tokenizers. This maps each token to the
whitespace-delimited word it starts (decoded from the model's own tokenizer) and fits the same
closed-form count table on word identity, restricted to word-initial tokens. The token-ID count
table is refit on exactly the same rows, so the two numbers differ only in the conditioning
variable. Train/val/test are the corpus doc splits; alpha is chosen on val; metrics are on test.

Scripts that never see a whitespace (zh/ja) have one "word" per whitespace run, so they contribute
few, long words; the result is reported pooled and per domain.

Usage: python -m scripts.phase9_word_f1 --tokenizers E:/temp/tok [--root E:/temp/traces]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from scripts.phase6_metrics import ENGINE_BY_MODEL
from src.probes.counts import CountTablePredictor
from src.probes.evaluate import evaluate_on_test
from src.probes.features import CorpusIndex
from src.traces.reader import TraceReader

TOKENIZER_REPO = {
    "gpt-oss-20b": "openai__gpt-oss-20b",
    "olmoe-0125": "allenai__OLMoE-1B-7B-0125",
    "olmoe-0125-instruct": "allenai__OLMoE-1B-7B-0125",
    "olmoe-0924": "allenai__OLMoE-1B-7B-0125",
    "qwen3-30b-a3b": "Qwen__Qwen3-30B-A3B",
    "gemma-4-26b-a4b": "google__gemma-4-26B-A4B",
    "deepseek-v2-lite": "deepseek-ai__DeepSeek-V2-Lite",
}
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)


_MARKERS = str.maketrans({"Ġ": " ", "▁": " ", "Ċ": "\n", "ĉ": "\t"})


def piece_table(tok: Tokenizer, vocab_size: int) -> list[str]:
    """Raw vocab strings with the whitespace markers mapped back to whitespace.

    Not ``decode([i])``: single-token decode strips the leading space under SentencePiece-style
    decoders (Gemma), which would erase exactly the word-boundary signal. Non-ASCII byte-level
    pieces stay as their byte-alphabet strings — they only need to be consistent keys.
    """
    return [(tok.id_to_token(i) or "").translate(_MARKERS) for i in range(vocab_size)]


def word_ids(index: CorpusIndex, pieces: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(is_word_initial[n], word_id[n]) — word_id is set on word-initial rows only (-1 elsewhere)."""
    n = index.n_tokens
    text = [pieces[t] if t < len(pieces) else "" for t in index.token_ids]
    initial = np.zeros(n, dtype=bool)
    for i in range(n):
        if index.pos_in_doc[i] == 0 or text[i][:1].isspace() or text[i - 1][-1:].isspace():
            initial[i] = bool(text[i].strip())
    starts = np.flatnonzero(initial)
    vocab: dict[str, int] = {}
    wid = np.full(n, -1, dtype=np.int64)
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else n
        parts = [text[s]]
        for i in range(s + 1, e):
            if index.doc_ids[i] != index.doc_ids[s] or text[i][:1].isspace():
                break
            parts.append(text[i])
        w = "".join(parts).strip().split()[0] if "".join(parts).strip() else ""
        wid[s] = vocab.setdefault(w, len(vocab))
    return initial, wid


def fit_eval(X, y, rows, n_experts, top_k, vocab, buckets):
    tr, va, te = rows
    p = CountTablePredictor(n_experts=n_experts, vocab_size=vocab)
    p.fit(X[tr], y[tr], X[va], y[va])
    m = evaluate_on_test(p, X[te], y[te], n_experts=n_experts, top_k=top_k, buckets=buckets)
    return {"mi_ratio": m["family_b"]["ratio"], "set_agreement@k": m["family_a"]["set_agreement@k"],
            "alpha": p.alpha, "n_test_rows": int(len(te)), "strata": m.get("strata")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/temp/traces")
    ap.add_argument("--tokenizers", default="E:/temp/tok")
    ap.add_argument("--corpus", default="corpora/mixed-v2.jsonl")
    ap.add_argument("--splits", default="corpora/mixed-v2.splits.json")
    ap.add_argument("--models", default=",".join(TOKENIZER_REPO))
    ap.add_argument("--out", default="results/phase9/t9_2_word_f1.json")
    args = ap.parse_args(argv)

    splits = {int(k): v for k, v in json.loads(Path(args.splits).read_text("utf-8")).items()}
    domain = {}
    for line in open(args.corpus, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            domain[int(d["doc_id"])] = d["domain"]
    dom_labels = sorted(set(domain.values()))

    out = {}
    for model in args.models.split(","):
        tok_path = Path(args.tokenizers) / TOKENIZER_REPO[model] / "tokenizer.json"
        if not tok_path.exists():
            print(f"!! {model}: no {tok_path}; skipped (hf download <repo> tokenizer.json)", flush=True)
            continue
        tok = Tokenizer.from_file(str(tok_path))
        root = Path(args.root) / ENGINE_BY_MODEL[model] / "traces"
        with TraceReader(root, model, "mixed-v2", doc_splits=splits, validate_sizes=False) as r:
            index = CorpusIndex.from_reader(r)
            pieces = piece_table(tok, max(tok.get_vocab_size(), index.vocab_size))
            initial, wid = word_ids(index, pieces)
            # words never seen in train share one reserved id with zero train counts -> marginal
            tr_all = index.rows("train")
            seen = np.zeros(wid.max() + 2, dtype=bool)
            seen[wid[tr_all][initial[tr_all]]] = True
            unseen_id = wid.max() + 1
            wX = np.where(seen[np.maximum(wid, 0)] & (wid >= 0), wid, unseen_id)
            rows = [r_[initial[r_]] for r_ in (index.rows(s) for s in ("train", "val", "test"))]
            te_dom = np.array([dom_labels.index(domain[int(d)]) for d in index.doc_ids[rows[2]]])
            n_layers = r.n_moe_layers
            res = {"n_word_initial_test": int(len(rows[2])),
                   "frac_word_initial": float(initial.mean()), "layers": {}}
            for f in DEPTHS:
                L = round(f * (n_layers - 1))
                y = r.topk_sets(L)
                y = np.asarray(y, dtype=np.int32)
                word = fit_eval(wX, y, rows, r.n_experts, r.top_k, unseen_id + 1, te_dom)
                tokn = fit_eval(index.token_ids, y, rows, r.n_experts, r.top_k, index.vocab_size, te_dom)
                for x in (word, tokn):
                    if x["strata"]:
                        x["strata"] = {dom_labels[int(k)]: v for k, v in x["strata"].items()}
                res["layers"][str(L)] = {"depth": f, "word": word, "token_same_rows": tokn}
                print(f"{model:22} L{L:2} word {word['mi_ratio']:.3f}  token(same rows) "
                      f"{tokn['mi_ratio']:.3f}", flush=True)
            res["mean_word_mi_ratio"] = float(np.mean([v["word"]["mi_ratio"] for v in res["layers"].values()]))
            res["mean_token_mi_ratio"] = float(np.mean([v["token_same_rows"]["mi_ratio"] for v in res["layers"].values()]))
            out[model] = res
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
