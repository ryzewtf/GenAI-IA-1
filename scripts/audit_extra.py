#!/usr/bin/env python3
"""Second-pass audit checks from ``audit/CHECKLISTS.md`` [H]: F5 tuning, the ε-mix target, language data volume.

Three experiments, each with ``run`` (fits, slow) and ``report`` (summary only):

* **f5tune** — F5 had its width and learning rate fixed while F4/FV/F4R had their regulariser
  chosen on val, so the F4R − F5 gap might be F5 being under-tuned. Refits F5 on the convergence
  study layers for a small grid, selects per model on summed val CE, and reports how far test
  attainable moves from the default (256 units, lr 3e-3; seed 0 of ``results/probes_v2``).
* **epsmix** — the CE ε-mix uses the test label marginal (audit L-k). Refits every feature on the
  study layers; :func:`src.probes.evaluate.evaluate_on_test` now also reports the bound with a
  uniform mix target, and the report gives the largest difference between the two.
* **langeq** — the smallest language has 10–13× fewer training tokens than English. Rebuilds the
  train split with the same number of tokens per language (FLORES and code documents dropped from
  train; val and test untouched), refits F1, F2 and F4R on every layer, and reports each
  language's drop against English with document-bootstrap intervals, next to the drop under the
  original split.

    python -m scripts.audit_extra {f5tune,epsmix,langeq} run --models m1,m2
    python -m scripts.audit_extra {f5tune,epsmix,langeq} report
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import audit_ridge_probe, phase7_sweep
from scripts.audit_convergence import N_LAYERS, study_layers
from scripts.phase6_metrics import ENGINE_BY_MODEL

ROOT = "E:/temp/traces"
OUT = Path("E:/temp2")
CANON = Path("results/probes_v2")
CANON_AGG = Path("results/probes_v2_docagg")
RESULTS = Path("results/audit")
CORPUS = Path("corpora/mixed-v2.jsonl")
SPLITS = Path("corpora/mixed-v2.splits.json")

F5_DEFAULT = (256, 3e-3)
F5_GRID = ((512, 3e-3), (1024, 3e-3), (256, 1e-3), (256, 1e-2))
LANGS = ("en", "ar", "de", "es", "fr", "hi", "ja", "ru", "zh")


def f4_l2(model: str) -> float:
    # val-selected in the convergence study (results/audit/convergence.md)
    return 1e-2 if model == "gpt-oss-20b" else 1e-3


def attainable(rec, key="family_b") -> float:
    m = rec["metrics"]
    k = m["n_slots"] // m["n_rows"]
    return m[key]["mi_bits"] / (m[key]["entropy_bits"] - math.log2(k))


def _layers(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["layers"] if path.exists() else {}


def _sweep(model, out, features, layers, *extra):
    return phase7_sweep.main(["--root", ROOT, "--no-validate-sizes", "--models", model,
                              "--features", *features, "--layers", *map(str, layers),
                              "--out", str(out), *extra])


# -- f5tune ----------------------------------------------------------------------------------------

def f5_dir(width, lr) -> Path:
    return OUT / "f5tune" / f"w{width}_lr{lr:g}"


def f5tune_run(args) -> int:
    rc = 0
    for model in args.models.split(","):
        for width, lr in F5_GRID:
            rc |= _sweep(model, f5_dir(width, lr), ["F5"], study_layers(N_LAYERS[model]),
                         "--probe-kwargs", json.dumps({"hidden_width": width, "lr": lr}))
    return rc


def f5tune_report(args) -> int:
    rows, per_model = [], {}
    for model in ENGINE_BY_MODEL:
        layers = [str(x) for x in study_layers(N_LAYERS[model])]
        configs = {F5_DEFAULT: _layers(CANON / f"{model}__F5.json")}
        for cfg in F5_GRID:
            configs[cfg] = _layers(f5_dir(*cfg) / f"{model}__F5.json")
        configs = {c: r for c, r in configs.items() if all(r.get(L, {}).get("status") == "ok" for L in layers)}
        if len(configs) < 2:
            continue
        val = {c: sum(r[L]["fit"]["best_val_ce_bits"] for L in layers) for c, r in configs.items()}
        test = {c: st.mean(attainable(r[L]) for L in layers) for c, r in configs.items()}
        best = min(val, key=val.get)
        f4r = _layers(CANON / f"{model}__F4R.json")
        f4r_mean = st.mean(attainable(f4r[L]) for L in layers) if all(L in f4r for L in layers) else None
        per_model[model] = {
            "layers": layers, "configs_fitted": len(configs),
            "val_ce_sum": {f"w{c[0]}_lr{c[1]:g}": v for c, v in val.items()},
            "test_attainable": {f"w{c[0]}_lr{c[1]:g}": v for c, v in test.items()},
            "default": f"w{F5_DEFAULT[0]}_lr{F5_DEFAULT[1]:g}", "selected": f"w{best[0]}_lr{best[1]:g}",
            "default_attainable": test[F5_DEFAULT] if F5_DEFAULT in test else None,
            "selected_attainable": test[best],
            "gain": test[best] - test[F5_DEFAULT] if F5_DEFAULT in test else None,
            "best_test_any_config": max(test.values()),
            "F4R_same_layers": f4r_mean,
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "f5_tuning.json").write_text(json.dumps(per_model, indent=2), encoding="utf-8")
    md = ["# F5 width / learning-rate selection on val (study layers, Î/(H − log2 k) on test)", "",
          "| model | default (w256, lr 3e-3) | selected on val | selected | gain | best any config | F4R same layers |",
          "|---|---:|---|---:|---:|---:|---:|"]
    for m, r in per_model.items():
        md.append(f"| {m} | {r['default_attainable']:.4f} | {r['selected']} | {r['selected_attainable']:.4f} | "
                  f"{r['gain']:+.4f} | {r['best_test_any_config']:.4f} | "
                  f"{(r['F4R_same_layers'] or float('nan')):.4f} |")
    (RESULTS / "f5_tuning.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


# -- epsmix ----------------------------------------------------------------------------------------

EPS_DIR = OUT / "epsmix"


def epsmix_run(args) -> int:
    rc = 0
    for model in args.models.split(","):
        layers = study_layers(N_LAYERS[model])
        rc |= _sweep(model, EPS_DIR, ["F0", "F1", "F2", "F3", "F6"], layers)
        rc |= _sweep(model, EPS_DIR, ["F4", "FV"], layers,
                     "--probe-kwargs", json.dumps({"lbfgs_maxiter": 2000, "l2": f4_l2(model)}))
        rc |= audit_ridge_probe.main(["--models", model, "--layers", *map(str, layers), "--no-curve",
                                      "--out", str(EPS_DIR), "--agg", str(OUT / "epsmix_agg")])
    return rc


def epsmix_report(args) -> int:
    cells = []
    sources = [EPS_DIR] + sorted((OUT / "f5tune").glob("w*"))
    for d in sources:
        for p in sorted(d.glob("*__*.json")):
            model, feature = p.stem.split("__")
            canon = _layers(CANON / p.name)
            for L, rec in _layers(p).items():
                if rec.get("status") != "ok" or "family_b_uniform_mix" not in rec["metrics"]:
                    continue
                a_test, a_uni = attainable(rec), attainable(rec, "family_b_uniform_mix")
                c = canon.get(L)
                cells.append({"model": model, "feature": feature, "layer": int(L), "source": d.name,
                              "test_mix": a_test, "uniform_mix": a_uni, "diff": a_uni - a_test,
                              "canonical": attainable(c) if c and c.get("status") == "ok" else None})
    out = {"n_cells": len(cells),
           "max_abs_diff": max((abs(c["diff"]) for c in cells), default=None),
           "max_abs_diff_by_feature": {f: max(abs(c["diff"]) for c in cells if c["feature"] == f)
                                       for f in sorted({c["feature"] for c in cells})},
           "refit_vs_canonical_max_abs": max((abs(c["test_mix"] - c["canonical"]) for c in cells
                                              if c["canonical"] is not None and c["source"] == "epsmix"), default=None),
           "cells": cells}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "epsilon_mix.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# ε-mix target: test label marginal vs uniform (Î/(H − log2 k), test)", "",
          f"{out['n_cells']} cells; largest |uniform − test-marginal| = {out['max_abs_diff']:.2e}.", "",
          "| feature | max abs difference |", "|---|---:|"]
    md += [f"| {f} | {v:.2e} |" for f, v in out["max_abs_diff_by_feature"].items()]
    (RESULTS / "epsilon_mix.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("refit vs canonical max |Δ| (same mix):", out["refit_vs_canonical_max_abs"])
    return 0


# -- langeq ----------------------------------------------------------------------------------------

LQ = OUT / "langeq"


def _corpus():
    meta = {}
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                meta[int(r["doc_id"])] = (str(r.get("lang") or ""), r.get("source"))
    return meta


def langeq_splits(model: str, seed: int = 0) -> dict:
    """Equal train tokens per language for this model's tokenizer; returns the summary."""
    from src.probes.features import CorpusIndex
    from src.traces.reader import TraceReader
    splits = {int(k): str(v) for k, v in json.loads(SPLITS.read_text(encoding="utf-8")).items()}
    meta = _corpus()
    root = Path(ROOT) / ENGINE_BY_MODEL[model] / "traces"
    with TraceReader(root, model, "mixed-v2", doc_splits=splits, validate_sizes=False) as reader:
        ids, counts = np.unique(CorpusIndex.from_reader(reader).doc_ids, return_counts=True)
    tokens = dict(zip(map(int, ids), map(int, counts)))
    by_lang = defaultdict(list)
    for d, s in splits.items():
        lang, source = meta[d]
        if s == "train" and lang in LANGS and source != "flores200" and d in tokens:
            by_lang[lang].append(d)
    full = {lang: sum(tokens[d] for d in docs) for lang, docs in by_lang.items()}
    target = min(full.values())
    rng = np.random.default_rng(seed)
    keep, got = set(), {}
    for lang in LANGS:
        docs = list(rng.permutation(sorted(by_lang[lang])))
        total = 0
        for d in docs:
            if total >= target:
                break
            keep.add(int(d))
            total += tokens[int(d)]
        got[lang] = total
    new = {d: (s if s != "train" or d in keep else "unused") for d, s in splits.items()}
    LQ.mkdir(parents=True, exist_ok=True)
    (LQ / f"splits_{model}.json").write_text(json.dumps({str(k): v for k, v in new.items()}), encoding="utf-8")
    return {"target_tokens": target, "full_train_tokens": full, "equalised_train_tokens": got,
            "train_docs_kept": len(keep)}


def langeq_run(args) -> int:
    rc, summary = 0, {}
    p = LQ / "splits_summary.json"
    if p.exists():
        summary = json.loads(p.read_text(encoding="utf-8"))
    for model in args.models.split(","):
        summary[model] = langeq_splits(model)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(model, summary[model], flush=True)
        sp = str(LQ / f"splits_{model}.json")
        rc |= phase7_sweep.main(["--root", ROOT, "--no-validate-sizes", "--models", model,
                                 "--features", "F1", "F2", "--splits", sp,
                                 "--out", str(LQ / "probes"), "--agg-dir", str(LQ / "agg")])
        rc |= audit_ridge_probe.main(["--models", model, "--no-curve", "--splits", sp,
                                      "--out", str(LQ / "probes"), "--agg", str(LQ / "agg")])
    return rc


def _lang_drops(agg_dir: Path, weights, mapping, models, features):
    from scripts.audit_bootstrap import subset
    from src.metrics.bootstrap import layer_mean, load_aggregates, replicate_metrics, summarize
    out = {}
    for model in models:
        for f in features:
            p = agg_dir / f"{model}__{f}.npz"
            if not p.exists():
                continue
            per_lang = defaultdict(list)
            for _, a in sorted(load_aggregates(p).items()):
                lab = np.array([mapping.get(int(d), "") for d in a.doc_ids])
                for lang in LANGS:
                    keep = lab == lang
                    if keep.sum() >= 3:
                        per_lang[lang].append(replicate_metrics(subset(a, keep), weights)["mi_attainable"])
            if "en" not in per_lang:
                continue
            reps = {lang: layer_mean(v) for lang, v in per_lang.items()}
            # bias-correct each replicate set the same way the strata table does (bc = 2θ − mean)
            bc = {lang: 2 * r[0] - r[1:].mean() for lang, r in reps.items()}
            drops = {lang: summarize(reps[lang] / reps["en"] - 1) for lang in reps if lang != "en"}
            mean_rep = np.mean([reps[lang] / reps["en"] - 1 for lang in reps if lang != "en"], axis=0)
            out[(model, f)] = {"bc_attainable": bc, "drop_vs_en": drops,
                               "mean_drop_vs_en": summarize(mean_rep),
                               "mean_drop_bc_points": float(np.mean([bc[l] / bc["en"] - 1 for l in bc if l != "en"]))}
    return out


def langeq_report(args) -> int:
    from src.metrics.bootstrap import draw_weights
    meta = _corpus()
    mapping = {d: lang for d, (lang, source) in meta.items() if source != "flores200"}
    test_docs = [int(d) for d, s in json.loads(SPLITS.read_text(encoding="utf-8")).items() if s == "test"]
    weights = draw_weights(np.array(test_docs), args.n_boot, seed=0)
    models = list(ENGINE_BY_MODEL)
    feats = ("F1", "F2", "F4R")
    eq = _lang_drops(LQ / "agg", weights, mapping, models, feats)
    orig = _lang_drops(CANON_AGG, weights, mapping, models, feats)
    summary = json.loads((LQ / "splits_summary.json").read_text(encoding="utf-8"))
    res = {"note": "drop = Î/(H−log2k) of language / English − 1, layer means, test, FLORES excluded; "
                   "equalised = same train tokens per language, code and FLORES dropped from train",
           "splits": summary, "cells": []}
    md = ["# Language drop vs English with equal training tokens per language", "",
          "Mean over the eight non-English languages of (language / English − 1), attainable-normalised, "
          "FLORES excluded. Points are bias-corrected (as in the strata table), with 95% basic intervals "
          "over test documents. en = English attainable (bias-corrected), original → equalised.", "",
          "| model | feature | en | original split | equalised train | change |", "|---|---|---|---|---|---:|"]
    for (model, f), e in sorted(eq.items()):
        o = orig.get((model, f))
        if o is None:
            continue
        res["cells"].append({"model": model, "feature": f, "original": o, "equalised": e})
        so, se = o["mean_drop_vs_en"], e["mean_drop_vs_en"]
        po, pe = o["mean_drop_bc_points"], e["mean_drop_bc_points"]
        md.append(f"| {model} | {f} | {o['bc_attainable']['en']:.3f} → {e['bc_attainable']['en']:.3f} | "
                  f"{po:+.3f} [{so['basic_lo']:+.3f}, {so['basic_hi']:+.3f}] | "
                  f"{pe:+.3f} [{se['basic_lo']:+.3f}, {se['basic_hi']:+.3f}] | {pe - po:+.3f} |")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "language_equalised.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (RESULTS / "language_equalised.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=("f5tune", "epsmix", "langeq"))
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--models", default=",".join(ENGINE_BY_MODEL))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    fn = {("f5tune", "run"): f5tune_run, ("f5tune", "report"): f5tune_report,
          ("epsmix", "run"): epsmix_run, ("epsmix", "report"): epsmix_report,
          ("langeq", "run"): langeq_run, ("langeq", "report"): langeq_report}[(args.experiment, args.cmd)]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
