"""Phase 7 panel sweep (T7.8) — fit F0-F6/FV across every model, layer and split-eval on test.

Loops the panel, opening each model's trace with the verified doc-splits and calling
``src.probes.train.sweep`` (resumable: cells already on disk are skipped unless --force). CPU
features (F0/F1/F2/F3/F6) run by default; add --hidden to also fit the router-input probes
(F4/F5/FV), which are heavier but still CPU-feasible.

Usage:
    python -m scripts.phase7_sweep [--root "FULL RESULTS"] [--corpus mixed-v2] \
        [--splits corpora/mixed-v2.splits.json] [--out results/probes] \
        [--models m1,m2] [--features F0 F3 ...] [--hidden] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scripts.phase6_metrics import ENGINE_BY_MODEL
from src.probes.train import CPU_FEATURES, sweep
from src.traces.reader import TraceReader

HIDDEN_FEATURES = ["F4", "F5", "FV"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 panel probe sweep (T7.8)")
    ap.add_argument("--root", default="FULL RESULTS")
    ap.add_argument("--corpus", default="mixed-v2")
    ap.add_argument("--splits", default="corpora/mixed-v2.splits.json")
    ap.add_argument("--out", default="results/probes")
    ap.add_argument("--models", default=None)
    ap.add_argument("--features", nargs="+", default=None, help="default = CPU features")
    ap.add_argument("--hidden", action="store_true", help="also fit F4/F5/FV router-input probes")
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--no-validate-sizes", action="store_true",
                    help="skip TraceReader size re-validation (data already T5.3-validated); "
                         "lets the sweep run against a trace copy that omits logits.bin, which "
                         "F4/F5/FV never read")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-kwargs", default=None,
                    help="JSON dict forwarded to the probe (e.g. '{\"solver\":\"lbfgs\"}'). "
                         "Keys apply only to the probe classes that define them (validated).")
    ap.add_argument("--doc-strata", default=None, metavar="CORPUS_JSONL",
                    help="also report test metrics per domain and per lang (T9.4), read from this "
                         "corpus file's doc_id/domain/lang fields")
    args = ap.parse_args(argv)

    doc_strata = None
    if args.doc_strata:
        doc_strata = {"domain": {}, "lang": {}}
        with open(args.doc_strata, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    doc_strata["domain"][int(d["doc_id"])] = str(d["domain"])
                    doc_strata["lang"][int(d["doc_id"])] = str(d["lang"])

    pkw = json.loads(args.probe_kwargs) if args.probe_kwargs else {}

    raw = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    doc_splits = {int(k): str(v) for k, v in raw.items()}

    if args.features:
        features = args.features
    else:
        features = list(CPU_FEATURES) + (HIDDEN_FEATURES if args.hidden else [])

    models = args.models.split(",") if args.models else list(ENGINE_BY_MODEL)
    print(f"sweep: {len(models)} model(s) x {features}", flush=True)

    rc = 0
    for model in models:
        engine = ENGINE_BY_MODEL.get(model)
        if engine is None:
            print(f"!! {model}: unknown model; skipped", file=sys.stderr, flush=True)
            continue
        trace_root = Path(args.root) / engine / "traces"
        t0 = time.perf_counter()
        try:
            with TraceReader(trace_root, model, args.corpus, doc_splits=doc_splits,
                             validate_sizes=not args.no_validate_sizes) as reader:
                result = sweep(
                    reader,
                    model=model,
                    out_dir=args.out,
                    features=features,
                    layers=args.layers,
                    force=args.force,
                    seed=args.seed,
                    probe_kwargs=pkw,
                    doc_strata=doc_strata,
                )
        except Exception as exc:
            print(f"!! {model}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            rc = 1
            continue
        dt = time.perf_counter() - t0
        print(
            f"{model:22} done={len(result.layers_done)}L skipped={len(result.layers_skipped)} "
            f"stopped_early={result.stopped_early} ({dt:.0f}s)",
            flush=True,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
