#!/usr/bin/env python3
"""T10.2 reproducibility bundle → release/ .

Writes, without touching any trace or result file:
  BUILD_COMMIT.txt     git HEAD + branch + dirty flag of this repo; llama.cpp / vLLM builds per engine
  requirements.lock    `pip freeze` of the analysis venv
  run_configs.json     configs/run*.yaml with their run_config_sha256 (as the manifests record it)
  checkpoints.json     per model: engine, weights hash (GGUF sha256 / HF model_sha256), quant,
                       checkpoint_status, revision — read from the shard manifests, not the yaml
  corpus.json          corpus path + sha256, split counts, seed/source revisions from the metadata
  invocations.json     exact collection CLI per run (collection_log.csv) and the analysis CLIs
  state/               every shard state.json / manifest.json, copied verbatim
  kaggle_image.txt     Kaggle base image tag(s) recorded in the manifests (``platform`` block)

Usage: python -m scripts.phase10_bundle [--traces "FULL RESULTS"] [--out release]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.phase6_metrics import ENGINE_BY_MODEL

ANALYSIS_CLI = [
    "python -m src.traces.validate <trace_dir>",
    "python -m scripts.phase6_metrics",
    "python -m scripts.phase7_temporal_locality",
    "python -m scripts.phase7_sweep --root <traces> --hidden   # + F6 refit at --probe-kwargs '{\"lr\":0.003}' for the 12 val-flagged layers",
    "python -m scripts.phase7_summary",
    "python -m scripts.phase7_sweep --root <traces> --hidden --layers <5 normalized depths> --doc-strata corpora/mixed-v2.jsonl",
    "python -m scripts.phase9_report",
    "python -m scripts.phase9_domain",
    "python -m scripts.phase9_word_f1",
    "python -m scripts.phase9_sample_size",
    "python -m scripts.phase8_conclusions",
    "python -m src.analysis.precision --reference-root ... --candidate-root ... [--cross-engine]",
    "python -m scripts.phase10_lookahead",
]


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # recorded, never fatal: a bundle with a gap beats no bundle
        return f"UNAVAILABLE ({type(exc).__name__})"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def analysis_distributions(code_dirs: tuple[str, ...] = ("src", "scripts", "tests")) -> set[str]:
    """Distributions the analysis code imports, plus everything they require, transitively.

    A plain ``pip freeze`` also pins whatever else lives in the venv. Here that was editor tooling
    (an editable markitdown checkout whose freeze line is not installable, and a dependency of it
    that PyPI no longer serves), so ``pip install -r requirements.lock`` failed as a whole.
    """
    import ast
    from importlib import metadata

    top = set()
    for d in code_dirs:
        for path in Path(d).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    top |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    top.add(node.module.split(".")[0])
    by_module = metadata.packages_distributions()
    todo = {_norm(dist) for mod in top for dist in by_module.get(mod, [])} | {"pytest"}
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen.add(name)
        for req in dist.requires or []:
            if "extra ==" in req:
                continue
            todo.add(_norm(re.split(r"[\s;<>=!~\[(]", req, maxsplit=1)[0]))
    return seen


def frozen_requirements() -> str:
    """``pip freeze`` of this interpreter, restricted to :func:`analysis_distributions`."""
    keep = analysis_distributions()
    text = sh(sys.executable, "-m", "pip", "freeze", "--exclude-editable")
    return "\n".join(line for line in text.splitlines()
                     if "==" in line and _norm(line.split("==")[0]) in keep)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="FULL RESULTS")
    ap.add_argument("--corpus", default="corpora/mixed-v2.jsonl")
    ap.add_argument("--out", default="release")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "state").mkdir(parents=True, exist_ok=True)

    manifests = {}
    for model, engine in ENGINE_BY_MODEL.items():
        mdir = Path(args.traces) / engine / "traces" / model / "mixed-v2"
        for f in sorted(mdir.glob("shard_*/*.json")) + sorted(mdir.glob("*.json")):
            if f.name in ("manifest.json", "state.json"):
                dst = out / "state" / engine / model / f.relative_to(mdir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
        first = sorted(mdir.glob("shard_*/manifest.json"))
        if first:
            manifests[model] = (engine, json.loads(first[0].read_text("utf-8")))

    engines = {}
    checkpoints = {}
    images = set()
    for model, (engine, m) in manifests.items():
        engines.setdefault(engine, m.get("engine_build") or m.get("llama_cpp_commit"))
        checkpoints[model] = {
            "engine": engine,
            "engine_build": m.get("engine_build"),
            "weights_sha256": m.get("model_sha256") or m.get("gguf_sha256"),
            "quant": m.get("quant"),
            "checkpoint_status": m.get("checkpoint_status"),
            "revision": m.get("revision") or m.get("hf_revision"),
            "run_config_sha256": m.get("run_config_sha256"),
            "logit_tensor_used": m.get("logit_tensor_used"),
        }
        plat = m.get("platform") or {}
        if isinstance(plat, dict):
            for k in ("kaggle_image", "image", "docker_image"):
                if plat.get(k):
                    images.add(str(plat[k]))

    status = sh("git", "status", "--porcelain")
    (out / "BUILD_COMMIT.txt").write_text(
        f"repo HEAD: {sh('git', 'rev-parse', 'HEAD')}\n"
        f"branch: {sh('git', 'rev-parse', '--abbrev-ref', 'HEAD')}\n"
        f"dirty: {bool(status) and not status.startswith('UNAVAILABLE')}\n"
        + "".join(f"{e}: {b}\n" for e, b in sorted(engines.items())),
        encoding="utf-8")
    (out / "requirements.lock").write_text(frozen_requirements() + "\n", encoding="utf-8")
    (out / "kaggle_image.txt").write_text(
        "\n".join(sorted(images)) or "not recorded in manifests — see notebooks/*.ipynb kernel metadata",
        encoding="utf-8")
    (out / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")

    cfgs = {}
    for p in sorted(Path("configs").glob("run*.yaml")):
        cfgs[p.name] = {"file_sha256": sha256(p)}
    try:
        from src.runtime.config import RunConfig  # noqa: WPS433
        for name in cfgs:
            cfgs[name]["run_config_sha256"] = RunConfig.load(Path("configs") / name).sha256
    except Exception as exc:
        cfgs["_note"] = f"run_config_sha256 not recomputed ({type(exc).__name__}); see checkpoints.json"
    (out / "run_configs.json").write_text(json.dumps(cfgs, indent=2), encoding="utf-8")

    corpus = Path(args.corpus)
    splits = json.loads(Path("corpora/mixed-v2.splits.json").read_text("utf-8"))
    meta_p = Path("corpora/moe-corpus-v2-metadata.json")
    (out / "corpus.json").write_text(json.dumps({
        "path": str(corpus), "sha256": sha256(corpus),
        "split_counts": {s: sum(1 for v in splits.values() if v == s) for s in set(splits.values())},
        "metadata": json.loads(meta_p.read_text("utf-8")) if meta_p.exists() else None,
    }, indent=2), encoding="utf-8")

    log = Path("results/collection_log.csv")
    runs = list(csv.DictReader(open(log, encoding="utf-8"))) if log.exists() else []
    (out / "invocations.json").write_text(json.dumps(
        {"collection_log": runs, "analysis": ANALYSIS_CLI}, indent=2), encoding="utf-8")

    print(f"bundle -> {out}/ ({len(manifests)} models, {len(runs)} collection rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
