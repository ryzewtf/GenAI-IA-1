#!/usr/bin/env python3
"""Render the v0.2 paper tables from the audit result files (no number is typed by hand).

Inputs (each table is skipped with a note when its input is missing):
  results/audit/bootstrap.json            cells, pairs, ladders, strata (document bootstrap)
  results/audit/convergence.json          F4/FV convergence study
  results/phase10/lookahead_conv_*.json   converged lookahead
  results/audit/seeds.json                F5 seed spread (scripts.audit_seeds)
  results/audit/word_control_attainable.json
  results/phase9_v2/t9_4_depth_mi_attainable.csv

Output: results/audit/paper_tables.md

Usage: python -m scripts.audit_paper_tables [--fill docs/PAPER_DRAFT.md]
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

A = Path("results/audit")
MODELS = ["gpt-oss-20b", "olmoe-0924", "olmoe-0125", "olmoe-0125-instruct", "deepseek-v2-lite",
          "qwen3-30b-a3b", "gemma-4-26b-a4b"]
LABEL = {"gpt-oss-20b": "gpt-oss-20b", "olmoe-0924": "OLMoE-0924", "olmoe-0125": "OLMoE-0125",
         "olmoe-0125-instruct": "OLMoE-0125-Instruct", "deepseek-v2-lite": "DeepSeek-V2-Lite",
         "qwen3-30b-a3b": "Qwen3-30B-A3B", "gemma-4-26b-a4b": "Gemma-4-26B-A4B"}
LADDER = ["F1", "F2", "F3", "F4", "F4R", "F5", "F6"]
PAIR_LABEL = {"pair_g": "Pair G (OLMoE-0125 − gpt-oss)", "pair_t": "Pair T (Gemma − Qwen3)",
              "pair_a": "Pair A (0125 − 0924)", "post_training": "Post-training (Instruct − 0125)"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ci(s: dict, bc: bool = False) -> str:
    pt = s["bc"] if bc else s["point"]
    return f"{pt:.3f} [{s['basic_lo']:.3f}, {s['basic_hi']:.3f}]"


def signed_ci(s: dict) -> str:
    mark = "" if s.get("resolved") else " (n.s.)"
    return f"{s['point']:+.3f} [{s['basic_lo']:+.3f}, {s['basic_hi']:+.3f}]{mark}"


def table(header: list[str], rows: list[list[str]], align_first: int = 1) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * align_first + "---:|" * (len(header) - align_first)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def ladder_tables(b: dict) -> list[str]:
    cells = {(c["model"], c["feature"]): c for c in b["cells"]}
    md = ["## T1. Predictability ladder, Î/(H − log₂k), mean over layers [95% CI]", ""]
    rows = [[LABEL[m]] + [ci(cells[(m, f)]["mi_attainable"]) if (m, f) in cells else "—" for f in LADDER]
            for m in MODELS]
    md += table(["Model"] + LADDER, rows) + [""]
    md += ["Validation probes (router input of layer ℓ itself): FV softmax / FVR ridge", ""]
    rows = [[LABEL[m]] + [ci(cells[(m, f)]["mi_attainable"]) if (m, f) in cells else "—" for f in ("FV", "FVR")]
            for m in MODELS]
    md += table(["Model", "FV", "FVR"], rows) + [""]

    md += ["## T2. Family A: top-k overlap (= recall at budget k), mean over layers; exact-set match", ""]
    feats = ["F0"] + LADDER
    rows = []
    for m in MODELS:
        r = [LABEL[m]] + [f"{cells[(m, f)]['recall@k']['point']:.3f}" if (m, f) in cells else "—" for f in feats]
        r += [f"{cells[(m, f)]['exact_match']['point']:.3f}" if (m, f) in cells else "—" for f in ("F4", "F4R", "F5")]
        rows.append(r)
    md += table(["Model"] + feats + ["exact F4", "exact F4R", "exact F5"], rows) + [""]

    md += ["## T4. Budgeted recall at k / 2k / 4k (mean over layers)", ""]
    rows = []
    for m in MODELS:
        r = [LABEL[m]]
        for f in ("F0", "F1", "F2", "F4", "F4R"):
            c = cells.get((m, f))
            r.append(" / ".join(f"{c[x]['point']:.3f}" for x in ("recall@k", "recall@2k", "recall@4k")) if c else "—")
        rows.append(r)
    md += table(["Model", "F0", "F1", "F2", "F4", "F4R"], rows) + [""]
    return md


def pair_tables(b: dict) -> list[str]:
    md = ["## T5. Pairs: Δ (second − first) of layer means [95% CI]; n.s. = interval contains 0", ""]
    by = defaultdict(dict)
    for p in b["pairs"]:
        by[p["pair"]][p["feature"]] = p
    for pair, label in PAIR_LABEL.items():
        if pair not in by:
            continue
        md += [f"**{label}**", ""]
        rows = [[f, signed_ci(by[pair][f]["mi_attainable"]), signed_ci(by[pair][f]["recall@k"]),
                 signed_ci(by[pair][f]["recall@2k"])] for f in ["F0"] + LADDER + ["FV", "FVR"] if f in by[pair]]
        md += table(["Feature", "Δ Î/(H−log₂k)", "Δ recall@k", "Δ recall@2k"], rows) + [""]
    md += ["## T5b. Within-model predictor differences, Δ Î/(H−log₂k) [95% CI]", ""]
    diffs = ["F4R − F4", "F4R − F5", "F5 − F4", "FVR − F4R", "FV − F4", "F6 − F2", "F2 − F1", "F4R − F2"]
    lad = defaultdict(dict)
    for r in b["ladders"]:
        lad[r["model"]][f"{r['a']} − {r['b']}"] = r[f"mi_attainable ({r['a']}-{r['b']})"]
    rows = [[LABEL[m]] + [signed_ci(lad[m][d]) if d in lad[m] else "—" for d in diffs] for m in MODELS]
    md += table(["Model"] + diffs, rows) + [""]
    return md


def strata_tables(b: dict) -> list[str]:
    md = ["## T6. Language: bias-corrected Î/(H−log₂k), FLORES-200 documents excluded", "",
          "Each language scored against its own marginal on the same rows. Mean of the eight non-English "
          "languages vs English; relative change; per-language range.", ""]
    s = defaultdict(dict)
    for x in b["strata"]:
        s[(x["kind"], x["feature"], x["model"])][x["stratum"]] = x
    rows = []
    for m in MODELS:
        r = [LABEL[m]]
        for f in ("F1", "F2", "F4", "F4R", "F5"):
            d = s.get(("lang_no_flores", f, m))
            if not d or "en" not in d:
                r.append("—")
                continue
            en = d["en"]["mi_attainable"]["bc"]
            other = [v["mi_attainable"]["bc"] for k, v in d.items() if k != "en"]
            rel = [(o - en) / en * 100 for o in other]
            r.append(f"{en:.3f} → {st.mean(other):.3f} ({st.mean(rel):+.0f}%; {min(rel):+.0f}…{max(rel):+.0f}%)")
        rows.append(r)
    md += table(["Model", "F1", "F2", "F4", "F4R", "F5"], rows) + [""]
    langs = ["en", "de", "es", "fr", "ru", "ar", "hi", "ja", "zh"]
    md += ["## A.5 Per-language Î/(H−log₂k), bias-corrected, FLORES excluded (F2 / F4R); test documents in header", ""]
    nd = {}
    for (kind, f, m), d in s.items():
        if kind == "lang_no_flores":
            for k, v in d.items():
                nd[k] = max(nd.get(k, 0), v["n_test_docs"])
    rows = []
    for m in MODELS:
        a, c = s.get(("lang_no_flores", "F2", m), {}), s.get(("lang_no_flores", "F4R", m), {})
        rows.append([LABEL[m]] + [f"{a[l]['mi_attainable']['bc']:.3f} / {c[l]['mi_attainable']['bc']:.3f}"
                                  if l in a and l in c else "—" for l in langs])
    md += table(["Model"] + [f"{l} ({nd.get(l, 0)})" for l in langs], rows) + [""]
    md += ["## T7. Domain: bias-corrected Î/(H−log₂k), own marginal on the same rows (F2 / F4R / F5)", ""]
    doms = ["code", "math", "prose", "multilingual"]
    rows = []
    for m in MODELS:
        r = [LABEL[m]]
        for dname in doms:
            vals = [s.get(("domain", f, m), {}).get(dname) for f in ("F2", "F4R", "F5")]
            r.append(" / ".join(f"{v['mi_attainable']['bc']:.3f}" if v else "—" for v in vals))
        rows.append(r)
    md += table(["Model"] + doms, rows) + [""]
    return md


def convergence_table(c: dict) -> list[str]:
    md = ["## T8. Convergence study (F4/FV at three depths, L2 chosen on val, maxiter 2000)", "",
          f"{c['n_cells']} cells, {c['n_converged']} converged (scipy success flag), "
          f"{c['n_selected_on_grid_edge']} selected at a grid edge; L2 grid {c['l2_grid']}.", ""]
    by = defaultdict(list)
    for r in c["cells"]:
        by[r["model"]].append(r)
    rows = []
    for m in MODELS:
        if m not in by:
            continue
        rs = by[m]
        d = [r["delta_attainable"] for r in rs]
        rows.append([LABEL[m], ", ".join(sorted({f"{r['selected_l2']:g}" for r in rs})),
                     f"{min(r['capped_150']['attainable'] for r in rs):.3f}–{max(r['capped_150']['attainable'] for r in rs):.3f}",
                     f"{min(r['selected']['attainable'] for r in rs):.3f}–{max(r['selected']['attainable'] for r in rs):.3f}",
                     f"{min(d):+.3f} … {max(d):+.3f}",
                     f"{min(r['selected']['n_iter'] for r in rs)}–{max(r['selected']['n_iter'] for r in rs)}"])
    md += table(["Model", "val-selected L2", "150-iteration fit", "converged fit", "Δ", "iterations"], rows) + [""]
    return md


def lookahead_table() -> list[str]:
    res = {}
    for p in sorted(Path("results/phase10").glob("lookahead_conv_*.json")):
        res.update(load(p))
    if not res:
        return ["## T9. Lookahead — not yet available", ""]
    md = ["## T9. Lookahead (converged softmax probe): Î/(H−log₂k) from the router input j layers back", "",
          "Mean over target depths {0.3, 0.5, 0.7, 0.9} valid at every lag (targets with ℓ − 8 ≥ 0).", ""]
    rows = []
    for m in MODELS:
        if m not in res:
            continue
        cells = list(res[m].values())
        have = {(c["target_layer"], c["lag"]) for c in cells}
        targets = sorted({t for t, _ in have if all((t, j) in have for j in (1, 2, 4, 8))})
        if not targets:  # run still in progress
            continue
        mean = {j: st.mean(c["mi_attainable"] for c in cells if c["lag"] == j and c["target_layer"] in targets)
                for j in (1, 2, 4, 8)}
        n_layers = {"gpt-oss-20b": 24, "deepseek-v2-lite": 26, "qwen3-30b-a3b": 48,
                    "gemma-4-26b-a4b": 30}.get(m, 16)
        conv = sum(bool(c.get("converged")) for c in cells)
        rows.append([LABEL[m]] + [f"{mean[j]:.3f}" for j in (1, 2, 4, 8)]
                    + [f"{mean[8] / mean[1] * 100:.0f}%", f"{mean[2] / mean[1] * 100:.0f}%",
                       f"{mean[4] / mean[1] * 100:.0f}%", f"{8 / (n_layers - 1):.2f}", f"{conv}/{len(cells)}"])
    md += table(["Model", "j=1", "j=2", "j=4", "j=8", "kept at 8", "kept at 2", "kept at 4",
                 "8/(L−1)", "converged"], rows) + [""]
    return md


def seeds_table(s: dict) -> list[str]:
    md = ["## T10. F5 seed spread (seeds 0, 1, 2; convergence-study layers)", "",
          "Per cell: max − min of Î/(H−log₂k) over seeds; summarised per model.", ""]
    rows = [[LABEL[m], f"{v['n_cells']}", f"{v['mean_range']:.4f}", f"{v['max_range']:.4f}",
             f"{v['mean_sd']:.4f}"] for m, v in s["per_model"].items()]
    md += table(["Model", "cells", "mean range", "max range", "mean SD"], rows) + [""]
    return md


def word_table(w: dict) -> list[str]:
    md = ["## T11. Word-level control, Î/(H−log₂k), word-initial tokens", ""]
    rows = [[LABEL[m], f"{v['word_attainable']:.3f}", f"{v['token_same_rows_attainable']:.3f}",
             f"{v['word_over_token']:.2f}", f"{v['frac_word_initial']:.2f}",
             f"{v['per_depth']['0.1']['word']:.3f}"] for m, v in w.items()]
    md += table(["Model", "word", "token, same rows", "word/token", "word-initial share", "word @ depth 0.1"], rows) + [""]
    return md


def depth_table() -> list[str]:
    p = Path("results/phase9_v2/t9_4_depth_mi_attainable.csv")
    if not p.exists():
        return ["## T3. Depth — not yet available", ""]
    rows_in = list(csv.DictReader(p.open(encoding="utf-8")))
    md = ["## T3. Î/(H−log₂k) by relative depth bin (F1 / F2 / F4R)", ""]
    bins = sorted({r["normalized_depth_mid"] for r in rows_in}, key=float)
    rows = []
    for m in MODELS:
        r = [LABEL[m]]
        for bmid in bins:
            x = next((y for y in rows_in if y["model"] == m and y["normalized_depth_mid"] == bmid), None)
            r.append(" / ".join(f"{float(x[f]):.2f}" if x and x.get(f) not in (None, "", "--") else "—"
                                for f in ("F1", "F2", "F4R")))
        rows.append(r)
    md += table(["Model"] + [f"depth {b}" for b in bins], rows) + [""]
    return md


def sections(md: list[str]) -> dict[str, list[str]]:
    """Split the rendered file into {key: body lines} on its '## <key>. ' headings (key = T1, A.5, ...)."""
    out, key = {}, None
    for line in md:
        if line.startswith("## "):
            key = line[3:].split(" ", 1)[0].rstrip(".")
            out[key] = []
        elif key is not None:
            out[key].append(line)
    return {k: v for k, v in out.items() if "not yet available" not in " ".join(v)}


def fill(draft: Path, secs: dict[str, list[str]]) -> int:
    """Replace every block between <!-- table:KEY --> and <!-- /table:KEY --> with the generated body."""
    lines = draft.read_text(encoding="utf-8").splitlines()
    out, i, n = [], 0, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if line.startswith("<!-- table:") and not line.startswith("<!-- table:/"):
            key = line[len("<!-- table:"):].split(" ")[0].rstrip("->").strip()
            end = next(j for j in range(i + 1, len(lines)) if lines[j].startswith(f"<!-- /table:{key} "))
            if key not in secs:
                raise SystemExit(f"no generated table for {key}")
            body = secs[key]
            while body and not body[0].strip():
                body = body[1:]
            while body and not body[-1].strip():
                body = body[:-1]
            out += body
            out.append(lines[end])
            i, n = end + 1, n + 1
            continue
        i += 1
    draft.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default=None, help="draft whose <!-- table:KEY --> blocks are replaced")
    args = ap.parse_args(argv)
    md = ["# v0.2 paper tables (generated by scripts/audit_paper_tables.py)", ""]
    b = load(A / "bootstrap.json")
    if b:
        md += [f"Bootstrap: {b['n_boot']} replicates over {b['n_docs_universe']} test documents; "
               "95% basic intervals.", ""]
        md += ladder_tables(b) + pair_tables(b) + strata_tables(b)
    md += depth_table()
    c = load(A / "convergence.json")
    if c:
        md += convergence_table(c)
    md += lookahead_table()
    s = load(A / "seeds.json")
    md += seeds_table(s) if s else ["## T10. Seeds — not yet available", ""]
    w = load(A / "word_control_attainable.json")
    if w:
        md += word_table(w)
    out = A / "paper_tables.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out} ({len(md)} lines)")
    if args.fill:
        print(f"filled {fill(Path(args.fill), sections(md))} table blocks in {args.fill}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
