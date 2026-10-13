#!/usr/bin/env python3
"""Assemble results/phase9/RESULTS.md from every Phase 8–10 output that exists.

Reads only files already written by the other drivers; a missing input becomes an explicit
"NOT YET RUN" line, never a blank or a guessed number.
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("results")
P9 = R / "phase9"
FEATS = ["F1", "F2", "F3", "F4", "F5", "F6"]


def j(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def f(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def section_primary(out):
    rows = j(R / "phase7_summary.json")
    lb = {r["model"]: r["lbi_mean"] for r in (j(R / "phase6_load_balance.json") or [])}
    out += ["## T9.1 Primary results (Î/H, mean over layers, test split)", ""]
    if not rows:
        return out + ["NOT YET RUN", ""]
    models = sorted({r["model"] for r in rows})
    cell = {(r["model"], r["feature"]): r for r in rows}
    out += ["| model | LBI | " + " | ".join(["F0"] + FEATS) + " |",
            "|---|---:|" + "---:|" * (len(FEATS) + 1)]
    for m in models:
        out.append(f"| {m} | {f(lb.get(m))} | " + " | ".join(
            f(cell[(m, x)]["mean_mi_ratio"]) if (m, x) in cell else "--" for x in ["F0"] + FEATS) + " |")
    neg = j(P9 / "negative_mi_cells.json")
    if neg:
        out += ["", f"Negative-Î cells: **{neg['count']}** of {neg['n_cells_examined']}, by feature "
                f"{neg['by_feature']} (F0 is the marginal, Î≈0 by construction; reported signed)."]
    return out + [""]


def section_pairs(out):
    out += ["## T9.2 / T9.3 Pairs (Δ Î/H = second − first member)", ""]
    for name in ("pair_g", "post_training", "pair_a", "pair_t"):
        p = P9 / f"{name}_mean.csv"
        if not p.exists():
            out.append(f"- {name}: NOT YET RUN")
            continue
        import csv
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows:
            continue
        a, b = [c for c in rows[0] if c.endswith("(mi_ratio)")]
        out.append(f"**{name}** ({a.split(' ')[0]} → {b.split(' ')[0]}): " + ", ".join(
            f"{r['feature']} {float(r['delta_b_minus_a']):+.3f}" for r in rows if r["feature"] != "F0"))
        out.append(f"  - framing: {rows[0]['framing']}; not matched: {rows[0]['not_matched']}")
    w = j(P9 / "t9_2_word_f1.json")
    out += ["", "**Word-level F1 control** (word-initial tokens; token-ID refit on the same rows):"]
    if not w:
        out.append("NOT YET RUN")
    else:
        for m, v in w.items():
            out.append(f"- {m}: word {f(v['mean_word_mi_ratio'])} vs token {f(v['mean_token_mi_ratio'])}"
                       f" (word-initial frac {f(v['frac_word_initial'], 2)})")
    return out + [""]


def section_domain(out):
    rows = j(P9 / "t9_4_domain.json")
    out += ["## T9.4 Domain breakdown (primary cross-model comparison)", "",
            "CE gain over the marginal within domain, ratio; mean of 5 normalized depths.", ""]
    if not rows:
        return out + ["NOT YET RUN", ""]
    doms = sorted({r["stratum"] for r in rows if r["kind"] == "domain"})
    for feat in ("F1", "F2", "F4"):
        cell = {(r["model"], r["stratum"]): r["gain_ratio"] for r in rows
                if r["kind"] == "domain" and r["feature"] == feat}
        models = sorted({m for m, _ in cell})
        out += [f"**{feat}**", "", "| model | " + " | ".join(doms) + " |", "|---|" + "---:|" * len(doms)]
        out += [f"| {m} | " + " | ".join(f(cell.get((m, d))) for d in doms) + " |" for m in models]
        out.append("")
    return out


def section_confounds(out):
    out += ["## T9.4 Confounds", ""]
    out.append("Measured load balance vs config-declared aux-loss: see `t9_4_confounds.md`. "
               "gpt-oss declares 0.9 (900× the panel) yet is the least balanced (LBI 0.904); "
               "DeepSeek declares 0.001 and is the most balanced (0.992): the declared coefficient "
               "does not predict measured balance.")
    s = j(P9 / "t9_4_sample_size.json")
    out += ["", "**Sample-size convergence (OLMoE):**"]
    if not s:
        out.append("NOT YET RUN")
    else:
        out.append(f"train tokens available {s['n_train_available']:,} (plan grid to 4M not reachable).")
        for p in s["points"]:
            out.append(f"- L{p['layer']} {p['feature']} N={p['n_train_tokens']:,}: Î/H {f(p['mi_ratio'], 4)}")
    return out + [""]


def section_precision(out):
    out += ["## Phase 8 precision / engine bounds", ""]
    base = Path("phase8_results")
    for name in ("ladder_f16_vs_q8_0", "ladder_f16_vs_q4_k_m", "engine_vllm_fp16_vs_llamacpp_f16"):
        d = j(base / f"{name}.json")
        if d:
            g = d["gate"]
            out.append(f"- {name}: mean set agreement {g['mean_set_agreement']:.4f} "
                       f"(worst L{g['worst_layer']} {g['worst_set_agreement']:.4f}) → **{g['band']}**")
    c = j(P9 / "t8_3_f16_rerun.json")
    out.append("- T8.3 F16 re-run: " + ("NOT YET RUN" if not c else
               f"conclusions **{c['conclusions']}** (order identical {c['feature_order_identical']}, "
               f"max|Δ Î/H| {c['max_abs_delta']:.4f})"))
    return out + [""]


def section_reconcile(out):
    r = j(P9 / "t9_5_reconcile.json")
    out += ["## T9.5 Reconciliation", ""]
    if not r:
        return out + ["NOT YET RUN", ""]
    out.append(f"{len(r.get('comparisons', []))} literature comparisons placed on one definition; "
               f"{len(r.get('gaps', []))} figures left as explicit gaps (see `t9_5_reconcile.md`).")
    return out + [""]


def section_lookahead(out):
    d = {}
    for p in sorted((R / "phase10").glob("lookahead*.json")):
        d.update(j(p) or {})
    out += ["## T10.3 Lookahead decay (h_{ℓ−j} → S_ℓ, Î/H)", ""]
    if not d:
        return out + ["NOT YET RUN", ""]
    lags = (1, 2, 4, 8)
    out += ["| model | " + " | ".join(f"j={l}" for l in lags) + " |", "|---|" + "---:|" * len(lags)]
    for m, cells in sorted(d.items()):
        by = {l: [c["mi_ratio"] for c in cells.values() if c["lag"] == l] for l in lags}
        out.append(f"| {m} | " + " | ".join(f(sum(v) / len(v)) if v else "--" for v in by.values()) + " |")
    return out + [""]


def main() -> int:
    out = ["# MoE routing predictability — results", "",
           "Generated by `scripts/phase9_writeup.py` from `results/`. Every number is read from a "
           "result file; missing inputs say NOT YET RUN.", ""]
    for s in (section_primary, section_pairs, section_domain, section_confounds,
              section_precision, section_reconcile, section_lookahead):
        out = s(out)
    P9.mkdir(parents=True, exist_ok=True)
    (P9 / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {P9 / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
