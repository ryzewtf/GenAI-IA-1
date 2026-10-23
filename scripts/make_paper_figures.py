#!/usr/bin/env python3
"""Build the paper-draft figures (docs/figures/*.png) from result files only — no hand-typed numbers.

v0.2 (audit): every Family B axis is the attainable share Î/(H − log2 k); the ridge router-input
probe F4R is shown next to the softmax/MLP probes; error bars are 95% document-bootstrap intervals;
lookahead uses the converged refit; figures are numbered in the order the draft cites them.

Inputs: results/audit/bootstrap.json, results/phase9_v2/, results/probes_v2/,
results/phase10/lookahead_conv_*.json, phase8_results/.

Usage: python -m scripts.make_paper_figures
"""
from __future__ import annotations

import collections
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results")
V2 = R / "phase9_v2"
PROBES = R / "probes_v2"
OUT = Path("docs/figures")
MODELS = ["gpt-oss-20b", "olmoe-0924", "olmoe-0125", "olmoe-0125-instruct",
          "deepseek-v2-lite", "qwen3-30b-a3b", "gemma-4-26b-a4b"]
SHORT = {"gpt-oss-20b": "gpt-oss-20b", "olmoe-0125": "OLMoE-0125", "olmoe-0125-instruct": "OLMoE-0125-Inst",
         "olmoe-0924": "OLMoE-0924", "deepseek-v2-lite": "DeepSeek-V2-Lite",
         "qwen3-30b-a3b": "Qwen3-30B-A3B", "gemma-4-26b-a4b": "Gemma-4-26B-A4B"}
# Categorical slots 1-7 of the validated reference palette, in fixed model order (dataviz skill:
# adjacent-pair CVD ΔE ≥ 9.1, normal-vision ≥ 19.6 on light). Seven hues cannot clear the all-pairs
# floor, so every model also has its own marker: identity never rests on colour alone.
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
MARKER = dict(zip(MODELS, ["o", "s", "^", "v", "D", "P", "X"]))
FEATS = ["F1", "F2", "F3", "F4", "F4R", "F5", "F6"]
FNAME = {"F0": "marginal", "F1": "token id", "F2": "prev-layer set", "F3": "prev-token set",
         "F4": "router input ℓ−1, softmax", "F4R": "router input ℓ−1, ridge", "F5": "router input ℓ−1, MLP",
         "F6": "F1+F2+F3"}
ATT = "Î/(H − log₂k)  (share of attainable information)"


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig1_ladder():
    cells = {(c["model"], c["feature"]): c["mi_attainable"] for c in j(R / "audit" / "bootstrap.json")["cells"]}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.115
    for i, m in enumerate(MODELS):
        xs, ys, lo, hi = [], [], [], []
        for x, f in enumerate(FEATS):
            c = cells.get((m, f))
            if c:
                xs.append(x + (i - 3) * w)
                ys.append(c["point"])
                lo.append(c["point"] - c["basic_lo"])
                hi.append(c["basic_hi"] - c["point"])
        ax.bar(xs, ys, w * 0.92, yerr=[lo, hi], capsize=1.5, error_kw={"lw": 0.7},
               label=SHORT[m], color=COLOR[m])
    ax.set_xticks(range(len(FEATS)), [f"{f}\n{FNAME[f].replace(', ', chr(10))}" for f in FEATS], fontsize=7.5)
    ax.set_ylabel(ATT, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 1 — Predictability ladder (test split, mean over layers, 95% document-bootstrap CI)",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=4, frameon=False)
    style(ax)
    save(fig, "fig1_ladder.png")


def fig2_depth():
    rs = rows(V2 / "t9_4_depth_mi_attainable.csv")
    feats = ["F1", "F2", "F3", "F4", "F4R", "F5"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, f in zip(axes.flat, feats):
        for m in MODELS:
            pts = [(float(r["normalized_depth_mid"]), float(r[f])) for r in rs
                   if r["model"] == m and r.get(f) not in (None, "", "--")]
            if pts:
                ax.plot(*zip(*pts), marker=MARKER[m], ms=4, lw=1.6, color=COLOR[m], label=SHORT[m])
        ax.set_title(f"{f} — {FNAME[f]}", fontsize=9)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("relative depth ℓ/(L−1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Î/(H − log₂k)")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("Fig. 2 — Predictability versus relative depth (5 bins)", fontsize=10)
    save(fig, "fig2_depth.png")


def fig3_budget():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, f in zip(axes, ["F1", "F2", "F4R"]):
        for m in MODELS:
            doc = j(PROBES / f"{m}__{f}.json")
            k = doc["top_k"]
            fa = [L["metrics"]["family_a"] for L in doc["layers"].values()
                  if "metrics" in L and "family_a" in L["metrics"]]
            ys = [st.mean(x[f"recall@{b * k}"] for x in fa) for b in (1, 2, 4)]
            ax.plot([1, 2, 4], ys, marker=MARKER[m], lw=1.6, color=COLOR[m], label=SHORT[m])
        ax.set_xticks([1, 2, 4], ["k", "2k", "4k"])
        ax.set_xlabel("prefetch budget (experts, multiples of k)")
        ax.set_title(f"{f} — {FNAME[f]}", fontsize=9)
        style(ax)
    axes[0].set_ylabel("recall of executed experts")
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Fig. 3 — Budgeted recall: the same routing looks near-perfect at a 4k budget", fontsize=10)
    save(fig, "fig3_budget.png")


def fig4_pairg():
    rs = rows(V2 / "pair_g_per_normalized_depth_attainable.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = sorted({r["depth_bin"] for r in rs})
    feat_color = dict(zip(FEATS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
    feat_marker = dict(zip(FEATS, ["o", "s", "^", "v", "D", "P", "X"]))
    for f in FEATS:
        ys = [next((float(r["delta_b_minus_a"]) for r in rs if r["depth_bin"] == b and r["feature"] == f), None)
              for b in bins]
        if all(y is not None for y in ys):
            ax.plot(range(len(bins)), ys, marker=feat_marker[f], lw=1.6, color=feat_color[f],
                    label=f"{f} {FNAME[f]}")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(range(len(bins)), bins, fontsize=8)
    ax.set_xlabel("relative depth bin")
    ax.set_ylabel("Δ Î/(H − log₂k)  (OLMoE-0125 − gpt-oss-20b)")
    ax.set_title("Fig. 4 — Pair G by depth: gpt-oss leads in the first fifth on every feature;\n"
                 "OLMoE leads on token/set features from depth 0.2 and on every feature from 0.6",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    style(ax)
    save(fig, "fig4_pair_g_depth.png")


def fig5_domain():
    strata = j(R / "audit" / "bootstrap.json")["strata"]
    val = {(s["feature"], s["model"], s["stratum"]): s["mi_attainable"]["bc"]
           for s in strata if s["kind"] == "domain"}
    doms = ["code", "math", "prose", "multilingual"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, f in zip(axes, ["F1", "F2", "F4R"]):
        mat = [[val.get((f, m, s), float("nan")) for s in doms] for m in MODELS]
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(4), doms, fontsize=8)
        ax.set_yticks(range(len(MODELS)), [SHORT[m] for m in MODELS], fontsize=7)
        lo, hi = im.get_clim()
        for i, row in enumerate(mat):
            for k, v in enumerate(row):
                ax.text(k, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > (lo + hi) / 2 else "#1a1a19")
        ax.set_title(f"{f} — {FNAME[f]}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig. 5 — Within-domain Î/(H − log₂k), each domain against its own marginal (bias-corrected)",
                 fontsize=10)
    save(fig, "fig5_domain.png")


def fig6_lookahead():
    la = {}
    for p in sorted((R / "phase10").glob("lookahead_conv_*.json")):
        la.update(j(p))
    nl = {m: j(PROBES / f"{m}__F4.json")["n_moe_layers"] for m in la}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in MODELS:
        if m not in la:
            continue
        tg = collections.defaultdict(dict)
        for c in la[m].values():
            tg[c["target_layer"]][c["lag"]] = c["mi_attainable"]
        common = [t for t in tg if all(lag in tg[t] for lag in (1, 2, 4, 8))]
        if not common:
            continue
        mean = {lag: st.mean(tg[t][lag] for t in common) for lag in (1, 2, 4, 8)}
        lags = (1, 2, 4, 8)
        axes[0].plot(lags, [mean[x] for x in lags], marker=MARKER[m], lw=1.6, color=COLOR[m], label=SHORT[m])
        axes[1].plot([x / (nl[m] - 1) for x in lags], [mean[x] / mean[1] for x in lags], marker=MARKER[m],
                     lw=1.6, color=COLOR[m], label=SHORT[m])
    axes[0].set_xlabel("lookahead j (layers)")
    axes[0].set_ylabel("Î/(H − log₂k)")
    axes[0].set_title("absolute lag", fontsize=9)
    axes[1].set_xlabel("lookahead as a fraction of depth, j/(L−1)")
    axes[1].set_ylabel("relative to j = 1")
    axes[1].set_title("fractional lag", fontsize=9)
    for ax in axes:
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Fig. 6 — Lookahead decay: router input of layer ℓ−j predicting the set at ℓ "
                 "(converged softmax probe, common targets)", fontsize=10)
    save(fig, "fig6_lookahead.png")


def fig7_balance():
    rs = rows(V2 / "t9_4_confounds.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in rs:
        m = r["model"]
        c = r["aux_loss_coef_declared"]
        declared = c not in ("", "--", "None")
        x = float(c) if declared else 1e-4
        ax.scatter(x, float(r["load_balance_measured"]), color=COLOR.get(m, "k"), s=55,
                   marker=MARKER.get(m, "o"), edgecolor="white", linewidth=1)
        ax.annotate(SHORT.get(m, m) + ("" if declared else " (none declared; x arbitrary)"),
                    (x, float(r["load_balance_measured"])), fontsize=7, xytext=(5, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("declared auxiliary-loss coefficient (config, log scale)")
    ax.set_ylabel("measured load-balance index H/log₂N")
    ax.set_title("Fig. 7 — Declared balancing pressure vs measured balance (all 7 checkpoints)", fontsize=9)
    style(ax)
    save(fig, "fig7_balance.png")


def fig8_precision():
    names = [("ladder_f16_vs_q8_0", "F16 vs Q8_0"), ("ladder_f16_vs_q4_k_m", "F16 vs Q4_K_M"),
             ("engine_vllm_fp16_vs_llamacpp_f16", "vLLM fp16 vs llama.cpp F16")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (n, lab), col, mk in zip(names, colors, markers):
        L = j(Path("phase8_results") / f"{n}.json")["comparison"]["layers"]
        q = [st.mean(x["flip_rate_by_margin"][i] for x in L) for i in range(5)]
        axes[0].plot(range(1, 6), q, marker=mk, lw=1.6, color=col, label=lab)
        axes[1].plot([x["depth"] for x in L], [1 - x["set_agreement"] for x in L], marker=mk, lw=1.6,
                     color=col, label=lab)
    axes[0].set_xticks(range(1, 6), ["Q1\n(smallest)", "Q2", "Q3", "Q4", "Q5\n(largest)"])
    axes[0].set_xlabel("k-th vs (k+1)-th router margin quintile (reference model)")
    axes[0].set_ylabel("rows whose expert set changed")
    axes[0].set_title("set-flip rate vs decision margin", fontsize=9)
    axes[1].set_xlabel("relative depth")
    axes[1].set_ylabel("slot disagreement (1 − top-k overlap)")
    axes[1].set_title("disagreement vs depth", fontsize=9)
    for ax in axes:
        style(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Fig. 8 — Precision and engine sensitivity (OLMoE-0125, 151,285 tokens)", fontsize=10)
    save(fig, "fig8_precision.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1_ladder, fig2_depth, fig3_budget, fig4_pairg, fig5_domain, fig6_lookahead, fig7_balance,
              fig8_precision):
        f()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
