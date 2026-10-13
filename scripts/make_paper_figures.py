#!/usr/bin/env python3
"""Build the paper-draft figures (docs/figures/*.png) from result files only — no hand-typed numbers.

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
OUT = Path("docs/figures")
MODELS = ["gpt-oss-20b", "olmoe-0125", "olmoe-0125-instruct", "olmoe-0924",
          "deepseek-v2-lite", "qwen3-30b-a3b", "gemma-4-26b-a4b"]
SHORT = {"gpt-oss-20b": "gpt-oss-20b", "olmoe-0125": "OLMoE-0125", "olmoe-0125-instruct": "OLMoE-0125-Inst",
         "olmoe-0924": "OLMoE-0924", "deepseek-v2-lite": "DeepSeek-V2-Lite",
         "qwen3-30b-a3b": "Qwen3-30B-A3B", "gemma-4-26b-a4b": "Gemma-4-26B-A4B"}
COLOR = {"gpt-oss-20b": "#d62728", "olmoe-0125": "#1f77b4", "olmoe-0125-instruct": "#6baed6",
         "olmoe-0924": "#08306b", "deepseek-v2-lite": "#2ca02c", "qwen3-30b-a3b": "#9467bd",
         "gemma-4-26b-a4b": "#ff7f0e"}
FEATS = ["F1", "F2", "F3", "F4", "F5", "F6"]
FNAME = {"F0": "marginal", "F1": "token id", "F2": "prev-layer set", "F3": "prev-token set",
         "F4": "router input (linear)", "F5": "router input (MLP)", "F6": "F1+F2+F3"}


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_ladder():
    cell = {(r["model"], r["feature"]): r["mean_mi_ratio"] for r in j(R / "phase7_summary.json")}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.12
    for i, m in enumerate(MODELS):
        xs = [x + (i - 3) * w for x in range(len(FEATS))]
        ax.bar(xs, [cell[(m, f)] for f in FEATS], w, label=SHORT[m], color=COLOR[m])
    ax.set_xticks(range(len(FEATS)), [f"{f}\n{FNAME[f]}" for f in FEATS], fontsize=8)
    ax.set_ylabel("Î/H (share of routing entropy explained)")
    ax.set_title("Fig. 1 — Predictability ladder (test split, mean over layers)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig1_ladder.png")


def fig_depth():
    rs = rows(R / "phase9" / "t9_4_depth_mi_ratio.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, f in zip(axes.flat, FEATS):
        for m in MODELS:
            pts = [(float(r["normalized_depth_mid"]), float(r[f])) for r in rs if r["model"] == m]
            ax.plot(*zip(*pts), marker="o", ms=3, color=COLOR[m], label=SHORT[m])
        ax.set_title(f"{f} — {FNAME[f]}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("relative depth ℓ/(L−1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Î/H")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Fig. 2 — Predictability versus relative depth (5 bins)")
    save(fig, "fig2_depth.png")


def fig_domain():
    d = j(R / "phase9" / "t9_4_domain.json")
    agg = collections.defaultdict(list)
    for r in d:
        if r["kind"] == "domain":
            agg[(r["feature"], r["model"], r["stratum"])].append(r["gain_ratio"])
    doms = ["code", "math", "prose", "multilingual"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, f in zip(axes, ["F1", "F2", "F4"]):
        mat = [[st.mean(agg[(f, m, s)]) for s in doms] for m in MODELS]
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(4), doms, fontsize=8)
        ax.set_yticks(range(len(MODELS)), [SHORT[m] for m in MODELS], fontsize=7)
        for i, row in enumerate(mat):
            for k, v in enumerate(row):
                ax.text(k, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5, color="w")
        ax.set_title(f"{f} — {FNAME[f]}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig. 3 — Within-domain gain over the marginal, (CE₀ − CE)/CE₀")
    save(fig, "fig3_domain.png")


def fig_budget():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, f in zip(axes, ["F1", "F2", "F4"]):
        for m in MODELS:
            doc = j(R / "probes" / f"{m}__{f}.json")
            k = doc["top_k"]
            fa = [L["metrics"]["family_a"] for L in doc["layers"].values()
                  if "metrics" in L and "family_a" in L["metrics"]]
            ys = [st.mean(x[f"recall@{b * k}"] for x in fa) for b in (1, 2, 4)]
            ax.plot([1, 2, 4], ys, marker="o", color=COLOR[m], label=SHORT[m])
        ax.set_xticks([1, 2, 4], ["k", "2k", "4k"])
        ax.set_xlabel("prefetch budget (experts, multiples of k)")
        ax.set_title(f"{f} — {FNAME[f]}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recall of executed experts")
    axes[0].legend(fontsize=6)
    fig.suptitle("Fig. 4 — Budgeted recall: the same predictors look near-perfect at 4k")
    save(fig, "fig4_budget.png")


def lookahead_common():
    la = {}
    for p in sorted((R / "phase10").glob("lookahead*.json")):
        la.update(j(p))
    out = {}
    for m, cells in la.items():
        tg = collections.defaultdict(dict)
        for c in cells.values():
            tg[c["target_layer"]][c["lag"]] = c["mi_ratio"]
        common = [t for t in tg if all(l in tg[t] for l in (1, 2, 4, 8))]
        out[m] = {l: st.mean(tg[t][l] for t in common) for l in (1, 2, 4, 8)}
    return out


def fig_lookahead():
    la = lookahead_common()
    nl = {m: j(R / "probes" / f"{m}__F4.json")["n_moe_layers"] for m in la}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in MODELS:
        if m not in la:
            continue
        ys = [la[m][l] / la[m][1] for l in (1, 2, 4, 8)]
        axes[0].plot([1, 2, 4, 8], [la[m][l] for l in (1, 2, 4, 8)], marker="o", color=COLOR[m], label=SHORT[m])
        axes[1].plot([l / (nl[m] - 1) for l in (1, 2, 4, 8)], ys, marker="o", color=COLOR[m], label=SHORT[m])
    axes[0].set_xlabel("lookahead j (layers)")
    axes[0].set_ylabel("Î/H")
    axes[0].set_title("absolute lag", fontsize=9)
    axes[1].set_xlabel("lookahead as a fraction of depth, j/(L−1)")
    axes[1].set_ylabel("Î/H relative to j=1")
    axes[1].set_title("fractional lag", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig. 5 — Lookahead decay: router input of layer ℓ−j predicting the set at ℓ (common targets)")
    save(fig, "fig5_lookahead.png")


def fig_precision():
    names = [("ladder_f16_vs_q8_0", "F16 vs Q8_0"), ("ladder_f16_vs_q4_k_m", "F16 vs Q4_K_M"),
             ("engine_vllm_fp16_vs_llamacpp_f16", "vLLM fp16 vs llama.cpp F16")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n, lab in names:
        L = j(Path("phase8_results") / f"{n}.json")["comparison"]["layers"]
        q = [st.mean(l["flip_rate_by_margin"][i] for l in L) for i in range(5)]
        axes[0].plot(range(1, 6), q, marker="o", label=lab)
        axes[1].plot([l["depth"] for l in L], [1 - l["set_agreement"] for l in L], marker="o", label=lab)
    axes[0].set_xticks(range(1, 6), ["Q1\n(smallest)", "Q2", "Q3", "Q4", "Q5\n(largest)"])
    axes[0].set_xlabel("k-th vs (k+1)-th router margin quintile (reference model)")
    axes[0].set_ylabel("rows whose expert set changed")
    axes[0].set_title("set-flip rate vs decision margin", fontsize=9)
    axes[1].set_xlabel("relative depth")
    axes[1].set_ylabel("slot disagreement (1 − top-k overlap)")
    axes[1].set_title("disagreement vs depth", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Fig. 6 — Precision and engine sensitivity (OLMoE-0125, 151,285 tokens)")
    save(fig, "fig6_precision.png")


def fig_balance():
    rs = rows(R / "phase9" / "t9_4_confounds.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rs:
        m = r["model"]
        c = r["aux_loss_coef_declared"]
        x = float(c) if c not in ("", "--", "None") else 1e-4
        ax.scatter(x, float(r["load_balance_measured"]), color=COLOR.get(m, "k"), s=50,
                   marker="o" if c not in ("", "--", "None") else "x")
        ax.annotate(SHORT.get(m, m) + ("" if c not in ("", "--", "None") else " (none declared; x arbitrary)"),
                    (x, float(r["load_balance_measured"])), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("declared auxiliary-loss coefficient (config, log scale)")
    ax.set_ylabel("measured load-balance index H/log₂N")
    ax.set_title("Fig. 7 — Declared balancing pressure vs measured balance", fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "fig7_balance.png")


def fig_pairg():
    rs = rows(R / "phase9" / "pair_g_per_normalized_depth.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = sorted({r["depth_bin"] for r in rs})
    for f in FEATS:
        ax.plot(range(len(bins)), [float(next(r["delta_b_minus_a"] for r in rs
                                             if r["depth_bin"] == b and r["feature"] == f)) for b in bins],
                marker="o", label=f"{f} {FNAME[f]}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-0.02, 0.02, color="grey", alpha=0.15, label="±0.02 probe-stability band")
    ax.set_xticks(range(len(bins)), bins, fontsize=8)
    ax.set_xlabel("relative depth bin")
    ax.set_ylabel("Δ Î/H (OLMoE-0125 − gpt-oss-20b)")
    ax.set_title("Fig. 8 — Pair G by depth: gpt-oss leads early, OLMoE catches up late", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, "fig8_pair_g_depth.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_ladder, fig_depth, fig_domain, fig_budget, fig_lookahead, fig_precision, fig_balance, fig_pairg):
        f()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
