"""
make_figures.py: every figure of Chapters Four and Five (static PNG, 150 dpi,
in results/figures/).

  fig01_s1_all_cells.png            Table 4.2 as small multiples (mean, 95% CI)
  fig02_s1_learning_curves.png      Figure 5.2 (with and without B0)
  fig03_lambda_sensitivity.png      Figure 5.1 on S1, S3 and S5
  fig04_primary_pairwise_a12.png    Tables 4.5 and 4.6 as an A12 matrix with significance
  fig05_rank_heatmap.png            rank of every algorithm in every synthetic cell
  fig06_s3_curves.png               S3 curves with the ten change-points
  fig07_s4_synthetic.png            S4 synthetic: curves and final regret
  fig08_pareto_fronts.png           Figure 4.1: Pareto fronts, measured latency
  fig09_guf_weight_sweep.png        Tables 4.16/4.18/4.20 as rank-vs-w1 lines
  fig10_latency_profile.png         Table 4.9 measured latency against the CCM regimes
  fig11_mpo_s3.png                  Layer 2: MPO on S3
  fig12_explainer.png               Layer 6: counterfactual explainer, three tolerance settings
  fig13_bcs_s4_bridge.png           BCS on S4 with every condition measured
  fig14_validation_scorecard.png    status of every framework component (FRAMEWORK_STATUS.json)

Colours: the validated categorical palette (obd_preprocessing/figures.py) in fixed
order; one colour per algorithm in every figure; identity is also carried by
legends, labels and marker shape.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import exp_common as E
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def plain_log(ax, axis="x"):
    """Log axis labelled at 1-2-5 steps with plain numbers; no crowded minor labels."""
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    for a in ([ax.xaxis] if axis == "x" else [ax.yaxis] if axis == "y" else [ax.xaxis, ax.yaxis]):
        lo, hi = (ax.get_xlim() if a is ax.xaxis else ax.get_ylim())
        if lo > 0 and hi / lo < 6:                  # under ~ three-quarters of a decade: a few round ticks
            from matplotlib.ticker import MaxNLocator
            a.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        else:
            a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.set_major_formatter(fmt)
        a.set_minor_formatter(NullFormatter())

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
COL = {"FG-TS": "#2a78d6", "B3_TS": "#eb6834", "B2_UCB1": "#1baf7a", "B1_EpsGreedy": "#eda100",
       "B0_Random": "#e87ba4", "B4_LinUCB": "#6250d6", "BernoulliTS_Zozotown": "#008300",
       "B4h_HybridLinUCB": "#e34948", "BernoulliTS_uniform": "#52514e"}
MARK = {"FG-TS": "o", "B3_TS": "s", "B2_UCB1": "^", "B1_EpsGreedy": "D", "B0_Random": "v",
        "B4_LinUCB": "P", "BernoulliTS_Zozotown": "*", "B4h_HybridLinUCB": "X", "BernoulliTS_uniform": "h"}
LAB = dict(E.LABEL, BernoulliTS_Zozotown="Bernoulli TS (ZOZOTOWN priors)", B4h_HybridLinUCB="B4h hybrid LinUCB")
FIVE = ["FG-TS", "B3_TS", "B2_UCB1", "B1_EpsGreedy", "B0_Random"]


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(E.FIG / name, dpi=150, facecolor=SURFACE)
    plt.close(fig); print("wrote", name)


def raw(cell, alg):
    p = E.RAW / f"{cell}__{alg}.npy"
    return np.load(p) if p.exists() else None


def title(fig, text, sub=None):
    h = fig.get_figheight()                      # place the subtitle a fixed 0.3 in below the title
    fig.suptitle(text, color=INK, fontsize=12, x=0.01, ha="left", va="top", y=1 - 0.12 / h)
    if sub:
        fig.text(0.01, 1 - 0.42 / h, sub, color=INK2, fontsize=9, ha="left", va="top")


def band(ax, cur, colour, label, lw=2):
    m = cur.mean(0); h = 1.96 * cur.std(0, ddof=1) / np.sqrt(cur.shape[0]); x = np.arange(1, cur.shape[1] + 1)
    ax.fill_between(x, m - h, m + h, color=colour, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=colour, linewidth=lw, label=label)


# ---------------------------------------------------------------------------
def fig01():
    cells = [c for c in E.CELLS if c.startswith("S1")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.6), facecolor=SURFACE)
    for ax, cell in zip(axes.flat, cells):
        style(ax, "x")
        for i, a in enumerate(FIVE):
            x = raw(cell, a); h = 1.96 * x.std(ddof=1) / np.sqrt(x.size)
            ax.plot([x.mean() - h, x.mean() + h], [i, i], color=COL[a], linewidth=2, solid_capstyle="round")
            ax.plot(x.mean(), i, MARK[a], color=COL[a], markersize=7, markeredgecolor=SURFACE)
        ax.set_xscale("log"); plain_log(ax); ax.set_yticks(range(5), [LAB[a] for a in FIVE] if cell.endswith("T500") else [""] * 5,
                                            fontsize=8.5, color=INK)
        ax.invert_yaxis(); K, T = cell.split("_")[1:]
        ax.set_title(f"K = {K[1:]}, T = {int(T[1:]):,}", color=INK, fontsize=10, loc="left")
    for ax in axes[2]:
        ax.set_xlabel("Mean cumulative regret (log scale), 95% CI", color=INK2, fontsize=9)
    title(fig, "S1 stationary Bernoulli: all nine cells (n = 30)",
          "Mean cumulative regret and 95% confidence interval per algorithm; log scale.")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, "fig01_s1_all_cells.png")


def fig02():
    cur = np.load(E.OUT / "curves_S1_K10_T10000.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    for ax, algs, t in ((a, FIVE, "(a) All five algorithms"), (b, FIVE[:4], "(b) Without B0 Random")):
        style(ax)
        for g in algs:
            band(ax, cur[g], COL[g], LAB[g])
        ax.set_title(t, color=INK, fontsize=10.5, loc="left"); ax.set_xlabel("Round t", color=INK2)
        ax.set_ylabel("Mean cumulative regret (95% CI band)", color=INK2)
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    for g in FIVE[:4]:
        y = cur[g].mean(0)[-1]
        b.annotate(f"{y:,.0f}", (10_000, y), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
    title(fig, "Figure 5.2. S1, K = 10, T = 10,000: cumulative regret over 30 runs")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "fig02_s1_learning_curves.png")


def fig03():
    lam = pd.read_csv(E.OUT / "lambda_sweep.csv")
    cells = ["S1_K10_T10000", "S3_K10_T10000", "S5_K500_T100"]
    names = {"S1_K10_T10000": "S1 stationary, K = 10, T = 10,000", "S3_K10_T10000": "S3 ten change-points, K = 10",
             "S5_K500_T100": "S5 cold start, K = 500, T = 100"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=SURFACE)
    for ax, c in zip(axes, cells):
        style(ax)
        g = lam[lam.cell == c].groupby("lambda").cumulative_regret
        m, h = g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(30)
        ax.fill_between(m.index, m - h, m + h, color=COL["FG-TS"], alpha=0.15, linewidth=0)
        ax.plot(m.index, m.values, "-o", color=COL["FG-TS"], linewidth=2, markersize=6, markeredgecolor=SURFACE,
                label="FG-TS (mean, 95% CI)")
        for ref in ("B3_TS", "B2_UCB1"):
            ax.axhline(raw(c, ref).mean(), color=COL[ref], linewidth=1.3, linestyle=(0, (4, 3)), label=LAB[ref])
        ax.set_xscale("symlog", linthresh=0.1); ax.set_xticks([0, 0.1, 0.3, 0.5, 1, 2], ["0", "0.1", "0.3", "0.5", "1", "2"])
        ax.set_title(names[c], color=INK, fontsize=10, loc="left"); ax.set_xlabel("Bonus weight λ", color=INK2)
    axes[0].set_ylabel("Mean cumulative regret", color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    title(fig, "Figure 5.1. FG-TS bonus weight: λ = 0 is exactly B3; the bonus helps only on S3")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "fig03_lambda_sensitivity.png")


def fig04():
    pw = json.loads((E.TAB / "pairwise_S1_K10_T10000.json").read_text())
    names = [a for a in FIVE]
    M = np.full((5, 5), np.nan); S = np.zeros((5, 5), bool)
    for k, v in pw.items():
        a, b = k.split(" vs ")
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            M[i, j] = v["a12"]; M[j, i] = 1 - v["a12"]; S[i, j] = S[j, i] = v["significant"]
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#e4e3df", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(7.2, 6), facecolor=SURFACE); style(ax, None)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.2f}{' *' if S[i, j] else ''}", ha="center", va="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(5), [LAB[a] for a in names], rotation=30, ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(5), [LAB[a] for a in names], fontsize=8.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.set_label("A12 = P(row has lower regret than column)", color=INK2, fontsize=8.5)
    title(fig, "Tables 4.5 and 4.6. S1 primary metric: pairwise A12",
          "* = significant after Holm-Bonferroni (two-sided Wilcoxon, α = 0.05). Ties count one half.")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig04_primary_pairwise_a12.png")


def fig05():
    r = pd.read_csv(E.TAB / "table_4_4_ranks.csv")
    algs = FIVE + ["B4_LinUCB"]; cells = list(E.CELLS)
    R = np.full((len(algs), len(cells)), np.nan); V = np.full_like(R, np.nan)
    for _, x in r.iterrows():
        if x.algorithm in algs:
            R[algs.index(x.algorithm), cells.index(x.setting)] = x["rank"]; V[algs.index(x.algorithm), cells.index(x.setting)] = x["mean"]
    cmap = LinearSegmentedColormap.from_list("seq", ["#0d3a73", "#2a78d6", "#a9cdf3", "#eef5fd"])
    fig, ax = plt.subplots(figsize=(15, 4.4), facecolor=SURFACE); style(ax, None)
    ax.imshow(R, cmap=cmap, vmin=1, vmax=6, aspect="auto")
    for i in range(len(algs)):
        for j in range(len(cells)):
            if not np.isnan(R[i, j]):
                ax.text(j, i, f"#{int(R[i, j])}", ha="center", va="center", fontsize=8.5,
                        color="#fcfcfb" if R[i, j] <= 2 else INK)
    ax.set_yticks(range(len(algs)), [LAB[a] for a in algs], fontsize=8.5, color=INK)
    ax.set_xticks(range(len(cells)), [c.replace("_SYNTHETIC", "syn").replace("_", " ") for c in cells],
                  rotation=55, ha="right", fontsize=7.8, color=INK2)
    title(fig, "Rank of every algorithm in every synthetic cell (1 = lowest mean regret; n = 30)",
          "Darker = better rank. B4 LinUCB runs only in S4, the only contextual setting.")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); save(fig, "fig05_rank_heatmap.png")


def fig06():
    cur = np.load(E.OUT / "curves_S3_K10_T10000.npz")
    from mab_benchmark.environments import PiecewiseStationaryBandit
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE); style(ax)
    for cp in PiecewiseStationaryBandit.CHANGE_POINTS:
        ax.axvline(cp, color=GRID, linewidth=1.2, linestyle=(0, (2, 2)))
    for g in FIVE:
        band(ax, cur[g], COL[g], LAB[g])
        ax.annotate(LAB[g].split(" (")[0], (10_000, cur[g].mean(0)[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center")
    ax.set_xlim(0, 11_300); ax.set_xlabel("Round t (dotted lines: change-points)", color=INK2)
    ax.set_ylabel("Mean cumulative regret (95% CI band)", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    title(fig, "S3 piecewise-stationary (K = 10, T = 10,000): UCB1 recovers best after change-points")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "fig06_s3_curves.png")


def fig07():
    cur = np.load(E.OUT / "curves_S4_SYNTHETIC_K50_T10000.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.4, 1]})
    style(a)
    for g in FIVE + ["B4_LinUCB"]:
        band(a, cur[g], COL[g], LAB[g])
    a.set_xlabel("Round t", color=INK2); a.set_ylabel("Mean cumulative regret", color=INK2)
    a.set_title("(a) Cumulative regret, 30 runs", color=INK, fontsize=10.5, loc="left")
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    style(b, "x")
    algs = FIVE + ["B4_LinUCB"]
    for i, g in enumerate(algs):
        x = raw("S4_SYNTHETIC_K50_T10000", g); h = 1.96 * x.std(ddof=1) / np.sqrt(x.size)
        b.plot([x.mean() - h, x.mean() + h], [i, i], color=COL[g], linewidth=2)
        b.plot(x.mean(), i, MARK[g], color=COL[g], markersize=9, markeredgecolor=SURFACE)
        b.annotate(f"{x.mean():,.0f}", (x.mean(), i), xytext=(0, 9), textcoords="offset points", ha="center",
                   fontsize=8, color=INK2)
    b.set_yticks(range(len(algs)), [LAB[g] for g in algs], fontsize=8.5, color=INK); b.invert_yaxis()
    b.set_xlabel("Mean cumulative regret at T = 10,000 (95% CI)", color=INK2)
    b.set_title("(b) Final regret", color=INK, fontsize=10.5, loc="left")
    title(fig, "S4 synthetic contextual (K = 50, d = 10): only LinUCB uses the context",
          "Rewards follow the context of each round; the context-free algorithms cannot exploit it and stay at Random level.")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig07_s4_synthetic.png")


def fig08():
    p = pd.read_csv(E.TAB / "table_4_14a_pareto.csv")
    sets = ["S1_K10_T10000", "S3_K10_T10000", "S4_SYNTHETIC_K50_T10000", "S4_OBD"]
    names = {"S1_K10_T10000": "S1 primary", "S3_K10_T10000": "S3 change-points",
             "S4_SYNTHETIC_K50_T10000": "S4 synthetic", "S4_OBD": "S4 OBD (replay)"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, s in zip(axes, sets):
        style(ax, "both"); d = p[p.setting == s]
        for _, r in d.iterrows():
            opt = r["Pareto-optimal"] == "yes"
            ax.plot(r.iloc[2], r.iloc[3], MARK.get(r.algorithm, "o"), color=COL.get(r.algorithm, INK2),
                    markersize=10 if opt else 7, markeredgecolor=INK if opt else SURFACE, markeredgewidth=1.4 if opt else 1,
                    alpha=1 if opt else 0.75)
        front = d[d["Pareto-optimal"] == "yes"].sort_values(d.columns[2])
        ax.plot(front.iloc[:, 2], front.iloc[:, 3], color=INK2, linewidth=1, linestyle=(0, (3, 2)))
        if s != "S4_OBD":
            ax.set_xscale("log"); plain_log(ax)
        ax.set_title(names[s], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Regret" if s != "S4_OBD" else "CTR deficit from best x 1000", color=INK2, fontsize=9)
    axes[0].set_ylabel("Latency violation L (δ = 0.5 ms)", color=INK2)
    algs = list(dict.fromkeys(p.algorithm))
    h = [plt.Line2D([], [], marker=MARK.get(a, "o"), color=COL.get(a, INK2), linestyle="none", label=LAB.get(a, a)) for a in algs]
    h.append(plt.Line2D([], [], marker="o", color=SURFACE, markeredgecolor=INK, linestyle="none", label="Pareto-optimal (outlined)"))
    fig.legend(handles=h, frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=5)
    title(fig, "Figure 4.1. Pareto fronts over regret and measured latency (CCM = 0.5 L, bandwidth term 0)")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93)); save(fig, "fig08_pareto_fronts.png")


def fig09():
    g = pd.read_csv(E.TAB / "table_4_15_4_17_4_19_guf_values.csv")
    sets = ["S1_K10_T10000", "S3_K10_T10000", "S4_SYNTHETIC_K50_T10000", "S4_OBD"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), facecolor=SURFACE, sharey=False)
    for ax, s in zip(axes, sets):
        style(ax); d = g[g.setting == s].copy()
        d["rank"] = d.groupby("w1").Gamma.rank(method="min")
        for a, x in d.groupby("algorithm"):
            ax.plot(x.w1, x["rank"], "-" + MARK.get(a, "o"), color=COL.get(a, INK2), linewidth=1.8, markersize=6,
                    markeredgecolor=SURFACE, label=LAB.get(a, a))
        ax.set_yticks(range(1, int(d["rank"].max()) + 1)); ax.invert_yaxis(); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_title(s.replace("_", " "), color=INK, fontsize=9.5, loc="left"); ax.set_xlabel("w1 (regret weight)", color=INK2)
    axes[0].set_ylabel("GUF rank (1 = best)", color=INK2)
    hs, ls = [], []
    for ax in axes:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ls:
                hs.append(h_); ls.append(l_)
    fig.legend(hs, ls, frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=5)
    title(fig, "GUF weight sweep: flat lines mean the ranking does not depend on the weights")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93)); save(fig, "fig09_guf_weight_sweep.png")


def fig11():
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    lat = pd.read_csv(E.OUT / "latency_profile.csv")
    cells = list(dict.fromkeys(lat.cell))
    algs = list(dict.fromkeys(lat.algorithm))          # one row per algorithm, aligned across panels
    fig, axes = plt.subplots(1, len(cells), figsize=(16, 4.6), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, c in zip(axes, cells):
        style(ax, "x"); d = lat[lat.cell == c]
        for _, r in d.iterrows():
            i = algs.index(r.algorithm)
            ax.plot([r.median_ms, r.p95_ms], [i, i], color=COL.get(r.algorithm, INK2), linewidth=2, solid_capstyle="round")
            ax.plot(r.median_ms, i, MARK.get(r.algorithm, "o"), color=COL.get(r.algorithm, INK2), markersize=7, markeredgecolor=SURFACE)
        for dlt in (0.5, 20):
            ax.axvline(dlt, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_xscale("log"); ax.set_xlim(0.001, 50)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(range(len(algs)), [LAB[a] for a in algs], fontsize=8.2, color=INK)
        ax.set_title(c.replace("_SYNTHETIC", " syn").replace("_", " "), color=INK, fontsize=9, loc="left")
        ax.set_xlabel("ms per round (median to p95)", color=INK2, fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].text(0.5, -0.55, " 0.5 ms", fontsize=7.5, color=INK2); axes[0].text(20, -0.55, " 20 ms", fontsize=7.5, color=INK2)
    over = lat[lat.median_ms > 0.5]
    msg = ", ".join(f"{LAB[r.algorithm].split(' (')[0]} on {r.cell.split('_T')[0].replace('_SYNTHETIC', ' syn').replace('_', ' ')}"
                    for _, r in over.iterrows()) or "none"
    title(fig, "Table 4.9 (measured). Decision latency per round against the stress (0.5 ms) and WiFi (20 ms) budgets",
          f"Every algorithm meets the 2G (200 ms), 4G (100 ms) and WiFi (20 ms) budgets. Median above the 0.5 ms stress budget: {msg}.")
    fig.tight_layout(rect=(0, 0, 1, 0.87)); save(fig, "fig10_latency_profile.png")


def fig12():
    p = E.REPO / "FRAMEWORK_STATUS.json"
    if not p.exists():
        print("skip fig12 (run framework_status.py first)"); return
    rows = json.loads(p.read_text())["components"]
    good = {"UP TO DATE": "#1baf7a", "PASSED": "#1baf7a", "NOT UP TO DATE": "#e34948", "FAILED": "#e34948",
            "PARTIAL": "#eda100", "NOT TESTED": "#b5b3ad", "N/A": "#e4e3df"}
    icon = {"UP TO DATE": "✓", "PASSED": "✓", "NOT UP TO DATE": "✗", "FAILED": "✗", "PARTIAL": "!", "NOT TESTED": "–", "N/A": "·"}
    n = len(rows)
    fig, ax = plt.subplots(figsize=(17, 0.42 * n + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE); ax.axis("off"); ax.set_xlim(0, 17); ax.set_ylim(n + 0.6, -1.2)
    ax.text(0.1, -0.6, "Component", fontsize=9.5, color=INK2, weight="bold")
    ax.text(7.25, -0.6, "Up to date?", fontsize=9.5, color=INK2, weight="bold", ha="center")
    ax.text(9.25, -0.6, "Validation", fontsize=9.5, color=INK2, weight="bold", ha="center")
    ax.text(10.5, -0.6, "Evidence", fontsize=9.5, color=INK2, weight="bold")
    for i, r in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, i - 0.45), 17, 0.9, color="#f3f2ef", linewidth=0))
        ax.text(0.1, i, r["component"], fontsize=8.6, color=INK, va="center")
        for x, key in ((7.25, "up_to_date"), (9.25, "validation")):
            v = r[key]
            ax.add_patch(plt.Rectangle((x - 0.85, i - 0.32), 1.7, 0.64, color=good.get(v, GRID), linewidth=0))
            ax.text(x, i, f"{icon.get(v, '')} {v}", fontsize=7.4, color="#fcfcfb" if v not in ("N/A", "NOT TESTED", "PARTIAL") else INK,
                    va="center", ha="center", weight="bold")
        ax.text(10.5, i, r["evidence_short"], fontsize=7.6, color=INK2, va="center")
    s = json.loads(p.read_text())["summary"]
    fig.suptitle(f"Framework status, mab_benchmark v1.2.0: {s['up_to_date']} of {s['total']} components up to date, "
                 f"{s['passed']} passed, {s['failed']} failed, {s['partial']} partial", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); save(fig, "fig14_validation_scorecard.png")


def fig13():
    s = pd.read_csv(E.OUT / "layers26_mpo_runs.csv"); cp = pd.read_csv(E.OUT / "layers26_mpo_changepoints.csv")
    s3 = s[s.cell == "S3_K10_T10000"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1]})
    style(a, "x"); style(b)
    for i, g in enumerate(FIVE):
        for mpo, y in ((False, i - 0.15), (True, i + 0.15)):
            x = s3[(s3.algorithm == g) & (s3.mpo == mpo)].regret.to_numpy()
            h = 1.96 * x.std(ddof=1) / np.sqrt(x.size)
            a.plot([x.mean() - h, x.mean() + h], [y, y], color=COL[g], linewidth=2)
            a.plot(x.mean(), y, MARK[g], color=COL[g] if mpo else SURFACE, markeredgecolor=COL[g],
                   markersize=8, markeredgewidth=1.6)
        p0 = s3[(s3.algorithm == g) & (~s3.mpo)].regret.mean(); p1 = s3[(s3.algorithm == g) & (s3.mpo)].regret.mean()
        if abs(p1 - p0) > 1:
            a.annotate(f"{100 * (p1 - p0) / p0:+.0f}% with the MPO", (p1, i + 0.15), xytext=(14, -3),
                       textcoords="offset points", fontsize=8.5, color=INK)
    a.set_yticks(range(5), [LAB[g] for g in FIVE], fontsize=8.5, color=INK); a.invert_yaxis()
    a.plot([], [], "o", color=SURFACE, markeredgecolor=INK2, label="plain"); a.plot([], [], "o", color=INK2, label="wrapped in the MPO")
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    a.set_xlabel("Mean cumulative regret on S3 (95% CI)", color=INK2)
    a.set_title("(a) Posterior reset after a flagged change-point", color=INK, fontsize=10.5, loc="left")
    bins = [-1.0, -0.4, -0.2, -0.1, 0.0, 1.0]; labels = ["< -0.4", "-0.4 to -0.2", "-0.2 to -0.1", "-0.1 to 0", "rise"]
    w = 0.38
    for k, g in enumerate(("B3_TS", "FG-TS")):
        d = cp[cp.algorithm == g].copy(); d["bin"] = pd.cut(d.mean_change_of_acting_arm, bins, labels=labels)
        r = d.groupby("bin", observed=False).detected.mean()
        n = d.groupby("bin", observed=False).size()
        xs = np.arange(len(labels)) + (k - 0.5) * w
        b.bar(xs, r.values, width=w * 0.92, color=COL[g], label=LAB[g])
        for x, v, m in zip(xs, r.values, n.values):
            b.text(x, (0 if np.isnan(v) else v) + 0.02, f"n={m}", ha="center", fontsize=7, color=INK2)
    b.set_xticks(range(len(labels)), labels, fontsize=8.5, color=INK2); b.set_ylim(0, 1.1)
    b.set_xlabel("Change of the acting arm's mean at the change-point", color=INK2)
    b.set_ylabel("Share detected within 500 rounds", color=INK2)
    b.set_title("(b) The monitor detects drops, as specified", color=INK, fontsize=10.5, loc="left")
    b.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    title(fig, "Layer 2 (Meta-Policy Orchestrator) on S3, 30 runs",
          "No flag was raised on stationary S1 for any algorithm, so S1 regret is unchanged by the MPO.")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); save(fig, "fig11_mpo_s3.png")


def fig14():
    j = json.loads((E.OUT / "layers26_results.json").read_text())["explainer"]
    s1 = pd.read_csv(E.OUT / "layers26_explainer_s1.csv")
    tols = ["statistical", "relative", "absolute"]
    tlab = {"statistical": "statistical\n(1.96 SE)", "relative": "relative\n(10%)", "absolute": "absolute\n(0.01)"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax in (a, b, c):
        style(ax)
    algs = ["B3_TS", "B2_UCB1", "B1_EpsGreedy"]; x = np.arange(3)
    a.bar(x - 0.2, [j["S1"][g]["sign_agreement_all"] for g in algs], 0.38, color=[COL[g] for g in algs], label="IPS gap (explainer)")
    a.bar(x + 0.2, [j["S1"][g]["sign_agreement_model_based"] for g in algs], 0.38, color=[COL[g] for g in algs],
          alpha=0.45, hatch="//", edgecolor=SURFACE, label="model-based gap, for comparison")
    for xi, g in zip(x, algs):
        for dx, key in ((-0.2, "sign_agreement_all"), (0.2, "sign_agreement_model_based")):
            a.text(xi + dx, j["S1"][g][key] + 0.02, f"{j['S1'][g][key]:.0%}", ha="center", fontsize=8, color=INK)
    a.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    a.set_xticks(x, [LAB[g] for g in algs], fontsize=8.2, color=INK); a.set_ylim(0, 1.12)
    a.set_ylabel("Share with the sign of the true margin", color=INK2)
    a.set_title("(a) S1: does the gap have the right sign?", color=INK, fontsize=10.5, loc="left")
    a.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    d = s1[s1.algorithm == "B3_TS"]
    kc = {"indistinguishable": INK2, "margin": COL["B2_UCB1"], "exploration": COL["B3_TS"]}
    for k, g in d.groupby("kind"):
        b.scatter(g.true_margin, np.clip(g.delta, -1.5, 1.5), s=14, color=kc[k], label=k, alpha=0.8, edgecolor="none")
    b.axhline(0, color=INK2, linewidth=0.7); b.axvline(0, color=INK2, linewidth=0.7)
    b.set_xlabel("True margin of the chosen arm", color=INK2); b.set_ylabel("Explainer gap Δt (clipped at ±1.5)", color=INK2)
    b.set_title("(b) S1, B3 TS: Δt against the truth (statistical)", color=INK, fontsize=10.5, loc="left")
    b.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    o = j["OBD_bts_log"]; xs = np.arange(3); bottom = np.zeros(3)
    for k, colr in (("share_margin", COL["B2_UCB1"]), ("share_exploration", COL["B3_TS"]), ("share_indistinguishable", INK2)):
        v = np.array([o[t][k] for t in tols])
        c.bar(xs, v, 0.6, bottom=bottom, color=colr, label=k.replace("share_", ""))
        bottom += v
    for xi, t in zip(xs, tols):
        sa = o[t]["sign_agreement_on_graded"]
        c.text(xi, 1.03, "sign right on graded:\n" + ("-" if sa is None else f"{sa:.0%}"), ha="center", fontsize=8, color=INK)
    c.set_xticks(xs, [tlab[t] for t in tols], fontsize=8.5, color=INK); c.set_ylim(0, 1.22)
    c.set_ylabel("Share of sampled rounds", color=INK2)
    c.set_title("(c) OBD (CTR 0.1% to 0.7%): tolerance settings", color=INK, fontsize=10.5, loc="left")
    c.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right", bbox_to_anchor=(1, 0.86))
    title(fig, "Layer 6 (counterfactual explainer) on sampled rounds of S1 and of the real ZOZOTOWN log",
          "The scale-aware tolerances grade explanations at click-through rates below 1%; the absolute tolerance of 0.01 cannot.")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); save(fig, "fig12_explainer.png")


def fig15():
    j = json.loads((E.OUT / "s4_bcs_bridge.json").read_text())["bcs_variants"]
    names = list(j)
    get = lambda b, k1, k2: b.get(k1, b.get(k2))
    vals = {n: [get(j[n], "c1_reward_proxy", "c1"), get(j[n], "c2_arm_independence", "c2"),
                get(j[n], "c3_sufficient_horizon", "c3"), j[n]["BCS"]] for n in names}
    fig, ax = plt.subplots(figsize=(13, 0.9 * len(names) + 1.8), facecolor=SURFACE); style(ax, "x")
    comp = ["c1 (rank agreement)", "c2 (arm independence)", "c3 (horizon)", "BCS"]
    cols = ["#2a78d6", "#1baf7a", "#eda100", INK]
    h = 0.19
    for k in range(4):
        y = np.arange(len(names)) + (k - 1.5) * h
        ax.barh(y, [vals[n][k] for n in names], height=h * 0.9, color=cols[k], label=comp[k])
        for yi, n in zip(y, names):
            ax.text(vals[n][k] + 0.01, yi, f"{vals[n][k]:.2f}", va="center", fontsize=7.5, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))
    ax.axvline(0.9, color=INK, linewidth=1, linestyle=(0, (4, 3))); ax.text(0.9, len(names) - 0.45, " PASS threshold 0.9", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), [n.replace(": ", "\n").replace(" (as reported)", " as reported\n(c2 assumed, 1/30 slices)")
                                      for n in names], fontsize=8.2, color=INK); ax.invert_yaxis()
    ax.set_xlim(0, 1.12); ax.set_xlabel("Score", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    prim = j["random_only: full-log Tier 2, domain-matched Tier 1"]
    title(fig, "Bridge Compliance Score on S4: each condition measured",
          f"Primary variant (fixed in advance): random-log replay, full-log horizon, domain-matched Tier 1. "
          f"rho = {prim['rho_S']:.2f}, 95% bootstrap CI {prim['rho_boot_ci95'][0]:.2f} to {prim['rho_boot_ci95'][1]:.2f}.")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig13_bcs_s4_bridge.png")


def main():
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig11, fig13, fig14, fig15, fig12):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main()
