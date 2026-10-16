"""
make_figures.py: figures for the Saito et al. validity check (PNG, 150 dpi).

  fig1_test1_on_policy_ctr.png       reproduced vs published on-policy CTR (Table 1)
  fig2_test2_external_validity.png   replay ratio vs ground truth, three campaigns
  fig2b_test2_learning_curves.png    running replay CTR per campaign
  fig3_test3_rmse_both_directions.png reproduced vs published RMSE (Tables 4 and 9)
  fig4_test4_sample_size.png         n = 10,000 vs n = 300,000 (Table 6 vs Table 4)
  fig5_rmse_agreement_scatter.png    all reproduced RMSE cells against the published cells

Colours: the validated categorical palette of obd_preprocessing/figures.py in
fixed order; reproduced values are filled markers, published values hollow
black diamonds, so identity never rests on colour alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import saito_common as C

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PUB = dict(marker="D", markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.3, linestyle="none")


def _style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def _save(fig, name):
    fig.savefig(C.FIG_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", C.FIG_DIR / name)


def fig1():
    d = json.loads((C.OUT_DIR / "test1_on_policy_ctr.json").read_text())["rows"]
    df = pd.DataFrame(d)
    logs = [f"{p}/{c}" for c in C.CAMPAIGNS for p in C.POLICIES]
    lab = {f"{p}/{c}": f"{'Bernoulli TS' if p == 'bts' else 'Random'}, {C.CAMPAIGN_LABEL[c]}" for p, c in
           (x.split("/") for x in logs)}
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE); _style(ax)
    for i, lg in enumerate(logs):
        for variant, col, dy in (("raw", BLUE, 0.12), ("clean", ORANGE, -0.12)):
            r = df[(df.log == lg) & (df.variant == variant)].iloc[0]
            ax.plot([r.ctr_percent - r.ci95_pm_percent_normal, r.ctr_percent + r.ci95_pm_percent_normal],
                    [i + dy] * 2, color=col, linewidth=2, solid_capstyle="round")
            ax.plot(r.ctr_percent, i + dy, "o", color=col, markersize=6, markeredgecolor=SURFACE,
                    label=("This study, raw log" if variant == "raw" else "This study, duplicates removed")
                    if i == 0 else None)
        r = df[(df.log == lg) & (df.variant == "raw")].iloc[0]
        ax.plot(r.published_ctr_percent, i, **PUB, markersize=7, label="Saito et al., Table 1" if i == 0 else None)
    ax.set_yticks(range(len(logs)), [lab[x] for x in logs], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("On-policy CTR (%) with 95% confidence interval", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fig.suptitle("Test 1. The raw logs reproduce every published CTR; removing duplicates does not",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); _save(fig, "fig1_test1_on_policy_ctr.png")


def fig2():
    res = json.loads((C.OUT_DIR / "test2_external_validity.json").read_text())["results"]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE); _style(ax)
    ylab = []
    for i, camp in enumerate(C.CAMPAIGNS):
        for j, (stream, col) in enumerate((("pooled", BLUE), ("random_only", GREEN))):
            r = next(x for x in res if x["variant"] == "clean" and x["campaign"] == camp and x["stream"] == stream)
            y = 3 * i + j
            lo, hi = r["replay_ratio_ci95"]
            ax.plot([lo, hi], [y, y], color=col, linewidth=2.2, solid_capstyle="round")
            ax.plot(r["replay_ratio"], y, "o", color=col, markersize=7, markeredgecolor=SURFACE,
                    label=("Replay, both logs" if stream == "pooled" else "Replay, random log only") if i == 0 else None)
            ax.plot(r["ground_truth_ratio"], y, "|", color=INK, markersize=16, markeredgewidth=2,
                    label="Ground truth, this data" if (i == 0 and j == 0) else None)
            ylab.append((y, f"{C.CAMPAIGN_LABEL[camp]}, {'both logs' if stream == 'pooled' else 'random log only'}"))
        ax.plot(r["published_ratio_table1"], 3 * i + 0.5, **PUB, markersize=7,
                label="Published ratio, Table 1" if i == 0 else None)
    ax.axvline(1, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks([y for y, _ in ylab], [t for _, t in ylab], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("CTR ratio, Bernoulli TS (ZOZOTOWN priors) / Random, with 95% bootstrap interval", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.suptitle("Test 2. S4 replay against the on-policy ratio (duplicates removed, 30 runs per stream)",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); _save(fig, "fig2_test2_external_validity.png")

    cur = pd.read_csv(C.OUT_DIR / "test2_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE, sharey=False)
    for ax, camp in zip(axes, C.CAMPAIGNS):
        _style(ax, "y")
        r = next(x for x in res if x["variant"] == "clean" and x["campaign"] == camp and x["stream"] == "pooled")
        for alg, col, lab in (("BernoulliTS_Zozotown", BLUE, "Bernoulli TS (ZOZOTOWN)"), ("Random", ORANGE, "Random")):
            g = cur[(cur.variant == "clean") & (cur.campaign == camp) & (cur.stream == "pooled") & (cur.algorithm == alg)]
            a = g.groupby("row_fraction")[["accepted", "clicks"]].sum()
            ax.plot(a.index * 100, 100 * a.clicks / a.accepted.replace(0, np.nan), color=col, linewidth=2, label=lab)
        for key, col in (("ground_truth_ctr_bts_log", BLUE), ("ground_truth_ctr_random_log", ORANGE)):
            ax.axhline(100 * r[key], color=col, linewidth=1.1, linestyle=(0, (4, 3)))
        ax.set_title(C.CAMPAIGN_LABEL[camp], color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("Share of each run replayed (%)", color=INK2)
    axes[0].set_ylabel("Running replay CTR, 30 runs pooled (%)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.suptitle("Test 2. Running replay CTR; dashed lines are the on-policy CTRs of the two logs",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); _save(fig, "fig2b_test2_learning_curves.png")


def _summary():
    return pd.DataFrame(json.loads((C.OUT_DIR / "ope" / "ope_summary_raw.json").read_text()))


def _rmse_panel(ax, row, published_key, col, title, show_labels):
    ys = np.arange(len(C.ESTIMATORS))
    for k, e in enumerate(C.ESTIMATORS):
        lo, hi = row[f"rmse_ci95_{e}"]
        ax.plot([lo, hi], [k, k], color=col, linewidth=2, solid_capstyle="round")
        ax.plot(row[f"rmse_{e}"], k, "o", color=col, markersize=6, markeredgecolor=SURFACE)
        p = row.get(published_key(e))
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            ax.plot(p, k, **PUB, markersize=6.5)
    ax.set_yticks(ys, C.ESTIMATORS if show_labels else [""] * len(ys), color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def fig3():
    s = _summary()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    for i, (beh, title) in enumerate((("bts", "Bernoulli TS log, evaluating Random (Table 4)"),
                                      ("random", "Random log, evaluating Bernoulli TS (Table 9)"))):
        for j, camp in enumerate(C.CAMPAIGNS):
            ax = axes[i, j]; _style(ax)
            r = s[(s.behaviour == beh) & (s.campaign == camp) & (s.n == 300_000)]
            if r.empty:
                ax.set_visible(False); continue
            _rmse_panel(ax, r.iloc[0], lambda e: f"published_{e}", BLUE if beh == "bts" else GREEN,
                        f"{C.CAMPAIGN_LABEL[camp]}", j == 0)
            if j == 0:
                ax.text(0, 1.16, title, transform=ax.transAxes, color=INK, fontsize=10.5, fontweight="bold")
            if i == 1:
                ax.set_xlabel("RMSE x 10³ over 200 bootstrap samples", color=INK2)
    h = [plt.Line2D([], [], marker="o", color=BLUE, linestyle="none", label="This study, Bernoulli TS log (95% bootstrap CI)"),
         plt.Line2D([], [], marker="o", color=GREEN, linestyle="none", label="This study, Random log (95% bootstrap CI)"),
         plt.Line2D([], [], **PUB, label="Saito et al.")]
    fig.legend(handles=h, frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncol=3,
               bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("Test 3. Reproduced RMSE of seven OPE estimators at n = 300,000, both directions",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=3); _save(fig, "fig3_test3_rmse_both_directions.png")


def fig4():
    s = _summary()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE)
    ys = np.arange(len(C.ESTIMATORS))
    for j, camp in enumerate(C.CAMPAIGNS):
        ax = axes[j]; _style(ax)
        for n, col, dy in ((10_000, ORANGE, -0.15), (300_000, BLUE, 0.15)):
            r = s[(s.behaviour == "bts") & (s.campaign == camp) & (s.n == n)]
            if r.empty:
                continue
            r = r.iloc[0]
            for k, e in enumerate(C.ESTIMATORS):
                lo, hi = r[f"rmse_ci95_{e}"]
                ax.plot([lo, hi], [k + dy] * 2, color=col, linewidth=2, solid_capstyle="round")
                ax.plot(r[f"rmse_{e}"], k + dy, "o", color=col, markersize=6, markeredgecolor=SURFACE,
                        label=f"This study, n = {n:,}" if k == 0 and j == 0 else None)
                p = r.get(f"published_{e}")
                if p is not None and not np.isnan(p):
                    ax.plot(p, k + dy, **PUB, markersize=6, label="Saito et al." if k == 0 and j == 0 and n == 10_000 else None)
        ax.set_yticks(ys, C.ESTIMATORS if j == 0 else [""] * len(ys), color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(C.CAMPAIGN_LABEL[camp], color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("RMSE x 10³ (Bernoulli TS log, evaluating Random)", color=INK2)
        if camp == "women":
            ax.set_title("Women's\n(published n = 10,000 values repeat Table 10, Men's)", color=INK, fontsize=10.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.suptitle("Test 4. Sample size changes which estimator is most accurate (Table 6)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); _save(fig, "fig4_test4_sample_size.png")


def fig5():
    s = _summary()
    fig, ax = plt.subplots(figsize=(6.6, 6.2), facecolor=SURFACE); _style(ax, "both")
    groups = [(("bts", 300_000), BLUE, "Bernoulli TS log -> Random, n = 300,000"),
              (("random", 300_000), GREEN, "Random log -> Bernoulli TS, n = 300,000"),
              (("bts", 10_000), ORANGE, "Bernoulli TS log -> Random, n = 10,000"),
              (("random", 10_000), AMBER, "Random log -> Bernoulli TS, n = 10,000")]
    marks = {"all": "o", "men": "s", "women": "^"}
    allv = []
    for (beh, n), col, lab in groups:
        sub = s[(s.behaviour == beh) & (s.n == n)]
        for _, r in sub.iterrows():
            x = [r[f"published_{e}"] for e in C.ESTIMATORS]; y = [r[f"rmse_{e}"] for e in C.ESTIMATORS]
            ax.plot(x, y, marks[r.campaign], color=col, markersize=6, linestyle="none", markeredgecolor=SURFACE,
                    label=lab if r.campaign == "all" else None)
            allv += x + y
    if allv:
        lo, hi = min(allv) * 0.8, max(allv) * 1.25
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
        tk = [0.3, 0.5, 1, 2, 3, 5]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(tk)); axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(FormatStrFormatter("%g"))
        ax.annotate("Table 6, Women's, n = 10,000:\npublished column repeats Table 10, Men's",
                    (3.0, 1.2), xytext=(1.25, 0.42), fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    for c, m in marks.items():
        ax.plot([], [], m, color=INK2, linestyle="none", label=C.CAMPAIGN_LABEL[c])
    ax.set_xlabel("Published RMSE x 10³", color=INK2); ax.set_ylabel("Reproduced RMSE x 10³", color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.suptitle("Tests 3 and 4. Every RMSE cell against its published value", color=INK, fontsize=11.5,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); _save(fig, "fig5_rmse_agreement_scatter.png")


def main():
    for f in (fig1, fig2, fig3, fig4, fig5):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "(missing input:", e.filename, ")")


if __name__ == "__main__":
    main()
