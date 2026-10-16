"""
figures.py: thesis figures for the S4 replay (static PNG, 150 dpi).

  learning_figure()  -> s4_obd_learning.png
      (a) running CTR (cumulative clicks / cumulative accepted events, summed
          over the 30 runs) against the share of each run's rows replayed;
      (b) final mean CTR per algorithm with its 95% confidence interval.
  external_validity_figure() -> s4_bernoulli_ts_learning.png
      BernoulliTS (ZOZOTOWN priors) and B0 Random running CTR, with the
      on-policy ground-truth CTRs of the two logs as reference lines.

Colours: validated categorical palette in fixed order (one hue per
algorithm, never re-assigned); text in neutral ink; identity is also given
by the legend and by the axis labels of panel (b).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDER = ["B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS", "FG-TS",
         "BernoulliTS_Zozotown", "B4_LinUCB", "B4h_HybridLinUCB"]
COLOUR = dict(zip(ORDER, PALETTE))           # colour follows the algorithm
LABEL = {"B0_Random": "B0 Random", "B1_EpsGreedy": "B1 ε-Greedy", "B2_UCB1": "B2 UCB1",
         "B3_TS": "B3 Thompson Sampling", "FG-TS": "FG-TS (λ = 0.5)",
         "BernoulliTS_Zozotown": "Bernoulli TS (ZOZOTOWN priors)",
         "B4_LinUCB": "B4 LinUCB", "B4h_HybridLinUCB": "B4h hybrid LinUCB"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def _running(curves, stream, alg):
    g = curves[(curves.stream == stream) & (curves.algorithm == alg)]
    agg = g.groupby("row_fraction")[["accepted", "clicks"]].sum()
    return agg.index.to_numpy(), (agg.clicks / agg.accepted.replace(0, np.nan)).to_numpy()


def learning_figure(curves: pd.DataFrame, summary: dict, stream: str, out, title: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]},
                               facecolor=SURFACE)
    for ax in (a, b):
        _style(ax)
    for alg in ORDER:
        if alg not in summary:
            continue
        x, y = _running(curves, stream, alg)
        a.plot(x * 100, y * 100, color=COLOUR[alg], linewidth=2, label=LABEL[alg])
    a.set_xlabel("Share of each run's rows replayed (%)", color=INK2)
    a.set_ylabel("Running CTR, all 30 runs pooled (%)", color=INK2)
    a.set_title("(a) Running CTR during replay", color=INK, fontsize=11, loc="left")
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right", ncol=2)

    algs = sorted([x for x in ORDER if x in summary], key=lambda k: summary[k]["mean_ctr"])
    for i, alg in enumerate(algs):
        s = summary[alg]
        b.plot([s["ci95_low"] * 100, s["ci95_high"] * 100], [i, i], color=COLOUR[alg],
               linewidth=2, solid_capstyle="round")
        b.plot(s["mean_ctr"] * 100, i, "o", color=COLOUR[alg], markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=2)
    b.set_yticks(range(len(algs)), [LABEL[x] for x in algs], color=INK, fontsize=9)
    b.set_xlabel("Mean CTR over 30 runs with 95% CI (%)", color=INK2)
    b.set_title("(b) Final CTR and uncertainty", color=INK, fontsize=11, loc="left")
    b.grid(axis="x", color=GRID, linewidth=0.8); b.grid(axis="y", visible=False)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def external_validity_figure(curves: pd.DataFrame, ev: dict, stream: str, out, title: str):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    _style(ax)
    for alg in ("BernoulliTS_Zozotown", "B0_Random"):
        x, y = _running(curves, stream, alg)
        ax.plot(x * 100, y * 100, color=COLOUR[alg], linewidth=2, label=LABEL[alg] + " (replay)")
        ax.annotate(LABEL[alg].split(" (")[0], (x[-1] * 100, y[-1] * 100), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    for key, alg, txt in (("ground_truth_ctr_bts_log", "BernoulliTS_Zozotown", "bts log, on-policy"),
                          ("ground_truth_ctr_random_log", "B0_Random", "random log, on-policy")):
        ax.axhline(ev[key] * 100, color=COLOUR[alg], linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"Ground truth, {txt}: {ev[key]:.3%}", (55, ev[key] * 100), xytext=(0, -5),
                    textcoords="offset points", fontsize=8.5, color=INK2, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))
    lo, hi = ev["replay_ratio_ci95_bootstrap"]
    ax.set_title(f"Replay ratio {ev['replay_ratio']:.2f}× (95% CI {lo:.2f} to {hi:.2f}); "
                 f"ground truth {ev['ground_truth_ratio_this_data']:.2f}×",
                 color=INK2, fontsize=10, loc="left")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    ax.set_xlabel("Share of each run's rows replayed (%)", color=INK2)
    ax.set_ylabel("Running CTR, all 30 runs pooled (%)", color=INK2)
    ax.set_xlim(0, 112)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
