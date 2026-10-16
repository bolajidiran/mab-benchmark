"""
thesis_figures.py: the figures of Chapter Four (mab_benchmark 1.3.0 results),
written as 600-dpi PNG files for the thesis LaTeX media folder.

  fig4_replay_bias_obd.png     OBD: matched events per algorithm relative to B0
                               Random, log as recorded vs shuffled within hour
  fig4_kuairec_conditions.png  KuaiRec: mean CTR per algorithm under ONLINE,
                               IID replay, BURSTY replay and SHUFFLED replay
  fig4_h1_comparability.png    H1: comparability C per protocol, with the
                               bootstrap interval of Delta = C_U - C_P
  fig4_guf_sweep.png           normalised GUF value against w1 per setting

Each figure is skipped (with a message) when its input file is missing.
Run: python thesis_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
# validated categorical palette (light surface): blue, orange, aqua, yellow
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARK = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 600
SHORT = {"B0_Random": "B0 Random", "B1_EpsGreedy": "B1 ε-Greedy", "B2_UCB1": "B2 UCB1", "B3_TS": "B3 TS",
         "FG-TS": "FG-TS", "BernoulliTS_Zozotown": "BernoulliTS\n(ZOZOTOWN)", "Greedy": "Greedy (ε=0)",
         "B4_LinUCB": "B4 LinUCB", "B4h_HybridLinUCB": "Hybrid LinUCB"}

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.facecolor": "white"})


def _save(fig, out, name):
    p = out / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written", p)


def _bars(ax, labels, series, names, fmt="{:.2f}", ref=None):
    n = len(series); w = 0.8 / n; x = np.arange(len(labels))
    for k, (vals, nm) in enumerate(zip(series, names)):
        xs = x - 0.4 + w / 2 + k * w
        ax.bar(xs, vals, w * 0.9, color=PAL[k], label=nm, edgecolor="white", linewidth=0.8)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.01, fmt.format(v), ha="center", va="bottom", fontsize=5.5, color=INK, rotation=90)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)


def fig_replay_bias(out):
    p = RES / "s4_replay_bias.json"
    if not p.exists():
        print("skip: no", p); return
    r = json.loads(p.read_text())["shuffle"]
    fig, axes = plt.subplots(1, len(r), figsize=(6.6 if len(r) > 1 else 5.2, 3.0), sharey=True, squeeze=False)
    for ax, (stream, d) in zip(axes[0], r.items()):
        algs = list(d["matched_ratio_vs_random_logged"])
        _bars(ax, [SHORT.get(a, a) for a in algs],
              [[d["matched_ratio_vs_random_logged"][a] for a in algs],
               [d["matched_ratio_vs_random_shuffled"][a] for a in algs]],
              ["log as recorded", "shuffled within hour"], ref=0.5)
        ax.set_title({"random_only": "OBD uniform-random log", "pooled": "OBD pooled log"}.get(stream, stream),
                      fontsize=9, color=INK)
        ax.tick_params(axis="x", labelsize=6.5, rotation=30)
        ax.set_ylim(0, 1.18); ax.text(len(algs) - 0.5, 0.52, "starvation floor 0.5", ha="right", va="bottom", fontsize=6.5, color=INK2)
    axes[0][0].set_ylabel("matched events ÷ B0 Random")
    axes[0][0].legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, 1.22), ncol=2)
    _save(fig, out, "fig4_replay_bias_obd.png")


def fig_kuairec(out):
    p = RES / "kuairec_bias.json"
    if not p.exists():
        print("skip: no", p); return
    s = json.loads(p.read_text())["summary"]
    conds = ["ONLINE", "IID", "BURSTY", "SHUFFLED"]
    lab = {"ONLINE": "online (truth)", "IID": "replay, i.i.d. log", "BURSTY": "replay, bursty log",
           "SHUFFLED": "replay, bursty log shuffled"}
    algs = s["ONLINE"]["ranking"]                       # truth order, best first
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(algs))[::-1]
    for k, c in enumerate(conds):
        v = [s[c]["mean_ctr"][a] for a in algs]
        ax.plot(v, y + (1.5 - k) * 0.16, MARK[k], color=PAL[k], ms=5, mec="white", mew=0.6,
                label=f"{lab[c]} (ρ vs online = {s[c]['spearman_vs_online']:.2f})")
    ax.set_yticks(y); ax.set_yticklabels([SHORT.get(a, a) for a in algs])
    ax.set_xlabel("mean CTR over 30 runs (reward = watch ratio ≥ 2)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, 1.2), ncol=2)
    _save(fig, out, "fig4_kuairec_conditions.png")


def fig_h1(out):
    p = RES / "h1_comparability.json"
    if not p.exists():
        print("skip: no", p); return
    h = json.loads(p.read_text())
    order = [x for x in ["U", "F30T", "F30", "F10", "F5", "F1", "F"] if x in h["protocols"]]
    C = [h["protocols"][x]["C"] for x in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
    cols = [PAL[0] if x == "U" else PAL[1] for x in order]
    a1.bar(order, C, color=cols, edgecolor="white")
    for i, v in enumerate(C):
        a1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    a1.set_ylim(0, 1.05); a1.set_ylabel("comparability C"); a1.grid(axis="x", visible=False)
    a1.set_title("Share of study pairs giving the same verdict", fontsize=8.5, color=INK)
    t = h["tests_U_vs"]; ps = [x for x in order if x != "U"]
    d = [t[x]["delta"] for x in ps]; lo = [t[x]["ci95"][0] for x in ps]; hi = [t[x]["ci95"][1] for x in ps]
    yy = np.arange(len(ps))[::-1]
    a2.errorbar(d, yy, xerr=[np.array(d) - lo, np.array(hi) - d], fmt="o", color=PAL[0], ms=4, capsize=2, lw=1)
    a2.axvline(0.10, color=INK2, ls="--", lw=0.8); a2.axvline(0, color=INK2, lw=0.6)
    a2.text(0.105, yy.max() + 0.45, "minimum meaningful Δ = 0.10", fontsize=6.5, color=INK2)
    a2.set_yticks(yy); a2.set_yticklabels([f"U − {x}" for x in ps])
    a2.set_xlabel("Δ = C_U − C_P with 95% bootstrap interval"); a2.grid(axis="y", visible=False)
    _save(fig, out, "fig4_h1_comparability.png")


def fig_guf(out):
    p = RES / "tables" / "table_4_15_4_17_4_19_guf_values.csv"
    if not p.exists():
        print("skip: no", p); return
    g = pd.read_csv(p)
    sets = list(dict.fromkeys(g.setting))
    fig, axes = plt.subplots(1, len(sets), figsize=(1.75 * len(sets) + 0.6, 2.6), squeeze=False)
    for ax, st in zip(axes[0], sets):
        d = g[g.setting == st]
        last = d[d.w1 == d.w1.max()].sort_values("Gamma")
        top = list(last.algorithm[:4])                  # four best at w1 = 0.8 keep the palette at four slots
        for k, a in enumerate(top):
            e = d[d.algorithm == a].sort_values("w1")
            ax.plot(e.w1, e.Gamma, marker=MARK[k], color=PAL[k], ms=3.5, lw=1.4, label=SHORT.get(a, a))
        ax.set_title(st.replace("_", " "), fontsize=7.5, color=INK)
        ax.set_xlabel("w₁"); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper left")
    axes[0][0].set_ylabel("Γ (lower is better)")
    _save(fig, out, "fig4_guf_sweep.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(RES / "figures"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_replay_bias, fig_kuairec, fig_h1, fig_guf):
        f(out)


if __name__ == "__main__":
    main()
