"""
STEP 5: S4 replay using the outputs of Fixes 1 to 5.

Method: rejection-sampling replay (Li et al., 2011, generalised to a
non-uniform logging policy). At each logged row t the algorithm chooses an
item a_t. The row is ACCEPTED (the algorithm observes the click and
updates) only if
    a_t == logged item   AND   u_t < min(1, c / propensity_t),
with c = 1/K and u_t ~ U(0, 1). For the random log (propensity = 1/K) this
is the standard replayer; for the Bernoulli TS log it down-weights items
the logging policy showed often. Rows whose propensity is below c are
accepted whenever matched, which slightly under-represents rarely shown
items (truncation bias); the B0 Random check and the random-log-only
sensitivity run below measure the size of that bias.

The algorithm's own round counter t is the number of ACCEPTED events + 1
(its real decision horizon).

Algorithms (same seeds, same slice, paired):
  B0 Random, B1 epsilon-Greedy (0.1), B2 UCB1, B3 Thompson Sampling
      -> unchanged classes from mab_benchmark.baselines
  FG-TS (lambda = 0.5)            -> common.FeelGoodTS
  BernoulliTS_Zozotown            -> common.BernoulliTSZozotown (ZOZOTOWN
                                     production priors; external-validity arm)
  B4  disjoint LinUCB (alpha = 1) -> common.LinUCBFast, user context x
  B4h hybrid LinUCB (alpha = 1)   -> common.HybridLinUCB, x + item + affinity

Statistics (thesis Requirements 1 to 4) use the package's own
StatisticalAnalyser on the 30 paired runs (it ranks "lower is better",
so it is given -CTR; signs are converted back in the report).

The Bridge Compliance Score of S4 is computed with every condition measured,
in experiments/s4_bcs_bridge.py, not here.

External-validity check: the replayed CTR of BernoulliTS_Zozotown relative to
B0 Random is compared with the ground-truth on-policy ratio CTR(bts log) /
CTR(random log) of the same cleaned data (and with the published 1.43x).
H1 to H3 are tested with paired statistics.

Outputs
  outputs/s4_external_validity.json / .md
  outputs/s4_learning_curves.csv      cumulative accepted events and clicks
  outputs/s4_replay_runs.csv          one row per (stream, run, algorithm)
  outputs/s4_replay_results.json      everything, machine-readable
  outputs/s4_replay_summary.md / .txt thesis-ready tables
  outputs/s4_selftest.json            equivalence tests of the fast LinUCBs

Run from the repository root (about 20 to 40 minutes on 2 cores):
  python obd_preprocessing/step5_run_s4_replay.py [--jobs 2] [--quick] [--variant keep_duplicates]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

import common as C
from mab_benchmark.baselines import (RandomPolicy, EpsilonGreedy, UCB1,
                                     ThompsonSampling, LinUCB)
from mab_benchmark.analysis.statistics import StatisticalAnalyser

ALGS = ["B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS", "FG-TS",
        "BernoulliTS_Zozotown", "B4_LinUCB", "B4h_HybridLinUCB"]
N_CHECKPOINTS = 50          # learning-curve checkpoints per run (every 2% of rows)
S1_RANKING = ["B3_TS", "FG-TS", "B2_UCB1", "B1_EpsGreedy", "B0_Random"]  # fgts_submission.json

_G = {}   # per-process data


def _init(stream_name, variant="main"):
    C.set_variant(variant)          # needed in child processes on Windows (spawn)
    s = dict(np.load(C.CACHE_DIR / "stream_pooled.npz"))
    ctx = dict(np.load(C.CACHE_DIR / "context.npz"))
    if stream_name == "random_only":
        keep = np.flatnonzero(s["source"] == 0)
        remap = np.full(s["ts"].size, -1, np.int64); remap[keep] = np.arange(keep.size)
        m = remap[s["aff_row"]] >= 0
        s = {k: (v[keep] if v.shape[:1] == s["ts"].shape else v) for k, v in s.items()
             if k not in ("aff_row", "aff_item", "aff_val")} | {
            "aff_row": remap[s["aff_row"][m]], "aff_item": s["aff_item"][m],
            "aff_val": s["aff_val"][m]}
        ctx["active"] = ctx["active"][keep]
        bounds = np.array_split(np.arange(keep.size), C.N_SLICES)
        starts = np.array([b[0] for b in bounds]); ends = np.array([b[-1] + 1 for b in bounds])
    else:
        sl = np.load(C.CACHE_DIR / "slices.npz"); starts, ends = sl["starts"], sl["ends"]
    _G.update(s=s, ctx=ctx, starts=starts, ends=ends, seeds=C.seed_table(),
              d=int(ctx["d"]), S=ctx["item_static"], stream=stream_name)


def _slice_arrays(a, b):
    s, ctx, d = _G["s"], _G["ctx"], _G["d"]
    n = b - a
    X = np.zeros((n, d))
    X[np.arange(n)[:, None], ctx["active"][a:b]] = 1.0
    X[:, 0] = 1.0                                          # bias
    lo, hi = np.searchsorted(s["aff_row"], [a, b])
    Aff = np.zeros((n, C.K))
    Aff[s["aff_row"][lo:hi] - a, s["aff_item"][lo:hi]] = np.log1p(s["aff_val"][lo:hi])
    return X, Aff


def make_alg(name, seed):
    d, S = _G["d"], _G["S"]
    return {"B0_Random": lambda: RandomPolicy(C.K, seed=seed),
            "B1_EpsGreedy": lambda: EpsilonGreedy(C.K, epsilon=0.1),
            "B2_UCB1": lambda: UCB1(C.K),
            "B3_TS": lambda: ThompsonSampling(C.K, reward_type="bernoulli"),
            "FG-TS": lambda: C.FeelGoodTS(C.K, lambda_=0.5),
            "BernoulliTS_Zozotown": lambda: C.BernoulliTSZozotown(C.K),
            "B4_LinUCB": lambda: C.LinUCBFast(C.K, d, alpha=1.0),
            "B4h_HybridLinUCB": lambda: C.HybridLinUCB(C.K, d, S, alpha=1.0)}[name]()


def replay_one(job):
    run, name = job
    s = _G["s"]; a, b = int(_G["starts"][run]), int(_G["ends"][run]); seed = _G["seeds"][run]
    X, Aff = _slice_arrays(a, b)
    item = s["item"][a:b].astype(int); click = s["click"][a:b].astype(float)
    pos = s["pos"][a:b]; src = s["source"][a:b]
    acc_p = np.minimum(1.0, C.ACCEPT_FLOOR / s["pscore"][a:b])
    u = np.random.default_rng([seed, 7]).random(b - a)     # common to all algorithms
    accept_ok = u < acc_p
    np.random.seed(seed % 2**32)
    alg = make_alg(name, seed)
    kind = "hyb" if name == "B4h_HybridLinUCB" else ("lin" if name == "B4_LinUCB" else "free")
    n_acc = 0; clicks = 0.0
    acc_pos = np.zeros(3, int); clk_pos = np.zeros(3)
    acc_src = np.zeros(2, int); clk_src = np.zeros(2)
    checkpoints = set(np.linspace(0, b - a, N_CHECKPOINTS + 1).astype(int)[1:] - 1)
    curve_acc, curve_clk = [], []
    t0 = time.time()
    for t in range(b - a):
        if kind == "free":
            ctx = None
        elif kind == "lin":
            ctx = X[t]
        else:
            ctx = (X[t], Aff[t])
        arm = alg.choose_arm(n_acc + 1, context=ctx)
        if arm == item[t] and accept_ok[t]:
            r = click[t]
            alg.update(arm, r, n_acc + 1, context=ctx)
            n_acc += 1; clicks += r
            acc_pos[pos[t] - 1] += 1; clk_pos[pos[t] - 1] += r
            acc_src[src[t]] += 1; clk_src[src[t]] += r
        if t in checkpoints:
            curve_acc.append(n_acc); curve_clk.append(int(clicks))
    return {"stream": _G["stream"], "run": run + 1, "seed": seed, "algorithm": name,
            "rows": b - a, "accepted": n_acc, "clicks": int(clicks),
            "ctr": clicks / n_acc if n_acc else np.nan,
            **{f"accepted_pos{q}": int(acc_pos[q - 1]) for q in C.POSITIONS},
            **{f"clicks_pos{q}": int(clk_pos[q - 1]) for q in C.POSITIONS},
            "accepted_random_rows": int(acc_src[0]), "accepted_bts_rows": int(acc_src[1]),
            "seconds": round(time.time() - t0, 1),
            "_curve_acc": curve_acc, "_curve_clk": curve_clk}


# ---------------------------------------------------------------------------
# Self-tests: fast LinUCB classes must decide exactly like the references
# ---------------------------------------------------------------------------
class _HybridReference:
    """Direct transcription of Li et al. (2010), Algorithm 2 (explicit inverses)."""

    def __init__(self, K, d, S, alpha=1.0):
        self.K, self.S, self.alpha = K, S, alpha
        k = S.shape[1] + 1
        self.A0, self.b0 = np.eye(k), np.zeros(k)
        self.A = [np.eye(d) for _ in range(K)]
        self.B = [np.zeros((d, k)) for _ in range(K)]
        self.b = [np.zeros(d) for _ in range(K)]

    def choose(self, x, aff):
        A0i = np.linalg.inv(self.A0); beta = A0i @ self.b0; p = np.zeros(self.K)
        for a in range(self.K):
            z = np.append(self.S[a], aff[a]); Ai = np.linalg.inv(self.A[a])
            th = Ai @ (self.b[a] - self.B[a] @ beta)
            s = (z @ A0i @ z - 2 * z @ A0i @ self.B[a].T @ Ai @ x + x @ Ai @ x
                 + x @ Ai @ self.B[a] @ A0i @ self.B[a].T @ Ai @ x)
            p[a] = z @ beta + x @ th + self.alpha * np.sqrt(max(s, 0))
        return int(np.argmax(p))

    def update(self, a, r, x, aff):
        z = np.append(self.S[a], aff[a]); Ai = np.linalg.inv(self.A[a])
        self.A0 += self.B[a].T @ Ai @ self.B[a]; self.b0 += self.B[a].T @ Ai @ self.b[a]
        self.A[a] += np.outer(x, x); self.B[a] += np.outer(x, z); self.b[a] += r * x
        Ai = np.linalg.inv(self.A[a])
        self.A0 += np.outer(z, z) - self.B[a].T @ Ai @ self.B[a]
        self.b0 += r * z - self.B[a].T @ Ai @ self.b[a]


def self_test(n_rows=4000):
    _init("pooled", C.VARIANT)
    s = _G["s"]; X, Aff = _slice_arrays(0, n_rows)
    item = s["item"][:n_rows].astype(int); click = s["click"][:n_rows].astype(float)
    rng = np.random.default_rng(0)
    # Force frequent updates so the test exercises many parameter states
    forced = rng.integers(0, C.K, n_rows)
    out = {}
    fast, ref = C.LinUCBFast(C.K, _G["d"]), LinUCB(C.K, _G["d"])
    same = 0; ties = 0; worst_gap = 0.0
    for t in range(n_rows):
        a1 = fast.choose_arm(t + 1, X[t]); a2 = ref.choose_arm(t + 1, X[t])
        if a1 == a2:
            same += 1
        else:   # accept only exact numerical ties (|UCB_a1 - UCB_a2| < 1e-9)
            u = [x @ np.linalg.inv(ref.A[a]) @ ref.b[a]
                 + np.sqrt(x @ np.linalg.inv(ref.A[a]) @ x) for a, x in ((a1, X[t]), (a2, X[t]))]
            gap = abs(u[0] - u[1]); worst_gap = max(worst_gap, gap)
            ties += gap < 1e-9
        arm = forced[t]; r = float(rng.random() < 0.3)
        fast.update(arm, r, t + 1, X[t]); ref.update(arm, r, t + 1, X[t])
    out["LinUCBFast_vs_package_LinUCB"] = {"rows": n_rows, "identical_choices": int(same),
                                           "numerical_ties": int(ties),
                                           "largest_gap_at_a_difference": float(worst_gap)}
    hyb, href = C.HybridLinUCB(C.K, _G["d"], _G["S"]), _HybridReference(C.K, _G["d"], _G["S"])
    same = 0; m = 1500
    Aff2 = Aff.copy(); Aff2[rng.random(Aff2.shape) < 0.02] = np.log1p(rng.integers(1, 5))
    for t in range(m):
        a1 = hyb.choose_arm(t + 1, (X[t], Aff2[t])); a2 = href.choose(X[t], Aff2[t])
        same += a1 == a2
        arm = forced[t]; r = float(rng.random() < 0.3)
        hyb.update(arm, r, t + 1, (X[t], Aff2[t])); href.update(arm, r, X[t], Aff2[t])
    out["HybridLinUCB_vs_reference"] = {"rows": m, "identical_choices": int(same)}
    C.write_json(out, C.OUT_DIR / "s4_selftest.json")
    print("self-test:", out)
    for v in out.values():
        if v["identical_choices"] + v.get("numerical_ties", 0) != v["rows"]:
            raise SystemExit("Self-test failed: fast LinUCB differs from reference")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def analyse(df, stream):
    sub = df[df.stream == stream]
    res = {a: sub[sub.algorithm == a].sort_values("run")["ctr"].to_numpy() for a in ALGS}
    rep = StatisticalAnalyser(alpha=0.05).analyse({a: -v for a, v in res.items()})
    summary = {a: {"mean_ctr": float(v.mean()), "sd_ctr": float(v.std(ddof=1)),
                   "ci95_low": float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                   "ci95_high": float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                   "pooled_ctr": float(sub[sub.algorithm == a].clicks.sum()
                                       / sub[sub.algorithm == a].accepted.sum()),
                   "mean_accepted": float(sub[sub.algorithm == a].accepted.mean()),
                   "mean_clicks": float(sub[sub.algorithm == a].clicks.mean())}
               for a, v in res.items()}
    pairs = []
    for (a, b), p in rep["pairwise"].items():
        es = rep["effect_sizes"].get((a, b)) or rep["effect_sizes"].get(f"{a} vs {b}") or {}
        pairs.append({"a": a, "b": b, "p_raw": p["p_raw"], "holm_threshold": p["p_threshold"],
                      "significant": p["significant"],
                      "higher_ctr": (a if summary[a]["mean_ctr"] > summary[b]["mean_ctr"] else b)
                      if p["significant"] else "not significant",
                      "cohens_d_ctr": -es.get("cohens_d", np.nan) if es else np.nan,
                      "a12_ctr": es.get("a12", np.nan) if es else np.nan})
    ranking = sorted(ALGS, key=lambda a: -summary[a]["mean_ctr"])
    return {"summary": summary, "pairwise": pairs, "ranking": ranking}


def external_validity(df, stream):
    """BernoulliTS_Zozotown vs B0 Random against the on-policy ground truth."""
    integ = json.loads((C.OUT_DIR / "fix5_integrity_report.json").read_text())["reports"]
    gt = {r["policy"]: r["ctr_clean"] for r in integ}
    gt_ratio = gt["bts"] / gt["random"]
    sub = df[df.stream == stream]
    z = sub[sub.algorithm == "BernoulliTS_Zozotown"].sort_values("run")
    r0 = sub[sub.algorithm == "B0_Random"].sort_values("run")
    zc, rc = z.ctr.to_numpy(), r0.ctr.to_numpy()
    ratio = zc.mean() / rc.mean()
    rng = np.random.default_rng(2024)
    idx = rng.integers(0, len(zc), (10_000, len(zc)))
    boots = zc[idx].mean(1) / rc[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    h1 = stats.wilcoxon(zc, rc, alternative="greater")
    ranking = sorted(ALGS, key=lambda a: -sub[sub.algorithm == a].ctr.mean())
    rank_z = ranking.index("BernoulliTS_Zozotown") + 1
    z_mean, z_ci = zc.mean(), 1.96 * zc.std(ddof=1) / np.sqrt(len(zc))
    return {
        "stream": stream,
        "ground_truth_ctr_bts_log": gt["bts"], "ground_truth_ctr_random_log": gt["random"],
        "ground_truth_ratio_this_data": gt_ratio,
        "published_ratio": C.PUBLISHED_CTR["bts"] / C.PUBLISHED_CTR["random"],
        "replay_ctr_BernoulliTS_Zozotown": z_mean, "replay_ctr_ci95": [z_mean - z_ci, z_mean + z_ci],
        "replay_ctr_B0_Random": rc.mean(),
        "replay_ratio": ratio, "replay_ratio_ci95_bootstrap": [float(lo), float(hi)],
        "rank_of_BernoulliTS_Zozotown": rank_z, "n_algorithms": len(ALGS),
        "H1_bts_beats_random": {"test": "one-sided paired Wilcoxon, BTS > Random",
                                "p": float(h1.pvalue), "supported": bool(h1.pvalue < 0.05)},
        "H2_rank_top_two": {"rank": rank_z, "supported": rank_z <= 2},
        "H3a_direction_matches": {"replay_ratio_above_1": bool(ratio > 1),
                                  "supported": bool(ratio > 1)},
        "H3b_ratio_matches": {"ground_truth_inside_ci95": bool(lo <= gt_ratio <= hi),
                              "supported": bool(lo <= gt_ratio <= hi)},
        "H3c_absolute_ctr_matches": {"ground_truth_bts_inside_ci95":
                                     bool(z_mean - z_ci <= gt["bts"] <= z_mean + z_ci)}}


def position_table(df, stream):
    sub = df[df.stream == stream]; out = {}
    for a in ALGS:
        g = sub[sub.algorithm == a]
        out[a] = {q: float(g[f"clicks_pos{q}"].sum() / max(g[f"accepted_pos{q}"].sum(), 1))
                  for q in C.POSITIONS}
    return out


def write_reports(df, results):
    L = ["# S4 replay: Open Bandit Dataset (random + bts, campaign all)", ""]
    for stream, R in results["analyses"].items():
        title = {"pooled": "Main result: both logs, all positions pooled",
                 "random_only": "Sensitivity: random log only (no truncation bias, lower power)"}[stream]
        L += [f"## {title}", "",
              "| Rank | Algorithm | Mean CTR | SD | 95% CI | Mean accepted events | Mean clicks |",
              "|---|---|---|---|---|---|---|"]
        for i, a in enumerate(R["ranking"], 1):
            s = R["summary"][a]
            L.append(f"| {i} | {a} | {s['mean_ctr']:.4%} | {s['sd_ctr']:.4%} | "
                     f"{s['ci95_low']:.4%} to {s['ci95_high']:.4%} | {s['mean_accepted']:,.0f} | "
                     f"{s['mean_clicks']:.1f} |")
        L += ["", "Pairwise tests (two-sided paired Wilcoxon over 30 runs, Holm-Bonferroni, "
              "alpha = 0.05; d and A12 oriented so that positive d / A12 > 0.5 favour the first algorithm):", "",
              "| Pair | p (raw) | Holm threshold | Significant | Higher CTR | Cohen's d | A12 |",
              "|---|---|---|---|---|---|---|"]
        for p in sorted(R["pairwise"], key=lambda r: r["p_raw"]):
            L.append(f"| {p['a']} vs {p['b']} | {p['p_raw']:.4g} | {p['holm_threshold']:.4g} | "
                     f"{'yes' if p['significant'] else 'no'} | {p['higher_ctr']} | "
                     f"{p['cohens_d_ctr']:.3f} | {p['a12_ctr']:.3f} |")
        L += [""]
        pt = results["positions"][stream]
        L += ["Per-position CTR from the same runs (clicks / accepted events, summed over runs):", "",
              "| Algorithm | Position 1 | Position 2 | Position 3 |", "|---|---|---|---|"]
        for a in R["ranking"]:
            q = {int(k): v for k, v in pt[a].items()}
            L.append(f"| {a} | {q[1]:.4%} | {q[2]:.4%} | {q[3]:.4%} |")
        L.append("")
    L += ["## External validity: BernoulliTS with ZOZOTOWN priors vs ground truth", "",
          "| Stream | Replay CTR BTS-Z | Replay CTR Random | Replay ratio (95% CI) | Ground-truth ratio "
          "(this data) | Published ratio | Rank of BTS-Z |", "|---|---|---|---|---|---|---|"]
    for stream, E in results["external_validity"].items():
        lo, hi = E["replay_ratio_ci95_bootstrap"]
        L.append(f"| {stream} | {E['replay_ctr_BernoulliTS_Zozotown']:.4%} | {E['replay_ctr_B0_Random']:.4%} | "
                 f"{E['replay_ratio']:.3f}x ({lo:.3f} to {hi:.3f}) | {E['ground_truth_ratio_this_data']:.3f}x | "
                 f"{E['published_ratio']:.2f}x | {E['rank_of_BernoulliTS_Zozotown']} of {E['n_algorithms']} |")
    L += ["", "| Hypothesis | " + " | ".join(results["external_validity"]) + " |",
          "|---|" + "---|" * len(results["external_validity"])]
    rowsH = [("H1: BTS-Z CTR > Random (one-sided paired Wilcoxon)",
              lambda E: f"p = {E['H1_bts_beats_random']['p']:.4f}, "
                        f"{'supported' if E['H1_bts_beats_random']['supported'] else 'not supported'}"),
             ("H2: BTS-Z ranks in the top two",
              lambda E: f"rank {E['H2_rank_top_two']['rank']}, "
                        f"{'supported' if E['H2_rank_top_two']['supported'] else 'not supported'}"),
             ("H3a: direction matches ground truth (ratio > 1)",
              lambda E: "supported" if E["H3a_direction_matches"]["supported"] else "not supported"),
             ("H3b: ground-truth ratio inside the 95% CI",
              lambda E: "supported" if E["H3b_ratio_matches"]["supported"] else "not supported")]
    for name, f in rowsH:
        L.append(f"| {name} | " + " | ".join(f(E) for E in results["external_validity"].values()) + " |")
    L += [""]
    chk = results["checks"]
    L += ["## Checks", "",
          f"- B0 Random replay CTR (pooled stream) = {chk['b0_pooled_ctr']:.4%} vs random-log CTR "
          f"{chk['random_log_ctr']:.4%}: relative difference {chk['b0_relative_bias']:+.1%} "
          "(size of the truncation bias from the bts rows).",
          f"- Self-tests: {chk['selftest']}.",
          f"- Runtime: {chk['runtime_min']:.1f} minutes on {chk['jobs']} process(es).", ""]
    md = "\n".join(L)
    ev = md[md.index("## External validity"):md.index("## Checks")]
    (C.OUT_DIR / "s4_external_validity.md").write_text(
        "# S4 external validity (variant: " + C.VARIANT + ")\n\n" + ev, encoding="utf-8")
    (C.OUT_DIR / "s4_replay_summary.md").write_text(md, encoding="utf-8")
    (C.OUT_DIR / "s4_replay_summary.txt").write_text(md.replace("**", ""), encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="3 runs only, for a smoke test")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--reports-only", action="store_true",
                    help="rebuild the reports from the saved results and runs, without replaying")
    C.add_variant_arg(ap)
    args = ap.parse_args()
    C.set_variant(args.variant or "main")
    if args.reports_only:
        results = json.loads((C.OUT_DIR / "s4_replay_results.json").read_text())
        write_reports(pd.read_csv(C.OUT_DIR / "s4_replay_runs.csv"), results)
        return
    print(f"variant: {C.VARIANT}  (cache: {C.CACHE_DIR}, outputs: {C.OUT_DIR})")
    t0 = time.time()
    if not args.skip_selftest:
        self_test()
    runs = range(3) if args.quick else range(C.N_SLICES)
    rows = []
    for stream in ("pooled", "random_only"):
        jobs = [(r, a) for r in runs for a in ALGS]
        # longest jobs first for better load balance
        jobs.sort(key=lambda j: j[1] != "B4h_HybridLinUCB")
        with Pool(args.jobs, initializer=_init, initargs=(stream, C.VARIANT)) as pool:
            for i, res in enumerate(pool.imap_unordered(replay_one, jobs), 1):
                rows.append(res)
                print(f"[{stream}] {i}/{len(jobs)} run {res['run']:>2} {res['algorithm']:<17} "
                      f"accepted {res['accepted']:>5} clicks {res['clicks']:>3} ({res['seconds']}s)",
                      flush=True)
    curves = []
    for r in rows:
        ca, cc = r.pop("_curve_acc"), r.pop("_curve_clk")
        for j, (x1, x2) in enumerate(zip(ca, cc), 1):
            curves.append({"stream": r["stream"], "algorithm": r["algorithm"], "run": r["run"],
                           "checkpoint": j, "row_fraction": j / N_CHECKPOINTS,
                           "accepted": x1, "clicks": x2})
    pd.DataFrame(curves).to_csv(C.OUT_DIR / "s4_learning_curves.csv", index=False)
    df = pd.DataFrame(rows).sort_values(["stream", "algorithm", "run"])
    df.to_csv(C.OUT_DIR / "s4_replay_runs.csv", index=False)
    if args.quick:
        print(df.groupby(["stream", "algorithm"])[["accepted", "clicks", "ctr", "seconds"]].mean())
        return
    random_ctr = float(np.load(C.CACHE_DIR / "clean_random.npz")["click"].mean())
    results = {"variant": C.VARIANT, "environment": C.environment_info(),
               "analyses": {}, "positions": {}, "external_validity": {}}
    for stream in ("pooled", "random_only"):
        results["analyses"][stream] = analyse(df, stream)
        results["positions"][stream] = position_table(df, stream)
        results["external_validity"][stream] = external_validity(df, stream)
    b0 = results["analyses"]["pooled"]["summary"]["B0_Random"]["pooled_ctr"]
    results["checks"] = {
        "b0_pooled_ctr": b0, "random_log_ctr": random_ctr,
        "b0_relative_bias": b0 / random_ctr - 1,
        "selftest": "skipped" if args.skip_selftest else
        "fast LinUCB classes made identical choices to the references",
        "runtime_min": (time.time() - t0) / 60, "jobs": args.jobs}
    C.write_json(results, C.OUT_DIR / "s4_replay_results.json")
    C.write_json(results["external_validity"], C.OUT_DIR / "s4_external_validity.json")
    write_reports(df, results)


if __name__ == "__main__":
    main()
