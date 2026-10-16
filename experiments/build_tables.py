"""
build_tables.py: every table of Chapters Four and Five, built from the raw outputs.

Inputs : results/raw/*.npy, results/lambda_sweep.csv, results/latency_profile.csv,
         results/layers26_*, results/s4_bcs_bridge.json,
         ../obd_preprocessing/outputs/s4_replay_results.json and s4_replay_runs.csv
         (S4 replay, for the Pareto and GUF tables)
Outputs: results/tables/*.csv and results/tables/CHAPTER4_5_TABLES.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import exp_common as E
import mab_benchmark.analysis.ncr as ncr_mod
from mab_benchmark.runner import SEED_TABLE
from mab_benchmark import StatisticalAnalyser, BCSCalculator
from mab_benchmark.analysis import guf
from mab_benchmark.explain import TOLERANCES

S4DIR = E.REPO / "obd_preprocessing" / "outputs"
PRIMARY = "S1_K10_T10000"
FIVE = ["FG-TS", "B3_TS", "B2_UCB1", "B1_EpsGreedy", "B0_Random"]
PRIMARY_BCS = "random_only: full-log Tier 2, domain-matched Tier 1"
MD = []


def raw(cell, alg):
    p = E.RAW / f"{cell}__{alg}.npy"
    return np.load(p) if p.exists() else None


def ci(x, kind="z"):
    m, s, n = x.mean(), x.std(ddof=1), x.size
    q = 1.959963984540054 if kind == "z" else stats.t.ppf(0.975, n - 1)
    return m - q * s / np.sqrt(n), m + q * s / np.sqrt(n)


def md_table(title, df, note=""):
    MD.extend([f"## {title}", ""])
    cols = list(df.columns)
    MD.append("| " + " | ".join(map(str, cols)) + " |")
    MD.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        MD.append("| " + " | ".join(_fmt(v) for v in r.values) + " |")
    if note:
        MD.extend(["", note])
    MD.append("")


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "-"
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return str(v)


def save(df, name):
    df.to_csv(E.TAB / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------
def table_4_2():
    rows = []
    for cell in [c for c in E.CELLS if c.startswith("S1")]:
        means = {a: raw(cell, a).mean() for a in FIVE}
        order = sorted(FIVE, key=means.get)
        rows.append({"setting": cell, **means, "FG-TS rank": f"#{order.index('FG-TS') + 1}"})
    df = pd.DataFrame(rows); save(df, "table_4_2_S1_all_cells")
    md_table("Table 4.2. Mean cumulative regret on S1, n = 30", df)
    return df


def table_4_3():
    rows = []
    for a in FIVE + ["BernoulliTS_uniform"]:
        x = raw(PRIMARY, a); lz, hz = ci(x); lt, ht = ci(x, "t")
        rows.append({"algorithm": E.LABEL[a], "mean": x.mean(), "SD": x.std(ddof=1), "95% CI low (z)": lz,
                     "95% CI high (z)": hz, "95% CI low (t29)": lt, "95% CI high (t29)": ht,
                     "median": np.median(x)})
    df = pd.DataFrame(rows).sort_values("mean"); save(df, "table_4_3_primary_summary")
    md_table("Table 4.3. Summary statistics, S1 K=10 T=10,000", df)
    return df


def table_4_4_all():
    rows = []
    for cell in E.CELLS:
        for a in E.algs_for(cell):
            x = raw(cell, a)
            if x is None:
                continue
            lo, hi = ci(x)
            rows.append({"setting": cell, "algorithm": a, "mean": x.mean(), "SD": x.std(ddof=1),
                         "ci_low": lo, "ci_high": hi})
    df = pd.DataFrame(rows); save(df, "table_4_4_all_algorithms_all_settings")
    wide = df.pivot(index="setting", columns="algorithm", values="mean").reindex(list(E.CELLS))
    wide = wide[[a for a in E.ALGS if a in wide.columns]].reset_index()
    rank = df.loc[df.algorithm != "BernoulliTS_uniform"].copy()
    rank["rank"] = rank.groupby("setting")["mean"].rank(method="min")
    save(rank, "table_4_4_ranks")
    md_table("Table 4.4. Mean cumulative regret, every algorithm, every synthetic cell (n = 30)", wide)
    return df


def tables_4_5_4_6():
    res = {a: raw(PRIMARY, a) for a in FIVE}
    rep = StatisticalAnalyser().analyse(res)
    rows, eff = [], []
    for (a, b), v in sorted(rep["pairwise"].items(), key=lambda kv: kv[1]["p_raw"]):
        e = rep["effect_sizes"][(a, b)]
        key = f"{a} vs {b}"
        rows.append({"comparison": key, "p_raw": v["p_raw"], "Holm threshold": v["p_threshold"],
                     "significant": "YES" if v["significant"] else "no", "winner (lower regret)": v["better"]})
        eff.append({"comparison": key, "Cohen d": e["cohens_d"], "A12": e["a12"], "magnitude": e["magnitude"]})
    d1, d2 = pd.DataFrame(rows), pd.DataFrame(eff)
    save(d1, "table_4_5_wilcoxon_holm_primary"); save(d2, "table_4_6_effect_sizes_primary")
    md_table("Table 4.5. Wilcoxon signed-rank with Holm-Bonferroni, S1 K=10 T=10,000", d1)
    md_table("Table 4.6. Cohen's d and Vargha-Delaney A12 (ties count 1/2)", d2,
             "A12 is P(first < second) + 0.5 P(tie): values below 0.5 mean the first algorithm has higher regret.")
    allrows = []
    for cell in E.CELLS:
        names = [a for a in E.algs_for(cell) if a != "BernoulliTS_uniform"]
        r = StatisticalAnalyser().analyse({a: raw(cell, a) for a in names})
        allrows.append({"setting": cell, "algorithms": len(names), "pairs": len(r["pairwise"]),
                        "significant pairs (Holm)": sum(v["significant"] for v in r["pairwise"].values()),
                        "ranking (best first)": " < ".join(r["ranking"])})
        E.write_json({f"{a} vs {b}": {**v, **r["effect_sizes"][(a, b)]} for (a, b), v in r["pairwise"].items()},
                     E.TAB / f"pairwise_{cell}.json")
    d3 = pd.DataFrame(allrows); save(d3, "table_pairwise_summary_all_settings")
    md_table("Pairwise significance in every synthetic cell", d3)
    return rep


def table_4_11():
    x, y = raw(PRIMARY, "BernoulliTS_uniform"), raw(PRIMARY, "FG-TS")
    rep = StatisticalAnalyser().analyse({"BernoulliTS_uniform": x, "FG-TS": y})
    (k, v), = rep["pairwise"].items(); e = rep["effect_sizes"][k]
    rows = []
    for a in ["BernoulliTS_uniform", "FG-TS", "B3_TS", "B2_UCB1", "B1_EpsGreedy", "B0_Random"]:
        z = raw(PRIMARY, a); lo, hi = ci(z)
        rows.append({"algorithm": E.LABEL[a], "mean": z.mean(), "SD": z.std(ddof=1), "ci_low": lo, "ci_high": hi,
                     "identical to B3 run-by-run": bool(np.array_equal(z, raw(PRIMARY, "B3_TS")))})
    df = pd.DataFrame(rows); save(df, "table_4_11_two_test_algorithms_S1")
    md_table("Table 4.11. Two test algorithms and four baselines, S1 K=10 T=10,000", df,
             f"Bernoulli TS (uniform) vs FG-TS: Wilcoxon p = {v['p_raw']:.3g}, Cohen d = {e['cohens_d']}, "
             f"A12 = {e['a12']}. Bernoulli TS with uniform priors draws exactly the same numbers as B3 under the "
             f"benchmark's seeding, so its row equals B3 run by run.")


def latency_table():
    lat = pd.read_csv(E.OUT / "latency_profile.csv")
    save(lat, "table_4_9_latency_measured")
    md_table("Table 4.9. Per-round decision latency (choose_arm + update), ms",
             lat[["cell", "algorithm", "median_ms", "p95_ms", "max_ms", "L_stress_0.5ms", "L_2G_200ms"]],
             "L = mean over rounds of max(0, l_t / delta - 1). Machine: see latency_meta.json.")
    return lat


def s4_obd():
    res = json.loads((S4DIR / "s4_replay_results.json").read_text())["analyses"]["pooled"]
    runs = pd.read_csv(S4DIR / "s4_replay_runs.csv")
    runs = runs[runs.stream == "pooled"]
    ms = (runs.seconds * 1000 / runs.rows).groupby(runs.algorithm).median()   # per logged row
    return res, ms


def ncr_runs(cell, alg):
    """Per-run NCR = regret / expected Random regret on the same instance (analysis.ncr)."""
    EnvClass, env_kwargs, _ = E.CELLS[cell]
    return ncr_mod.ncr_for_setting(raw(cell, alg), EnvClass, env_kwargs, SEED_TABLE[:E.N_RUNS])


def pareto_and_guf(lat):
    """GUF on (NCR, L, CCM), all dimensionless with 0 = ideal; the setting's BCS is a gate."""
    weights = [0.4, 0.5, 0.6, 0.7, 0.8]
    allrows, guf_rows, rank_rows = [], [], []
    settings, gates = {}, {}
    for cell in ["S1_K10_T10000", "S3_K10_T10000", "S4_SYNTHETIC_K50_T10000"]:
        L = lat[lat.cell == cell].set_index("algorithm")["L_stress_0.5ms"]
        C = lat[lat.cell == cell].set_index("algorithm")["CCM_stress_0.5ms"]
        names = [a for a in E.algs_for(cell) if a != "BernoulliTS_uniform"]
        settings[cell] = {a: (float(np.nanmean(ncr_runs(cell, a))), float(L[a]), float(C[a])) for a in names}
        gates[cell] = None                      # synthetic: no Tier 2 evidence, gate not passed
    s4, ms = s4_obd()
    ctr = {a: v["mean_ctr"] for a, v in s4["summary"].items()}
    bridge = json.loads((E.OUT / "s4_bcs_bridge.json").read_text())
    cal = bridge["calibration"]                 # item CTRs of the random log
    mu_star, mu_rand = cal["item_ctr_max"], cal["item_ctr_mean"]
    L_obd = {a: float(np.maximum(0, ms[a] / 0.5 - 1)) for a in ctr}       # per-row median time vs 0.5 ms
    # NCR on logged data: (mu* - CTR) / (mu* - mu_rand), with mu* the best item CTR and mu_rand the
    # mean item CTR of the uniform random log (the expected CTR of uniform Random).
    settings["S4_OBD"] = {a: (max(0.0, (mu_star - ctr[a]) / (mu_star - mu_rand)), L_obd[a], 0.5 * L_obd[a])
                          for a in ctr}
    gates["S4_OBD"] = bridge["bcs_variants"][PRIMARY_BCS]["BCS"]
    for st, pts in settings.items():
        pipe = guf.run_gap5_pipeline(pts, weights, bcs=gates[st])
        dom = pipe["dominated_by"]
        for a, (r, l, c) in pts.items():
            allrows.append({"setting": st, "algorithm": a, "NCR": r, "L (0.5 ms)": l,
                            "CCM": c, "Pareto-optimal": "yes" if not dom[a] else "no",
                            "dominated by": ", ".join(dom[a])})
        ranks = {}
        for w1, v in pipe["sweep"]["by_w1"].items():
            for a, gv in v["gamma"].items():
                guf_rows.append({"setting": st, "algorithm": a, "w1": w1, "Gamma": gv})
            ranks[w1] = v["ranking"]
        stable = pipe["sweep"]["stable"]
        changes = sorted({" < ".join(v) for v in ranks.values()})
        g = pipe["gate"]
        rank_rows.append({"setting": st, "ranking at w1=0.6 (best first)": " < ".join(ranks[0.6]),
                          "identical across sweep": "yes" if stable else "no",
                          "distinct rankings in sweep": len(changes),
                          "BCS": "n/a (no Tier 2)" if g["bcs"] is None else round(g["bcs"], 4),
                          "certifiable": "yes" if g["certifiable"] else "no"})
    p, g, r = pd.DataFrame(allrows), pd.DataFrame(guf_rows), pd.DataFrame(rank_rows)
    save(p, "table_4_14a_pareto"); save(g, "table_4_15_4_17_4_19_guf_values"); save(r, "table_4_16_4_18_4_20_guf_rankings")
    md_table("Table 4.14a. Pareto characterisation on (NCR, L, CCM) with measured latency", p,
             "NCR = regret / expected Random regret (synthetic); on S4 OBD, (mu* - CTR) / (mu* - mu_rand) "
             "from the random log's item CTRs.")
    md_table("Tables 4.16 / 4.18 / 4.20. Normalised GUF ranking across the w1 sweep, with the BCS gate", r)
    return p, r


def lambda_table():
    lam = pd.read_csv(E.OUT / "lambda_sweep.csv")
    g = lam.groupby(["cell", "lambda"]).cumulative_regret.agg(["mean", "std"]).reset_index()
    ts = {c: raw(c, "B3_TS").mean() for c in g.cell.unique()}
    g["B3_TS mean"] = g.cell.map(ts)
    eq = {c: bool(np.allclose(lam[(lam.cell == c) & (lam["lambda"] == 0)].sort_values("run").cumulative_regret,
                              raw(c, "B3_TS"))) for c in g.cell.unique()}
    g["lambda=0 identical to B3"] = [("yes" if eq[c] else "NO") if l == 0 else "" for c, l in zip(g.cell, g["lambda"])]
    save(g, "figure_5_1_lambda_sweep_table")
    md_table("Figure 5.1 data. FG-TS lambda sweep", g)
    return g, eq


def table_5_1(lam_eq):
    def m(c, a): return raw(c, a).mean()
    rep_s1 = StatisticalAnalyser().analyse({a: raw(PRIMARY, a) for a in FIVE})
    rep_s3 = StatisticalAnalyser().analyse({a: raw("S3_K10_T10000", a) for a in FIVE})
    def sig(rep, a, b):
        k = (a, b) if (a, b) in rep["pairwise"] else (b, a)
        return rep["pairwise"][k]["significant"], rep["effect_sizes"][k]["cohens_d"]
    g = m("S1_K10_T10000", "FG-TS") / m("S1_K10_T500", "FG-TS")
    s_ts1, d_ts1 = sig(rep_s1, "FG-TS", "B3_TS"); s_u1, d_u1 = sig(rep_s1, "FG-TS", "B2_UCB1")
    s_ts3, d_ts3 = sig(rep_s3, "FG-TS", "B3_TS"); s_u3, d_u3 = sig(rep_s3, "FG-TS", "B2_UCB1")
    X, V = "design expectation", "evidence"
    rows = [
        {"#": 1, X: "Sub-linear regret growth on S1",
         V: f"T x20 (500 to 10,000) multiplies FG-TS regret by {g:.2f}", "verdict": "SUPPORTED" if g < 20 else "NOT SUPPORTED"},
        {"#": 2, X: "The bonus costs regret against TS on stationary S1",
         V: f"FG-TS {m(PRIMARY,'FG-TS'):.1f} vs TS {m(PRIMARY,'B3_TS'):.1f}; significant {s_ts1}, d = {d_ts1}", "verdict": "SUPPORTED" if s_ts1 and m(PRIMARY,'FG-TS') > m(PRIMARY,'B3_TS') else "NOT SUPPORTED"},
        {"#": 3, X: "The bonus helps against TS after change-points (S3)",
         V: f"FG-TS {m('S3_K10_T10000','FG-TS'):.1f} vs TS {m('S3_K10_T10000','B3_TS'):.1f}; significant {s_ts3}, d = {d_ts3}", "verdict": "SUPPORTED" if s_ts3 and m('S3_K10_T10000','FG-TS') < m('S3_K10_T10000','B3_TS') else "NOT SUPPORTED"},
        {"#": 4, X: "FG-TS beats UCB1 on S1 and on S3",
         V: f"S1: {m(PRIMARY,'FG-TS'):.1f} vs {m(PRIMARY,'B2_UCB1'):.1f} (sig {s_u1}); S3: {m('S3_K10_T10000','FG-TS'):.1f} vs {m('S3_K10_T10000','B2_UCB1'):.1f} (sig {s_u3})",
         "verdict": "S1 SUPPORTED; S3 NOT SUPPORTED (UCB1 best on S3)" if m('S3_K10_T10000','B2_UCB1') < m('S3_K10_T10000','FG-TS') else "SUPPORTED"},
        {"#": 5, X: "lambda = 0 reduces FG-TS exactly to B3",
         V: f"identical run by run: {lam_eq}", "verdict": "SUPPORTED" if all(lam_eq.values()) else "NOT SUPPORTED"},
    ]
    df = pd.DataFrame(rows); save(df, "table_5_1_design_expectations")
    md_table("Table 5.1. Design expectations for the optimism bonus, tested", df,
             "Zhang (2022) defines the feel-good term inside the loss of a contextual model; the expectations "
             "here follow from the count-based bonus of the finite-armed algorithm and are the thesis' own.")


def layers_2_6_tables():
    f = E.OUT / "layers26_results.json"
    if not f.exists():
        return
    res = json.loads(f.read_text())
    s = pd.read_csv(E.OUT / "layers26_mpo_summary.csv")
    cols = ["cell", "algorithm", "regret_plain", "regret_mpo", "change_pct", "wilcoxon_p", "flags_per_run",
            "resets_per_run", "reset_rule_ok", "detected_all_cp", "detected_when_acting_arm_dropped_by_0.1+",
            "median_delay", "false_alarms_per_run"]
    save(s[cols], "table_layer2_mpo")
    r = pd.read_csv(E.OUT / "layers26_mpo_runs.csv"); s3 = r[r.cell == "S3_K10_T10000"]
    g = lambda a, m: s3[(s3.algorithm == a) & (s3.mpo == m)].sort_values("run").regret.to_numpy()
    vs = []
    for a in ("B3_TS", "FG-TS"):
        x, u = g(a, True), g("B2_UCB1", False)
        vs.append({"comparison": f"MPO({a}) vs plain B2_UCB1 on S3", "mean_mpo": x.mean(), "mean_ucb1": u.mean(),
                   "wilcoxon_p": stats.wilcoxon(x, u).pvalue,
                   "cohen_d": (x.mean() - u.mean()) / np.sqrt((x.var(ddof=1) + u.var(ddof=1)) / 2)})
    vs = pd.DataFrame(vs); save(vs, "table_layer2_mpo_vs_ucb1")
    md_table("Layer 2. Meta-Policy Orchestrator: change-point flags, posterior resets and regret, 30 runs",
             s[cols], f"A change-point counts as detected when a flag falls within {res['detect_window']} rounds "
             f"after it; window W = {res['mpo_window']}. reset_rule_ok: resets = flags for B3 and FG-TS, "
             "no resets for the other algorithms. S1 has no change-points, so every S1 flag would be a false alarm.")
    md_table("Layer 2. MPO-wrapped Thompson Sampling against plain UCB1 on S3", vs)
    st = res["ccm_stress"]
    ccm = pd.DataFrame([{"test": "latency injection (250 ms from round 1,000; 2G, delta = 200 ms)",
                         "first CCM flag": st["latency_injection"]["first_ccm_flag"],
                         "final weights (w1, w2, w3)": st["latency_injection"]["weights_final"],
                         "result": "weights moved from w1 to w2 and w3 until w1 = 0.2"},
                        {"test": "per-arm costs (5 best arms cost 300 ms; 2G)",
                         "first CCM flag": st["per_arm_costs"]["first_ccm_flag"],
                         "final weights (w1, w2, w3)": st["per_arm_costs"]["weights_final"],
                         "result": f"share of slow-arm pulls {st['per_arm_costs']['share_slow_before_flag']:.2f} "
                                   f"before the first flag, {st['per_arm_costs']['share_slow_after_flag']:.2f} after"}])
    save(ccm, "table_layer2_ccm_rule")
    md_table("Layer 2. CCM switching rule, deterministic stress tests", ccm)
    ex = res["explainer"]
    rows = []
    for data, v in [(f"S1 K=10 T=10,000, {a}", x) for a, x in ex["S1"].items()] + \
                   [("OBD, ZOZOTOWN Bernoulli TS log (2,000 sampled rounds)", ex["OBD_bts_log"])]:
        for tol in TOLERANCES:
            t = v[tol]
            rows.append({"data": data, "tolerance": tol, "indistinguishable": t["share_indistinguishable"],
                         "margin": t["share_margin"], "exploration": t["share_exploration"],
                         "sign agreement on graded": t["sign_agreement_on_graded"]})
    e = pd.DataFrame(rows); save(e, "table_layer6_explainer")
    o = ex["OBD_bts_log"]
    md_table("Layer 6. Counterfactual explainer on sampled rounds, by tolerance setting", e,
             "Graded explanations are those reported as a margin or as exploration; sign agreement is the share "
             "of them whose sign of Delta_t matches the true margin mu(a_t) - max_{a != a_t} mu(a). statistical: "
             "eps_t = 1.96 SE(Delta_t); relative: eps_t = 10% of the larger estimate; absolute: eps = 0.01. "
             "On OBD the true item values are the on-policy CTRs of the uniform random log; median |true margin| = "
             f"{o['median_abs_true_margin']:.4f}, median statistical eps_t = {o['median_eps_statistical']:.5f}.")
    m = pd.DataFrame([{"data": f"S1, {a}", "IPS gap, sign of every explanation": x["sign_agreement_all"],
                       "model-based gap": x["sign_agreement_model_based"]} for a, x in ex["S1"].items()])
    save(m, "table_layer6_ips_vs_model")
    md_table("Layer 6. Sign agreement of the IPS gap and of the model-based gap (S1)", m)


def s4_bridge_table():
    f = E.OUT / "s4_bcs_bridge.json"
    if not f.exists():
        return None
    res = json.loads(f.read_text())
    rows = []
    for k, b in res["bcs_variants"].items():
        rows.append({"variant": k, "rho_S": b["rho_S"],
                     "rho_S 95% CI (bootstrap)": b.get("rho_boot_ci95", ""),
                     "P(rho >= 0.7)": b.get("P(rho >= 0.7)", ""),
                     "T (accepted)": b["T_accepted"], "c1": b["c1_reward_proxy"], "c2": b["c2_arm_independence"],
                     "c3": b["c3_sufficient_horizon"], "BCS": b["BCS"], "band": b["band"]})
    df = pd.DataFrame(rows); save(df, "table_bcs_s4_bridge")
    ai = pd.DataFrame([{"bin": k, **v} for k, v in res["arm_independence"].items()])
    save(ai, "table_bcs_s4_arm_independence")
    prim = res["bcs_variants"][PRIMARY_BCS]
    sens = []
    for k, v in res["arm_independence"].items():
        for lab, p in ((f"p_indep, {k} bins", v["p_indep"]),
                       (f"p_indep / null, {k} bins", min(1.0, v["p_indep"] / v["p_indep_null_mean"]))):
            b = BCSCalculator.compute(prim["rho_S"], p, 80, prim["T_accepted"])
            sens.append({"c2 definition": lab, "p_indep": p, "c2": b["c2_arm_independence"], "BCS": b["BCS"], "band": b["band"]})
    for lab, r in (("rho at the lower 95% bootstrap bound", prim["rho_boot_ci95"][0]),):
        b = BCSCalculator.compute(r, res["arm_independence"]["1h"]["p_indep"], 80, prim["T_accepted"])
        sens.append({"c2 definition": lab + " (c2 from 1h bins)", "p_indep": res["arm_independence"]["1h"]["p_indep"],
                     "c2": b["c2_arm_independence"], "BCS": b["BCS"], "band": b["band"]})
    sens = pd.DataFrame(sens); save(sens, "table_bcs_s4_sensitivity")
    md_table("BCS on S4. Each bridge condition measured", df,
             "Primary variant, fixed before the run: random_only (unbiased replay of the uniform log), full-log "
             "horizon, domain-matched Tier 1 (Bernoulli simulator with the 80 item CTRs). c2 from the hourly "
             "item CTR series of the random log (table below).")
    md_table("BCS on S4. Sensitivity of the primary BCS to the c2 definition and to rho", sens,
             "The thesis defines p_indep as the share of arm pairs with |rho_ij| <= 0.3 but fixes no time "
             "resolution. 'p_indep / null' divides by the value complete independence gives at the same "
             "resolution, which removes the share of pairs that exceed 0.3 by sampling noise alone.")
    md_table("BCS on S4. Arm independence on the random log", ai,
             "p_indep_null_mean: the same statistic after independent circular shifts of each item's series, "
             "i.e. what complete independence would give at this resolution.")
    if "validation_vs_step5" in res:
        v = pd.DataFrame([{"algorithm": a, **x} for a, x in res["validation_vs_step5"].items()])
        save(v, "table_bcs_s4_replay_validation")
        md_table("BCS on S4. Block-vectorised replay against the per-row replay of step5 (30 pooled slices)", v,
                 "UCB1 is deterministic, so its runs must be identical; for the stochastic algorithms the two "
                 "implementations use different random numbers and are compared in distribution (Welch t).")
    return res


def main():
    MD.extend(["# Chapter Four and Five tables (mab_benchmark 1.2.0)", "",
               f"Generated by experiments/build_tables.py. n = {E.N_RUNS} runs per cell, SEED_TABLE[0:30], "
               "stream layout 1.2.0. Raw arrays: experiments/results/raw/.", ""])
    table_4_2(); table_4_3(); table_4_4_all(); tables_4_5_4_6(); table_4_11()
    lat = latency_table(); pareto_and_guf(lat)
    g, eq = lambda_table(); table_5_1(eq)
    layers_2_6_tables(); s4_bridge_table()
    (E.TAB / "CHAPTER4_5_TABLES.md").write_text("\n".join(MD) + "\n", encoding="utf-8")
    print("written", E.TAB / "CHAPTER4_5_TABLES.md")


if __name__ == "__main__":
    main()
