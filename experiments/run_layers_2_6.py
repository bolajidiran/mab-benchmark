"""
run_layers_2_6.py: empirical exercise of the Meta-Policy Orchestrator (Layer 2)
and the counterfactual explainer (Layer 6), Steps 3 and 5 of the thesis'
experimental protocol (Chapter Three, "The Experimental Phase").

Part A (Step 3). S3 (ten change-points) and S1 K = 10, T = 10,000 (none), 30
    runs, the five algorithms of the case study each wrapped in the MPO. Logs
    every change-point flag and every posterior reset, scores flags against the
    known change-points, and compares regret with and without the MPO on
    identical random streams (paired Wilcoxon).
Part B. The CCM rule: a deterministic stress test with injected latency and
    per-arm costs (2G regime), since S1 to S5 have no per-arm costs.
Part C (Step 5). The explainer on sampled rounds of S1 K = 10, T = 10,000
    (B3 TS, UCB1, epsilon-Greedy; propensities from choice_propensity) and on
    the real Bernoulli TS log of the Open Bandit Dataset (logged pscore), with
    the true arm values used to check whether each explanation is correct.

Outputs: results/layers26_*.csv / .json, read by build_tables.py and
make_figures.py. Run: python run_layers_2_6.py --jobs 2
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

import exp_common as E
from mab_benchmark.runner import SEED_TABLE, run_streams
from mab_benchmark.orchestration import MetaPolicyOrchestrator
from mab_benchmark.explain import CounterfactualExplainer, choice_propensity, TOLERANCES
from mab_benchmark.environments import BernoulliBandit, PiecewiseStationaryBandit

FIVE = ["FG-TS", "B3_TS", "B2_UCB1", "B1_EpsGreedy", "B0_Random"]
CELL_ENV = {"S3_K10_T10000": (PiecewiseStationaryBandit, False), "S1_K10_T10000": (BernoulliBandit, True)}
DETECT_WINDOW = 500           # a flag within 500 rounds after a change-point counts as a detection


def make(name, K, seed):
    cls, kw = E.ALGS[name]
    kw = dict(kw)
    if name == "B0_Random":
        kw["seed"] = seed
    return cls(n_arms=K, **kw)


def run_one(job):
    cell, name, i, wrapped = job
    EnvCls, stationary = CELL_ENV[cell]
    seed = SEED_TABLE[i]
    rng, alg_seed = run_streams(seed)
    env = EnvCls(K=10, T=10_000, seed=seed)
    np.random.seed(alg_seed)
    base = make(name, env.K, alg_seed)
    alg = MetaPolicyOrchestrator(base, measure_latency=False) if wrapped else base
    reg = 0.0
    pulls = np.zeros(env.T, np.int16)
    for t in range(1, env.T + 1):
        a = alg.choose_arm(t)
        r = env.pull(a, rng) if stationary else env.pull(a, t, rng)
        reg += env.regret_increment(a) if stationary else env.regret_increment(a, t)
        alg.update(a, r, t)
        pulls[t - 1] = a
    out = {"cell": cell, "algorithm": name, "run": i + 1, "seed": seed, "mpo": wrapped, "regret": reg}
    if wrapped:
        out["flags"] = alg.log["change_point_flags"]
        out["resets"] = alg.log["resets"]
        if not stationary:
            out["cp_info"] = cp_scoring(env, pulls, alg.log["change_point_flags"])
    return out


def cp_scoring(env, pulls, flags):
    """For each change-point: did the acting arm's mean drop, and was it flagged in time?"""
    rows = []
    flags = np.asarray(flags)
    for j, cp in enumerate(env.CHANGE_POINTS):
        before = pulls[max(0, cp - 200):cp]
        arm = int(np.bincount(before, minlength=env.K).argmax())
        drop = float(env.segment_means[j + 1, arm] - env.segment_means[j, arm])
        nxt = env.CHANGE_POINTS[j + 1] if j + 1 < len(env.CHANGE_POINTS) else env.T
        hit = flags[(flags > cp) & (flags <= min(cp + DETECT_WINDOW, nxt))]
        rows.append({"cp": cp, "acting_arm": arm, "mean_change_of_acting_arm": drop,
                     "detected": bool(hit.size), "delay": int(hit[0] - cp) if hit.size else None})
    windows = [(cp, cp + DETECT_WINDOW) for cp in env.CHANGE_POINTS]
    false = [int(f) for f in flags if not any(lo < f <= hi for lo, hi in windows)]
    return {"change_points": rows, "false_alarms": false}


# --------------------------------------------------------------------------- Part B
def ccm_stress():
    from mab_benchmark.baselines import ThompsonSampling
    res = {}
    # B1: injected device latency of 250 ms from round 1,000 (2G budget delta = 200 ms)
    seed = SEED_TABLE[0]; rng, s = run_streams(seed); np.random.seed(s)
    env = BernoulliBandit(K=10, T=2_000, seed=seed)
    mpo = MetaPolicyOrchestrator(ThompsonSampling(10), regime="2G", measure_latency=False,
                                 latency_fn=lambda t: 250.0 if t > 1_000 else 20.0)
    for t in range(1, env.T + 1):
        a = mpo.choose_arm(t); mpo.update(a, env.pull(a, rng), t)
    res["latency_injection"] = {"first_ccm_flag": mpo.log["ccm_flags"][:1],
                                "weights_final": mpo.weights, "n_weight_changes": len(mpo.log["weight_changes"]),
                                "ccm_last_window": mpo.monitor.ccm(100), "monitor": mpo.monitor.summary()}
    # B2: per-arm costs; arms 5 to 9 cost 300 ms (over the 2G budget) and the best arm is among them
    rng, s = run_streams(seed); np.random.seed(s)
    env = BernoulliBandit(K=10, T=2_000, seed=seed)
    best = int(np.argmax(env.means)) if hasattr(env, "means") else None
    slow = sorted(int(a) for a in np.argsort(env.means)[-5:])      # the five best arms are the slow ones
    cost = np.zeros((10, 2)); cost[:, 0] = 50.0; cost[slow, 0] = 300.0
    mpo = MetaPolicyOrchestrator(ThompsonSampling(10), regime="2G", measure_latency=False, arm_cost=cost)
    picks = np.zeros(env.T, int)
    for t in range(1, env.T + 1):
        a = mpo.choose_arm(t)
        mpo.update(a, env.pull(a, rng), t)                # records the chosen arm's latency cost
        picks[t - 1] = a
    first = mpo.log["ccm_flags"][0] if mpo.log["ccm_flags"] else None
    after = picks[first:] if first else picks[:0]
    res["per_arm_costs"] = {"slow_arms": slow, "best_arm": best, "first_ccm_flag": first,
                            "share_slow_before_flag": float(np.isin(picks[:first or env.T], slow).mean()),
                            "share_slow_after_flag": float(np.isin(after, slow).mean()) if after.size else None,
                            "weights_final": mpo.weights}
    return res


# --------------------------------------------------------------------------- Part C
SAMPLE_T = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]


def explain_s1(job):
    name, i = job
    seed = SEED_TABLE[i]; rng, s = run_streams(seed); np.random.seed(s)
    env = BernoulliBandit(K=10, T=10_000, seed=seed)
    alg = make(name, 10, s)
    ex = CounterfactualExplainer(10)
    prng = np.random.default_rng([seed, 99])
    mu = np.array([env.means[a] for a in range(10)]) if hasattr(env, "means") else None
    rows = []
    for t in range(1, env.T + 1):
        a = alg.choose_arm(t)
        p = choice_propensity(alg, t, n_mc=200, rng=prng)[a]
        if t in SAMPLE_T:
            kinds = {}
            for tol in TOLERANCES:
                ex.tolerance = tol; kinds["kind_" + tol] = ex.explain(t, a).kind
            ex.tolerance = "statistical"
            e = ex.explain(t, a)
            true = float(mu[a] - np.max(np.delete(mu, a)))
            est = (alg.alpha / (alg.alpha + alg.beta)) if hasattr(alg, "alpha") else alg.estimates
            dm = float(est[a] - np.max(np.delete(est, a)))          # model-based gap, for comparison
            rows.append({"algorithm": name, "run": i + 1, "t": t, "delta": e.delta, "kind": e.kind, **kinds, "dm_delta": dm,
                         "true_margin": true, "best_alt_is_true_best_alt": int(e.best_alternative == int(
                             np.delete(np.arange(10), a)[np.argmax(np.delete(mu, a))]))})
        r = env.pull(a, rng)
        alg.update(a, r, t)
        ex.observe(a, r, max(p, 1e-12))
    return rows


def explain_obd(n_sample=2_000):
    """Explain the logged choices of the ZOZOTOWN Bernoulli TS policy, from its own log."""
    cache = E.REPO / "obd_preprocessing" / "cache"
    b = np.load(cache / "clean_bts.npz"); r = np.load(cache / "clean_random.npz")
    K = 80
    # true item values: on-policy CTR of each item in the uniform random log (unbiased)
    imp = np.bincount(r["item"], minlength=K); clk = np.bincount(r["item"], weights=r["click"], minlength=K)
    true = clk / np.maximum(imp, 1)
    order = np.argsort(b["ts"], kind="stable")
    item, click, ps = b["item"][order].astype(int), b["click"][order].astype(float), b["pscore"][order]
    sample = set(np.unique(np.linspace(10_000, item.size - 1, n_sample).astype(int)).tolist())
    ex = CounterfactualExplainer(K)
    rows = []
    for t in range(item.size):
        if t in sample:
            kinds = {}
            for tol in TOLERANCES:
                ex.tolerance = tol; kinds["kind_" + tol] = ex.explain(t, item[t]).kind
            ex.tolerance = "statistical"
            e = ex.explain(t, item[t])
            rows.append({"t": t, "arm": int(item[t]), "delta": e.delta, "kind": e.kind, **kinds, "eps": e.eps,
                         "true_margin": float(true[item[t]] - np.max(np.delete(true, item[t])))})
        ex.observe(item[t], click[t], ps[t])
    return rows, {"items": K, "rows": int(item.size), "true_ctr_range": [float(true.min()), float(true.max())]}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(c, a, i, w) for c in CELL_ENV for a in FIVE for i in range(E.N_RUNS) for w in (False, True)]
    with ProcessPoolExecutor(args.jobs) as ex:
        out = list(ex.map(run_one, jobs, chunksize=4))
    runs = pd.DataFrame([{k: v for k, v in o.items() if k not in ("flags", "resets", "cp_info")} |
                         {"n_flags": len(o.get("flags", [])), "n_resets": len(o.get("resets", []))}
                         for o in out])
    runs.to_csv(E.OUT / "layers26_mpo_runs.csv", index=False)
    cps = []
    for o in out:
        if o["mpo"] and "cp_info" in o:
            for r in o["cp_info"]["change_points"]:
                cps.append({"algorithm": o["algorithm"], "run": o["run"], **r})
    cps = pd.DataFrame(cps); cps.to_csv(E.OUT / "layers26_mpo_changepoints.csv", index=False)
    false = {(o["algorithm"], o["run"]): len(o["cp_info"]["false_alarms"]) for o in out
             if o["mpo"] and "cp_info" in o}

    summary = []
    for c in CELL_ENV:
        for a in FIVE:
            p = runs[(runs.cell == c) & (runs.algorithm == a) & (~runs.mpo)].sort_values("run").regret.to_numpy()
            m = runs[(runs.cell == c) & (runs.algorithm == a) & (runs.mpo)].sort_values("run")
            q = m.regret.to_numpy()
            w = stats.wilcoxon(q, p) if not np.allclose(p, q) else None
            row = {"cell": c, "algorithm": a, "regret_plain": p.mean(), "regret_mpo": q.mean(),
                   "change_pct": 100 * (q.mean() - p.mean()) / p.mean(),
                   "wilcoxon_p": None if w is None else float(w.pvalue),
                   "flags_per_run": m.n_flags.mean(), "resets_per_run": m.n_resets.mean(),
                   "reset_rule_ok": bool(((m.n_resets == m.n_flags) if a in ("B3_TS", "FG-TS")
                                          else (m.n_resets == 0)).all())}
            if c.startswith("S3"):
                s = cps[cps.algorithm == a]
                dropped = s[s.mean_change_of_acting_arm < -0.1]
                row |= {"detected_all_cp": s.detected.mean(),
                        "detected_when_acting_arm_dropped_by_0.1+": dropped.detected.mean() if len(dropped) else None,
                        "cp_with_drop_0.1+": len(dropped) / max(len(s), 1),
                        "median_delay": float(s.delay.dropna().median()) if s.delay.notna().any() else None,
                        "false_alarms_per_run": float(np.mean([false[(a, r)] for r in range(1, E.N_RUNS + 1)]))}
            summary.append(row)
    summary = pd.DataFrame(summary); summary.to_csv(E.OUT / "layers26_mpo_summary.csv", index=False)
    print(summary.round(3).to_string())

    stress = ccm_stress()
    jobs = [(a, i) for a in ("B3_TS", "B2_UCB1", "B1_EpsGreedy") for i in range(E.N_RUNS)]
    with ProcessPoolExecutor(args.jobs) as ex:
        s1 = [r for rows in ex.map(explain_s1, jobs) for r in rows]
    s1 = pd.DataFrame(s1); s1.to_csv(E.OUT / "layers26_explainer_s1.csv", index=False)
    obd_rows, obd_meta = explain_obd()
    obd = pd.DataFrame(obd_rows); obd.to_csv(E.OUT / "layers26_explainer_obd.csv", index=False)

    def by_tol(df, extra):
        out = {}
        for tol in TOLERANCES:
            k = df["kind_" + tol]; graded = k != "indistinguishable"
            ok = np.sign(df.delta[graded]) == np.sign(df.true_margin[graded])
            out[tol] = {"share_indistinguishable": float((k == "indistinguishable").mean()),
                        "share_margin": float((k == "margin").mean()),
                        "share_exploration": float((k == "exploration").mean()),
                        "sign_agreement_on_graded": float(ok.mean()) if graded.any() else None}
        out["sign_agreement_all"] = float((np.sign(df.delta) == np.sign(df.true_margin)).mean())
        out["median_abs_delta"] = float(df.delta.abs().median())
        return out | extra
    exp_sum = {"S1": {a: by_tol(g, {"sign_agreement_model_based": float((np.sign(g.dm_delta) == np.sign(g.true_margin)).mean()),
                                    "best_alternative_correct": float(g.best_alt_is_true_best_alt.mean())})
                      for a, g in s1.groupby("algorithm")},
               "OBD_bts_log": by_tol(obd, {"median_abs_true_margin": float(obd.true_margin.abs().median()),
                                           "median_eps_statistical": float(obd.eps.median()), **obd_meta})}
    E.write_json({"provenance": E.provenance(), "detect_window": DETECT_WINDOW, "mpo_window": 100,
                  "mpo_summary": summary.to_dict("records"), "ccm_stress": stress, "explainer": exp_sum,
                  "seconds": round(time.time() - t0, 1)}, E.OUT / "layers26_results.json")
    print(stress); print(exp_sum)


if __name__ == "__main__":
    main()
