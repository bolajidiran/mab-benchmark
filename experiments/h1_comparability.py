"""
h1_comparability.py: the test of Hypothesis H1 (Chapter Three, Section 3.2).

H1  Independent evaluations run under the unified protocol reach the same
    conclusion about a pair of MAB algorithms significantly more often than
    independent evaluations run under the fragmented practice documented in
    the SLR.   H0: C_U <= C_F.   H1: C_U > C_F (one-sided).

Pre-registered design (fixed before the first run; values in PRE below)
  Algorithms   B0 Random, B1 epsilon-Greedy (0.1), B2 UCB1, B3 TS, FG-TS (0.5):
               10 pairs. Problem family: S1 stationary Bernoulli (the setting
               every protocol can express), so disagreement comes from the
               protocol, not from the problem.
  Protocol U   every study: S1, K = 10, T = 10,000, all five algorithms, 30
               runs, verdict per pair from the two-sided Wilcoxon signed-rank
               test with Holm correction over the 10 pairs (alpha = 0.05):
               "A better", "B better" or "no significant difference".
  Protocol F   every study draws its own design in the proportions of the SLR:
               - primary metric, among the metrics a Bernoulli simulator can
                 compute, re-weighted from the 54 comparing papers (Section
                 2.7.1): pseudo-regret 21, average reward (= CTR) 13 + 5,
                 cumulative realised reward (throughput) 8; F1, nDCG and Pareto
                 regret cannot be emulated and are left out;
               - baselines, each included independently with its SLR
                 frequency: UCB1 28/62, TS 17/62, Random 17/62, eps-Greedy
                 11/62; FG-TS, the proposed algorithm, is always included;
               - a significance test with probability 4/54 (uncorrected
                 Wilcoxon, alpha = 0.05); otherwise "the higher mean wins";
               - K in {10, 50, 200} and T in {500, 2,000, 10,000}, uniformly
                 (the SLR matrix does not record K and T, so the benchmark
                 grid is used and this is disclosed);
               - number of runs uniform on {1, 5, 10} (only 4 of 62 papers
                 report it; sensitivity with 1, 5 and 10 fixed is reported).
  Ablations    F30  : F design with 30 runs, verdict by the higher mean.
               F30T : F design with 30 runs, Wilcoxon + Holm verdicts.
  Seeds        study s of protocol P uses SeedSequence([2026, P, s]); every run
               draws a fresh environment and fresh reward and algorithm streams.
               No study reuses SEED_TABLE, so U's agreement is not built in.
  Measure      C = share of study pairs that (a) both include the algorithm
               pair and (b) give the identical verdict; averaged over the 10
               pairs. Secondary: directional agreement (pairs where both
               studies give a direction), and accuracy against the truth.
  Truth        per (K, T) cell, 100 independent runs (seed block 2027): a pair
               whose true mean difference is below 0.2 pooled SD (Cohen's d <
               0.2) is "no difference", otherwise the lower-regret algorithm.
  Test         R = 100 studies per protocol. Delta = C_U - C_F. One-sided
               permutation test on study labels (10,000 permutations) and a
               95% bootstrap interval for Delta (2,000 resamples of studies
               within each protocol). Supporting: one-sided Wilcoxon over the
               10 algorithm pairs.
  Decision     H1 supported iff p < 0.05 AND the lower bound of the interval
               for Delta exceeds the minimum meaningful difference 0.10.

Outputs: results/h1_studies.csv, results/h1_comparability.json
Run: python h1_comparability.py --jobs 2      (about 2 h on 2 cores)
"""
from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

import exp_common as E
from mab_benchmark.baselines import EpsilonGreedy, RandomPolicy, ThompsonSampling, UCB1
from mab_benchmark.environments import BernoulliBandit
from FeelGoodThompsonSampling import FeelGoodThompsonSampling

ALG = {"B0_Random": (RandomPolicy, {}), "B1_EpsGreedy": (EpsilonGreedy, {"epsilon": 0.1}),
       "B2_UCB1": (UCB1, {}), "B3_TS": (ThompsonSampling, {"reward_type": "bernoulli"}),
       "FG-TS": (FeelGoodThompsonSampling, {"lam": 0.5})}
NAMES = list(ALG)
PAIRS = list(itertools.combinations(NAMES, 2))
PRE = {"metric_weights": {"pseudo_regret": 21, "average_reward": 18, "cumulative_reward": 8},
       "baseline_inclusion": {"B2_UCB1": 28 / 62, "B3_TS": 17 / 62, "B0_Random": 17 / 62, "B1_EpsGreedy": 11 / 62},
       "p_significance_test": 4 / 54, "K": [10, 50, 200], "T": [500, 2_000, 10_000], "runs_F": [1, 5, 10],
       "alpha": 0.05, "R": 100, "n_perm": 10_000, "n_boot": 2_000, "min_meaningful_delta": 0.10,
       "truth_runs": 100, "truth_negligible_d": 0.2}
PCODE = {"U": 1, "F": 2, "F30": 3, "F30T": 4, "F1": 5, "F5": 6, "F10": 7}


# --------------------------------------------------------------------------- one run
def one_run(name, K, T, seed):
    """Pseudo-regret and realised cumulative reward of one run (fresh instance)."""
    cls, kw = ALG[name]
    env = BernoulliBandit(K, T, seed=int(seed))
    rs, as_ = np.random.SeedSequence([int(seed), 1]).spawn(2)
    rng = np.random.default_rng(rs); a_seed = int(as_.generate_state(1, dtype=np.uint32)[0])
    np.random.seed(a_seed)
    try:
        alg = cls(K, seed=a_seed, **kw)
    except TypeError:
        alg = cls(K, **kw)
    reg = 0.0; rew = 0.0
    for t in range(1, T + 1):
        a = alg.choose_arm(t)
        r = env.pull(a, rng); alg.update(a, r, t)
        reg += env.regret_increment(a); rew += r
    return reg, rew


def run_seeds(proto, s, n):
    ss = np.random.SeedSequence([2026, PCODE[proto], s])
    return [int(x) for x in ss.generate_state(n, dtype=np.uint32)]


# --------------------------------------------------------------------------- verdicts
def holm_verdicts(scores, algs, test_alpha, correct=True):
    """scores: {alg: array (higher is better)}. Wilcoxon per pair; Holm if correct."""
    out, pv = {}, {}
    for a, b in itertools.combinations(algs, 2):
        d = scores[a] - scores[b]
        p = 1.0 if np.allclose(d, 0) or d.size < 6 else float(stats.wilcoxon(scores[a], scores[b]).pvalue)
        pv[(a, b)] = p
    keys = sorted(pv, key=pv.get); m = len(keys); sig = {}
    stop = False
    for j, k in enumerate(keys):
        thr = test_alpha / (m - j) if correct else test_alpha
        if correct:
            if stop or pv[k] > thr:
                stop = True; sig[k] = False
            else:
                sig[k] = True
        else:
            sig[k] = pv[k] <= thr
    for (a, b) in pv:
        if sig[(a, b)]:
            out[(a, b)] = a if scores[a].mean() > scores[b].mean() else b
        else:
            out[(a, b)] = "none"
    return out


def mean_verdicts(scores, algs):
    out = {}
    for a, b in itertools.combinations(algs, 2):
        ma, mb = scores[a].mean(), scores[b].mean()
        out[(a, b)] = "none" if ma == mb else (a if ma > mb else b)
    return out


# --------------------------------------------------------------------------- one study
def study(args):
    proto, s = args
    rng = np.random.default_rng([2026, PCODE[proto], s, 99])                 # design draws
    if proto == "U":
        K, T, n, metric = 10, 10_000, 30, "pseudo_regret"
        algs = list(NAMES); test, correct = True, True
    else:
        K = int(rng.choice(PRE["K"])); T = int(rng.choice(PRE["T"]))
        w = PRE["metric_weights"]; metric = str(rng.choice(list(w), p=np.array(list(w.values())) / sum(w.values())))
        algs = ["FG-TS"] + [a for a, p in PRE["baseline_inclusion"].items() if rng.random() < p]
        if len(algs) == 1:                                                    # a paper compares against >= 1 baseline
            algs.append(str(rng.choice(list(PRE["baseline_inclusion"]),
                                       p=np.array(list(PRE["baseline_inclusion"].values()))
                                       / sum(PRE["baseline_inclusion"].values()))))
        if proto == "F":
            n = int(rng.choice(PRE["runs_F"])); test = rng.random() < PRE["p_significance_test"]; correct = False
        elif proto in ("F1", "F5", "F10"):
            n = int(proto[1:]); test = rng.random() < PRE["p_significance_test"]; correct = False
        elif proto == "F30":
            n, test, correct = 30, False, False
        else:                                                                 # F30T
            n, test, correct = 30, True, True
    seeds = run_seeds(proto, s, n)
    scores = {}
    for a in algs:
        reg, rew = zip(*[one_run(a, K, T, sd) for sd in seeds])
        reg, rew = np.array(reg), np.array(rew)
        scores[a] = {"pseudo_regret": -reg, "average_reward": rew / T, "cumulative_reward": rew}[metric]
    v = holm_verdicts(scores, algs, PRE["alpha"], correct) if test else mean_verdicts(scores, algs)
    row = {"protocol": proto, "study": s, "K": K, "T": T, "n_runs": n, "metric": metric,
           "algorithms": "|".join(algs), "test": bool(test), "holm": bool(correct and test)}
    row.update(verdict_columns(v))
    return row


def verdict_columns(v):
    """Map verdicts keyed in a study's own algorithm order onto the canonical PAIRS
    columns. The value is the winner's name or "none", so it does not depend on the
    orientation of the key; "" means the study did not compare the pair."""
    return {f"{a}~{b}": v.get((a, b), v.get((b, a), "")) for a, b in PAIRS}


# --------------------------------------------------------------------------- truth
def truth_job(args):
    K, T, name, i = args
    seed = int(np.random.SeedSequence([2027, K, T, i]).generate_state(1, dtype=np.uint32)[0])
    return K, T, name, one_run(name, K, T, seed)[0]


def truth_verdicts(tr):
    out = {}
    for (K, T), g in tr.groupby(["K", "T"]):
        reg = {a: g[g.alg == a].regret.to_numpy() for a in NAMES}
        for a, b in PAIRS:
            sp = np.sqrt((reg[a].var(ddof=1) + reg[b].var(ddof=1)) / 2)
            d = (reg[a].mean() - reg[b].mean()) / sp if sp > 0 else 0.0
            out[(K, T, a, b)] = "none" if abs(d) < PRE["truth_negligible_d"] else (a if d < 0 else b)
    return out


# --------------------------------------------------------------------------- comparability
def pair_agreement(df, directional=False):
    """Per algorithm pair: share of study pairs that both compare it and agree."""
    res = {}
    for a, b in PAIRS:
        col = df[f"{a}~{b}"].fillna("").to_numpy()
        n = col.size; tot = n * (n - 1) / 2; agree = 0
        vals, cnt = np.unique(col[col != ""], return_counts=True)
        for v, c in zip(vals, cnt):
            if directional and v == "none":
                continue
            agree += c * (c - 1) / 2
        if directional:
            k = int(np.sum((col != "") & (col != "none")))
            tot = k * (k - 1) / 2
        res[(a, b)] = agree / tot if tot else np.nan
    return res


def C(df, directional=False):
    return float(np.nanmean(list(pair_agreement(df, directional).values())))


def accuracy(df, truth):
    hits = tot = 0
    for _, r in df.iterrows():
        for a, b in PAIRS:
            v = r[f"{a}~{b}"]
            if isinstance(v, str) and v:
                tot += 1; hits += v == truth[(int(r["K"]), int(r["T"]), a, b)]   # r.T would be the transpose
    return hits / tot if tot else np.nan


def perm_test(dU, dF, n_perm, seed=0):
    obs = C(dU) - C(dF)
    pool = pd.concat([dU, dF], ignore_index=True); nU = len(dU)
    rng = np.random.default_rng(seed); ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pool))
        ge += (C(pool.iloc[idx[:nU]]) - C(pool.iloc[idx[nU:]])) >= obs
    return obs, (ge + 1) / (n_perm + 1)


def boot_ci(dU, dF, n_boot, seed=1):
    rng = np.random.default_rng(seed); out = np.empty(n_boot)
    for k in range(n_boot):
        out[k] = C(dU.iloc[rng.integers(0, len(dU), len(dU))]) - C(dF.iloc[rng.integers(0, len(dF), len(dF))])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--R", type=int, default=PRE["R"])
    ap.add_argument("--protocols", default="U,F,F30,F30T,F1,F5,F10")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--n-perm", type=int, default=PRE["n_perm"])
    args = ap.parse_args()
    t0 = time.time()
    path = E.OUT / "h1_studies.csv"
    if not args.analyse_only:
        done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["protocol", "study"])
        have = set(zip(done.protocol, done.study))
        jobs = [(p, s) for p in args.protocols.split(",") for s in range(args.R) if (p, s) not in have]
        rows = []
        with ProcessPoolExecutor(args.jobs) as ex:
            for k, r in enumerate(ex.map(study, jobs, chunksize=1)):
                rows.append(r)
                if (k + 1) % 20 == 0:
                    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
                    print(k + 1, "of", len(jobs), round(time.time() - t0), flush=True)
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
        tpath = E.OUT / "h1_truth.csv"
        if not tpath.exists():
            tj = [(K, T, a, i) for K in PRE["K"] for T in PRE["T"] for a in NAMES for i in range(PRE["truth_runs"])]
            with ProcessPoolExecutor(args.jobs) as ex:
                tr = pd.DataFrame(list(ex.map(truth_job, tj, chunksize=20)), columns=["K", "T", "alg", "regret"])
            tr.to_csv(tpath, index=False)
            print("truth done", round(time.time() - t0), flush=True)

    df = pd.read_csv(path).fillna({f"{a}~{b}": "" for a, b in PAIRS})
    for a, b in PAIRS:
        df[f"{a}~{b}"] = df[f"{a}~{b}"].astype(str).replace("nan", "")
    truth = truth_verdicts(pd.read_csv(E.OUT / "h1_truth.csv"))
    dU = df[df.protocol == "U"].reset_index(drop=True)
    out = {"pre_registration": PRE, "protocols": {}}
    for p, g in df.groupby("protocol"):
        g = g.reset_index(drop=True)
        out["protocols"][p] = {"studies": int(len(g)), "C": C(g), "C_directional": C(g, True),
                               "accuracy_vs_truth": accuracy(g, truth),
                               "per_pair_C": {f"{a}~{b}": v for (a, b), v in pair_agreement(g).items()},
                               "share_pairs_compared": float(np.mean([(g[f"{a}~{b}"] != "").mean() for a, b in PAIRS]))}
    tests = {}
    for p in [x for x in out["protocols"] if x != "U"]:
        dF = df[df.protocol == p].reset_index(drop=True)
        obs, pval = perm_test(dU, dF, args.n_perm)
        lo, hi = boot_ci(dU, dF, PRE["n_boot"])
        cu = pair_agreement(dU); cf = pair_agreement(dF)
        x = np.array([cu[k] for k in PAIRS]); y = np.array([np.nan_to_num(cf[k]) for k in PAIRS])
        w = stats.wilcoxon(x, y, alternative="greater")
        tests[p] = {"delta": obs, "p_perm_one_sided": pval, "ci95": [lo, hi],
                    "wilcoxon_pairs_p": float(w.pvalue),
                    "H1_supported": bool(pval < 0.05 and lo > PRE["min_meaningful_delta"])}
    out["tests_U_vs"] = tests
    out["seconds"] = round(time.time() - t0, 1)
    E.write_json(out, E.OUT / "h1_comparability.json")
    print({p: (round(v["delta"], 3), v["p_perm_one_sided"], [round(z, 3) for z in v["ci95"]], v["H1_supported"])
           for p, v in tests.items()})


if __name__ == "__main__":
    main()
