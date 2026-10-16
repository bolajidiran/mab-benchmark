"""
s4_bcs_bridge.py: the Bridge Compliance Score on S4, with every bridge
condition measured on the Open Bandit Dataset.

What this script measures
  M1  Horizon. Each run replays the FULL log (Li et al., 2011, the replayer
      protocol uses the whole log); runs differ in the algorithm's random stream
      and the acceptance coins, from SEED_TABLE[0:30]. Two streams:
        random_only : the uniform-random log only (exact, unbiased replay);
        pooled      : random + Bernoulli TS logs (truncated rejection sampling).
      A block-vectorised replay gives the same process as the per-row replay
      of step5 (validated below: UCB1 decision-for-decision, the others in
      distribution on the 30 slices).
  M2  Arm independence. p_indep = share of item pairs with |Spearman rho| <= 0.3
      between hourly CTR series in the random log, with a permutation null.
  M3  Domain-matched Tier 1. Tier 1 regret on a Bernoulli simulator with K = 80
      and the 80 item CTRs of the random log as arm means, horizon equal to the
      Tier 2 horizon. Compared with the generic S1 ranking (S1, K = 10, T = 10,000).
  c1 is reported with a bootstrap interval over runs (2,000 resamples).

Algorithms: B0 to B3, FG-TS (lambda = 0.5) and Bernoulli TS with the ZOZOTOWN
priors (the six context-free algorithms that run in both tiers).

Outputs: results/s4_bcs_bridge.json, results/s4_bcs_bridge_runs.csv,
         results/s4_bcs_validation.csv
Run: python s4_bcs_bridge.py --jobs 2      (about 1 h on 2 cores)
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

import exp_common as E

sys.path.insert(0, str(E.REPO / "obd_preprocessing"))
import common as C                                               # noqa: E402
from mab_benchmark.runner import SEED_TABLE, run_streams         # noqa: E402
from mab_benchmark.analysis.bcs import BCSCalculator             # noqa: E402
from mab_benchmark.baselines import RandomPolicy, EpsilonGreedy, UCB1, ThompsonSampling  # noqa: E402

K = C.K
ALGS6 = ["B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS", "FG-TS", "BernoulliTS_Zozotown"]
_S1 = ["B3_TS", "FG-TS", "B2_UCB1", "B1_EpsGreedy", "B0_Random"]
S1_RANK5 = sorted(_S1, key=lambda a: np.load(E.RAW / f"S1_K10_T10000__{a}.npy").mean())   # S1 regret ranking
_G = {}


# --------------------------------------------------------------------------- data
def _load(stream):
    s = np.load(C.CACHE_DIR / "stream_pooled.npz")
    keep = np.flatnonzero(s["source"] == 0) if stream == "random_only" else slice(None)
    item = s["item"][keep].astype(np.int64); click = s["click"][keep].astype(float)
    ps = s["pscore"][keep]
    return item, click, ps


def _init(stream):
    _G["stream"] = stream
    _G["item"], _G["click"], _G["ps"] = _load(stream)
    _G["acc_p"] = np.minimum(1.0, C.ACCEPT_FLOOR / _G["ps"])
    pri = __import__("json").loads(C.ZOZOTOWN_PRIOR_FILE.read_text())
    _G["prior"] = (np.array(pri["alpha"], float), np.array(pri["beta"], float))
    if stream == "pooled":
        sl = np.load(C.CACHE_DIR / "slices.npz"); _G["starts"], _G["ends"] = sl["starts"], sl["ends"]


# --------------------------------------------------------------------------- block replay
class _State:
    def __init__(self, name, rng, prior):
        self.name, self.rng = name, rng
        if name == "BernoulliTS_Zozotown":
            self.a, self.b = prior[0].copy(), prior[1].copy()
        else:
            self.a, self.b = np.ones(K), np.ones(K)
        self.n = np.zeros(K); self.est = np.zeros(K)

    def choose(self, B, t):
        nm, rng = self.name, self.rng
        if nm == "B0_Random":
            return rng.integers(0, K, B)
        if nm == "B1_EpsGreedy":
            ch = np.full(B, int(np.argmax(self.est)))
            ex = rng.random(B) < 0.1
            ch[ex] = rng.integers(0, K, int(ex.sum()))
            return ch
        if nm == "B2_UCB1":
            z = np.flatnonzero(self.n == 0)
            if z.size:
                return np.full(B, int(z[0]))
            return np.full(B, int(np.argmax(self.est + np.sqrt(2.0 * np.log(t) / self.n))))
        th = rng.beta(self.a, self.b, size=(B, K))
        if nm == "FG-TS":
            th = th + 0.5 * np.sqrt(np.log(t + 1) / (self.n + 1))
        return th.argmax(1)

    def update(self, arm, r):
        self.a[arm] += r; self.b[arm] += 1 - r
        self.n[arm] += 1; self.est[arm] += (r - self.est[arm]) / self.n[arm]


def block_replay(name, seed, lo, hi, u, block=256):
    """Replay rows lo..hi. A row can only be accepted when its acceptance coin
    passes (u_t < min(1, c / pscore_t)); on every other row the algorithm's
    choice has no effect, so only the rows whose coin passes are visited. Each
    visited row still gets its own independent choice, exactly as in step5."""
    ok = np.flatnonzero(u < _G["acc_p"][lo:hi]) + lo
    item, click = _G["item"][ok], _G["click"][ok]
    st = _State(name, np.random.default_rng([seed, 11, ALGS6.index(name)]), _G["prior"])
    n_acc, clk, pos, N = 0, 0.0, 0, ok.size
    while pos < N:
        end = min(pos + block, N)
        ch = st.choose(end - pos, n_acc + 1)
        hit = np.flatnonzero(ch == item[pos:end])
        if not hit.size:
            pos = end; continue
        j = pos + int(hit[0]); r = click[j]
        st.update(int(ch[hit[0]]), r); n_acc += 1; clk += r; pos = j + 1
    return n_acc, clk


def job_full(args):
    name, i = args
    seed = SEED_TABLE[i]; N = _G["item"].size
    u = np.random.default_rng([seed, 7]).random(N)
    t0 = time.time(); n, c = block_replay(name, seed, 0, N, u)
    return {"stream": _G["stream"], "run": i + 1, "seed": seed, "algorithm": name, "rows": N,
            "accepted": n, "clicks": c, "ctr": c / n if n else np.nan, "seconds": round(time.time() - t0, 1)}


def job_slice(args):
    name, i = args
    seed = SEED_TABLE[i]; a, b = int(_G["starts"][i]), int(_G["ends"][i])
    u = np.random.default_rng([seed, 7]).random(b - a)          # same coins as step5
    n, c = block_replay(name, seed, a, b, u)
    return {"run": i + 1, "algorithm": name, "accepted": n, "clicks": c, "ctr": c / n if n else np.nan}


# --------------------------------------------------------------------------- Tier 1 (domain-matched simulator)
class _Calibrated:
    def __init__(self, means, T):
        self.means, self.T, self.K = means, T, means.size
        self.mu_star = means.max()


def _tier1_alg(name, seed):
    return {"B0_Random": lambda: RandomPolicy(K, seed=seed), "B1_EpsGreedy": lambda: EpsilonGreedy(K, epsilon=0.1),
            "B2_UCB1": lambda: UCB1(K), "B3_TS": lambda: ThompsonSampling(K, reward_type="bernoulli"),
            "FG-TS": lambda: C.FeelGoodTS(K, lambda_=0.5),
            "BernoulliTS_Zozotown": lambda: C.BernoulliTSZozotown(K)}[name]()


def job_tier1(args):
    name, i, T, means = args
    seed = SEED_TABLE[i]; rng, a_seed = run_streams(seed)
    np.random.seed(a_seed)
    alg = _tier1_alg(name, a_seed)
    u = rng.random(T); reg = 0.0; mstar = means.max()
    for t in range(1, T + 1):
        a = alg.choose_arm(t)
        alg.update(a, float(u[t - 1] < means[a]), t)
        reg += mstar - means[a]
    return {"T": T, "run": i + 1, "algorithm": name, "regret": reg}


# --------------------------------------------------------------------------- M2
def arm_independence():
    r = np.load(C.CACHE_DIR / "clean_random.npz")
    ts, item, click = r["ts"], r["item"].astype(int), r["click"].astype(float)
    out = {}
    rng = np.random.default_rng(2024)
    for hours in (1, 3, 6):
        b = ((ts - ts.min()) // (hours * 3_600 * 10**9)).astype(int); nb = b.max() + 1
        imp = np.zeros((K, nb)); clk = np.zeros((K, nb))
        np.add.at(imp, (item, b), 1); np.add.at(clk, (item, b), click)
        ctr = clk / np.maximum(imp, 1)
        rho = stats.spearmanr(ctr.T).statistic
        iu = np.triu_indices(K, 1); obs = np.abs(rho[iu])
        null = []
        for _ in range(20):                                       # independent circular shifts per item
            sh = np.stack([np.roll(ctr[k], rng.integers(nb)) for k in range(K)])
            null.append((np.abs(stats.spearmanr(sh.T).statistic[iu]) <= 0.3).mean())
        out[f"{hours}h"] = {"bins": int(nb), "mean_impressions_per_item_bin": float(imp.mean()),
                            "mean_clicks_per_item_bin": float(clk.mean()),
                            "p_indep": float((obs <= 0.3).mean()), "median_abs_rho": float(np.median(obs)),
                            "p_indep_null_mean": float(np.mean(null)),
                            "share_pairs_rho_gt_0.3_positive": float((rho[iu] > 0.3).mean())}
    return out


# --------------------------------------------------------------------------- statistics
def spearman_rank(a_names, b_names):
    common = [n for n in a_names if n in b_names]
    return float(stats.spearmanr([a_names.index(n) for n in common], [b_names.index(n) for n in common]).statistic)


def boot_rho(t1: dict, t2: dict, B=2000, seed=0):
    """t1: {alg: regrets (lower better)}, t2: {alg: ctr (higher better)}; resample runs in each tier."""
    rng = np.random.default_rng(seed); names = list(t1)
    out = np.empty(B)
    for k in range(B):
        m1 = [t1[a][rng.integers(0, t1[a].size, t1[a].size)].mean() for a in names]
        m2 = [-t2[a][rng.integers(0, t2[a].size, t2[a].size)].mean() for a in names]
        out[k] = stats.spearmanr(m1, m2).statistic
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    t0 = time.time(); res = {"provenance": E.provenance()}

    # ---- validation of the block replay against step5 (pooled slices) ----
    if not args.skip_validation:
        with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=("pooled",)) as ex:
            val = pd.DataFrame(list(ex.map(job_slice, [(a, i) for a in ALGS6 for i in range(30)])))
        ref = pd.read_csv(E.REPO / "obd_preprocessing" / "outputs" / "s4_replay_runs.csv")
        ref = ref[ref.stream == "pooled"][["run", "algorithm", "accepted", "clicks", "ctr"]]
        m = val.merge(ref, on=["run", "algorithm"], suffixes=("_block", "_step5"))
        m.to_csv(E.OUT / "s4_bcs_validation.csv", index=False)
        vrows = {}
        for a, g in m.groupby("algorithm"):
            d = g.ctr_block - g.ctr_step5
            vrows[a] = {"identical_runs": int(((g.accepted_block == g.accepted_step5) & (g.clicks_block == g.clicks_step5)).sum()),
                        "mean_ctr_block": float(g.ctr_block.mean()), "mean_ctr_step5": float(g.ctr_step5.mean()),
                        "mean_accepted_block": float(g.accepted_block.mean()), "mean_accepted_step5": float(g.accepted_step5.mean()),
                        "welch_p_ctr": float(stats.ttest_ind(g.ctr_block, g.ctr_step5, equal_var=False).pvalue),
                        "welch_p_accepted": float(stats.ttest_ind(g.accepted_block, g.accepted_step5, equal_var=False).pvalue)}
        res["validation_vs_step5"] = vrows
        print("validation:", vrows, flush=True)

    # ---- M1: full-log Tier 2 ----
    runs = []
    for stream in ("random_only", "pooled"):
        jobs = [(a, i) for a in ALGS6 for i in range(30)]
        with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(stream,)) as ex:
            runs += list(ex.map(job_full, jobs))
        print(stream, "done", round(time.time() - t0), flush=True)
    runs = pd.DataFrame(runs); runs.to_csv(E.OUT / "s4_bcs_bridge_runs.csv", index=False)

    # ---- M2 ----
    res["arm_independence"] = arm_independence()

    # ---- M3: Tier 1, domain-matched simulator ----
    r = np.load(C.CACHE_DIR / "clean_random.npz")
    imp = np.bincount(r["item"], minlength=K); clk = np.bincount(r["item"], weights=r["click"], minlength=K)
    means = clk / imp
    res["calibration"] = {"item_ctr_min": float(means.min()), "item_ctr_max": float(means.max()),
                          "item_ctr_mean": float(means.mean())}
    t1 = {}
    for stream in ("random_only", "pooled"):
        T = int(round(runs[runs.stream == stream].accepted.mean()))
        jobs = [(a, i, T, means) for a in ALGS6 for i in range(30)]
        with ProcessPoolExecutor(args.jobs) as ex:
            out = pd.DataFrame(list(ex.map(job_tier1, jobs)))
        t1[stream] = out; out.assign(stream=stream).to_csv(E.OUT / f"s4_bcs_tier1_{stream}.csv", index=False)
        print("tier1", stream, T, round(time.time() - t0), flush=True)

    # ---- BCS variants ----
    p_ind = res["arm_independence"]["1h"]["p_indep"]
    variants = {}
    for stream in ("random_only", "pooled"):
        g = runs[runs.stream == stream]
        ctr = {a: g[g.algorithm == a].sort_values("run").ctr.to_numpy() for a in ALGS6}
        T = int(round(g.accepted.mean()))
        rank2 = sorted(ALGS6, key=lambda a: -ctr[a].mean())
        reg = {a: t1[stream][t1[stream].algorithm == a].sort_values("run").regret.to_numpy() for a in ALGS6}
        rank1 = sorted(ALGS6, key=lambda a: reg[a].mean())
        # generic S1 Tier 1, five shared algorithms, full-horizon Tier 2
        rho_s1 = spearman_rank(S1_RANK5, [a for a in rank2 if a in S1_RANK5])
        b_s1 = BCSCalculator.compute(rho_s1, p_ind, K, T)
        rho_m = spearman_rank(rank1, rank2)
        bs = boot_rho(reg, ctr)
        b_m = BCSCalculator.compute(rho_m, p_ind, K, T)
        pw = []                                                    # how many Tier 2 pairs are resolved?
        for x in range(len(ALGS6)):
            for y in range(x + 1, len(ALGS6)):
                a, b = ALGS6[x], ALGS6[y]
                pw.append(stats.wilcoxon(ctr[a], ctr[b]).pvalue)
        holm = np.sort(pw) <= 0.05 / (len(pw) - np.arange(len(pw)))
        n_sig = int(np.argmin(holm)) if not holm.all() else len(pw)
        integ = __import__("json").loads((C.OUT_DIR / "fix5_integrity_report.json").read_text())["reports"]
        gt = {x["policy"]: x["ctr_clean"] for x in integ}             # on-policy CTR of the cleaned logs
        variants[f"{stream}: full-log Tier 2, generic S1 Tier 1"] = b_s1 | {"rho_S": rho_s1, "T_accepted": T}
        variants[f"{stream}: full-log Tier 2, domain-matched Tier 1"] = b_m | {
            "rho_S": rho_m, "rho_boot_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "P(rho >= 0.7)": float((bs >= 0.7).mean()), "T_accepted": T,
            "tier1_ranking": rank1, "tier2_ranking": rank2,
            "tier2_pairs_significant_holm": f"{n_sig} of {len(pw)}",
            "replay_ratio_BTS_over_Random": float(ctr["BernoulliTS_Zozotown"].mean() / ctr["B0_Random"].mean()),
            "on_policy_ratio": gt["bts"] / gt["random"],
            "tier2_mean_ctr": {a: float(ctr[a].mean()) for a in ALGS6},
            "tier1_mean_regret": {a: float(reg[a].mean()) for a in ALGS6}}
    res["bcs_variants"] = variants
    res["seconds"] = round(time.time() - t0, 1)
    E.write_json(res, E.OUT / "s4_bcs_bridge.json")
    for k, v in variants.items():
        print(k, {x: v.get(x) for x in ("rho_S", "c1_reward_proxy", "c2_arm_independence", "c3_sufficient_horizon", "BCS", "band", "rho_boot_ci95")})


if __name__ == "__main__":
    main()
