"""
kuairec_bias.py: a bias-free test of the replayer on fully observed real data.

Question. Does offline replay of a log with OBD-like bursts change the ranking
of MAB algorithms, compared with the ranking the same algorithms earn when they
interact online with the same users and items?

Why KuaiRec. The KuaiRec small matrix (Gao et al., 2022) records the reaction
of 1,411 users to 3,327 videos with 99.6 per cent of all user-item pairs
observed, so every counterfactual reward is known. That permits an online
ground truth on real preferences, which no logged dataset (OBD, Yahoo! R6B)
permits.

Instance. mab_benchmark.environments.KuaiRecBandit: K = 80 items (item_seed
2026, coverage >= 0.99), reward 1[watch_ratio >= 2.0], one uniformly random
user per round.

Conditions (each run r uses SEED_TABLE[r]; 30 runs)
  ONLINE   the algorithm interacts with KuaiRecBandit for T = 14,315 rounds.
           Ground truth: realised CTR and expected (pseudo-)regret.
  IID      replay (Li et al., 2011) of an i.i.d. uniform-random log of
           n = K * T rows: user uniform, item uniform, reward from the matrix.
           The replayer's assumptions hold, so its ranking should equal ONLINE.
  BURSTY   replay of a uniform-random log with OBD's burst structure
           (mab_benchmark.replay.bursty_uniform_log, slate 3): one user per
           session, the session's slate repeated a geometric number of times.
           Every row's item is still uniform on K (propensity 1/K), only the
           ORDER is not i.i.d.; a repeated (user, item) pair repeats its reward,
           so a click predicts a click on the next appearance, as on OBD.
           mean_repeats is calibrated so that the sweep-to-random matched-event
           ratio equals the OBD random log's (0.175, results/s4_replay_bias.json).
  SHUFFLED the BURSTY log permuted within windows of 8,180 rows (the mean
           number of OBD random-log rows per clock hour), the mandatory
           within-window shuffle diagnostic of Section 3.8.3.

Algorithms: B0 Random, B1 epsilon-Greedy (0.1), B2 UCB1, B3 Thompson sampling,
FG-TS and pure Greedy (epsilon = 0), the S4 bridge algorithms without the
OBD-specific Zozotown prior, plus the pure exploiter the bias question needs.

Reported per condition: mean CTR per algorithm, ranking, Spearman correlation
with the ONLINE ranking, matched-event ratio to B0 Random (starved < 0.5),
and, for BURSTY vs SHUFFLED, rho_shuffle and the BCS_R replay-integrity term.

Outputs: results/kuairec_bias_runs.csv, results/kuairec_bias.json
Run: python kuairec_bias.py --jobs 2      (--n-runs 2 for a smoke test)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

import exp_common as E
from mab_benchmark.environments import KuaiRecBandit
from mab_benchmark.runner import SEED_TABLE
from mab_benchmark import replay as RP

ALGS = ["B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS", "FG-TS", "Greedy"]
T = 14_315
SLATE = 3
WINDOW = 8_180
OBD_SWEEP_OVER_RANDOM = 0.175
_G = {}


class State:
    """Context-free algorithm state (same update rules as the S4 bridge)."""

    def __init__(self, name, K, rng):
        self.name, self.K, self.rng = name, K, rng
        self.a, self.b = np.ones(K), np.ones(K)
        self.n = np.zeros(K); self.est = np.zeros(K)

    def choose(self, B, t):
        nm, rng, K = self.name, self.rng, self.K
        if nm == "B0_Random":
            return rng.integers(0, K, B)
        if nm in ("B1_EpsGreedy", "Greedy"):
            z = np.flatnonzero(self.n == 0)
            if nm == "Greedy" and z.size:                      # one pull of each item, then exploit
                return np.full(B, int(z[0]))
            best = np.flatnonzero(self.est == self.est.max())
            ch = np.full(B, int(rng.choice(best)))
            if nm == "B1_EpsGreedy":
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


# --------------------------------------------------------------------------- logs
def make_log(kind, seed, env, mean_repeats):
    rng = np.random.default_rng([seed, 31])
    n = env.K * T
    if kind == "IID":
        arms = rng.integers(0, env.K, n); users = rng.integers(0, env.n_users, n)
        return arms, env.R[users, arms]
    arms, sess = RP.bursty_uniform_log(n, env.K, slate=SLATE, mean_repeats=mean_repeats, rng=rng)
    users = rng.integers(0, env.n_users, sess.max() + 1)[sess]
    rew = env.R[users, arms]
    if kind == "SHUFFLED":
        p = RP.shuffle_within_windows(np.arange(n) // WINDOW, np.random.default_rng([seed, 13]))
        arms, rew = arms[p], rew[p]
    return arms, rew


def block_replay(st, arms, rew, block=256):
    n_acc, clk, pos, N = 0, 0.0, 0, arms.size
    while pos < N:
        end = min(pos + block, N)
        ch = st.choose(end - pos, n_acc + 1)
        hit = np.flatnonzero(ch == arms[pos:end])
        if not hit.size:
            pos = end; continue
        j = pos + int(hit[0]); r = rew[j]
        st.update(int(arms[j]), r); n_acc += 1; clk += r; pos = j + 1
    return n_acc, clk


def online(st, env, rng):
    clk = reg = 0.0
    for t in range(1, T + 1):
        a = int(st.choose(1, t)[0])
        r = env.pull(a, rng)
        st.update(a, r); clk += r; reg += env.regret_increment(a)
    return T, clk, reg


def _init(mean_repeats):
    _G["env"] = KuaiRecBandit(T=T); _G["mr"] = mean_repeats


def job(args):
    kind, name, i = args
    seed = SEED_TABLE[i]; env = _G["env"]
    st = State(name, env.K, np.random.default_rng([seed, 11, ALGS.index(name)]))
    t0 = time.time()
    if kind == "ONLINE":
        n, c, reg = online(st, env, np.random.default_rng([seed, 5]))
    else:
        arms, rew = make_log(kind, seed, env, _G["mr"])
        n, c = block_replay(st, arms, rew); reg = np.nan
    return {"condition": kind, "run": i + 1, "seed": seed, "algorithm": name, "matched": n,
            "clicks": c, "ctr": c / n if n else np.nan, "regret": reg, "seconds": round(time.time() - t0, 2)}


def calibrate(env, target=OBD_SWEEP_OVER_RANDOM, grid=(2, 3, 4, 5, 6, 8, 10, 12, 16)):
    out = {}
    for m in grid:
        arms, sess = RP.bursty_uniform_log(env.K * T, env.K, slate=SLATE, mean_repeats=m,
                                           rng=np.random.default_rng(99))
        users = np.random.default_rng(98).integers(0, env.n_users, sess.max() + 1)[sess]
        out[m] = RP.burst_profile(arms, env.R[users, arms], env.K)
        print("calibrate", m, {k: round(v, 3) for k, v in out[m].items()}, flush=True)
    best = min(out, key=lambda m: abs(out[m]["sweep_over_random"] - target))
    return best, out


def rank(means):
    return sorted(means, key=lambda a: -means[a])


def rho(r1, r2):
    return float(stats.spearmanr([r1.index(a) for a in ALGS], [r2.index(a) for a in ALGS]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--mean-repeats", type=float, default=None)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    t0 = time.time()
    env = KuaiRecBandit(T=T)
    res = {"instance": {"K": env.K, "T": T, "n_users": env.n_users, "mu_star": env.mu_star,
                        "mean_mu": float(env.means.mean()), "item_ids": env.item_ids.tolist(),
                        "missing_share": env.missing_share, "threshold": 2.0, "item_seed": 2026,
                        "log_rows": env.K * T, "slate": SLATE, "window": WINDOW}}
    if args.mean_repeats is None:
        mr, prof = calibrate(env)
        res["calibration"] = {"target_sweep_over_random": OBD_SWEEP_OVER_RANDOM, "chosen_mean_repeats": mr,
                              "profiles": {str(k): v for k, v in prof.items()}}
    else:
        mr = args.mean_repeats
    res["instance"]["mean_repeats"] = mr
    conds = ["ONLINE", "IID", "BURSTY", "SHUFFLED"]
    jobs = [(c, a, i) for c in conds for a in ALGS for i in range(args.n_runs)]
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(mr,)) as ex:
        runs = pd.DataFrame(list(ex.map(job, jobs, chunksize=4)))
    runs.to_csv(E.OUT / f"kuairec_bias_runs{args.out_suffix}.csv", index=False)

    # profiles of the three logs actually replayed (run 1)
    res["log_profiles"] = {}
    for c in ("IID", "BURSTY", "SHUFFLED"):
        arms, rew = make_log(c, SEED_TABLE[0], env, mr)
        res["log_profiles"][c] = RP.burst_profile(arms, rew, env.K)

    summ = {}
    for c, g in runs.groupby("condition"):
        ctr = {a: g[g.algorithm == a].ctr.to_numpy() for a in ALGS}
        mt = {a: g[g.algorithm == a].sort_values("run").matched.to_numpy() for a in ALGS}
        means = {a: float(np.nanmean(v)) for a, v in ctr.items()}
        ratios = RP.matched_ratios(mt) if c != "ONLINE" else None
        summ[c] = {"mean_ctr": means, "sd_ctr": {a: float(np.nanstd(v, ddof=1)) for a, v in ctr.items()},
                   "ranking": rank(means), "mean_matched": {a: float(v.mean()) for a, v in mt.items()},
                   "matched_ratio_vs_random": ratios,
                   "starved": [a for a, v in (ratios or {}).items() if v < RP_STARVE]}
        if c == "ONLINE":
            summ[c]["mean_regret"] = {a: float(g[g.algorithm == a].regret.mean()) for a in ALGS}
            summ[c]["ranking_by_regret"] = sorted(ALGS, key=lambda a: summ[c]["mean_regret"][a])
    truth = summ["ONLINE"]["ranking"]
    for c in summ:
        summ[c]["spearman_vs_online"] = rho(summ[c]["ranking"], truth)
    # pairwise sign agreement with ONLINE (per-pair mean CTR difference)
    for c in ("IID", "BURSTY", "SHUFFLED"):
        agree = tot = 0
        for x in range(len(ALGS)):
            for y in range(x + 1, len(ALGS)):
                a, b = ALGS[x], ALGS[y]
                d0 = summ["ONLINE"]["mean_ctr"][a] - summ["ONLINE"]["mean_ctr"][b]
                d1 = summ[c]["mean_ctr"][a] - summ[c]["mean_ctr"][b]
                tot += 1; agree += int(np.sign(d0) == np.sign(d1))
        summ[c]["pair_sign_agreement_vs_online"] = agree / tot
    rs = rho(summ["BURSTY"]["ranking"], summ["SHUFFLED"]["ranking"])
    res["summary"] = summ
    res["rho_shuffle"] = rs
    res["c_R_replay_integrity"] = float(np.clip(rs / 0.7, 0, 1))
    res["seconds"] = round(time.time() - t0)
    (E.OUT / f"kuairec_bias{args.out_suffix}.json").write_text(json.dumps(res, indent=1))
    for c in conds:
        s = summ[c]
        print(c, "rank", s["ranking"], "rho_vs_online", round(s["spearman_vs_online"], 3),
              "ratios", None if s["matched_ratio_vs_random"] is None else
              {a: round(v, 3) for a, v in s["matched_ratio_vs_random"].items()}, flush=True)
    print("rho_shuffle", rs, "seconds", res["seconds"])


RP_STARVE = 0.5

if __name__ == "__main__":
    main()
