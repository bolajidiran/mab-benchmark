"""
s4_replay_bias.py: replay-bias diagnostics for S4 (mandatory since 1.3.0),
and off-policy evaluation of the policies the algorithms learn on OBD.

What it measures
  D1  Burst structure of each log: how the logged item sequence departs from
      the independent, identically distributed (i.i.d.) sequence the replayer
      method assumes (Li et al., 2011). Per log: median and mean gap before an
      item reappears, the share of reappearances within 20 rows, the
      matched-event ratio of a deterministic item sweep against a randomised
      chooser, and the click lift after a click on the same item.
  D2  Within-hour shuffle test. Each run replays the full log twice: as logged
      and with its rows permuted at random inside every clock hour (which
      removes the bursts but keeps hour-to-hour drift). Reported per
      algorithm: matched events and CTR in both orders, and the Spearman
      correlation of the two rankings (rho_shuffle).
  D3  Starvation. Matched events of each algorithm divided by those of B0
      Random in the same run (m_a / m_rand). Random's matched-event count is
      the unbiased rate n / K for any policy that randomises every decision.
  O1  OPE of the learned policies. After the full random-log replay, each
      algorithm's final policy is frozen and its action distribution computed
      (exactly for Random, epsilon-Greedy and UCB1; 20,000 Monte Carlo draws for
      the Thompson-sampling family). Its value is estimated on a 100,000-row
      sample of the Bernoulli TS log with IPW, SNIPW, DM, DR, SNDR, Switch-DR
      and DRos (mab_benchmark.analysis.ope: obp when installed, else the
      native estimators that match obp exactly), and compared with its true
      value under the random log's item-position CTRs.

Algorithms: the six context-free algorithms of the S4 bridge.
Runs: SEED_TABLE[0:30]; shuffles use SeedSequence([seed, 13]).

Outputs: results/s4_replay_bias.json, s4_replay_bias_runs.csv,
         s4_ope_final_policies.csv
Run: python s4_replay_bias.py --jobs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

import exp_common as E

sys.path.insert(0, str(E.REPO / "obd_preprocessing"))
import common as C                                               # noqa: E402
import s4_bcs_bridge as B                                        # noqa: E402
from mab_benchmark.runner import SEED_TABLE                      # noqa: E402
from mab_benchmark.analysis import ope                           # noqa: E402

K = C.K
ALGS6 = B.ALGS6
HOUR_NS = 3_600 * 10**9
_G = {}


# --------------------------------------------------------------------------- data
def _init(stream):
    B._init(stream)
    s = np.load(C.CACHE_DIR / "stream_pooled.npz")
    keep = np.flatnonzero(s["source"] == 0) if stream == "random_only" else slice(None)
    _G["hour"] = ((s["ts"][keep] - s["ts"].min()) // HOUR_NS).astype(np.int64)
    _G["stream"] = stream


def _perm_within_hour(seed):
    rng = np.random.default_rng([seed, 13])
    h = _G["hour"]
    return np.lexsort((rng.random(h.size), h))          # sort by hour, random order inside the hour


# --------------------------------------------------------------------------- replay with final state
def _replay(name, seed, item, click, acc_p, u, block=256):
    """Same process as s4_bcs_bridge.block_replay; also returns the final state."""
    ok = np.flatnonzero(u < acc_p)
    it, ck = item[ok], click[ok]
    st = B._State(name, np.random.default_rng([seed, 11, ALGS6.index(name)]), B._G["prior"])
    n_acc, clk, pos, N = 0, 0.0, 0, ok.size
    while pos < N:
        end = min(pos + block, N)
        ch = st.choose(end - pos, n_acc + 1)
        hit = np.flatnonzero(ch == it[pos:end])
        if not hit.size:
            pos = end; continue
        j = pos + int(hit[0]); r = ck[j]
        st.update(int(ch[hit[0]]), r); n_acc += 1; clk += r; pos = j + 1
    return n_acc, clk, st


def policy_distribution(st, t, n_mc=20_000, seed=0):
    """Action distribution of a frozen _State at decision time t."""
    nm = st.name
    if nm == "B0_Random":
        return np.full(K, 1.0 / K)
    if nm == "B1_EpsGreedy":
        p = np.full(K, 0.1 / K); p[int(np.argmax(st.est))] += 0.9
        return p
    if nm == "B2_UCB1":
        z = np.flatnonzero(st.n == 0); p = np.zeros(K)
        p[int(z[0]) if z.size else int(np.argmax(st.est + np.sqrt(2.0 * np.log(t) / st.n)))] = 1.0
        return p
    rng = np.random.default_rng([seed, 17])
    th = rng.beta(st.a, st.b, size=(n_mc, K))
    if nm == "FG-TS":
        th = th + 0.5 * np.sqrt(np.log(t + 1) / (st.n + 1))
    return np.bincount(th.argmax(1), minlength=K) / n_mc


def job(args):
    name, i = args
    seed = SEED_TABLE[i]
    item, click, acc_p = B._G["item"], B._G["click"], B._G["acc_p"]
    N = item.size
    u = np.random.default_rng([seed, 7]).random(N)                     # same coins as the bridge
    t0 = time.time()
    n0, c0, st = _replay(name, seed, item, click, acc_p, u)
    pi = policy_distribution(st, n0 + 1, seed=seed) if _G["stream"] == "random_only" else None
    p = _perm_within_hour(seed)
    n1, c1, _ = _replay(name, seed, item[p], click[p], acc_p[p], u[p])
    return {"stream": _G["stream"], "run": i + 1, "seed": seed, "algorithm": name, "rows": N,
            "accepted": n0, "clicks": c0, "ctr": c0 / n0 if n0 else np.nan,
            "accepted_shuffled": n1, "clicks_shuffled": c1, "ctr_shuffled": c1 / n1 if n1 else np.nan,
            "seconds": round(time.time() - t0, 1), "pi": None if pi is None else pi.tolist()}


# --------------------------------------------------------------------------- D1
def burst_diagnostics(item, click, n_rows=None, w=20):
    it = item if n_rows is None else item[:n_rows]
    ck = click if n_rows is None else click[:n_rows]
    gaps_med, gaps_mean = [], []
    last = np.full(K, -1); within = 0; reapp = 0; pc = []
    for t, (x, c) in enumerate(zip(it, ck)):
        p = last[x]
        if p >= 0:
            reapp += 1; g = t - p
            if g <= w:
                within += 1; pc.append((ck[p], c))
        last[x] = t
    for a in range(K):
        idx = np.flatnonzero(it == a)
        if idx.size > 2:
            d = np.diff(idx); gaps_med.append(np.median(d)); gaps_mean.append(d.mean())
    j = 0; sweep = 0
    for x in it:                                            # deterministic sweep 0, 1, ..., K-1, 0, ...
        if x == j:
            sweep += 1; j = (j + 1) % K
    rnd = int((np.random.default_rng(0).integers(0, K, it.size) == it).sum())
    pc = np.array(pc) if pc else np.zeros((0, 2))
    after_click = pc[pc[:, 0] == 1, 1]; after_none = pc[pc[:, 0] == 0, 1]
    return {"rows": int(it.size), "median_gap": float(np.median(gaps_med)), "mean_gap": float(np.mean(gaps_mean)),
            f"share_reappear_within_{w}": within / max(reapp, 1),
            "sweep_matches": int(sweep), "random_matches": rnd, "sweep_over_random": sweep / max(rnd, 1),
            "ctr": float(ck.mean()),
            "p_click_after_click_short_gap": float(after_click.mean()) if after_click.size else None,
            "n_after_click_short_gap": int(after_click.size),
            "p_click_after_no_click_short_gap": float(after_none.mean()) if after_none.size else None}


# --------------------------------------------------------------------------- O1
def ope_final_policies(runs, n_eval=100_000):
    s = np.load(C.CACHE_DIR / "stream_pooled.npz")
    src, item, pos, click, ps = s["source"], s["item"].astype(int), s["pos"].astype(int) - 1, \
        s["click"].astype(float), s["pscore"]
    L = int(pos.max()) + 1
    rnd = src == 0
    clk = np.zeros((K, L)); imp = np.zeros((K, L))
    np.add.at(clk, (item[rnd], pos[rnd]), click[rnd]); np.add.at(imp, (item[rnd], pos[rnd]), 1.0)
    mu = clk / np.maximum(imp, 1)                                   # truth: random-log item-position CTR
    bts = np.flatnonzero(src == 1)
    sel = np.sort(np.random.default_rng(2026).choice(bts, n_eval, replace=False))
    a, k, r, pb = item[sel], pos[sel], click[sel], ps[sel]
    q = ope.cross_fitted_item_position_model(a, k, r, K, L, seed=2026)
    rows = []
    g = runs[runs.stream == "random_only"]
    for _, row in g.iterrows():
        pi = np.asarray(row["pi"], float)
        ad = np.broadcast_to(pi[None, :, None], (n_eval, K, L))
        out = ope.estimate(r, a, pb, ad, position=k, estimated_rewards=q)
        truth = float((pi @ mu)[k].mean())
        rows.append({"run": row["run"], "algorithm": row["algorithm"], "true_value": truth,
                     **{f"{e}": v for e, v in out["values"].items()},
                     "lambda_switch": out["lambdas"]["Switch-DR"], "lambda_dros": out["lambdas"]["DRos"],
                     "backend": out["backend"]})
    return pd.DataFrame(rows)


def rank_from(means: dict, higher_better=True):
    return sorted(means, key=lambda a: -means[a] if higher_better else means[a])


def spearman(a_names, b_names):
    return float(stats.spearmanr([a_names.index(n) for n in a_names], [b_names.index(n) for n in a_names]).statistic)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--streams", default="random_only,pooled")
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    t0 = time.time(); res = {"provenance": E.provenance()}

    # D1 on both logs
    s = np.load(C.CACHE_DIR / "stream_pooled.npz")
    res["bursts"] = {}
    for name, m in (("random_log", s["source"] == 0), ("bts_log", s["source"] == 1)):
        res["bursts"][name] = burst_diagnostics(s["item"][m].astype(int), s["click"][m].astype(float),
                                                n_rows=None if name == "random_log" else 1_374_165)
    print("bursts", res["bursts"], flush=True)
    del s

    runs = []
    for stream in args.streams.split(","):
        with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(stream,)) as ex:
            runs += list(ex.map(job, [(a, i) for a in ALGS6 for i in range(args.n_runs)]))
        print(stream, "done", round(time.time() - t0), flush=True)
    runs = pd.DataFrame(runs)
    runs.drop(columns=["pi"]).to_csv(E.OUT / f"s4_replay_bias_runs{args.out_suffix}.csv", index=False)

    # agreement with the bridge runs (same process, same coins)
    bridge = pd.read_csv(E.OUT / "s4_bcs_bridge_runs.csv")
    m = runs.merge(bridge, on=["stream", "run", "algorithm"], suffixes=("", "_bridge"))
    res["identical_to_bridge_runs"] = f"{int(((m.accepted == m.accepted_bridge) & (m.clicks == m.clicks_bridge)).sum())} of {len(m)}"

    res["shuffle"] = {}
    for stream, g in runs.groupby("stream"):
        mean0 = {a: g[g.algorithm == a].ctr.mean() for a in ALGS6}
        mean1 = {a: g[g.algorithm == a].ctr_shuffled.mean() for a in ALGS6}
        acc0 = {a: g[g.algorithm == a].accepted.mean() for a in ALGS6}
        acc1 = {a: g[g.algorithm == a].accepted_shuffled.mean() for a in ALGS6}
        r0, r1 = rank_from(mean0), rank_from(mean1)
        ratio0 = {a: float((g[g.algorithm == a].sort_values("run").accepted.to_numpy()
                            / g[g.algorithm == "B0_Random"].sort_values("run").accepted.to_numpy()).mean()) for a in ALGS6}
        ratio1 = {a: float((g[g.algorithm == a].sort_values("run").accepted_shuffled.to_numpy()
                            / g[g.algorithm == "B0_Random"].sort_values("run").accepted_shuffled.to_numpy()).mean()) for a in ALGS6}
        pvals = {a: float(stats.wilcoxon(g[g.algorithm == a].sort_values("run").ctr,
                                         g[g.algorithm == a].sort_values("run").ctr_shuffled).pvalue) for a in ALGS6}
        res["shuffle"][stream] = {"ranking_logged": r0, "ranking_shuffled": r1, "rho_shuffle": spearman(r0, r1),
                                  "mean_ctr_logged": mean0, "mean_ctr_shuffled": mean1,
                                  "mean_matched_logged": acc0, "mean_matched_shuffled": acc1,
                                  "matched_ratio_vs_random_logged": ratio0,
                                  "matched_ratio_vs_random_shuffled": ratio1,
                                  "min_matched_ratio_logged": min(ratio0.values()),
                                  "wilcoxon_p_ctr_logged_vs_shuffled": pvals}
    print(json.dumps(res["shuffle"], indent=1)[:3000], flush=True)

    if (runs.stream == "random_only").any():
        o = ope_final_policies(runs)
        o.to_csv(E.OUT / f"s4_ope_final_policies{args.out_suffix}.csv", index=False)
        est = [c for c in ope.ESTIMATORS if c in o.columns]
        summ = {}
        for e in est + ["true_value"]:
            summ[e] = {a: float(o[o.algorithm == a][e].mean()) for a in ALGS6}
        err = {e: {"rmse": float(np.sqrt(((o[e] - o.true_value) ** 2).mean())),
                   "bias": float((o[e] - o.true_value).mean())} for e in est}
        rt = rank_from(summ["true_value"]); rdr = rank_from(summ["DR"])
        rr = res["shuffle"]["random_only"]["ranking_logged"]
        res["ope_final_policies"] = {"backend": str(o.backend.iloc[0]), "n_eval_rows_bts_log": 100_000,
                                     "mean_value": summ, "error_vs_truth": err,
                                     "ranking_true_value": rt, "ranking_DR": rdr,
                                     "rho_DR_vs_true": spearman(rt, rdr), "rho_replay_vs_true": spearman(rt, rr)}
    res["seconds"] = round(time.time() - t0, 1)
    E.write_json(res, E.OUT / f"s4_replay_bias{args.out_suffix}.json")
    print("done", res["seconds"])


if __name__ == "__main__":
    main()
