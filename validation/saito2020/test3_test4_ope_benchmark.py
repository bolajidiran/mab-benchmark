"""
Tests 3 and 4: reproduce the OPE estimator benchmark of Saito et al.

Test 3: RMSE (x 10^3) of IPW, SNIPW, DM, DR, SNDR, Switch-DR and DRos at
        n = 300,000 in both directions:
          behaviour bts    -> evaluation Random   (paper Table 4)
          behaviour random -> evaluation BTS      (paper Table 9)
Test 4: the same with n = 10,000, behaviour bts -> Random (paper Table 6),
        to test the published switch in the best estimator with sample size;
        and n = 10,000, behaviour random -> BTS (paper Table 10), because one
        column of Table 6 repeats a column of Table 10.

For each configuration: 200 bootstrap seeds (0 to 199), each a bootstrap
sample of the first n rows of the behaviour log, a cross-fitted reward
model and the seven estimators; the ground truth is the on-policy CTR of
the full evaluation-policy log. RMSE = sqrt(mean over seeds of the squared
error). Results are appended seed by seed, so a run can be resumed.

Usage (from the repository root):
  python validation/saito2020/test3_test4_ope_benchmark.py --behaviour bts --campaign all --n 300000
  python validation/saito2020/test3_test4_ope_benchmark.py --all            # all nine configurations
Options: --seeds 200  --variant raw|clean  --summarise-only
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import ope_native as O
import saito_common as C

OPE_DIR = C.OUT_DIR / "ope"
OPE_DIR.mkdir(exist_ok=True)
CONFIGS = ([("bts", c, 300_000) for c in C.CAMPAIGNS] +
           [("random", c, 300_000) for c in C.CAMPAIGNS] +
           [("bts", c, 10_000) for c in C.CAMPAIGNS] +
           [("random", c, 10_000) for c in C.CAMPAIGNS])     # Table 10, to locate a duplicated column


def tag(beh, camp, n, variant):
    ev = "random" if beh == "bts" else "bts"
    return f"{beh}_to_{ev}_{camp}_n{n}_{variant}"


def prepare(beh, camp, n, variant):
    d = C.load_cache(beh, camp)
    keep = np.flatnonzero(~d["dup"]) if variant == "clean" else np.arange(d["ts"].size)
    rows = keep[:n]
    if rows.size < n:
        raise SystemExit(f"{beh}/{camp} has only {rows.size} rows")
    data = {"ctx_codes": d["user_codes"][rows], "Xctx": C.one_hot_context(d, rows),
            "action": d["item"][rows].astype(np.int64), "reward": d["click"][rows].astype(np.int64),
            "position": d["pos"][rows].astype(np.int64), "pscore": d["pscore"][rows]}
    ev = "random" if beh == "bts" else "bts"
    e = C.load_cache(ev, camp)
    gt = float(e["click"][~e["dup"]].mean()) if variant == "clean" else float(e["click"].mean())
    K = C.N_ITEMS[camp]
    if ev == "bts":
        pr = C.PRIORS[camp]
        pi_e = O.bts_action_dist(pr["alpha"], pr["beta"], C.LEN_LIST, 100_000, 12345)
    else:
        pi_e = O.random_action_dist(K, C.LEN_LIST)
    return data, pi_e, gt


def run(beh, camp, n, n_seeds=200, variant="raw"):
    t = tag(beh, camp, n, variant)
    path = OPE_DIR / f"{t}_seeds.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    todo = [s for s in range(n_seeds) if done.empty or s not in set(done.seed)]
    if not todo:
        print(f"[{t}] complete"); return
    data, pi_e, gt = prepare(beh, camp, n, variant)
    lam = None
    if not done.empty and 0 in set(done.seed):
        r0 = done[done.seed == 0].iloc[0]
        lam = (r0["lambda_own_Switch-DR"], r0["lambda_own_DRos"])
    print(f"[{t}] ground truth {gt:.6f}; {len(todo)} seeds to run", flush=True)
    for s in todo:
        t0 = time.time()
        r, iw, dm, qf = O.run_seed(s, data, pi_e, n)
        ls, _ = O.tune(r, iw, dm, qf, "Switch-DR")
        ld, _ = O.tune(r, iw, dm, qf, "DRos")
        if s == 0:
            lam = (ls, ld)
        if lam is None:
            raise SystemExit("seed 0 must run first (its tuned lambdas are reused, as in obp 0.5.0)")
        est = O.estimate_all(r, iw, dm, qf, lam[0], lam[1])
        own = O.estimate_all(r, iw, dm, qf, ls, ld)
        row = {"seed": s, "ground_truth": gt, **est,
               "lambda_used_Switch-DR": lam[0], "lambda_used_DRos": lam[1],
               "lambda_own_Switch-DR": ls, "lambda_own_DRos": ld,
               "Switch-DR_own_lambda": own["Switch-DR"], "DRos_own_lambda": own["DRos"],
               "max_iw": float(iw.max()), "mean_iw": float(iw.mean()),
               "seconds": round(time.time() - t0, 2)}
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        if s % 10 == 0 or s == todo[-1]:
            print(f"[{t}] seed {s} done in {row['seconds']} s", flush=True)


def summarise(variant="raw"):
    pub = {300_000: {"bts": C.PUBLISHED["table4_bts_to_random_n300000"],
                     "random": C.PUBLISHED["table9_random_to_bts_n300000"]},
           10_000: {"bts": C.PUBLISHED["table6_bts_to_random_n10000"],
                    "random": C.PUBLISHED["table10_random_to_bts_n10000"]}}
    out = []
    for beh, camp, n in CONFIGS:
        path = OPE_DIR / f"{tag(beh, camp, n, variant)}_seeds.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("seed")
        gt = df.ground_truth.iloc[0]
        rec = {"behaviour": beh, "evaluation": "random" if beh == "bts" else "bts", "campaign": camp,
               "n": n, "variant": variant, "n_seeds": len(df), "ground_truth": gt,
               "lambda_Switch-DR": df["lambda_used_Switch-DR"].iloc[0], "lambda_DRos": df["lambda_used_DRos"].iloc[0],
               "mean_seconds_per_seed": df.seconds.mean()}
        rng = np.random.default_rng(2026)
        bidx = rng.integers(0, len(df), (2000, len(df)))
        for e in C.ESTIMATORS + ["Switch-DR_own_lambda", "DRos_own_lambda"]:
            se = (df[e] - gt) ** 2
            rmse = np.sqrt(se.mean()) * 1e3
            boots = np.sqrt(se.to_numpy()[bidx].mean(1)) * 1e3
            rec[f"rmse_{e}"] = rmse
            rec[f"rmse_ci95_{e}"] = np.percentile(boots, [2.5, 97.5]).tolist()
            rec[f"bias_{e}"] = (df[e].mean() - gt) * 1e3
            rec[f"sd_{e}"] = df[e].std(ddof=1) * 1e3
            if e in C.ESTIMATORS:
                p = pub.get(n, {}).get(beh, {}).get(e, {}).get(camp)
                rec[f"published_{e}"] = p
        out.append(rec)
    if out:
        C.write_json(out, OPE_DIR / f"ope_summary_{variant}.json")
        pd.DataFrame(out).to_csv(OPE_DIR / f"ope_summary_{variant}.csv", index=False)
        print(pd.DataFrame(out)[["behaviour", "campaign", "n", "n_seeds"] +
                                [f"rmse_{e}" for e in C.ESTIMATORS]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour", choices=C.POLICIES)
    ap.add_argument("--campaign", choices=C.CAMPAIGNS)
    ap.add_argument("--n", type=int, choices=[10_000, 300_000])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--variant", choices=["raw", "clean"], default="raw")
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args()
    if not a.summarise_only:
        jobs = CONFIGS if a.all else [(a.behaviour, a.campaign, a.n)]
        for beh, camp, n in jobs:
            run(beh, camp, n, a.seeds, a.variant)
    summarise(a.variant)


if __name__ == "__main__":
    main()
