"""
run_curves_lambda.py: cumulative-regret curves and the FG-TS lambda sweep.

  curves   S1 K=10 T=10,000; S3 K=10 T=10,000; S4 synthetic K=50 T=10,000
           (all algorithms; LinUCB on S4), 30 runs each -> Figure 5.2 and the S3, S4 curves
  lambda   FG-TS with lambda in {0, 0.1, 0.3, 0.5, 1, 2} on S1 K=10 T=10,000,
           S3 and S5 K=500 T=100 -> Figure 5.1
Output: results/curves_<cell>.npz (float32, shape (30, T) per algorithm),
        results/lambda_sweep.csv, results/lambda_curves_S1.npz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import exp_common as E
from mab_benchmark.runner import BenchmarkRunner

CURVE_CELLS = ["S1_K10_T10000", "S3_K10_T10000", "S4_SYNTHETIC_K50_T10000"]
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
LAMBDA_CELLS = ["S1_K10_T10000", "S3_K10_T10000", "S5_K500_T100"]


def curve_job(a):
    cell, alg, lam = a
    Env, ekw, rkw = E.CELLS[cell]
    A, akw = E.ALGS[alg]
    if lam is not None:
        akw = {"lam": lam}
    return a, BenchmarkRunner(E.N_RUNS).run_setting_curves(A, akw, Env, ekw, **rkw).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2); a = ap.parse_args()
    jobs = [(c, g, None) for c in CURVE_CELLS for g in E.algs_for(c)]
    jobs += [(c, "FG-TS", l) for c in LAMBDA_CELLS for l in LAMBDAS]
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for key, cur in ex.map(curve_job, jobs):
            res[key] = cur
            print(key, float(cur[:, -1].mean()), flush=True)
    for c in CURVE_CELLS:
        np.savez_compressed(E.OUT / f"curves_{c}.npz", **{g: res[(c, g, None)] for g in E.algs_for(c)})
    rows = []
    for c in LAMBDA_CELLS:
        for l in LAMBDAS:
            fin = res[(c, "FG-TS", l)][:, -1]
            for i, v in enumerate(fin):
                rows.append({"cell": c, "lambda": l, "run": i + 1, "seed": E.SEED_TABLE[i], "cumulative_regret": float(v)})
    pd.DataFrame(rows).to_csv(E.OUT / "lambda_sweep.csv", index=False)
    np.savez_compressed(E.OUT / "lambda_curves_S1.npz",
                        **{f"lam_{l}": res[("S1_K10_T10000", "FG-TS", l)] for l in LAMBDAS})
    E.write_json(E.provenance() | {"curve_cells": CURVE_CELLS, "lambdas": LAMBDAS, "lambda_cells": LAMBDA_CELLS},
                 E.OUT / "curves_lambda_meta.json")


if __name__ == "__main__":
    main()
