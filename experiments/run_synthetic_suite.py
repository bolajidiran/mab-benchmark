"""
run_synthetic_suite.py: every algorithm on every synthetic cell (S1 x 9, S2 x 6,
S3, S4 synthetic x 2, S5 x 3), 30 runs each.

Produces the raw data behind Tables 4.2 to 4.6 and 4.11.
Output: results/raw/<cell>__<alg>.npy (30 final regrets), results/suite_runs.csv,
        results/suite_meta.json. Resumable: finished jobs are skipped.
Usage: python experiments/run_synthetic_suite.py [--jobs 2] [--only S1_K10_T10000 ...]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import exp_common as E
from mab_benchmark.runner import BenchmarkRunner


def job(args):
    cell, alg = args
    path = E.RAW / f"{cell}__{alg}.npy"
    if path.exists():
        return cell, alg, None
    Env, ekw, rkw = E.CELLS[cell]
    A, akw = E.ALGS[alg]
    t0 = time.time()
    reg = BenchmarkRunner(E.N_RUNS).run_setting(A, akw, Env, ekw, **rkw)
    np.save(path, reg)
    return cell, alg, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cells = a.only or list(E.CELLS)
    jobs = [(c, g) for c in cells for g in E.algs_for(c)]
    # long jobs first so the pool stays busy
    jobs.sort(key=lambda j: -(E.CELLS[j[0]][1]["T"] * E.CELLS[j[0]][1]["K"] * (20 if j[1] == "B4_LinUCB" else 1)))
    timing = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for cell, alg, sec in ex.map(job, jobs):
            if sec is not None:
                timing[f"{cell}__{alg}"] = round(sec, 1)
                print(f"{cell:26s} {alg:20s} {sec:7.1f} s", flush=True)
    rows = []
    for cell in E.CELLS:
        for alg in E.algs_for(cell):
            p = E.RAW / f"{cell}__{alg}.npy"
            if p.exists():
                for i, v in enumerate(np.load(p)):
                    rows.append({"cell": cell, "setting": cell.split("_")[0], "algorithm": alg,
                                 "run": i + 1, "seed": E.SEED_TABLE[i], "cumulative_regret": float(v)})
    pd.DataFrame(rows).to_csv(E.OUT / "suite_runs.csv", index=False)
    meta = E.provenance() | {"cells": list(E.CELLS), "algorithms": {k: str(v[1]) for k, v in E.ALGS.items()},
                             "seconds": timing}
    E.write_json(meta, E.OUT / "suite_meta.json")


if __name__ == "__main__":
    main()
