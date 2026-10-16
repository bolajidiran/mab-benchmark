"""
run_latency.py: measured per-round decision latency (choose_arm + update) of
every algorithm, for the latency term L and the CCM of the GUF (Tables 4.9,
4.14a to 4.20).

For each algorithm and setting: 5 runs x min(T, 2,000) rounds, wall-clock time
per round with time.perf_counter_ns. Then, for each deployment regime of
Section 3.6 (and the 0.5 ms stress threshold used in Chapter Four),
L = mean over rounds of max(0, l_t / delta - 1). The bandwidth term of the CCM
is zero: an arm index is the only payload the benchmark transmits (a few bytes
against budgets of 50 KB or more).
Output: results/latency_profile.csv, results/latency_rounds_summary.json
Note: latency depends on the machine; the hardware is recorded in the output.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import exp_common as E
from mab_benchmark.runner import run_streams, _make_algorithm

REGIMES = {"stress_0.5ms": 0.5, "WiFi_20ms": 20.0, "4G_100ms": 100.0, "2G_200ms": 200.0}
BANDWIDTH_KB = {"stress_0.5ms": 5000, "WiFi_20ms": 5000, "4G_100ms": 500, "2G_200ms": 50}
CELLS = ["S1_K10_T10000", "S1_K200_T10000", "S3_K10_T10000", "S4_SYNTHETIC_K50_T10000", "S5_K500_T200"]
N_RUNS, MAX_T = 5, 2000


def profile(cell, alg):
    Env, ekw, rkw = E.CELLS[cell]
    A, akw = E.ALGS[alg]
    ctx_mode = rkw.get("use_context", False); stat = rkw.get("is_stationary", True)
    lat = []
    for i in range(N_RUNS):
        seed = E.SEED_TABLE[i]; rng, aseed = run_streams(seed)
        env = Env(**ekw, seed=seed); np.random.seed(aseed)
        alg_o = _make_algorithm(A, env.K, akw, aseed, getattr(env, "context_dim", 0) if ctx_mode else 0)
        for t in range(1, min(env.T, MAX_T) + 1):
            ctx = env.context(t) if ctx_mode else None
            t0 = time.perf_counter_ns()
            arm = alg_o.choose_arm(t, ctx)
            t1 = time.perf_counter_ns()
            r = env.pull(arm, t, rng) if (ctx_mode or not stat) else env.pull(arm, rng)
            t2 = time.perf_counter_ns()
            alg_o.update(arm, r, t, ctx)
            t3 = time.perf_counter_ns()
            lat.append(((t1 - t0) + (t3 - t2)) / 1e6)       # ms, excluding the environment
    return np.array(lat)


def main():
    rows = []
    for cell in CELLS:
        for alg in E.algs_for(cell):
            l = profile(cell, alg)
            rec = {"cell": cell, "algorithm": alg, "rounds_timed": l.size, "median_ms": np.median(l),
                   "mean_ms": l.mean(), "p95_ms": np.percentile(l, 95), "max_ms": l.max()}
            for name, d in REGIMES.items():
                L = float(np.maximum(0, l / d - 1).mean())
                rec[f"L_{name}"] = L
                rec[f"CCM_{name}"] = 0.5 * L + 0.5 * 0.0     # alpha = 0.5; bandwidth term zero
            rows.append(rec); print(cell, alg, f"{rec['median_ms']:.4f} ms", flush=True)
    pd.DataFrame(rows).to_csv(E.OUT / "latency_profile.csv", index=False)
    E.write_json(E.provenance() | {"regimes_ms": REGIMES, "bandwidth_budget_kb": BANDWIDTH_KB,
                                   "runs": N_RUNS, "max_rounds": MAX_T,
                                   "processor": __import__("platform").processor()},
                 E.OUT / "latency_meta.json")


if __name__ == "__main__":
    main()
