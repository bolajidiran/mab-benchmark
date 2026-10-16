"""
STEP 4 = FIX 1 + FIX 2: use the full log, split into 30 disjoint runs.

Fix 1 (full log). Every cleaned row of both logs is used: 13.7M rows in one
time-ordered stream (step 2).

Fix 2 (independent runs). Runs that replay the same rows give a deterministic
algorithm such as UCB1 identical results (SD = 0), and the Wilcoxon test is
then not meaningful. The stream is therefore cut into 30 consecutive,
non-overlapping time slices of (almost) equal size. Run i
replays slice i only, in time order, with seed SEED_TABLE[i]. Every
algorithm sees the same slice in run i, so the comparison is paired.

The script records each slice's boundaries and the expected number of
accepted events under rejection-sampling replay (acceptance probability
min(1, c / propensity), c = 1/K), so the statistical power of every run is
known before step 5 starts.

Outputs
  outputs/fix1_fix2_slices.csv / .json / .md
  cache/slices.npz

Run from the repository root:
  python obd_preprocessing/step4_fix1_fix2_slices.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C


def main():
    import argparse
    ap = argparse.ArgumentParser()
    C.add_variant_arg(ap)
    C.set_variant(ap.parse_args().variant or "main")
    s = dict(np.load(C.CACHE_DIR / "stream_pooled.npz"))
    n = s["ts"].size
    seeds = C.seed_table()
    bounds = np.array_split(np.arange(n), C.N_SLICES)
    starts = np.array([b[0] for b in bounds]); ends = np.array([b[-1] + 1 for b in bounds])

    # Disjoint and exhaustive (Fix 1 + Fix 2 together)
    assert starts[0] == 0 and ends[-1] == n
    assert np.all(starts[1:] == ends[:-1])

    accept = np.minimum(1.0, C.ACCEPT_FLOOR / s["pscore"])
    rows = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        src = s["source"][a:b]
        rows.append({
            "run": i + 1, "seed": seeds[i], "row_start": int(a), "row_end": int(b),
            "rows": int(b - a),
            "rows_random": int((src == 0).sum()), "rows_bts": int((src == 1).sum()),
            "first_timestamp": str(pd.Timestamp(s["ts"][a], tz="UTC")),
            "last_timestamp": str(pd.Timestamp(s["ts"][b - 1], tz="UTC")),
            "logged_clicks": int(s["click"][a:b].sum()),
            # expected accepted events for a uniform-random target policy:
            # sum over rows of P(choose logged item) * P(accept) = (1/K) * min(1, c/p)
            "expected_accepted_events": float(accept[a:b].sum() / C.K),
        })
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / "fix1_fix2_slices.csv", index=False)
    np.savez(C.CACHE_DIR / "slices.npz", starts=starts, ends=ends, seeds=np.array(seeds))

    summary = {
        "step": "Fix 1 + Fix 2 slices", "environment": C.environment_info(),
        "total_rows": int(n), "n_runs": C.N_SLICES,
        "rows_per_run_min": int(df.rows.min()), "rows_per_run_max": int(df.rows.max()),
        "expected_accepted_per_run_mean": float(df.expected_accepted_events.mean()),
        "expected_accepted_total": float(df.expected_accepted_events.sum()),
        "share_rows_accept_prob_below_1": float(np.mean(accept < 1)),
        "acceptance_floor_c": C.ACCEPT_FLOOR,
        "runs": rows}
    C.write_json(summary, C.OUT_DIR / "fix1_fix2_slices.json")

    L = ["# Fix 1 + Fix 2: full log in 30 disjoint time slices", "",
         "| Quantity | Value |", "|---|---|",
         f"| Rows used | {n:,} (all cleaned rows, both logs) |",
         f"| Distinct rows across the 30 runs | {n:,} (each row in exactly one run) |",
         f"| Rows per run | {df.rows.min():,} to {df.rows.max():,} |",
         f"| Expected accepted events per run (uniform target) | about {df.expected_accepted_events.mean():,.0f} |",
         "", f"Acceptance probability min(1, c / propensity) with c = 1/{C.K}; "
         f"{np.mean(accept < 1):.1%} of rows (all from the bts log) have acceptance below 1.", "",
         "| Run | Seed | Rows | Rows random | Rows bts | From (UTC) | To (UTC) | Logged clicks | Expected accepted |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run']} | {r['seed']} | {r['rows']:,} | {r['rows_random']:,} | {r['rows_bts']:,} | "
                 f"{r['first_timestamp'][:19]} | {r['last_timestamp'][:19]} | {r['logged_clicks']:,} | "
                 f"{r['expected_accepted_events']:,.0f} |")
    (C.OUT_DIR / "fix1_fix2_slices.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
