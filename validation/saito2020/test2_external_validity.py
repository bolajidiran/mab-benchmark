"""
Test 2: external validity of the S4 replay protocol in all three campaigns.

The question: when the thesis' S4 replay protocol evaluates the policy that
actually produced the Bernoulli TS log (Bernoulli TS with the ZOZOTOWN
production priors) against the uniform Random policy, does it recover the
CTR advantage that Saito et al. observed on-policy? The ground truth is
CTR(bts log) / CTR(random log) of the same campaign (Table 1).

Protocol (the S4 protocol of obd_preprocessing/, applied per campaign):
  data      "clean" (primary): exact duplicates removed, as Fix 5 does for
            S4; "raw" (secondary): every row, as in Saito et al.'s Table 1;
  stream    both logs of the campaign merged in time order; a
            random-log-only stream is reported as a check;
  runs      30 disjoint, consecutive slices; slice i uses seed SEED_TABLE[i]
            (1000 to 1029) of mab_benchmark.runner;
  replay    rejection sampling: a row counts for an algorithm if the item it
            chooses equals the logged item and u < min(1, c / propensity),
            c = 1/K; u is common to both algorithms within a slice;
  policies  Random (uniform over K items) and Bernoulli TS with the ZOZOTOWN
            priors of that campaign (Beta posterior updated on accepted rows).

Pass criteria, set before running (per campaign):
  H1   Bernoulli TS (ZOZOTOWN) beats Random: one-sided paired Wilcoxon, p < 0.05;
  H3a  replay ratio > 1;
  H3b  the ground-truth ratio lies inside the 95% paired-bootstrap interval
       of the replay ratio (10,000 resamples of the 30 runs);
  H3c  each on-policy CTR lies inside the 95% interval of the replay CTR of
       the matching policy (descriptive).

Option --analyse-only recomputes the statistics from outputs/test2_runs.csv.
Output: outputs/test2_external_validity.json / .md, outputs/test2_runs.csv,
        outputs/test2_curves.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

import saito_common as C

N_SLICES = 30
VARIANTS = ("clean", "raw")
N_CURVE = 50


def _drop_dups(d):
    k = ~d["dup"]
    return {key: (v[k] if isinstance(v, np.ndarray) and v.shape[:1] == k.shape else v) for key, v in d.items()}


def stream(camp, which, variant):
    r, b = C.load_cache("random", camp), C.load_cache("bts", camp)
    if variant == "clean":
        r, b = _drop_dups(r), _drop_dups(b)
    if which == "random_only":
        logs = [(r, 0)]
    else:
        logs = [(r, 0), (b, 1)]
    ts = np.concatenate([d["ts"] for d, _ in logs])
    o = np.argsort(ts, kind="stable")
    cat = lambda k: np.concatenate([d[k] for d, _ in logs])[o]
    src = np.concatenate([np.full(d["ts"].size, s, np.int8) for d, s in logs])[o]
    return {"item": cat("item").astype(np.int64), "click": cat("click").astype(np.int64),
            "pscore": cat("pscore"), "src": src}


def replay_random(item, click, ok, K, seed):
    rng = np.random.default_rng([seed, 11])
    choice = rng.integers(0, K, item.size)
    acc = (choice == item) & ok
    return acc, click * acc


def replay_bts(item, click, ok, alpha0, beta0, seed):
    rng = np.random.default_rng([seed, 13])
    a = alpha0.copy(); b = beta0.copy()
    n = item.size; K = a.size
    acc = np.zeros(n, bool)
    t, block = 0, 256
    while t < n:
        m = min(block, n - t)
        draws = rng.beta(a, b, size=(m, K))
        ch = draws.argmax(1)
        hit = np.flatnonzero((ch == item[t:t + m]) & ok[t:t + m])
        if hit.size == 0:
            t += m
            continue
        j = t + hit[0]                     # first accepted row; later draws are discarded
        acc[j] = True
        a[item[j]] += click[j]; b[item[j]] += 1 - click[j]
        t = j + 1
    return acc, click * acc


def run_campaign(camp, which, variant):
    s = stream(camp, which, variant)
    K = C.N_ITEMS[camp]
    pr = C.PRIORS[camp]
    alpha0, beta0 = np.array(pr["alpha"], float), np.array(pr["beta"], float)
    seeds = C.seed_table(N_SLICES)
    bounds = np.array_split(np.arange(s["item"].size), N_SLICES)
    runs, curves = [], []
    for i, rows in enumerate(bounds):
        a0, a1 = rows[0], rows[-1] + 1
        item, click, ps, src = s["item"][a0:a1], s["click"][a0:a1], s["pscore"][a0:a1], s["src"][a0:a1]
        u = np.random.default_rng([seeds[i], 7]).random(a1 - a0)
        ok = u < np.minimum(1.0, (1.0 / K) / ps)
        cps = np.linspace(0, a1 - a0, N_CURVE + 1).astype(int)[1:]
        for name, fn in (("Random", lambda: replay_random(item, click, ok, K, seeds[i])),
                         ("BernoulliTS_Zozotown", lambda: replay_bts(item, click, ok, alpha0, beta0, seeds[i]))):
            t0 = time.time()
            acc, clk = fn()
            na, nc = int(acc.sum()), int(clk.sum())
            runs.append({"variant": variant, "campaign": camp, "stream": which, "run": i + 1, "seed": seeds[i], "algorithm": name,
                         "rows": int(a1 - a0), "accepted": na, "clicks": nc, "ctr": nc / na if na else np.nan,
                         "accepted_bts_rows": int((acc & (src == 1)).sum()),
                         "seconds": round(time.time() - t0, 2)})
            ca, cc = np.cumsum(acc)[cps - 1], np.cumsum(clk)[cps - 1]
            curves += [{"variant": variant, "campaign": camp, "stream": which, "run": i + 1, "algorithm": name,
                        "row_fraction": k / N_CURVE, "accepted": int(x), "clicks": int(y)}
                       for k, (x, y) in enumerate(zip(ca, cc), start=1)]
    return runs, curves


def analyse(df, camp, which, variant, rep):
    sub = df[(df.variant == variant) & (df.campaign == camp) & (df.stream == which)]
    z = sub[sub.algorithm == "BernoulliTS_Zozotown"].sort_values("run").ctr.to_numpy()
    r = sub[sub.algorithm == "Random"].sort_values("run").ctr.to_numpy()
    ratio = z.mean() / r.mean()
    rng = np.random.default_rng(2024)
    idx = rng.integers(0, z.size, (10_000, z.size))
    boots = z[idx].mean(1) / r[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    w = stats.wilcoxon(z, r, alternative="greater")
    gt_b = rep[f"bts/{camp}"][f"ctr_{variant}"]; gt_r = rep[f"random/{camp}"][f"ctr_{variant}"]
    pub = C.PUBLISHED["table1"]["rows"]
    pub_ratio = pub[f"bts/{camp}"]["ctr_percent"] / pub[f"random/{camp}"]["ctr_percent"]
    ci = lambda x: [x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(x.size), x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(x.size)]
    zc, rc = ci(z), ci(r)
    return {"variant": variant, "campaign": camp, "stream": which, "n_runs": int(z.size),
            "rows_per_run": int(sub.rows.mean()),
            "accepted_per_run_BernoulliTS_Zozotown": float(sub[sub.algorithm == "BernoulliTS_Zozotown"].accepted.mean()),
            "accepted_per_run_Random": float(sub[sub.algorithm == "Random"].accepted.mean()),
            "replay_ctr_BernoulliTS_Zozotown": z.mean(), "replay_ctr_ci95_BernoulliTS_Zozotown": zc,
            "replay_ctr_Random": r.mean(), "replay_ctr_ci95_Random": rc,
            "ground_truth_ctr_bts_log": gt_b, "ground_truth_ctr_random_log": gt_r,
            "replay_ratio": ratio, "replay_ratio_ci95": [float(lo), float(hi)],
            "ground_truth_ratio": gt_b / gt_r, "published_ratio_table1": pub_ratio,
            "relative_error_of_ratio": ratio / (gt_b / gt_r) - 1,
            "replay_vs_on_policy_BernoulliTS_Zozotown": z.mean() / gt_b - 1,
            "replay_vs_on_policy_Random": r.mean() / gt_r - 1,
            "H1_bts_beats_random": {"p_one_sided_wilcoxon": float(w.pvalue), "pass": bool(w.pvalue < 0.05)},
            "H3a_ratio_above_1": {"pass": bool(ratio > 1)},
            "H3b_ground_truth_ratio_inside_ci95": {"pass": bool(lo <= gt_b / gt_r <= hi)},
            "H3c_absolute_ctrs_inside_ci95": {
                "bts": bool(zc[0] <= gt_b <= zc[1]),
                "random": bool(rc[0] <= gt_r <= rc[1])}}


def main():
    rep = {f"{r['policy']}/{r['campaign']}": r
           for r in __import__("json").loads((C.OUT_DIR / "s0_logs_report.json").read_text())["logs"]}
    import sys
    all_runs, all_curves = [], []
    for variant in ([] if "--analyse-only" in sys.argv else VARIANTS):
        for camp in C.CAMPAIGNS:
            for which in ("pooled", "random_only"):
                t0 = time.time()
                runs, curves = run_campaign(camp, which, variant)
                all_runs += runs; all_curves += curves
                print(f"[{variant}/{camp}/{which}] {time.time() - t0:.0f} s", flush=True)
    if all_runs:
        df = pd.DataFrame(all_runs)
        df.to_csv(C.OUT_DIR / "test2_runs.csv", index=False)
        pd.DataFrame(all_curves).to_csv(C.OUT_DIR / "test2_curves.csv", index=False)
    else:
        df = pd.read_csv(C.OUT_DIR / "test2_runs.csv")
    res = [analyse(df, c, w, v, rep) for v in VARIANTS for c in C.CAMPAIGNS for w in ("pooled", "random_only")]
    C.write_json({"test": "Test 2: external validity of the S4 replay protocol", "results": res},
                 C.OUT_DIR / "test2_external_validity.json")
    L = ["# Test 2: external validity of the S4 replay protocol", "",
         "| Data | Campaign | Stream | Replay CTR BTS-Z % | Replay CTR Random % | Replay ratio (95% CI) | Ground truth | Published | H1 p | H3b | Random replay vs log |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['variant']} | {C.CAMPAIGN_LABEL[r['campaign']]} | {r['stream']} | {100 * r['replay_ctr_BernoulliTS_Zozotown']:.3f} | "
                 f"{100 * r['replay_ctr_Random']:.3f} | {r['replay_ratio']:.3f} ({r['replay_ratio_ci95'][0]:.3f} to "
                 f"{r['replay_ratio_ci95'][1]:.3f}) | {r['ground_truth_ratio']:.3f} | {r['published_ratio_table1']:.3f} | "
                 f"{r['H1_bts_beats_random']['p_one_sided_wilcoxon']:.2g} | "
                 f"{'pass' if r['H3b_ground_truth_ratio_inside_ci95']['pass'] else 'fail'} | "
                 f"{100 * r['replay_vs_on_policy_Random']:+.1f}% |")
    (C.OUT_DIR / "test2_external_validity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
