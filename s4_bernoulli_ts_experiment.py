"""
=============================================================================
EXTERNAL VALIDITY EXPERIMENT - Bernoulli TS (ZOZOTOWN priors) vs ground truth
=============================================================================
PhD Thesis : Developing a Performance Model of Multi-Armed-Bandit
             Recommender Systems (LASU 2026)
Student    : Ojediran Alaba Bolaji
Supervisors: Professor O.A. Enikuomehin | Professor A. Rahman

Purpose
  The Bernoulli TS policy with ZOZOTOWN production priors is the policy that
  generated the OBD bts log, so its on-policy CTR is KNOWN. If offline replay
  is a valid S4 protocol, replaying this policy should recover the known
  advantage over uniform random:
      ground-truth ratio = CTR(bts log) / CTR(random log)   (published 1.43x)

Two tracks
  Track 0  S1, S3 and S5 synthetic benchmark of Bernoulli TS with a uniform
           prior, plus FG-TS on S1, seeded as in mab_benchmark.runner.
  Track 1  S4 external validity, taken from the S4 pipeline
           (obd_preprocessing/step5_run_s4_replay.py): rejection-sampling
           replay on both logs, 30 disjoint runs, paired statistics.

Hypotheses
  H1  Bernoulli TS (ZOZOTOWN) replay CTR > B0 Random (one-sided paired Wilcoxon)
  H2  Bernoulli TS (ZOZOTOWN) ranks in the top two of the eight algorithms
  H3a Replay ratio > 1 (direction matches the ground truth)
  H3b Ground-truth ratio lies inside the 95% bootstrap CI of the replay ratio

Outputs (repository root for the main variant)
  s4_bernoulli_ts_results.json, s4_bernoulli_ts_summary.txt,
  s4_bernoulli_ts_learning.png

HOW TO RUN (from the repository root):
    python s4_bernoulli_ts_experiment.py
    python s4_bernoulli_ts_experiment.py --variant keep_duplicates
    python s4_bernoulli_ts_experiment.py --skip-track0
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

REPO = Path(__file__).resolve().parent
PIPE = REPO / "obd_preprocessing"
sys.path.insert(0, str(PIPE))
import common as C          # noqa: E402
import figures as F         # noqa: E402
from mab_benchmark.core import BanditAlgorithm                       # noqa: E402
from mab_benchmark.runner import SEED_TABLE                          # noqa: E402
from mab_benchmark.environments import (BernoulliBandit,              # noqa: E402
                                        PiecewiseStationaryBandit, ColdStartBandit)


# ---------------------------------------------------------------------------
# Track 0
# ---------------------------------------------------------------------------
def cohens_d(a, b):
    n = len(a)
    sp = np.sqrt(((n - 1) * a.std(ddof=1) ** 2 + (n - 1) * b.std(ddof=1) ** 2) / (2 * n - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def a12_stat(a, b):
    """Vargha-Delaney A12 on regret: P(a < b) + 0.5 P(tie), the convention of
    mab_benchmark.analysis.statistics (values above 0.5 favour a)."""
    wins = sum(1 for xi in a for xj in b if xi < xj)
    ties = sum(0.5 for xi in a for xj in b if xi == xj)
    return float((wins + ties) / (len(a) * len(b)))


class BernoulliTSUniform(BanditAlgorithm):
    """Bernoulli TS with a uniform Beta(1, 1) prior (same as B3 on S1, S3, S5)."""

    def __init__(self, n_arms, alpha_0=1.0, beta_0=1.0, **kwargs):
        self.alpha_0, self.beta_0 = alpha_0, beta_0
        super().__init__(n_arms, context_dim=0)

    def reset(self):
        self.alpha = np.full(self.n_arms, self.alpha_0)
        self.beta = np.full(self.n_arms, self.beta_0)
        self.counts = np.zeros(self.n_arms, dtype=int)

    def choose_arm(self, t, context=None):
        return int(np.argmax(np.random.beta(self.alpha, self.beta)))

    def update(self, arm, reward, t, context=None):
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += r; self.beta[arm] += 1.0 - r; self.counts[arm] += 1


def _run(env_cls, AlgClass, kwargs, K, T, n_runs=30, timed=False):
    regrets = np.zeros(n_runs)
    for i in range(n_runs):
        seed = SEED_TABLE[i]
        # reward stream and Thompson draws come from child streams of
        # SeedSequence([seed, 1]), as in mab_benchmark.runner.run_streams.
        reward_ss, alg_ss = np.random.SeedSequence([seed, 1]).spawn(2)
        np.random.seed(int(alg_ss.generate_state(1, dtype=np.uint32)[0]))
        rng = np.random.default_rng(reward_ss)
        env = env_cls(K=K, T=T, seed=seed)
        alg = AlgClass(n_arms=K, **kwargs)
        total = 0.0
        for t in range(1, T + 1):
            arm = alg.choose_arm(t)
            reward = env.pull(arm, t, rng) if timed else env.pull(arm, rng)
            alg.update(arm, reward, t)
            total += env.regret_increment(arm, t) if timed else env.regret_increment(arm)
        regrets[i] = total
    return regrets


def track0():
    print("TRACK 0 - S1, S3, S5 synthetic benchmark: Bernoulli TS (uniform prior)")
    t0 = time.time()
    r = {
        "S1_K10_T500": _run(BernoulliBandit, BernoulliTSUniform, {}, 10, 500),
        "S1_K10_T2000": _run(BernoulliBandit, BernoulliTSUniform, {}, 10, 2_000),
        "S1_K10_T10000": _run(BernoulliBandit, BernoulliTSUniform, {}, 10, 10_000),
        "S3_K10_T10000": _run(PiecewiseStationaryBandit, BernoulliTSUniform, {}, 10, 10_000, timed=True),
        "S5_K500_T100": _run(ColdStartBandit, BernoulliTSUniform, {}, 500, 100),
    }
    fgts = _run(BernoulliBandit, C.FeelGoodTS, {"lambda_": 0.5}, 10, 10_000)
    w = scipy_stats.wilcoxon(r["S1_K10_T10000"], fgts, alternative="two-sided")
    out = {k: {"mean": round(float(v.mean()), 2), "sd": round(float(v.std()), 2)} for k, v in r.items()}
    out.update({
        "FG-TS_S1_K10_T10000": {"mean": round(float(fgts.mean()), 2), "sd": round(float(fgts.std()), 2)},
        "Wilcoxon_vs_FG-TS": {"p_raw": float(w.pvalue),
                              "cohens_d": round(cohens_d(r["S1_K10_T10000"], fgts), 4),
                              "a12": round(a12_stat(r["S1_K10_T10000"], fgts), 4),
                              "lower_regret": "BernoulliTS_uniform"
                              if r["S1_K10_T10000"].mean() < fgts.mean() else "FG-TS"},
        "seconds": round(time.time() - t0, 1)})
    return out


# ---------------------------------------------------------------------------
# Track 1 (S4 pipeline)
# ---------------------------------------------------------------------------
def ensure_pipeline(variant, rerun):
    if rerun or not (C.OUT_DIR / "s4_external_validity.json").exists():
        cmd = [sys.executable, str(REPO / "s4_obd_experiment.py"), "--variant", variant, "--rerun"]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    C.add_variant_arg(ap)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--skip-track0", action="store_true")
    args = ap.parse_args()
    variant = args.variant or "main"
    C.set_variant(variant)
    ensure_pipeline(variant, args.rerun)
    res = json.loads((C.OUT_DIR / "s4_replay_results.json").read_text())
    curves = pd.read_csv(C.OUT_DIR / "s4_learning_curves.csv")
    ev = res["external_validity"]
    out_dir = REPO if variant == "main" else C.OUT_DIR
    t0 = None if args.skip_track0 else track0()

    payload = {"experiment": "External validity: Bernoulli TS (ZOZOTOWN priors) vs ground truth",
               "variant": variant,
               "evaluation_policy": "Bernoulli TS with ZOZOTOWN production priors "
                                    "(obp 0.4.1 prior_bts.yaml, campaign all)",
               "ground_truth": {"published": {"bts": C.PUBLISHED_CTR["bts"], "random": C.PUBLISHED_CTR["random"],
                                              "ratio": C.PUBLISHED_CTR["bts"] / C.PUBLISHED_CTR["random"]},
                                "this_data": {"bts": ev["pooled"]["ground_truth_ctr_bts_log"],
                                              "random": ev["pooled"]["ground_truth_ctr_random_log"],
                                              "ratio": ev["pooled"]["ground_truth_ratio_this_data"]}},
               "track1_s4_external_validity": ev,
               "track1_s4_ranking": res["analyses"]["pooled"]["ranking"],
               "track0_synthetic": t0,
               "source": str((C.OUT_DIR / "s4_replay_results.json").relative_to(REPO))}
    C.write_json(payload, out_dir / "s4_bernoulli_ts_results.json")

    L = ["EXTERNAL VALIDITY EXPERIMENT - Ojediran, A.B. - LASU 2026", "=" * 78, "",
         f"Variant : {variant}",
         f"Ground truth (published, Saito et al.): BTS {C.PUBLISHED_CTR['bts']:.2%}, "
         f"Random {C.PUBLISHED_CTR['random']:.2%}, ratio {payload['ground_truth']['published']['ratio']:.2f}x",
         f"Ground truth (this cleaned data)      : BTS {payload['ground_truth']['this_data']['bts']:.4%}, "
         f"Random {payload['ground_truth']['this_data']['random']:.4%}, "
         f"ratio {payload['ground_truth']['this_data']['ratio']:.3f}x", ""]
    for stream, E in ev.items():
        lo, hi = E["replay_ratio_ci95_bootstrap"]
        L += [f"[{stream}]",
              f"  Replay CTR  BTS (ZOZOTOWN) = {E['replay_ctr_BernoulliTS_Zozotown']:.4%}   "
              f"B0 Random = {E['replay_ctr_B0_Random']:.4%}",
              f"  Replay ratio = {E['replay_ratio']:.3f}x  (95% bootstrap CI {lo:.3f} to {hi:.3f})",
              f"  Rank of BTS (ZOZOTOWN): {E['rank_of_BernoulliTS_Zozotown']} of {E['n_algorithms']}",
              f"  H1  BTS > Random       : p = {E['H1_bts_beats_random']['p']:.4f}  "
              f"{'SUPPORTED' if E['H1_bts_beats_random']['supported'] else 'NOT SUPPORTED'}",
              f"  H2  rank in top two    : {'SUPPORTED' if E['H2_rank_top_two']['supported'] else 'NOT SUPPORTED'}",
              f"  H3a direction (> 1)    : {'SUPPORTED' if E['H3a_direction_matches']['supported'] else 'NOT SUPPORTED'}",
              f"  H3b ratio inside CI    : {'SUPPORTED' if E['H3b_ratio_matches']['supported'] else 'NOT SUPPORTED'}", ""]
    if t0:
        L += ["TRACK 0 (synthetic, Bernoulli TS uniform prior; mean cumulative regret, 30 runs)",
              *[f"  {k:<15} mean = {t0[k]['mean']:>9.2f}  SD = {t0[k]['sd']:.2f}"
                for k in ("S1_K10_T500", "S1_K10_T2000", "S1_K10_T10000", "S3_K10_T10000", "S5_K500_T100")],
              f"  FG-TS S1 K=10 T=10,000: mean = {t0['FG-TS_S1_K10_T10000']['mean']:.2f}",
              f"  Wilcoxon BTS vs FG-TS: p = {t0['Wilcoxon_vs_FG-TS']['p_raw']:.3g}, "
              f"d = {t0['Wilcoxon_vs_FG-TS']['cohens_d']}, lower regret = {t0['Wilcoxon_vs_FG-TS']['lower_regret']}", ""]
    (out_dir / "s4_bernoulli_ts_summary.txt").write_text("\n".join(L), encoding="utf-8")
    F.external_validity_figure(curves, ev["pooled"], "pooled", out_dir / "s4_bernoulli_ts_learning.png",
                               "S4 external validity: Bernoulli TS with ZOZOTOWN priors vs B0 Random")
    print("\n".join(L))


if __name__ == "__main__":
    main()
