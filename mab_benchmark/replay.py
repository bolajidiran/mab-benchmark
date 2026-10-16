"""
mab_benchmark.replay
====================

The replayer method of Li et al. (2011) for logged bandit data, and the
replay-bias diagnostics that are mandatory for every replay-based Tier 2
result since mab_benchmark 1.3.0 (Chapter Three, Section 3.8.3).

Why diagnostics are mandatory
-----------------------------
The replayer is unbiased when the logged (context, action, reward) triples are
independent and identically distributed and the logging policy is known
(Li et al., 2011). Real logs are not always i.i.d. in their ORDER: on the Open
Bandit Dataset the same items reappear in bursts (median gap 6 rows against an
i.i.d. expectation of K = 80), and a click makes the next appearance of the
same item about nine times more likely to be clicked. Under such a log an
algorithm that keeps choosing the item that just paid off ("win-stay") is fed
more, and better, events than an algorithm that explores deterministically,
which waits for an out-of-burst item. The replay then favours exploitation,
the effect Pires et al. (2025) report across nine datasets.

Diagnostics
-----------
burst_profile          how far the logged order departs from i.i.d.
shuffle_within_windows a permutation that keeps every row inside its time
                       window (for example the clock hour), which removes
                       bursts but keeps slow drift
matched_ratios         matched events of each algorithm relative to a
                       randomised reference (B0 Random): < 0.5 means starved
replay_integrity       Spearman correlation between the Tier 2 ranking on the
                       log as recorded and on the shuffled log
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def replay(alg, arms, rewards, accept_prob=None, rng=None, contexts=None) -> dict:
    """Replay one algorithm over a log (Li et al., 2011, Algorithm 3; with
    rejection sampling, accept_prob_t = min(1, c / pscore_t), for a
    non-uniform logging policy).

    alg     : a BanditAlgorithm instance (choose_arm(t, context), update(arm, r, t, context))
    arms    : logged arm per row; rewards: logged reward per row
    The algorithm's clock is its number of accepted events + 1.
    """
    arms = np.asarray(arms, int); rewards = np.asarray(rewards, float)
    n = arms.size
    if accept_prob is not None:
        rng = rng or np.random.default_rng(0)
        coin = rng.random(n) < np.asarray(accept_prob, float)
    acc = 0; clk = 0.0
    for t in range(n):
        if accept_prob is not None and not coin[t]:
            continue
        ctx = None if contexts is None else contexts[t]
        a = alg.choose_arm(acc + 1, ctx)
        if a == arms[t]:
            alg.update(a, rewards[t], acc + 1, ctx)
            acc += 1; clk += rewards[t]
    return {"accepted": acc, "clicks": clk, "ctr": clk / acc if acc else float("nan")}


def burst_profile(arms, rewards, K: int, window: int = 20, seed: int = 0) -> dict:
    """Order statistics of a log. Under an i.i.d. uniform log the median gap is
    about K ln 2, sweep_over_random is about 1 and the click lift is about 1."""
    a = np.asarray(arms, int); r = np.asarray(rewards, float)
    last = np.full(K, -1); within = reapp = 0; prev_click, nxt = [], []
    for t in range(a.size):
        p = last[a[t]]
        if p >= 0:
            reapp += 1
            if t - p <= window:
                within += 1; prev_click.append(r[p]); nxt.append(r[t])
        last[a[t]] = t
    gaps = [np.median(np.diff(np.flatnonzero(a == k))) for k in range(K) if (a == k).sum() > 2]
    j = sweep = 0
    for x in a:
        if x == j:
            sweep += 1; j = (j + 1) % K
    rnd = int((np.random.default_rng(seed).integers(0, K, a.size) == a).sum())
    pc, nx = np.array(prev_click), np.array(nxt)
    after_click = nx[pc == 1].mean() if (pc == 1).any() else float("nan")
    base = r.mean()
    return {"rows": int(a.size), "median_gap": float(np.median(gaps)) if gaps else float("nan"),
            "iid_median_gap": float(K * np.log(2)),
            "share_reappear_within_window": within / max(reapp, 1),
            "sweep_over_random": sweep / max(rnd, 1),
            "click_lift_after_click": float(after_click / base) if base > 0 else float("nan")}


def shuffle_within_windows(window_ids, rng=None) -> np.ndarray:
    """Permutation that sorts rows by window and randomly inside each window."""
    w = np.asarray(window_ids)
    rng = rng or np.random.default_rng(0)
    return np.lexsort((rng.random(w.size), w))


def matched_ratios(accepted: dict, reference: str = "B0_Random") -> dict:
    """accepted: {algorithm: array of matched events per run} (paired by run)."""
    ref = np.asarray(accepted[reference], float)
    return {a: float(np.mean(np.asarray(v, float) / ref)) for a, v in accepted.items()}


def ranking(values: dict, higher_is_better: bool = True) -> list:
    return sorted(values, key=lambda a: -values[a] if higher_is_better else values[a])


def replay_integrity(rank_logged: list, rank_shuffled: list) -> float:
    """Spearman correlation between two rankings of the same algorithms."""
    if set(rank_logged) != set(rank_shuffled):
        raise ValueError("rankings must contain the same algorithms")
    return float(stats.spearmanr([rank_logged.index(a) for a in rank_logged],
                                 [rank_shuffled.index(a) for a in rank_logged]).statistic)


def bursty_uniform_log(n: int, K: int, slate: int = 3, mean_repeats: float = 4.0, rng=None):
    """A uniform-random log with the burst structure observed on OBD: each
    impression shows a slate of `slate` distinct random arms, and the same
    slate is shown again a geometric number of times (mean `mean_repeats`)
    before a new slate is drawn. Every row's arm is still uniform on K, so the
    marginal logging policy is uniform (propensity 1/K) and only the ORDER of
    the log is non-i.i.d. Returns (arms, session_id) of length n."""
    rng = rng or np.random.default_rng(0)
    arms = np.empty(n, int); sess = np.empty(n, int)
    i = s = 0
    p = 1.0 / mean_repeats
    while i < n:
        sl = rng.choice(K, slate, replace=False); reps = rng.geometric(p)
        for _ in range(reps):
            for a in sl:
                if i >= n:
                    break
                arms[i] = a; sess[i] = s; i += 1
        s += 1
    return arms, sess
