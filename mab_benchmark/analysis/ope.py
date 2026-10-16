"""
mab_benchmark.analysis.ope
==========================

Off-policy evaluation (OPE) of a fixed target policy on logged bandit data,
through the Open Bandit Pipeline (obp; Saito et al., 2021) when it is
installed, and through a numerically identical native implementation when it
is not.

Why both. obp is the reference implementation of these estimators, and
mab_benchmark uses it rather than competing with it. But obp (0.5.x) requires
Python < 3.11, while the rest of the benchmark runs on current Python. The
native backend reproduces obp's formulas exactly; tests/test_ope.py checks
that the two backends agree to floating-point precision whenever obp is
importable, and the continuous-integration job runs that check on Python 3.10.

Estimators (obp 0.5 definitions; w_i = pi_e(a_i | x_i, k_i) / pi_b(a_i | x_i, k_i)):
  IPW        mean(w r)
  SNIPW      sum(w r) / sum(w)
  DM         mean_i sum_a pi_e(a | x_i, k_i) q(x_i, a, k_i)
  DR         DM_i + w (r - q(x_i, a_i, k_i))
  SNDR       DM_i + w (r - q_i) / mean(w)
  Switch-DR  DM_i + 1[w <= lambda] w (r - q_i)
  DRos       DM_i + lambda w / (w^2 + lambda) (r - q_i)
For Switch-DR and DRos, lambda is either given or tuned by obp 0.5's
estimated-MSE rule (sample variance / n + squared high-probability bias bound,
delta = 0.05) over obp's default grid.

Inputs follow obp's bandit-feedback convention: reward, action, pscore and
position are length-n arrays; action_dist has shape (n, K, L) and gives the
target policy's probability of each item at each position for each round;
estimated_rewards (same shape) is the reward model q, needed by DM and the DR
family.
"""

from __future__ import annotations

import numpy as np

ESTIMATORS = ("IPW", "SNIPW", "DM", "DR", "SNDR", "Switch-DR", "DRos")
LAMBDAS = (1, 5, 10, 50, 100, 500, 1000, 5000, 10000)
DELTA = 0.05


def obp_available() -> bool:
    try:
        import obp.ope  # noqa: F401
        return True
    except Exception:          # ImportError, or an incompatible dependency stack
        return False


# --------------------------------------------------------------------------- helpers
def _prep(reward, action, pscore, action_dist, position, estimated_rewards):
    r = np.asarray(reward, float); a = np.asarray(action, int); ps = np.asarray(pscore, float)
    ad = np.asarray(action_dist, float)
    if ad.ndim == 2:
        ad = ad[:, :, None]
    n, K, L = ad.shape
    k = np.zeros(n, int) if position is None else np.asarray(position, int)
    if not (r.size == a.size == ps.size == k.size == n):
        raise ValueError("reward, action, pscore, position and action_dist must have n rows")
    if np.any(ps <= 0):
        raise ValueError("pscore must be positive (positivity assumption)")
    if not np.allclose(ad.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("action_dist must sum to 1 over items for every round and position")
    rows = np.arange(n)
    w = ad[rows, a, k] / ps
    dm = qf = None
    if estimated_rewards is not None:
        q = np.asarray(estimated_rewards, float)
        if q.ndim == 2:
            q = q[:, :, None]
        if q.shape != ad.shape:
            raise ValueError("estimated_rewards must have the shape of action_dist")
        dm = np.average(q[rows, :, k], weights=ad[rows, :, k], axis=1)
        qf = q[rows, a, k]
    return r, w, dm, qf


def _bias_upper_bound(r, w, w_hat, qf, delta=DELTA):
    n = r.size
    b = np.abs(((w - w_hat) * (r - qf)).mean())
    b += np.sqrt((2 * (w ** 2).mean() * np.log(2 / delta)) / n)
    b += (2 * w.max() * np.log(2 / delta)) / (3 * n)
    return b


def _switch_rounds(r, w, dm, qf, lam):
    return dm + np.array(w <= lam, dtype=int) * w * (r - qf)


def _dros_w(w, lam):
    return (lam * w) / (w ** 2 + lam) if lam < np.inf else w


def _dros_rounds(r, w, dm, qf, lam):
    return dm + _dros_w(w, lam) * (r - qf)


def tune_lambda(r, w, dm, qf, which: str, lambdas=LAMBDAS) -> float:
    """obp 0.5 estimated-MSE tuning; ties go to the first lambda in grid order."""
    n = r.size
    scores = {}
    for lam in lambdas:
        if which == "Switch-DR":
            rr = _switch_rounds(r, w, dm, qf, lam); w_hat = w * np.array(w <= lam, dtype=int)
        else:
            rr = _dros_rounds(r, w, dm, qf, lam); w_hat = _dros_w(w, lam)
        scores[lam] = np.var(rr) / n + _bias_upper_bound(r, w, w_hat, qf) ** 2
    return float(min(scores.items(), key=lambda x: x[1])[0])


# --------------------------------------------------------------------------- native backend
def _round_values(r, w, dm, qf, name, lam):
    if name == "IPW":
        return r * w
    if name == "SNIPW":
        return r * w / w.mean()
    if dm is None:
        raise ValueError(f"{name} needs estimated_rewards (a reward model)")
    if name == "DM":
        return dm
    if name == "DR":
        return dm + w * (r - qf)
    if name == "SNDR":
        return dm + w * (r - qf) / w.mean()
    if name == "Switch-DR":
        return _switch_rounds(r, w, dm, qf, lam)
    if name == "DRos":
        return _dros_rounds(r, w, dm, qf, lam)
    raise ValueError(f"unknown estimator {name}")


def _native(r, w, dm, qf, estimators, lambdas_used):
    return {e: float(_round_values(r, w, dm, qf, e, lambdas_used.get(e)).mean()) for e in estimators}


# --------------------------------------------------------------------------- obp backend
def _obp(reward, action, pscore, action_dist, position, estimated_rewards, estimators, lambdas_used):
    from obp.ope import (DirectMethod, DoublyRobust, DoublyRobustWithShrinkage, InverseProbabilityWeighting,
                         OffPolicyEvaluation, SelfNormalizedDoublyRobust,
                         SelfNormalizedInverseProbabilityWeighting, SwitchDoublyRobust)
    make = {"IPW": lambda: InverseProbabilityWeighting(),
            "SNIPW": lambda: SelfNormalizedInverseProbabilityWeighting(),
            "DM": lambda: DirectMethod(),
            "DR": lambda: DoublyRobust(),
            "SNDR": lambda: SelfNormalizedDoublyRobust(),
            "Switch-DR": lambda: SwitchDoublyRobust(lambda_=lambdas_used["Switch-DR"]),
            "DRos": lambda: DoublyRobustWithShrinkage(lambda_=lambdas_used["DRos"])}
    ad = np.asarray(action_dist, float)
    if ad.ndim == 2:
        ad = ad[:, :, None]
    n = ad.shape[0]
    fb = {"n_rounds": n, "action": np.asarray(action, int), "reward": np.asarray(reward, float),
          "pscore": np.asarray(pscore, float),
          "position": np.zeros(n, int) if position is None else np.asarray(position, int),
          "context": np.zeros((n, 1))}
    est = [make[e]() for e in estimators]
    ope = OffPolicyEvaluation(bandit_feedback=fb, ope_estimators=est)
    q = None
    if estimated_rewards is not None:
        q = np.asarray(estimated_rewards, float)
        q = q[:, :, None] if q.ndim == 2 else q
    out = ope.estimate_policy_values(action_dist=ad, estimated_rewards_by_reg_model=q)
    names = {"IPW": "ipw", "SNIPW": "snipw", "DM": "dm", "DR": "dr", "SNDR": "sndr",
             "Switch-DR": "switch-dr", "DRos": "dr-os"}
    return {e: float(out[names[e]]) for e in estimators}


# --------------------------------------------------------------------------- public API
def estimate(reward, action, pscore, action_dist, position=None, estimated_rewards=None,
             estimators=None, lam_switch=None, lam_dros=None, backend: str = "auto") -> dict:
    """Policy-value estimates of the target policy action_dist.

    estimators : subset of ESTIMATORS (default: all that the inputs allow).
    lam_switch, lam_dros : fixed lambdas, or None to tune them (obp 0.5 rule).
    backend    : "obp", "native" or "auto" (obp when importable).
    Returns {"values": {estimator: value}, "lambdas": {...}, "backend": str}.
    """
    r, w, dm, qf = _prep(reward, action, pscore, action_dist, position, estimated_rewards)
    if estimators is None:
        estimators = ESTIMATORS if dm is not None else ("IPW", "SNIPW")
    estimators = tuple(estimators)
    bad = [e for e in estimators if e not in ESTIMATORS]
    if bad:
        raise ValueError(f"unknown estimators {bad}")
    if dm is None and any(e not in ("IPW", "SNIPW") for e in estimators):
        raise ValueError("DM and the DR family need estimated_rewards (a reward model)")
    lambdas_used = {}
    if "Switch-DR" in estimators:
        lambdas_used["Switch-DR"] = lam_switch if lam_switch is not None else tune_lambda(r, w, dm, qf, "Switch-DR")
    if "DRos" in estimators:
        lambdas_used["DRos"] = lam_dros if lam_dros is not None else tune_lambda(r, w, dm, qf, "DRos")
    use = backend
    if backend == "auto":
        use = "obp" if obp_available() else "native"
    if use == "obp":
        vals = _obp(reward, action, pscore, action_dist, position, estimated_rewards, estimators, lambdas_used)
    elif use == "native":
        vals = _native(r, w, dm, qf, estimators, lambdas_used)
    else:
        raise ValueError("backend must be 'obp', 'native' or 'auto'")
    return {"values": vals, "lambdas": lambdas_used, "backend": use}


def bootstrap_ci(reward, action, pscore, action_dist, position=None, estimated_rewards=None,
                 estimators=("IPW", "SNIPW", "DR"), n_boot=500, alpha=0.05, seed=0,
                 lam_switch=None, lam_dros=None) -> dict:
    """Percentile bootstrap interval over rounds (native backend; lambdas held fixed at
    the full-sample tuned values so the interval reflects estimation noise only)."""
    r, w, dm, qf = _prep(reward, action, pscore, action_dist, position, estimated_rewards)
    lam = {}
    if "Switch-DR" in estimators:
        lam["Switch-DR"] = lam_switch if lam_switch is not None else tune_lambda(r, w, dm, qf, "Switch-DR")
    if "DRos" in estimators:
        lam["DRos"] = lam_dros if lam_dros is not None else tune_lambda(r, w, dm, qf, "DRos")
    rng = np.random.default_rng(seed); n = r.size
    draws = {e: np.empty(n_boot) for e in estimators}
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        sub = (r[i], w[i], None if dm is None else dm[i], None if qf is None else qf[i])
        for e in estimators:
            draws[e][b] = _round_values(*sub, e, lam.get(e)).mean()
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {e: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for e, v in draws.items()}


def cross_fitted_item_position_model(action, position, reward, K: int, L: int, n_folds: int = 2,
                                     seed: int = 0, prior: float = 1.0):
    """A context-free reward model q(a, k): the smoothed click rate of item a at
    position k, cross-fitted (each round's prediction comes from the folds that
    do not contain it, as obp's RegressionModel does). Returns q of shape (n, K, L).
    Appropriate for context-free target policies; contextual policies need a
    contextual model (obp.ope.RegressionModel)."""
    a = np.asarray(action, int); k = np.asarray(position, int); r = np.asarray(reward, float)
    n = a.size
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, n)
    q = np.empty((n, K, L))
    glob = r.mean()
    for f in range(n_folds):
        tr = fold != f; te = ~tr
        clk = np.zeros((K, L)); imp = np.zeros((K, L))
        np.add.at(clk, (a[tr], k[tr]), r[tr]); np.add.at(imp, (a[tr], k[tr]), 1.0)
        m = (clk + prior * glob) / (imp + prior)
        q[te] = m[None, :, :]
    return q
