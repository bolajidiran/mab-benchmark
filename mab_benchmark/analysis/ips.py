"""
mab_benchmark.analysis.ips
==========================

Inverse Propensity Score (IPS) estimator of Chapter Three, Section 3.9.

    mu_IPS(a)  = (1/T) sum_t r_t 1{a_t = a} / pi_b(a_t | x_t)
    mu_cIPS(a) = (1/T) sum_t r_t 1{a_t = a} / max(pi_b(a_t | x_t), c_min),   c_min = 0.01
    V_IPS(pi)  = (1/T) sum_t r_t pi(a_t | x_t) / pi_b(a_t | x_t)            (policy value)
"""

from __future__ import annotations

import numpy as np

C_MIN = 0.01


def arm_value_ips(rewards, logged_arms, pscores, n_arms: int, clip: float | None = None) -> np.ndarray:
    r = np.asarray(rewards, float); a = np.asarray(logged_arms, int); p = np.asarray(pscores, float)
    if clip is not None:
        p = np.maximum(p, clip)
    return np.bincount(a, weights=r / p, minlength=n_arms) / r.size


def policy_value_ips(rewards, pscores, target_probs, clip: float | None = None) -> float:
    """target_probs[t] = pi(a_t | x_t) of the evaluation policy for the logged action."""
    r = np.asarray(rewards, float); p = np.asarray(pscores, float)
    if clip is not None:
        p = np.maximum(p, clip)
    return float(np.mean(r * np.asarray(target_probs, float) / p))
