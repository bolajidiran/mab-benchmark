"""
mab_benchmark.analysis.guf
==========================

Grand Unified Optimisation Function (GUF), Contextual Constraint Metric (CCM),
extended BCS and the two-layer (Pareto + weighted-sum) evaluation of
Chapter Three, Sections 3.5 and 3.6.

Functions: compute_pareto_front(), is_dominated(), dominators(), weight_sweep(),
bcs_extended() and run_gap5_pipeline(), with the definitions of Chapter Three.

Definitions
-----------
L(pi; delta)       = (1/T) sum_t max(0, l_t / delta - 1)                (latency violation)
CCM(pi; B, delta)  = alpha L(pi; delta) + (1 - alpha) (1/T) sum_t max(0, bw_t / B - 1)
GUF Gamma(pi)      = w1 NCR + w2 L + w3 CCM,  w1 + w2 + w3 = 1, w_i > 0
c4                 = clip(1 - CCM / tau_CCM, 0, 1),  tau_CCM = 0.05
BCS_ext            = (c1 + c2 + c3 + c4) / 4

All three GUF terms are dimensionless and equal 0 for the ideal policy:
NCR = R / R_rand (analysis.ncr; 1 for uniform Random), L and CCM are mean
relative budget violations, each in [0, inf). Using raw regret instead of NCR
(as in the September 2026 draft) let the regret term dominate the latency term
by two to three orders of magnitude, so the weights had no effect.

The BCS is a property of the evaluation SETTING (environment, data and
horizon), not of a policy: c2 and c3 do not depend on pi, and c1 depends on the
whole set of algorithms. It therefore enters as a gate on certification, not as
a constraint of the minimisation:
    pi_GUF = argmin_pi Gamma(pi)          (always computed and reported)
    certifiable = 1[BCS(setting) >= 0.9]  (a GUF ranking on a setting below the
                                           gate is reported as uncertified)
"""

from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = (0.6, 0.2, 0.2)
TAU_CCM = 0.05
REGIMES = {"2G": {"delta_ms": 200.0, "B_kb": 50.0},
           "4G": {"delta_ms": 100.0, "B_kb": 500.0},
           "WiFi": {"delta_ms": 20.0, "B_kb": 5000.0}}


def latency_violation(latency_ms, delta_ms: float) -> float:
    l = np.asarray(latency_ms, dtype=float)
    if delta_ms <= 0:
        raise ValueError("delta_ms must be > 0")
    return float(np.maximum(0.0, l / delta_ms - 1.0).mean()) if l.size else 0.0


def ccm(latency_ms, bandwidth_kb, delta_ms: float, B_kb: float, alpha: float = 0.5) -> float:
    """Contextual Constraint Metric, in [0, inf). CCM = 0 iff every round meets both
    budgets. (The draft's stated range [0, 1) was wrong: max(0, l/delta - 1) is unbounded.)"""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    bw = np.asarray(bandwidth_kb, dtype=float)
    v_bw = float(np.maximum(0.0, bw / B_kb - 1.0).mean()) if bw.size else 0.0
    return alpha * latency_violation(latency_ms, delta_ms) + (1.0 - alpha) * v_bw


def guf_value(ncr_value: float, L: float, ccm_value: float, weights=DEFAULT_WEIGHTS) -> float:
    """Gamma = w1 NCR + w2 L + w3 CCM. Pass NCR (analysis.ncr), not raw regret."""
    w1, w2, w3 = weights
    if min(weights) <= 0 or abs(w1 + w2 + w3 - 1.0) > 1e-9:
        raise ValueError("weights must be positive and sum to 1")
    if ncr_value < 0 or L < 0 or ccm_value < 0:
        raise ValueError("NCR, L and CCM are non-negative")
    return w1 * ncr_value + w2 * L + w3 * ccm_value


def is_dominated(v, w) -> bool:
    """True if objective vector v is dominated by w (lower is better on every objective)."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    return bool(np.all(w <= v) and np.any(w < v))


def dominators(points: dict, name) -> list:
    return [b for b, w in points.items() if b != name and is_dominated(points[name], w)]


def compute_pareto_front(points: dict) -> list:
    """points: {policy: (NCR, L, CCM, ...)}. Returns the Pareto-optimal policies, input order."""
    return [a for a in points if not dominators(points, a)]


def weight_sweep(points: dict, w1_values=(0.4, 0.5, 0.6, 0.7, 0.8)) -> dict:
    """Layer 2: rank by Gamma for each w1, the remainder split evenly between w2 and w3."""
    out = {}
    for w1 in w1_values:
        w = (w1, (1 - w1) / 2, (1 - w1) / 2)
        g = {a: guf_value(v[0], v[1], v[2], w) for a, v in points.items()}
        out[w1] = {"gamma": g, "ranking": sorted(g, key=g.get)}
    rankings = [v["ranking"] for v in out.values()]
    return {"by_w1": out, "stable": all(r == rankings[0] for r in rankings)}


def bcs_extended(bcs_result: dict, ccm_value: float, tau: float = TAU_CCM) -> dict:
    c4 = max(0.0, min(1.0, 1.0 - ccm_value / tau))
    c = [bcs_result["c1_reward_proxy"], bcs_result["c2_arm_independence"], bcs_result["c3_sufficient_horizon"], c4]
    return {"c4_ccm": round(c4, 4), "BCS_ext": round(sum(c) / 4.0, 4)}


BCS_GATE = 0.9


def bcs_gate(bcs_value, threshold: float = BCS_GATE) -> dict:
    """Certification gate of a setting. bcs_value None means the BCS was not
    computed (no Tier 2 evidence), which does not pass the gate."""
    passed = bcs_value is not None and float(bcs_value) >= threshold
    return {"bcs": None if bcs_value is None else float(bcs_value), "threshold": threshold,
            "certifiable": bool(passed)}


def run_gap5_pipeline(points: dict, w1_values=(0.4, 0.5, 0.6, 0.7, 0.8), bcs=None) -> dict:
    """Two-layer GUF evaluation: Pareto front first (no weights), then the
    weighted-sum sweep, both on (NCR, L, CCM); the setting's BCS decides only
    whether the resulting ranking is certifiable."""
    front = compute_pareto_front(points)
    return {"pareto_front": front,
            "dominated_by": {a: dominators(points, a) for a in points},
            "sweep": weight_sweep(points, w1_values),
            "gate": bcs_gate(bcs)}
