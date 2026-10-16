"""
mab_benchmark.orchestration
===========================

Layer 2 of the architecture: the Meta-Policy Orchestrator (MPO), and the
system performance monitor of Layer 6 that feeds it. Implemented in v1.2.0
from the specification of Chapter Three ("Layer 2: Meta-Policy Orchestrator";
"System Performance Monitor: Real-Time CCM Inputs").

Function 1, ingestion
    Any BanditAlgorithm can be wrapped: MetaPolicyOrchestrator(base) is itself
    a BanditAlgorithm, so the runner treats it like any other algorithm.

Function 2, state monitoring (two binary flags per round)
    Change-point flag. The monitor keeps its own Beta(1, 1) posterior mean per
    arm (for Bernoulli TS this equals the algorithm's own posterior mean) and,
    for every pulled arm, the residual r_t - mu_hat(a_t) computed BEFORE the
    update. A change-point is flagged when the mean of the last W residuals
    falls below -h_t, a sustained drop of realised reward below the posterior
    mean. The threshold is UCB-shaped and parameterised by lambda:
        h_t = sqrt(2 ln t / W) / (1 + lambda),
    with lambda = 0 for every algorithm that has no lambda (B0 to B3), so the
    rule is parameter-free for the mandatory baselines, and lambda = the
    feel-good weight for FG-TS (a larger bonus gives a faster response).
    After a flag the residual window and the monitor's own posterior restart,
    so the next flag needs W fresh observations.
    CCM flag. The running CCM over the last W rounds, from the per-round
    latency and bandwidth recorded by the SystemPerformanceMonitor, exceeds
    tau_CCM.

Function 3, switching engine
    Rule A, change-point: reset the posterior of every Thompson-Sampling-family
    algorithm (an algorithm with Beta or Normal-Inverse-Gamma state: B3, FG-TS);
    other algorithms are monitored and the flag is logged, but not reset.
    Rule B, CCM: while the CCM flag holds, move `weight_step` of GUF weight from
    w1 to w2 and w3 (equally), keeping w1 + w2 + w3 = 1 and w1 >= w_min. If the
    caller supplies per-arm costs (latency, bandwidth), arms whose cost breaks
    the budget are also excluded from the choice while the flag holds; the
    benchmark settings S1 to S5 have no per-arm costs, so there Rule B changes
    only the weights.

Choices made here that the thesis text leaves open (stated so that they can be
changed or defended): the window W = 100, the exact lambda function of h_t,
the restart of the window after a flag, and weight_step = 0.05.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from mab_benchmark.core import BanditAlgorithm
from mab_benchmark.analysis.guf import DEFAULT_WEIGHTS, REGIMES, TAU_CCM


def is_ts_family(alg) -> bool:
    return (hasattr(alg, "alpha") and hasattr(alg, "beta") and hasattr(alg, "reset")) or \
        hasattr(alg, "n_obs") and hasattr(alg, "sum_r")


class SystemPerformanceMonitor:
    """Layer 6: per-round latency l_t (ms) and bandwidth bw_t (KB), with regime flags."""

    LATENCY_LIMIT_MS = 100.0

    def __init__(self, regime: str = "WiFi", alpha: float = 0.5) -> None:
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {sorted(REGIMES)}")
        self.regime, self.alpha = regime, alpha
        self.delta_ms = REGIMES[regime]["delta_ms"]
        self.B_kb = REGIMES[regime]["B_kb"]
        self.latency_ms: list[float] = []
        self.bandwidth_kb: list[float] = []

    def record(self, latency_ms: float, bandwidth_kb: float = 0.0) -> dict:
        self.latency_ms.append(float(latency_ms)); self.bandwidth_kb.append(float(bandwidth_kb))
        return {"regime": self.regime, "l_t": latency_ms, "bw_t": bandwidth_kb,
                "over_100ms": latency_ms >= self.LATENCY_LIMIT_MS,
                "over_delta": latency_ms > self.delta_ms, "over_B": bandwidth_kb > self.B_kb}

    def ccm(self, last: int | None = None) -> float:
        l = np.asarray(self.latency_ms[-last:] if last else self.latency_ms, float)
        b = np.asarray(self.bandwidth_kb[-last:] if last else self.bandwidth_kb, float)
        if l.size == 0:
            return 0.0
        L = np.maximum(0.0, l / self.delta_ms - 1.0).mean()
        V = np.maximum(0.0, b / self.B_kb - 1.0).mean()
        return float(self.alpha * L + (1 - self.alpha) * V)

    def summary(self) -> dict:
        l = np.asarray(self.latency_ms, float)
        return {"regime": self.regime, "rounds": int(l.size),
                "median_ms": float(np.median(l)) if l.size else float("nan"),
                "p95_ms": float(np.percentile(l, 95)) if l.size else float("nan"),
                "share_over_100ms": float((l >= self.LATENCY_LIMIT_MS).mean()) if l.size else 0.0,
                "CCM": self.ccm()}


class MetaPolicyOrchestrator(BanditAlgorithm):
    """Wrap a BanditAlgorithm with the MPO state monitor and switching engine.

    Parameters
    ----------
    base : a constructed BanditAlgorithm (the acting algorithm).
    window : W, rounds per residual and CCM window.
    lam : lambda of the threshold; None reads `base.lam` / `base.lambda_` (FG-TS) or uses 0.
    regime : connectivity regime of the performance monitor ("2G", "4G", "WiFi").
    tau_ccm : CCM threshold of the CCM flag.
    latency_fn : optional callable(t) -> extra latency in ms added to the measured time
                 (used to emulate slow devices in tests; None = measured time only).
    arm_cost : optional (K, 2) array of per-arm (latency_ms, bandwidth_kb) costs; the
               chosen arm's cost is added to the round's recorded latency and bandwidth.
    """

    def __init__(self, base: BanditAlgorithm, window: int = 100, lam: float | None = None,
                 regime: str = "WiFi", tau_ccm: float = TAU_CCM, weight_step: float = 0.05,
                 w_min: float = 0.2, latency_fn=None, arm_cost=None, measure_latency: bool = True) -> None:
        self.base, self.window = base, int(window)
        if lam is None:
            lam = getattr(base, "lam", getattr(base, "lambda_", 0.0)) or 0.0
        self.lam = float(lam)
        self.ts_family = is_ts_family(base)
        self.tau_ccm, self.weight_step, self.w_min = tau_ccm, weight_step, w_min
        self.latency_fn, self.measure_latency = latency_fn, measure_latency
        self.arm_cost = None if arm_cost is None else np.asarray(arm_cost, float)
        self.regime = regime
        super().__init__(base.n_arms, getattr(base, "context_dim", 0))   # calls self.reset()

    # --------------------------------------------------------------
    def reset(self) -> None:
        self.base.reset()
        self.monitor = SystemPerformanceMonitor(self.regime)
        self._a = np.ones(self.n_arms); self._b = np.ones(self.n_arms)
        self._res = deque(maxlen=self.window)
        self.weights = list(DEFAULT_WEIGHTS)
        self.ccm_flag = False
        self.log = {"change_point_flags": [], "resets": [], "ccm_flags": [], "weight_changes": []}
        self._t_choice = 0.0

    def threshold(self, t: int) -> float:
        return math.sqrt(2.0 * math.log(max(t, 2)) / self.window) / (1.0 + self.lam)

    def _budget_ok(self) -> np.ndarray:
        if self.arm_cost is None:
            return np.ones(self.n_arms, bool)
        m = self.monitor
        return (self.arm_cost[:, 0] <= m.delta_ms) & (self.arm_cost[:, 1] <= m.B_kb)

    # --------------------------------------------------------------
    def choose_arm(self, t: int, context=None) -> int:
        t0 = time.perf_counter()
        arm = self.base.choose_arm(t, context)
        if self.ccm_flag and self.arm_cost is not None:
            ok = self._budget_ok()
            if ok.any() and not ok[arm]:
                cand = np.flatnonzero(ok)
                post = self._a[cand] / (self._a[cand] + self._b[cand])
                arm = int(cand[int(np.argmax(post))])
        self._t_choice = time.perf_counter() - t0
        return int(arm)

    def update(self, arm: int, reward: float, t: int, context=None) -> None:
        t0 = time.perf_counter()
        self.base.update(arm, reward, t, context)
        elapsed_ms = (self._t_choice + time.perf_counter() - t0) * 1000.0 if self.measure_latency else 0.0
        extra = self.latency_fn(t) if self.latency_fn else 0.0
        lat_arm = float(self.arm_cost[arm, 0]) if self.arm_cost is not None else 0.0
        bw = float(self.arm_cost[arm, 1]) if self.arm_cost is not None else 0.0
        self.monitor.record(elapsed_ms + extra + lat_arm, bw)

        # Function 2a: change-point monitor
        r = float(np.clip(reward, 0.0, 1.0))
        mu = self._a[arm] / (self._a[arm] + self._b[arm])
        self._res.append(r - mu)
        self._a[arm] += r; self._b[arm] += 1.0 - r
        if len(self._res) == self.window and np.mean(self._res) < -self.threshold(t):
            self.log["change_point_flags"].append(int(t))
            if self.ts_family:                                    # Function 3, rule A
                self.base.reset()
                self.log["resets"].append(int(t))
            self._res.clear(); self._a[:] = 1.0; self._b[:] = 1.0

        # Function 2b: CCM monitor, Function 3 rule B
        flag = len(self.monitor.latency_ms) >= self.window and self.monitor.ccm(self.window) > self.tau_ccm
        if flag and not self.ccm_flag:
            self.log["ccm_flags"].append(int(t))
        self.ccm_flag = flag
        if flag and self.weights[0] - self.weight_step >= self.w_min - 1e-12:
            self.weights[0] -= self.weight_step
            self.weights[1] += self.weight_step / 2; self.weights[2] += self.weight_step / 2
            self.log["weight_changes"].append((int(t), tuple(round(w, 4) for w in self.weights)))

    def __repr__(self) -> str:
        return f"MetaPolicyOrchestrator({self.base!r}, W={self.window}, lambda={self.lam})"


def with_mpo(BaseClass, base_kwargs: dict | None = None, **mpo_kwargs):
    """Return a class the BenchmarkRunner can construct: BaseClass wrapped in the MPO.

    >>> runner.run_setting(with_mpo(ThompsonSampling), {}, PiecewiseStationaryBandit,
    ...                    {"K": 10, "T": 10_000}, is_stationary=False)
    """
    import inspect
    takes_seed = "seed" in inspect.signature(BaseClass.__init__).parameters

    class _MPO(MetaPolicyOrchestrator):
        def __init__(self, n_arms: int, seed: int | None = None):
            kw = dict(base_kwargs or {})
            if takes_seed and seed is not None and "seed" not in kw:
                kw["seed"] = seed
            super().__init__(BaseClass(n_arms=n_arms, **kw), **mpo_kwargs)

    _MPO.__name__ = _MPO.__qualname__ = f"MPO_{BaseClass.__name__}"
    return _MPO
