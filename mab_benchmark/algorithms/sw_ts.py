"""
Sliding-window Thompson sampling for piecewise-stationary Bernoulli bandits
(Trovo, Paladino, Restelli & Gatti, 2020).

The Beta(1 + successes, 1 + failures) posterior of each arm uses only the
rewards observed in the last `window` rounds, so evidence older than the
window is forgotten and the posterior can follow a change-point. Rewards are
clipped to [0, 1].
"""
from __future__ import annotations

from collections import deque

import numpy as np

from mab_benchmark.core import BanditAlgorithm


class SlidingWindowTS(BanditAlgorithm):
    def __init__(self, n_arms: int, window: int = 1000, alpha_0: float = 1.0, beta_0: float = 1.0) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window); self.alpha_0 = float(alpha_0); self.beta_0 = float(beta_0)
        super().__init__(n_arms)

    def reset(self) -> None:
        self.succ = np.zeros(self.n_arms); self.fail = np.zeros(self.n_arms)
        self.hist: deque = deque()

    def choose_arm(self, t: int, context=None) -> int:
        th = np.random.beta(self.alpha_0 + self.succ, self.beta_0 + self.fail)
        return int(np.argmax(th))

    def update(self, arm: int, reward: float, t: int, context=None) -> None:
        r = min(max(float(reward), 0.0), 1.0)
        self.hist.append((arm, r)); self.succ[arm] += r; self.fail[arm] += 1 - r
        if len(self.hist) > self.window:
            a, x = self.hist.popleft(); self.succ[a] -= x; self.fail[a] -= 1 - x

    def __repr__(self) -> str:
        return f"SlidingWindowTS(n_arms={self.n_arms}, window={self.window})"
