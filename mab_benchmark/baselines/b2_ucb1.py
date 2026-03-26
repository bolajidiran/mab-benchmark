"""
B2: UCB1
=========

Upper Confidence Bound algorithm from Auer, Cesa-Bianchi & Fischer (2002).
Achieves O(log T) cumulative regret — the theoretical ceiling that all
benchmark papers claim to match or beat.

Algorithm
---------
For round t, select arm:
    a_t = argmax_a [ mu_hat_a + sqrt(2 * log(t) / n_a) ]

where mu_hat_a is the empirical mean and n_a is the pull count.
Each arm is pulled once in the first K rounds to initialise.

Role in benchmark
-----------------
The O(log T) regret bound makes UCB1 the theoretical reference point.
It was published in 2002 but omitted from Vermorel (2005) — the most
consequential gap in that benchmark.

Source: BenchmarkSpec Gap A-1; GAP5 Contested Zone B.
"""

from __future__ import annotations
import numpy as np
from mab_benchmark.core import BanditAlgorithm


class UCB1(BanditAlgorithm):
    """
    Baseline B2: UCB1 (Auer et al., 2002).

    Parameters
    ----------
    n_arms      : int   — number of arms K
    context_dim : int   — ignored

    State
    -----
    counts    : np.ndarray (K,) — pull counts n_a
    estimates : np.ndarray (K,) — empirical mean rewards
    t_total   : int             — total rounds elapsed
    """

    def reset(self) -> None:
        self.counts    = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.t_total   = 0

    def choose_arm(
        self,
        t: int,
        context: np.ndarray | None = None,
    ) -> int:
        self.t_total = t

        # Initialisation: pull each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB1 index: mu_hat + sqrt(2 * log(t) / n_a)
        log_t  = np.log(t)
        ucb    = self.estimates + np.sqrt(2.0 * log_t / self.counts)
        return int(np.argmax(ucb))

    def update(
        self,
        arm: int,
        reward: float,
        t: int,
        context: np.ndarray | None = None,
    ) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / n

    def ucb_values(self, t: int) -> np.ndarray:
        """Return current UCB indices for all arms (diagnostic)."""
        if np.any(self.counts == 0):
            return np.full(self.n_arms, np.inf)
        return self.estimates + np.sqrt(2.0 * np.log(t) / self.counts)

    def __repr__(self) -> str:
        return f"UCB1(n_arms={self.n_arms})"
