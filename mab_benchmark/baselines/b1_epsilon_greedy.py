"""
B1: Epsilon-Greedy
==================

With probability epsilon, select a random arm.
With probability (1 - epsilon), select the arm with the highest
empirical mean reward.

Role in benchmark
-----------------
Simplest possible non-trivial policy.  Tests whether algorithmic
sophistication adds real value.  Benchmark mandates epsilon=0.1.

Source: GAP5 Contested Zone B; BenchmarkSpec Gap A-1.
"""

from __future__ import annotations
import numpy as np
from mab_benchmark.core import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
    """
    Baseline B1: Epsilon-greedy with epsilon=0.1.

    Parameters
    ----------
    n_arms      : int   — number of arms K
    context_dim : int   — ignored (non-contextual)
    epsilon     : float — exploration probability (default 0.1 per spec)

    State
    -----
    counts    : np.ndarray (K,) — number of times each arm was pulled
    estimates : np.ndarray (K,) — empirical mean reward per arm
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int = 0,
        epsilon: float = 0.1,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0,1), got {epsilon}.")
        self.epsilon = epsilon
        super().__init__(n_arms, context_dim)

    def reset(self) -> None:
        self.counts    = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)

    def choose_arm(
        self,
        t: int,
        context: np.ndarray | None = None,
    ) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_arms))
        return int(np.argmax(self.estimates))

    def update(
        self,
        arm: int,
        reward: float,
        t: int,
        context: np.ndarray | None = None,
    ) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental mean update: avoids storing all rewards
        self.estimates[arm] += (reward - self.estimates[arm]) / n

    def __repr__(self) -> str:
        return f"EpsilonGreedy(n_arms={self.n_arms}, epsilon={self.epsilon})"
