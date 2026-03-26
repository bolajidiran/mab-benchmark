"""
B0: Random Policy
=================

Selects an arm uniformly at random every round.

Role in benchmark
-----------------
Lower bound on performance.  Every other algorithm must beat this.
Also serves as the denominator in the NCR formula.
"""

from __future__ import annotations
import numpy as np
from mab_benchmark.core import BanditAlgorithm


class RandomPolicy(BanditAlgorithm):
    """
    Baseline B0: uniform random arm selection.

    Parameters
    ----------
    n_arms      : int — number of arms K
    context_dim : int — ignored (random policy is context-free)
    seed        : int — optional seed for reproducibility
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int = 0,
        seed: int | None = None,
    ) -> None:
        # Store seed BEFORE calling super().__init__(), because
        # super().__init__() calls self.reset() which needs self._seed.
        self._seed = seed
        super().__init__(n_arms, context_dim)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def choose_arm(
        self,
        t: int,
        context: np.ndarray | None = None,
    ) -> int:
        return int(self._rng.integers(0, self.n_arms))

    def update(
        self,
        arm: int,
        reward: float,
        t: int,
        context: np.ndarray | None = None,
    ) -> None:
        pass  # Random policy does not learn

    def __repr__(self) -> str:
        return f"RandomPolicy(n_arms={self.n_arms})"
