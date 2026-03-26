"""
S1: Stationary Bernoulli Bandit
================================

Source: BenchmarkSpec_Vermorel2005.docx, Gap D-1; GAP5 Pillar 4.

What it is
----------
K arms, each with a fixed unknown reward probability drawn from
Beta(1,1) = Uniform(0,1).  Pull an arm, receive 1 (success) or
0 (failure).  Nothing changes over time.

Why it is included
------------------
This is the foundational MAB setting.  The Lai-Robbins lower bound
was proved exactly for this case.  Every algorithm must work here.

Parameters (from benchmark specification)
-----------------------------------------
K in {10, 50, 200}
T in {500, 2_000, 10_000}
mu_a ~ Beta(1, 1) i.i.d.
"""

from __future__ import annotations
import numpy as np


class BernoulliBandit:
    """
    Stationary K-armed Bernoulli bandit (Setting S1).

    Parameters
    ----------
    K    : int   — number of arms
    T    : int   — time horizon
    seed : int   — random seed (from pre-registered seed table)

    Attributes
    ----------
    means   : np.ndarray shape (K,)  — true reward probabilities
    mu_star : float                  — optimal arm mean (oracle)
    a_star  : int                    — index of optimal arm
    """

    VALID_K = {10, 50, 200}
    VALID_T = {500, 2_000, 10_000}

    def __init__(self, K: int, T: int, seed: int) -> None:
        if K not in self.VALID_K:
            raise ValueError(
                f"S1 requires K in {self.VALID_K}, got {K}. "
                "Custom K values are accepted for research use — "
                "remove this check if needed."
            )
        if T not in self.VALID_T:
            raise ValueError(
                f"S1 requires T in {self.VALID_T}, got {T}."
            )
        self.K    = K
        self.T    = T
        self.seed = seed

        rng        = np.random.default_rng(seed)
        # mu_a ~ Beta(1,1) = Uniform(0,1)
        self.means = rng.uniform(0.0, 1.0, size=K)
        self.mu_star = float(self.means.max())
        self.a_star  = int(self.means.argmax())

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        """
        Pull an arm and observe a Bernoulli reward.

        Parameters
        ----------
        arm : int               — arm index in {0, ..., K-1}
        rng : np.random.Generator  — caller's RNG (ensures reproducibility)

        Returns
        -------
        float : 1.0 (success) or 0.0 (failure)
        """
        if not (0 <= arm < self.K):
            raise IndexError(f"arm {arm} out of range [0, {self.K-1}]")
        return float(rng.uniform() < self.means[arm])

    def regret_increment(self, arm: int) -> float:
        """Return mu_star - mu_arm (instantaneous expected regret)."""
        return self.mu_star - self.means[arm]

    def __repr__(self) -> str:
        return (
            f"BernoulliBandit(K={self.K}, T={self.T}, "
            f"seed={self.seed}, mu_star={self.mu_star:.4f})"
        )
