"""
S2: Stationary Gaussian Bandit
================================

Source: BenchmarkSpec_Vermorel2005.docx, Gap A-3; Vermorel (2005) baseline.

What it is
----------
K arms with fixed unknown mean rewards.  Rewards are continuous values
drawn from a normal distribution N(mu_a, sigma_a^2).

Why it is included
------------------
Continuous rewards are standard in network optimisation, clinical dosing,
and recommendation scoring.  This is a direct replacement for the
Vermorel (2005) synthetic dataset — now with UCB-Normal added as a
mandatory baseline, which Vermorel omitted despite it being the
theoretically correct algorithm for Gaussian rewards.

Parameters (from benchmark specification)
-----------------------------------------
K in {10, 50, 200}
T in {1_000, 10_000}
mu_a    ~ Uniform(0, 1)
sigma_a ~ Uniform(0, 1)
"""

from __future__ import annotations
import numpy as np


class GaussianBandit:
    """
    Stationary K-armed Gaussian bandit (Setting S2).

    Parameters
    ----------
    K    : int   — number of arms
    T    : int   — time horizon
    seed : int   — random seed

    Attributes
    ----------
    means   : np.ndarray shape (K,) — true arm means
    stds    : np.ndarray shape (K,) — true arm std devs
    mu_star : float                 — maximum arm mean (oracle)
    a_star  : int                   — index of optimal arm
    """

    VALID_K = {10, 50, 200}
    VALID_T = {1_000, 10_000}

    def __init__(self, K: int, T: int, seed: int) -> None:
        if K not in self.VALID_K:
            raise ValueError(f"S2 requires K in {self.VALID_K}, got {K}.")
        if T not in self.VALID_T:
            raise ValueError(f"S2 requires T in {self.VALID_T}, got {T}.")

        self.K    = K
        self.T    = T
        self.seed = seed

        rng       = np.random.default_rng(seed)
        self.means = rng.uniform(0.0, 1.0, size=K)
        self.stds  = rng.uniform(0.0, 1.0, size=K)
        self.mu_star = float(self.means.max())
        self.a_star  = int(self.means.argmax())

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        """
        Pull an arm and observe a Gaussian reward.

        Returns
        -------
        float : sample from N(mu_arm, sigma_arm^2)
        """
        if not (0 <= arm < self.K):
            raise IndexError(f"arm {arm} out of range [0, {self.K-1}]")
        return float(rng.normal(self.means[arm], self.stds[arm]))

    def regret_increment(self, arm: int) -> float:
        return self.mu_star - self.means[arm]

    def __repr__(self) -> str:
        return (
            f"GaussianBandit(K={self.K}, T={self.T}, "
            f"seed={self.seed}, mu_star={self.mu_star:.4f})"
        )
