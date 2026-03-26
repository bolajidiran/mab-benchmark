"""
S5: Cold-Start Bandit (K > T)
==============================

Source: BenchmarkSpec_Vermorel2005.docx, Gap E-3;
        GAP5_EvalFramework.docx, Pillar 4.

What it is
----------
More arms than rounds.  The algorithm cannot possibly try every arm
even once.  It must make intelligent choices about which arms to explore
and which to ignore from the very first round.

Why it is included
------------------
In recommendation systems, advertising, and drug screening, there are
far more options than available time.  Vermorel (2005) explicitly
acknowledged this is 'an important case in practice' and then tested
only K=1000, T=100 without systematically studying it.  Algorithms
that require pulling every arm at initialisation (e.g., Poker) fail
catastrophically here.

Parameters (from benchmark specification)
-----------------------------------------
K  = 500  (fixed)
T in {50, 100, 200}
mu_a ~ Beta(1,1) = Uniform(0,1)
"""

from __future__ import annotations
import numpy as np


class ColdStartBandit:
    """
    Cold-start K-armed Bernoulli bandit (Setting S5, K > T).

    Parameters
    ----------
    K    : int   — number of arms (must be 500)
    T    : int   — time horizon (must be in {50, 100, 200})
    seed : int   — random seed

    Attributes
    ----------
    means   : np.ndarray shape (K,) — true reward probabilities
    mu_star : float                 — optimal arm mean (oracle)
    a_star  : int                   — index of optimal arm

    Notes
    -----
    With K=500 and T<=200, the algorithm can explore at most 40% of arms.
    The optimal strategy allocates exploration budget carefully.
    Random exploration (B0) is extremely costly here.
    """

    K_REQUIRED = 500
    VALID_T    = {50, 100, 200}

    def __init__(self, K: int = 500, T: int = 100, seed: int = 0) -> None:
        if K != self.K_REQUIRED:
            raise ValueError(
                f"S5 requires K={self.K_REQUIRED}, got {K}."
            )
        if T not in self.VALID_T:
            raise ValueError(
                f"S5 requires T in {self.VALID_T}, got {T}."
            )

        self.K    = K
        self.T    = T
        self.seed = seed

        rng        = np.random.default_rng(seed)
        self.means = rng.uniform(0.0, 1.0, size=K)
        self.mu_star = float(self.means.max())
        self.a_star  = int(self.means.argmax())

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        """
        Pull an arm and observe a Bernoulli reward.

        Returns
        -------
        float : 1.0 (success) or 0.0 (failure)
        """
        if not (0 <= arm < self.K):
            raise IndexError(f"arm {arm} out of range [0, {self.K-1}]")
        return float(rng.uniform() < self.means[arm])

    def regret_increment(self, arm: int) -> float:
        return self.mu_star - self.means[arm]

    def exploration_coverage(self, arms_explored: set[int]) -> float:
        """Fraction of arms explored at least once."""
        return len(arms_explored) / self.K

    def __repr__(self) -> str:
        return (
            f"ColdStartBandit(K={self.K}, T={self.T}, "
            f"seed={self.seed}, mu_star={self.mu_star:.4f})"
        )
