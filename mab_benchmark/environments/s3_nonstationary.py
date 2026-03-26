"""
S3: Piecewise-Stationary Bandit
================================

Source: BenchmarkSpec_Vermorel2005.docx, Gap E-1; GAP5 Pillar 4.

What it is
----------
Like S1 (Bernoulli), but the reward probabilities change abruptly at
10 known time-points.  An algorithm that does not detect these changes
will lock onto the arm that was best before the change, even though it
is now suboptimal.

Why it is included
------------------
Real environments change.  User preferences shift.  Market conditions
change.  Non-stationarity is the default condition of real bandit
problems, not an extension.  The GAP5 corpus identified this as
Contested Zone C, yet most papers test only stationary settings.

Parameters (from benchmark specification)
-----------------------------------------
K  = 10  (fixed)
T  = 10_000 (fixed)
10 abrupt change-points at rounds:
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9500
mu_a^(j) ~ Beta(1,1) i.i.d. for each segment j = 0, ..., 10
"""

from __future__ import annotations
import numpy as np


class PiecewiseStationaryBandit:
    """
    Piecewise-stationary K-armed Bernoulli bandit (Setting S3).

    Parameters
    ----------
    K    : int   — number of arms (must be 10)
    T    : int   — time horizon (must be 10_000)
    seed : int   — random seed

    Attributes
    ----------
    CHANGE_POINTS : list[int]
        Rounds at which arm means are redrawn.
    segment_means : np.ndarray shape (n_segments, K)
        True arm means for each segment.
    """

    CHANGE_POINTS: list[int] = [
        1_000, 2_000, 3_000, 4_000, 5_000,
        6_000, 7_000, 8_000, 9_000, 9_500,
    ]

    def __init__(self, K: int = 10, T: int = 10_000, seed: int = 0) -> None:
        if K != 10:
            raise ValueError(f"S3 requires K=10, got {K}.")
        if T != 10_000:
            raise ValueError(f"S3 requires T=10_000, got {T}.")

        self.K    = K
        self.T    = T
        self.seed = seed

        n_seg = len(self.CHANGE_POINTS) + 1          # 11 segments
        rng   = np.random.default_rng(seed)
        self.segment_means = rng.uniform(0.0, 1.0, size=(n_seg, K))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def current_segment(self, t: int) -> int:
        """Return the segment index (0-indexed) for round t."""
        return sum(1 for cp in self.CHANGE_POINTS if t > cp)

    def mu_star(self, t: int) -> float:
        """Oracle mean at round t (changes with segment)."""
        seg = self.current_segment(t)
        return float(self.segment_means[seg].max())

    def a_star(self, t: int) -> int:
        """Oracle optimal arm at round t."""
        seg = self.current_segment(t)
        return int(self.segment_means[seg].argmax())

    def pull(self, arm: int, t: int, rng: np.random.Generator) -> float:
        """
        Pull an arm at round t and observe a Bernoulli reward.

        Note: t is required so the environment knows which segment
        is currently active.

        Returns
        -------
        float : 1.0 or 0.0
        """
        if not (0 <= arm < self.K):
            raise IndexError(f"arm {arm} out of range [0, {self.K-1}]")
        seg = self.current_segment(t)
        p   = self.segment_means[seg, arm]
        return float(rng.uniform() < p)

    def regret_increment(self, arm: int, t: int) -> float:
        return self.mu_star(t) - self.segment_means[self.current_segment(t), arm]

    def __repr__(self) -> str:
        return (
            f"PiecewiseStationaryBandit(K={self.K}, T={self.T}, "
            f"seed={self.seed}, n_segments={len(self.CHANGE_POINTS)+1})"
        )
