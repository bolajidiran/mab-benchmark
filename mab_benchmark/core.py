"""
mab_benchmark.core
==================

Defines the mandatory interface every algorithm must implement.

Source: BenchmarkSpec_Vermorel2005.docx, Gap I-1.
Rationale: The absence of a shared interface is the structural root
cause of 20 years of incomparable results. Every researcher had to
re-implement all baselines from scratch. With this interface, any
algorithm is automatically compared against every algorithm already
on the leaderboard.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BanditAlgorithm(ABC):
    """
    Abstract base class for every MAB algorithm in the benchmark.

    Implement exactly two methods:

        choose_arm(t, context)  →  int
        update(arm, reward, t, context)  →  None

    Nothing else is required.

    Parameters
    ----------
    n_arms : int
        Number of arms K.
    context_dim : int
        Dimension of the context vector.  0 = non-contextual setting.

    Examples
    --------
    >>> class MyAlgorithm(BanditAlgorithm):
    ...     def choose_arm(self, t, context=None):
    ...         return int(np.argmax(self.estimates))
    ...     def update(self, arm, reward, t, context=None):
    ...         self.counts[arm] += 1
    ...         n = self.counts[arm]
    ...         self.estimates[arm] += (reward - self.estimates[arm]) / n
    ...     def reset(self):
    ...         self.counts    = np.zeros(self.n_arms)
    ...         self.estimates = np.zeros(self.n_arms)
    """

    def __init__(self, n_arms: int, context_dim: int = 0) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        self.n_arms      = n_arms
        self.context_dim = context_dim
        self.reset()

    @abstractmethod
    def choose_arm(
        self,
        t: int,
        context: np.ndarray | None = None,
    ) -> int:
        """
        Called once per round before the reward is observed.

        Parameters
        ----------
        t       : int
            Current round number, 1-indexed.
        context : np.ndarray of shape (context_dim,) or None
            Feature vector for the current round.
            None for non-contextual settings (S1, S2, S3, S5).

        Returns
        -------
        int
            Arm index in {0, 1, ..., K-1}.
        """
        ...

    @abstractmethod
    def update(
        self,
        arm: int,
        reward: float,
        t: int,
        context: np.ndarray | None = None,
    ) -> None:
        """
        Called once per round after the reward is observed.

        Parameters
        ----------
        arm     : int
            The arm that was pulled (as returned by choose_arm).
        reward  : float
            The reward received.
        t       : int
            Current round number (same as in choose_arm).
        context : np.ndarray or None
            Same context vector passed to choose_arm.
        """
        ...

    def reset(self) -> None:
        """
        Reset all internal state.

        Called automatically between independent runs by BenchmarkRunner.
        Override to reset algorithm-specific state.  Always call
        super().reset() if you override.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_arms={self.n_arms})"
