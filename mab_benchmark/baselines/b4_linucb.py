"""
B4: LinUCB (Disjoint Model)
============================

Contextual bandit algorithm from Li et al. (2010).
Mandatory baseline for Setting S4 only.

Algorithm
---------
Maintains a separate linear model for each arm a:
    E[r | x, a] = x^T theta_a

Confidence bound for arm a at round t:
    UCB_a(x) = x^T theta_hat_a + alpha * sqrt(x^T A_a^{-1} x)

where A_a = I_d + sum of outer products of contexts for arm a,
      b_a = sum of context * reward for arm a,
      theta_hat_a = A_a^{-1} b_a.

The algorithm selects: a_t = argmax_a UCB_a(x_t).

Role in benchmark
-----------------
The canonical contextual baseline.  Any contextual algorithm that
does not beat LinUCB on S4 has not demonstrated added value from its
contextual component.

Source: BenchmarkSpec Gap E-5; TwoTierProtocol S4 baseline.
"""

from __future__ import annotations
import numpy as np
from mab_benchmark.core import BanditAlgorithm


class LinUCB(BanditAlgorithm):
    """
    Baseline B4: LinUCB with disjoint linear models (Li et al., 2010).

    Parameters
    ----------
    n_arms      : int   — number of arms K
    context_dim : int   — dimension d of context vector
    alpha       : float — exploration parameter (default 1.0 per spec)

    State
    -----
    A : np.ndarray (K, d, d) — per-arm Gram matrices
    b : np.ndarray (K, d)    — per-arm reward-weighted context sums
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
    ) -> None:
        if context_dim == 0:
            raise ValueError(
                "LinUCB requires context_dim > 0. "
                "Use UCB1 or ThompsonSampling for non-contextual settings."
            )
        self.alpha = alpha
        super().__init__(n_arms, context_dim)

    def reset(self) -> None:
        d = self.context_dim
        # A_a = I_d for each arm (identity matrix = unit regularisation)
        self.A = np.stack([np.eye(d) for _ in range(self.n_arms)])
        # b_a = 0_d for each arm
        self.b = np.zeros((self.n_arms, d), dtype=float)

    def choose_arm(
        self,
        t: int,
        context: np.ndarray | None = None,
    ) -> int:
        if context is None:
            raise ValueError("LinUCB requires a context vector.")
        x   = context.astype(float)
        ucb = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv        = np.linalg.inv(self.A[a])
            theta_hat    = A_inv @ self.b[a]
            exploration  = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb[a]       = x @ theta_hat + exploration

        return int(np.argmax(ucb))

    def update(
        self,
        arm: int,
        reward: float,
        t: int,
        context: np.ndarray | None = None,
    ) -> None:
        if context is None:
            raise ValueError("LinUCB requires a context vector for update.")
        x              = context.astype(float)
        self.A[arm]   += np.outer(x, x)
        self.b[arm]   += reward * x

    def theta_hat(self, arm: int) -> np.ndarray:
        """Return the current parameter estimate for arm a."""
        return np.linalg.inv(self.A[arm]) @ self.b[arm]

    def __repr__(self) -> str:
        return (
            f"LinUCB(n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, alpha={self.alpha})"
        )
