"""
Disjoint linear contextual bandits sharing LinUCB's model (Li et al., 2010):
    E[r | x, a] = x^T theta_a,  A_a = I_d + sum x x^T,  b_a = sum r x.

LinTS   Thompson sampling with linear payoffs (Agrawal & Goyal, 2013):
        theta~_a ~ N(A_a^{-1} b_a, v^2 A_a^{-1}),  a_t = argmax_a x^T theta~_a.
Lin     the purely greedy linear model, a_t = argmax_a x^T A_a^{-1} b_a, which is
        epsilon-greedy with epsilon = 0 (the "Lin" of Pires et al., 2025). It is
        the reference that tests whether an offline protocol rewards exploration.
"""
from __future__ import annotations

import numpy as np

from mab_benchmark.core import BanditAlgorithm


class _DisjointLinear(BanditAlgorithm):
    def __init__(self, n_arms: int, context_dim: int) -> None:
        if context_dim == 0:
            raise ValueError(f"{type(self).__name__} requires context_dim > 0")
        super().__init__(n_arms, context_dim)

    def reset(self) -> None:
        d = self.context_dim
        self.A_inv = np.stack([np.eye(d) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, d))

    def _theta(self):
        return np.einsum("kij,kj->ki", self.A_inv, self.b)

    def update(self, arm: int, reward: float, t: int, context=None) -> None:
        if context is None:
            raise ValueError(f"{type(self).__name__} requires a context vector")
        x = np.asarray(context, float)
        Ai = self.A_inv[arm]; Ax = Ai @ x
        self.A_inv[arm] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)      # Sherman-Morrison
        self.b[arm] += reward * x


class Lin(_DisjointLinear):
    def choose_arm(self, t: int, context=None) -> int:
        if context is None:
            raise ValueError("Lin requires a context vector")
        return int(np.argmax(self._theta() @ np.asarray(context, float)))

    def __repr__(self) -> str:
        return f"Lin(n_arms={self.n_arms}, context_dim={self.context_dim})"


class LinTS(_DisjointLinear):
    def __init__(self, n_arms: int, context_dim: int, v: float = 0.5) -> None:
        if v <= 0:
            raise ValueError("v must be > 0")
        self.v = float(v)
        super().__init__(n_arms, context_dim)

    def choose_arm(self, t: int, context=None) -> int:
        if context is None:
            raise ValueError("LinTS requires a context vector")
        x = np.asarray(context, float)
        th = self._theta()
        z = np.random.standard_normal((self.n_arms, self.context_dim))
        L = np.linalg.cholesky(self.A_inv + 1e-12 * np.eye(self.context_dim))
        sample = th + self.v * np.einsum("kij,kj->ki", L, z)
        return int(np.argmax(sample @ x))

    def __repr__(self) -> str:
        return f"LinTS(n_arms={self.n_arms}, context_dim={self.context_dim}, v={self.v})"
