"""
KL-UCB for Bernoulli rewards (Garivier & Cappe, 2011).

Index of arm a at round t:
    U_a(t) = max { q in [mu_hat_a, 1] : N_a * kl(mu_hat_a, q) <= ln t + c ln ln t }
with kl(p, q) the Bernoulli Kullback-Leibler divergence. c = 0 is the variant
Garivier and Cappe recommend in practice (c = 3 is the one their bound covers).
The index is found by bisection to 1e-6. Rewards outside [0, 1] are clipped,
so on Gaussian S2 the algorithm runs but is not covered by the theory.
"""
from __future__ import annotations

import math

import numpy as np

from mab_benchmark.core import BanditAlgorithm

_EPS = 1e-12


def kl_bernoulli(p: float, q: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS); q = min(max(q, _EPS), 1 - _EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_ucb_index(mean: float, n: float, level: float, tol: float = 1e-6) -> float:
    if n == 0:
        return 1.0
    lo, hi = mean, 1.0
    bound = level / n
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if kl_bernoulli(mean, mid) > bound:
            hi = mid
        else:
            lo = mid
    return lo


class KLUCB(BanditAlgorithm):
    def __init__(self, n_arms: int, c: float = 0.0) -> None:
        if c < 0:
            raise ValueError("c must be >= 0")
        self.c = float(c)
        super().__init__(n_arms)

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms); self.sums = np.zeros(self.n_arms)

    def choose_arm(self, t: int, context=None) -> int:
        z = np.flatnonzero(self.counts == 0)
        if z.size:
            return int(z[0])
        level = math.log(t) + (self.c * math.log(math.log(t)) if t > math.e else 0.0)
        means = self.sums / self.counts
        idx = [kl_ucb_index(means[a], self.counts[a], level) for a in range(self.n_arms)]
        return int(np.argmax(idx))

    def update(self, arm: int, reward: float, t: int, context=None) -> None:
        self.counts[arm] += 1; self.sums[arm] += min(max(float(reward), 0.0), 1.0)

    def __repr__(self) -> str:
        return f"KLUCB(n_arms={self.n_arms}, c={self.c})"
