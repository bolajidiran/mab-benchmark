"""
S4-SIM: the bias-free Tier 1 environment of S4, a Bernoulli bandit whose K = 80
arm means are the item click-through rates of the Open Bandit Dataset's
uniform-random log (Saito et al., 2021).

It keeps the domain of the S4 replay (same items, same CTR scale of 0.1 to 0.7
per cent, same horizon) and removes everything that can bias a replay: there is
no logging policy and every round's reward is drawn independently. The BCS on
S4 compares the replay ranking with the ranking on this environment (condition
c1), so a Tier 1 that is itself biased cannot certify a biased Tier 2.

means: the 80 CTRs, computed by obd_preprocessing (cache/clean_random.npz), or
passed explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "obd_preprocessing" / "cache" / "clean_random.npz"


def obd_item_ctr(path: Path = CACHE) -> np.ndarray:
    r = np.load(path)
    item = r["item"].astype(int); click = r["click"].astype(float)
    K = int(item.max()) + 1
    return np.bincount(item, weights=click, minlength=K) / np.bincount(item, minlength=K)


class OBDCalibratedBandit:
    def __init__(self, K: int = 80, T: int = 14_315, seed: int = 0, means=None) -> None:
        m = obd_item_ctr() if means is None else np.asarray(means, float)
        if m.size != K:
            raise ValueError(f"expected {K} item CTRs, got {m.size}")
        self.means, self.K, self.T, self.seed = m, K, T, seed
        self.mu_star = float(m.max()); self.a_star = int(m.argmax())

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        return float(rng.uniform() < self.means[arm])

    def regret_increment(self, arm: int) -> float:
        return self.mu_star - self.means[arm]

    def expected_random_regret(self) -> float:
        return float(self.T * (self.mu_star - self.means.mean()))

    def __repr__(self) -> str:
        return f"OBDCalibratedBandit(K={self.K}, T={self.T}, mu*={self.mu_star:.5f})"
