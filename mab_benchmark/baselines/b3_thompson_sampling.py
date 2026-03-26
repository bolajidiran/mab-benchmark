"""
B3: Thompson Sampling
======================

Bayesian probability-matching algorithm.  At each round, sample a
reward estimate for each arm from its posterior distribution and select
the arm with the highest sample.

Two conjugate models are implemented:
  - Beta-Bernoulli  for S1, S3, S5 (binary rewards)
  - Normal-InvGamma for S2          (Gaussian rewards)

Role in benchmark
-----------------
Theoretically equivalent to UCB1 in O(log T) regret, but often
empirically superior.  Papers that compare only against UCB1 and
omit Thompson Sampling tend to report misleadingly large advantages.
Mandating TS in the comparison pool prevents this.

Source: GAP5 Contested Zone B; BenchmarkSpec Gap A-2.
"""

from __future__ import annotations
import numpy as np
from mab_benchmark.core import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    """
    Baseline B3: Thompson Sampling.

    Automatically selects the conjugate model based on reward_type.

    Parameters
    ----------
    n_arms       : int   — number of arms K
    context_dim  : int   — ignored
    reward_type  : str   — 'bernoulli' (default) or 'gaussian'

    Beta-Bernoulli model (reward_type='bernoulli')
    -----------------------------------------------
    Prior: Beta(alpha_0, beta_0) = Beta(1, 1) = Uniform(0,1)
    Update: alpha_a += r_t, beta_a += (1 - r_t)
    Sample:  theta_a ~ Beta(alpha_a, beta_a)

    Normal-InvGamma model (reward_type='gaussian')
    -----------------------------------------------
    Prior: mu_a ~ N(mu_0, sigma^2 / kappa_0), sigma^2 ~ InvGamma(a_0, b_0)
    Uses a Normal-InvGamma conjugate update with a vague prior.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int = 0,
        reward_type: str = "bernoulli",
    ) -> None:
        if reward_type not in ("bernoulli", "gaussian"):
            raise ValueError(
                f"reward_type must be 'bernoulli' or 'gaussian', "
                f"got '{reward_type}'."
            )
        self.reward_type = reward_type
        super().__init__(n_arms, context_dim)

    # ------------------------------------------------------------------
    # Reset (initialise / re-initialise prior)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if self.reward_type == "bernoulli":
            # Beta(1,1) prior — equivalent to Uniform(0,1)
            self.alpha = np.ones(self.n_arms, dtype=float)
            self.beta  = np.ones(self.n_arms, dtype=float)
        else:
            # Normal-InvGamma vague prior
            self.n_obs  = np.zeros(self.n_arms, dtype=float)
            self.sum_r  = np.zeros(self.n_arms, dtype=float)
            self.sum_r2 = np.zeros(self.n_arms, dtype=float)
            # Hyperparameters
            self.mu0     = 0.5    # prior mean
            self.kappa0  = 1.0    # prior pseudo-observations
            self.alpha0  = 1.0    # InvGamma shape
            self.beta0   = 1.0    # InvGamma rate

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def choose_arm(
        self,
        t: int,
        context: np.ndarray | None = None,
    ) -> int:
        if self.reward_type == "bernoulli":
            samples = np.random.beta(self.alpha, self.beta)
        else:
            samples = self._sample_gaussian()
        return int(np.argmax(samples))

    def update(
        self,
        arm: int,
        reward: float,
        t: int,
        context: np.ndarray | None = None,
    ) -> None:
        if self.reward_type == "bernoulli":
            self.alpha[arm] += reward
            self.beta[arm]  += 1.0 - reward
        else:
            self.n_obs[arm]  += 1.0
            self.sum_r[arm]  += reward
            self.sum_r2[arm] += reward ** 2

    # ------------------------------------------------------------------
    # Gaussian posterior sampling
    # ------------------------------------------------------------------

    def _sample_gaussian(self) -> np.ndarray:
        """
        Sample from the Normal-InvGamma posterior for each arm.

        For arm a with n observations, sample_mean, and sample variance:
            kappa_n = kappa0 + n
            mu_n    = (kappa0 * mu0 + sum_r) / kappa_n
            alpha_n = alpha0 + n / 2
            beta_n  = beta0 + 0.5 * (sum_r2 - n * mu_n^2)
                           + 0.5 * kappa0 * n * (mu_n - mu0)^2 / kappa_n
        Then: sigma^2 ~ InvGamma(alpha_n, beta_n)
              mu      ~ N(mu_n, sigma^2 / kappa_n)
        """
        samples = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            n = self.n_obs[a]
            if n == 0:
                # No data: sample from prior
                sigma2 = 1.0 / np.random.gamma(self.alpha0, 1.0 / self.beta0)
                samples[a] = np.random.normal(self.mu0, np.sqrt(sigma2))
                continue

            kappa_n = self.kappa0 + n
            mu_n    = (self.kappa0 * self.mu0 + self.sum_r[a]) / kappa_n
            alpha_n = self.alpha0 + n / 2.0
            beta_n  = (
                self.beta0
                + 0.5 * (self.sum_r2[a] - n * mu_n ** 2)
                + 0.5 * self.kappa0 * n * (mu_n - self.mu0) ** 2 / kappa_n
            )
            beta_n  = max(beta_n, 1e-8)   # numerical safety

            sigma2    = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)
            samples[a] = np.random.normal(mu_n, np.sqrt(sigma2 / kappa_n))

        return samples

    def __repr__(self) -> str:
        return (
            f"ThompsonSampling(n_arms={self.n_arms}, "
            f"reward_type='{self.reward_type}')"
        )
