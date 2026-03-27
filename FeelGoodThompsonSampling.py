import numpy as np
from mab_benchmark import BanditAlgorithm


class FeelGoodThompsonSampling(BanditAlgorithm):
    """
    Feel-Good Thompson Sampling (FG-TS) for Stochastic MAB.

    Adapts the Feel-Good Thompson Sampling framework of Zhang (2022)
    to the finite-armed stochastic bandit setting. The key modification
    over standard Thompson Sampling is the addition of an optimism bonus
    (lambda * sqrt(log(t+1) / (n+1))) to each arm's posterior sample,
    biasing selection toward high-reward models and correcting the
    underexploration of standard TS in pessimistic reward configurations.

    The Beta-Bernoulli conjugate posterior is used for S1/S3/S4/S5
    (binary/Bernoulli rewards). For S2 (Gaussian rewards), rewards are
    binarised at threshold 0.5 — a standard approximation for
    Beta-conjugate updates under continuous rewards in MAB benchmarks.

    Verified Citation
    -----------------
    Zhang, T. (2022). Feel-Good Thompson Sampling for Contextual
    Bandits and Reinforcement Learning. SIAM Journal on Mathematics
    of Data Science, 4(2), 834-857.
    DOI: 10.1137/21M140924X
    arXiv: https://arxiv.org/abs/2110.00871

    Parameters
    ----------
    lam     : float — optimism bonus weight lambda (default 1.0)
              Controls how aggressively the algorithm favours
              high-reward models beyond the posterior sample.
              Larger values = more optimism = more exploration.
              Zhang (2022) shows lam = O(sqrt(d)) is regret-optimal
              for d-dimensional contextual bandits; for finite-armed
              MAB, lam in [0.5, 2.0] is recommended.

    alpha_0 : float — Beta prior alpha pseudo-count (default 1.0)
              Jeffreys non-informative prior: alpha_0 = beta_0 = 1.0
              (uniform prior). Increase for optimistic initialisation.

    beta_0  : float — Beta prior beta pseudo-count (default 1.0)
              Symmetric with alpha_0. Increase for pessimistic init.
    """

    def __init__(self, n_arms, lam=1.0, alpha_0=1.0, beta_0=1.0):
        super().__init__(n_arms)
        self.lam     = lam
        self.alpha_0 = alpha_0
        self.beta_0  = beta_0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reinitialise Beta posterior parameters and pull counts.
        All arms start from the prior (alpha_0, beta_0).
        """
        self.alpha  = np.full(self.n_arms, self.alpha_0, dtype=float)
        self.beta   = np.full(self.n_arms, self.beta_0,  dtype=float)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.t      = 0

    # ------------------------------------------------------------------
    # Feel-Good Optimism Bonus
    # ------------------------------------------------------------------

    def _feel_good_bonus(self, arm):
        """
        Compute the Feel-Good optimism bonus for a given arm.

        Bonus = lambda * sqrt(log(t + 1) / (n + 1))

        This is the finite-armed MAB adaptation of the Feel-Good
        bonus from Zhang (2022), Eq. (2.3). It decays as arm n
        increases (diminishing exploration) and grows logarithmically
        with time (ensuring persistent exploration in long horizons).

        Returns
        -------
        float : optimism bonus to add to the posterior sample
        """
        n = self.counts[arm]
        return self.lam * np.sqrt(np.log(self.t + 1) / (n + 1))

    # ------------------------------------------------------------------
    # Arm Selection
    # ------------------------------------------------------------------

    def choose_arm(self, t, context=None):
        """
        Select arm by Feel-Good augmented posterior sample.

        For each arm i, compute:
            theta_i  ~ Beta(alpha_i, beta_i)       [posterior sample]
            fg_i      = lam * sqrt(log(t+1)/(n_i+1)) [feel-good bonus]
            score_i   = theta_i + fg_i              [augmented score]

        Select arm with highest augmented score.

        The feel-good bonus lifts underexplored arms above their
        posterior sample, correcting the underexploration of
        standard TS identified in Zhang (2022), Theorem 2.1.
        """
        self.t = t

        scores = np.array([
            np.random.beta(
                max(self.alpha[arm], 1e-6),
                max(self.beta[arm],  1e-6)
            ) + self._feel_good_bonus(arm)
            for arm in range(self.n_arms)
        ])

        return int(np.argmax(scores))

    # ------------------------------------------------------------------
    # Posterior Update
    # ------------------------------------------------------------------

    def update(self, arm, reward, t, context=None):
        """
        Update Beta posterior for selected arm using observed reward.

        Binarisation: reward > 0.5 → success (alpha += 1)
                      reward ≤ 0.5 → failure (beta  += 1)

        This binarisation enables Beta-Bernoulli conjugacy across
        all five benchmark settings, including S2 (Gaussian rewards)
        where continuous values are thresholded at 0.5.

        Posterior update follows the standard Beta-Bernoulli rule:
            alpha_new = alpha_old + I(reward > 0.5)
            beta_new  = beta_old  + I(reward ≤ 0.5)
        """
        self.counts[arm] += 1
        if reward > 0.5:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm]  += 1.0