import numpy as np
from mab_benchmark.core import BanditAlgorithm


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
    clipped to [0, 1] before the update, exactly as in B3.

    Naming note: Zhang (2022) defines Feel-Good TS by adding the term
    -lambda * min(b, f_theta(x)) to the loss inside the posterior. The
    count-based bonus used here, lambda * sqrt(log(t+1)/(n+1)), is a
    UCB-style optimism bonus added to the posterior sample; it is an
    adaptation inspired by Zhang (2022), not the estimator of that paper.

    Verified Citation
    -----------------
    Zhang, T. (2022). Feel-Good Thompson Sampling for Contextual
    Bandits and Reinforcement Learning. SIAM Journal on Mathematics
    of Data Science, 4(2), 834-857.
    DOI: 10.1137/21M140924X
    arXiv: https://arxiv.org/abs/2110.00871

    Parameters
    ----------
    lam     : float — optimism bonus weight lambda (default 0.5)
              Larger values = more optimism = more exploration;
              lam = 0 reproduces B3 Thompson Sampling exactly.

    alpha_0 : float — Beta prior alpha pseudo-count (default 1.0)
              alpha_0 = beta_0 = 1.0 is the uniform prior.
              Increase for optimistic initialisation.

    beta_0  : float — Beta prior beta pseudo-count (default 1.0)
              Symmetric with alpha_0. Increase for pessimistic init.
    """

    def __init__(self, n_arms, lam=0.5, alpha_0=1.0, beta_0=1.0):
        # Parameters must exist before super().__init__(), which calls reset().
        self.lam     = lam
        self.alpha_0 = alpha_0
        self.beta_0  = beta_0
        super().__init__(n_arms)

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

        r = clip(reward, 0, 1);  alpha += r;  beta += 1 - r
        For Bernoulli rewards (S1, S3, S5, S4) this is the exact conjugate
        update; for S2 it is the clipped approximation also used by B3.
        """
        # Same update as the benchmarked notebook class (FeelGoodTS): clip to
        # [0, 1] and add fractional counts; identical for Bernoulli rewards.
        r = float(np.clip(reward, 0.0, 1.0))
        self.counts[arm] += 1
        self.alpha[arm]  += r
        self.beta[arm]   += 1.0 - r