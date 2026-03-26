"""
S4: Contextual Bandit — Yahoo! R6B Replayer and Synthetic Fallback
===================================================================

Source: BenchmarkSpec_Vermorel2005.docx, Gap E-5; GAP5 Pillar 4;
        TwoTierProtocol_GAP5.docx, Pillar 2; Li et al. (2011).

What it is
----------
Each round the algorithm sees a feature vector (user context) and must
choose which news article (arm) to recommend.  The reward is a binary
click signal.  Unlike S1-S3, arms are not interchangeable: the same
arm may be good for one context and bad for another.

Two variants
------------
YahooReplayer          — uses the real Yahoo! R6B dataset (requires
                          a data-access agreement from Yahoo! Webscope).
SyntheticContextualBandit — uses a synthetic linear contextual bandit
                          (fully reproducible, no data required).
                          This is the approved fallback when Yahoo! R6B
                          is not available (documented in MAB_BenchmarkSuite).

Replayer method (Li et al., 2011)
----------------------------------
The replayer accepts only rounds where the logged arm matches the
algorithm's chosen arm, producing an unbiased CTR estimate.  This is
the mandatory evaluation method for S4 (TwoTierProtocol, Pillar 2).

Parameters
----------
K  ~ 270 (Yahoo! R6B articles)
T in {10_000, 50_000}
r_t in {0, 1}  (click / no click)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ==========================================================================
# Yahoo! R6B Replayer
# ==========================================================================

class YahooReplayer:
    """
    Offline replayer evaluation on the Yahoo! Front Page dataset (R6B).

    The dataset must be downloaded separately from Yahoo! Webscope:
        https://webscope.sandbox.yahoo.com/
    Expected format: tab-separated with columns
        [timestamp, article_id, click, user_features..., article_features...]

    Parameters
    ----------
    data_path : str | Path
        Path to the R6B data file.
    T : int
        Number of accepted interactions to simulate.
        Must be in {10_000, 50_000}.
    max_pool : int
        How many logged rows to read for filtering (default: T * 15,
        accounting for typical replayer acceptance rate of ~5-10%).

    Usage
    -----
    >>> replayer = YahooReplayer("path/to/R6B.txt", T=10_000)
    >>> results  = replayer.run(my_algorithm)
    >>> print(results["ctr"])
    """

    VALID_T = {10_000, 50_000}

    def __init__(
        self,
        data_path: str | Path,
        T: int = 10_000,
        max_pool: int | None = None,
    ) -> None:
        if not _HAS_PANDAS:
            raise ImportError(
                "pandas is required for YahooReplayer. "
                "Install with: pip install pandas"
            )
        if T not in self.VALID_T:
            raise ValueError(f"S4 requires T in {self.VALID_T}, got {T}.")

        self.T        = T
        self.data_path = Path(data_path)
        self._pool    = max_pool or (T * 15)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}.\n"
                "Download from: https://webscope.sandbox.yahoo.com/\n"
                "As a fallback, use SyntheticContextualBandit."
            )

        self._df = self._load()
        self.n_arms = self._df["arm"].nunique()
        self._arm_index = {
            arm_id: idx
            for idx, arm_id in enumerate(sorted(self._df["arm"].unique()))
        }
        self.context_dim = len(self._parse_context(self._df.iloc[0]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> "pd.DataFrame":
        rows = []
        with open(self.data_path) as f:
            for i, line in enumerate(f):
                if i >= self._pool:
                    break
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                rows.append({
                    "t":      int(parts[0]),
                    "arm":    int(parts[1]),
                    "reward": int(parts[2]),
                    "raw":    parts[3:],
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_context(row: "pd.Series") -> np.ndarray:
        """Parse raw feature strings into a float array."""
        raw = row.get("raw", [])
        features = []
        for token in raw:
            try:
                # format: "feature_id:value"
                val = float(token.split(":")[-1])
                features.append(val)
            except (ValueError, IndexError):
                continue
        return np.array(features, dtype=float) if features else np.zeros(6)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, algorithm: "BanditAlgorithm") -> dict:
        """
        Run one replayer pass.

        The algorithm must already be initialised with n_arms=self.n_arms.
        For each logged row, the algorithm chooses an arm.  If it matches
        the logged arm, the interaction is accepted and the algorithm
        updates on the observed reward.

        Returns
        -------
        dict with keys:
            cumulative_reward : float
            n_accepted        : int
            ctr               : float   (cumulative_reward / n_accepted)
            cumulative_regret : list[float]   regret curve, length n_accepted
        """
        n_accepted      = 0
        cum_reward      = 0.0
        regret_curve    = []
        best_ctr_so_far = 0.0      # running oracle CTR estimate

        for _, row in self._df.iterrows():
            if n_accepted >= self.T:
                break

            context = self._parse_context(row)
            logged_arm_raw = int(row["arm"])
            logged_arm     = self._arm_index.get(logged_arm_raw, 0)

            chosen = algorithm.choose_arm(n_accepted + 1, context)

            # Replayer rule: only accept matching rounds
            if chosen == logged_arm:
                reward = float(row["reward"])
                algorithm.update(chosen, reward, n_accepted + 1, context)
                cum_reward += reward
                n_accepted += 1

                # Running oracle estimate
                best_ctr_so_far = max(best_ctr_so_far, reward)
                instantaneous_regret = best_ctr_so_far - reward
                regret_curve.append(instantaneous_regret)

        ctr = cum_reward / n_accepted if n_accepted > 0 else 0.0
        return {
            "cumulative_reward": cum_reward,
            "n_accepted":        n_accepted,
            "ctr":               ctr,
            "cumulative_regret": np.cumsum(regret_curve).tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"YahooReplayer(T={self.T}, n_arms={self.n_arms}, "
            f"context_dim={self.context_dim})"
        )


# ==========================================================================
# Synthetic Contextual Bandit (approved fallback for S4)
# ==========================================================================

class SyntheticContextualBandit:
    """
    Synthetic linear contextual bandit — approved fallback for S4.

    Use this when the Yahoo! R6B dataset is not available.
    Results from this environment are labelled
    'S4_SYNTHETIC' (not 'S4_YAHOO') in leaderboard submissions.

    Model
    -----
    Each arm a has a hidden parameter theta_a in R^d.
    At round t, context x_t ~ N(0, I_d).
    Expected reward: mu_a(x_t) = sigmoid(x_t @ theta_a).
    Observed reward: r_t ~ Bernoulli(mu_a(x_t)).

    Parameters
    ----------
    K            : int   — number of arms (default 50)
    T            : int   — time horizon (10_000 or 50_000)
    context_dim  : int   — dimension of context vector (default 10)
    seed         : int   — random seed
    """

    VALID_T = {10_000, 50_000}

    def __init__(
        self,
        K: int = 50,
        T: int = 10_000,
        context_dim: int = 10,
        seed: int = 0,
    ) -> None:
        if T not in self.VALID_T:
            raise ValueError(f"S4 requires T in {self.VALID_T}, got {T}.")

        self.K           = K
        self.T           = T
        self.context_dim = context_dim
        self.seed        = seed
        self.n_arms      = K

        rng = np.random.default_rng(seed)
        # Each arm has a d-dimensional weight vector
        self.thetas = rng.normal(0.0, 0.5, size=(K, context_dim))
        # Pre-generate all contexts for reproducibility
        self._contexts = rng.normal(0.0, 1.0, size=(T, context_dim))
        self._t = 0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def reset(self) -> None:
        self._t = 0

    def context(self, t: int) -> np.ndarray:
        """Return the context vector for round t (1-indexed)."""
        return self._contexts[t - 1]

    def expected_reward(self, arm: int, context: np.ndarray) -> float:
        return float(self._sigmoid(context @ self.thetas[arm]))

    def pull(self, arm: int, rng_or_t, rng: np.random.Generator | None = None) -> float:
        # Accept both pull(arm, rng) and pull(arm, t, rng) signatures
        if rng is None:
            rng = rng_or_t
            t = self._t + 1
        else:
            t = int(rng_or_t)
        ctx = self.context(t)
        mu  = self.expected_reward(arm, ctx)
        return float(rng.uniform() < mu)

    def mu_star(self, t: int) -> float:
        ctx = self.context(t)
        return max(self.expected_reward(a, ctx) for a in range(self.K))

    def regret_increment(self, arm: int, t: int) -> float:
        ctx = self.context(t)
        return self.mu_star(t) - self.expected_reward(arm, ctx)

    def __repr__(self) -> str:
        return (
            f"SyntheticContextualBandit(K={self.K}, T={self.T}, "
            f"context_dim={self.context_dim}, seed={self.seed})"
        )
