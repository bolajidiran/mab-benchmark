"""
mab_benchmark.runner
====================

BenchmarkRunner — evaluates any BanditAlgorithm on the full benchmark
suite with the mandatory number of independent runs.

Source: GAP5_EvalFramework.docx, Step 5;
        BenchmarkSpec_Vermorel2005.docx, Gaps I-1, I-2.

Design
------
The runner enforces all five benchmark settings (S1-S5), all mandatory
(K, T) combinations, and the pre-registered seed table.  No run uses
the same seed as any other run for any setting.  The seed table is
fixed across all papers so that results are directly comparable.

The runner returns raw arrays of shape (n_runs,) containing cumulative
regret per run.  Statistical analysis is handled by StatisticalAnalyser.
"""

from __future__ import annotations
from typing import Type
import numpy as np

from mab_benchmark.core import BanditAlgorithm
from mab_benchmark.environments import (
    BernoulliBandit,
    GaussianBandit,
    PiecewiseStationaryBandit,
    SyntheticContextualBandit,
    ColdStartBandit,
)
from mab_benchmark.baselines import (
    MANDATORY_BASELINES,
    CONTEXTUAL_BASELINE,
)


# Pre-registered seed table (200 seeds; never change once published)
SEED_TABLE: list[int] = list(range(1000, 1200))


class BenchmarkRunner:
    """
    Runs any BanditAlgorithm on all five benchmark settings.

    Parameters
    ----------
    n_runs : int
        Number of independent runs per (algorithm, setting, K, T).
        Must be >= 30 (Requirement 1 from GAP5 Pillar 3).

    Usage
    -----
    >>> runner  = BenchmarkRunner(n_runs=30)
    >>> results = runner.full_suite(UCB1, {})
    >>> # results is a dict: setting_key -> np.ndarray(n_runs,)

    To run a single setting:
    >>> regrets = runner.run_setting(
    ...     UCB1, {},
    ...     BernoulliBandit, {"K": 10, "T": 500}
    ... )
    """

    def __init__(self, n_runs: int = 30) -> None:
        if n_runs < 30:
            raise ValueError(
                f"n_runs must be >= 30 (GAP5 Pillar 3). Got {n_runs}.\n"
                "This requirement ensures the Wilcoxon signed-rank test "
                "has adequate statistical power."
            )
        if n_runs > len(SEED_TABLE):
            raise ValueError(
                f"n_runs={n_runs} exceeds the pre-registered seed table "
                f"({len(SEED_TABLE)} seeds)."
            )
        self.n_runs = n_runs

    # ------------------------------------------------------------------
    # Core method: run one (algorithm, environment) combination
    # ------------------------------------------------------------------

    def run_setting(
        self,
        AlgClass: Type[BanditAlgorithm],
        alg_kwargs: dict,
        EnvClass: type,
        env_kwargs: dict,
        is_stationary: bool = True,
        use_context: bool = False,
    ) -> np.ndarray:
        """
        Evaluate AlgClass on EnvClass for self.n_runs independent runs.

        Returns
        -------
        np.ndarray of shape (n_runs,)
            Cumulative regret at horizon T for each run.
        """
        regrets = np.zeros(self.n_runs, dtype=float)

        for i in range(self.n_runs):
            seed = SEED_TABLE[i]
            rng  = np.random.default_rng(seed)

            env = EnvClass(**env_kwargs, seed=seed)
            T   = env.T

            alg = AlgClass(
                n_arms=env.K,
                **alg_kwargs
            )

            cum_regret = 0.0

            for t in range(1, T + 1):
                # Contextual settings: get context from environment
                if use_context and hasattr(env, "context"):
                    ctx = env.context(t)
                else:
                    ctx = None

                arm = alg.choose_arm(t, ctx)

                # Non-stationary: pass t to pull so env knows segment
                if not is_stationary:
                    reward = env.pull(arm, t, rng)
                    reg    = env.regret_increment(arm, t)
                elif use_context:
                    reward = env.pull(arm, rng)
                    reg    = env.regret_increment(arm, t)
                else:
                    reward = env.pull(arm, rng)
                    reg    = env.regret_increment(arm)

                alg.update(arm, reward, t, ctx)
                cum_regret += reg

            regrets[i] = cum_regret

        return regrets

    # ------------------------------------------------------------------
    # Full suite: all five settings
    # ------------------------------------------------------------------

    def full_suite(
        self,
        AlgClass: Type[BanditAlgorithm],
        alg_kwargs: dict,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Run all five benchmark settings and return results dict.

        Keys follow the convention  "S{n}_K{k}_T{t}".

        Parameters
        ----------
        AlgClass   : BanditAlgorithm subclass
        alg_kwargs : keyword arguments passed to AlgClass (besides n_arms)
        verbose    : print progress to stdout

        Returns
        -------
        dict mapping setting_key -> np.ndarray(n_runs,) cumulative regrets
        """
        results = {}

        # --- S1: Stationary Bernoulli -----------------------------------
        for K in (10, 50, 200):
            for T in (500, 2_000, 10_000):
                key = f"S1_K{K}_T{T}"
                if verbose:
                    print(f"  Running {key} ...", flush=True)
                results[key] = self.run_setting(
                    AlgClass, alg_kwargs,
                    BernoulliBandit, {"K": K, "T": T},
                )

        # --- S2: Stationary Gaussian ------------------------------------
        for K in (10, 50, 200):
            for T in (1_000, 10_000):
                key = f"S2_K{K}_T{T}"
                if verbose:
                    print(f"  Running {key} ...", flush=True)
                results[key] = self.run_setting(
                    AlgClass, alg_kwargs,
                    GaussianBandit, {"K": K, "T": T},
                )

        # --- S3: Piecewise-Stationary -----------------------------------
        key = "S3_K10_T10000"
        if verbose:
            print(f"  Running {key} ...", flush=True)
        results[key] = self.run_setting(
            AlgClass, alg_kwargs,
            PiecewiseStationaryBandit, {"K": 10, "T": 10_000},
            is_stationary=False,
        )

        # --- S4: Contextual (Synthetic fallback) -----------------------
        for T in (10_000, 50_000):
            key = f"S4_SYNTHETIC_K50_T{T}"
            if verbose:
                print(f"  Running {key} (synthetic fallback) ...",
                      flush=True)
            results[key] = self.run_setting(
                AlgClass, alg_kwargs,
                SyntheticContextualBandit, {"K": 50, "T": T,
                                             "context_dim": 10},
                use_context=True,
            )

        # --- S5: Cold-Start --------------------------------------------
        for T in (50, 100, 200):
            key = f"S5_K500_T{T}"
            if verbose:
                print(f"  Running {key} ...", flush=True)
            results[key] = self.run_setting(
                AlgClass, alg_kwargs,
                ColdStartBandit, {"K": 500, "T": T},
            )

        return results

    # ------------------------------------------------------------------
    # Baseline sweep: run all mandatory baselines for comparison
    # ------------------------------------------------------------------

    def baseline_sweep(
        self,
        verbose: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Run all four mandatory baselines on the full suite.

        Returns
        -------
        dict: alg_name -> {setting_key -> regret_array}
        """
        all_results = {}
        for name, AlgClass in MANDATORY_BASELINES.items():
            if verbose:
                print(f"\nBaseline: {name}", flush=True)
            all_results[name] = self.full_suite(
                AlgClass, {}, verbose=verbose
            )
        return all_results

    def __repr__(self) -> str:
        return f"BenchmarkRunner(n_runs={self.n_runs})"
