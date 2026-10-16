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

# Identifier of the random-stream layout used by run_setting (see run_streams).
STREAM_LAYOUT = "1.2.0"


def run_streams(seed: int) -> tuple[np.random.Generator, int]:
    """Independent random streams for one run, derived from one table seed.

    The environment draws its arm means and contexts from default_rng(seed)
    internally. The reward stream and the
    algorithm's own randomness come from two child streams of
    SeedSequence([seed, 1]), so they are reproducible and statistically
    independent of the environment draws.
    """
    reward_ss, alg_ss = np.random.SeedSequence([seed, 1]).spawn(2)
    alg_seed = int(alg_ss.generate_state(1, dtype=np.uint32)[0])
    return np.random.default_rng(reward_ss), alg_seed


def _make_algorithm(AlgClass, n_arms: int, alg_kwargs: dict, alg_seed: int, context_dim: int = 0):
    """Instantiate an algorithm, passing context_dim and seed when it accepts them."""
    import inspect
    params = inspect.signature(AlgClass.__init__).parameters
    kw = dict(alg_kwargs)
    if context_dim and "context_dim" in params and "context_dim" not in kw:
        kw["context_dim"] = context_dim
    if "seed" in params and "seed" not in kw:
        kw["seed"] = alg_seed
    return AlgClass(n_arms=n_arms, **kw)


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

    def __init__(self, n_runs: int = 30, seeds=None, enforce_min_runs: bool = True) -> None:
        """
        n_runs           : number of independent runs (>= 30 unless enforce_min_runs=False).
        seeds            : optional explicit seed list (default: the pre-registered SEED_TABLE).
                           Used by the H1 comparability study, which needs fresh, independent
                           seeds for every simulated study.
        enforce_min_runs : False only to emulate under-powered practice (H1 study);
                           certification always uses the default.
        """
        self.seeds = list(SEED_TABLE) if seeds is None else [int(x) for x in seeds]
        if enforce_min_runs and n_runs < 30:
            raise ValueError(
                f"n_runs must be >= 30 (GAP5 Pillar 3). Got {n_runs}.\n"
                "This requirement ensures the Wilcoxon signed-rank test "
                "has adequate statistical power."
            )
        if n_runs > len(self.seeds):
            raise ValueError(
                f"n_runs={n_runs} exceeds the seed list "
                f"({len(self.seeds)} seeds)."
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
        return self._run(AlgClass, alg_kwargs, EnvClass, env_kwargs,
                         is_stationary, use_context, record_curve=False)

    def run_setting_curves(
        self,
        AlgClass: Type[BanditAlgorithm],
        alg_kwargs: dict,
        EnvClass: type,
        env_kwargs: dict,
        is_stationary: bool = True,
        use_context: bool = False,
    ) -> np.ndarray:
        """Same runs and random streams as run_setting, returning the full
        cumulative-regret curve: array of shape (n_runs, T); the last column
        equals run_setting's output exactly."""
        return self._run(AlgClass, alg_kwargs, EnvClass, env_kwargs,
                         is_stationary, use_context, record_curve=True)

    def _run(self, AlgClass, alg_kwargs, EnvClass, env_kwargs,
             is_stationary, use_context, record_curve):
        regrets = np.zeros(self.n_runs, dtype=float)
        curves = None

        for i in range(self.n_runs):
            seed = self.seeds[i]
            rng, alg_seed = run_streams(seed)

            env = EnvClass(**env_kwargs, seed=seed)
            T   = env.T
            if record_curve and curves is None:
                curves = np.zeros((self.n_runs, T))

            # Algorithms in this package draw from numpy's global stream
            # (np.random.*); seed it per run so every run is reproducible.
            np.random.seed(alg_seed)
            alg = _make_algorithm(AlgClass, env.K, alg_kwargs, alg_seed,
                                  getattr(env, "context_dim", 0) if use_context else 0)

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
                    reward = env.pull(arm, t, rng)      # reward drawn at round t's context
                    reg    = env.regret_increment(arm, t)
                else:
                    reward = env.pull(arm, rng)
                    reg    = env.regret_increment(arm)

                alg.update(arm, reward, t, ctx)
                cum_regret += reg
                if record_curve:
                    curves[i, t - 1] = cum_regret

            regrets[i] = cum_regret

        return curves if record_curve else regrets

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
