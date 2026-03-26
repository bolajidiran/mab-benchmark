"""
tests/test_mab_benchmark.py
============================

Complete test suite for the mab_benchmark package.

Run with:
    pytest tests/ -v

Or with coverage:
    pytest tests/ -v --cov=mab_benchmark --cov-report=term-missing
"""

import numpy as np
import pytest

from mab_benchmark.core import BanditAlgorithm
from mab_benchmark.environments import (
    BernoulliBandit,
    GaussianBandit,
    PiecewiseStationaryBandit,
    SyntheticContextualBandit,
    ColdStartBandit,
)
from mab_benchmark.baselines import (
    RandomPolicy, EpsilonGreedy, UCB1, ThompsonSampling, LinUCB
)
from mab_benchmark.runner import BenchmarkRunner, SEED_TABLE
from mab_benchmark.analysis.statistics import StatisticalAnalyser
from mab_benchmark.analysis.bcs import BCSCalculator
from mab_benchmark.analysis.leaderboard import LeaderboardSubmitter


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def small_runner():
    """Runner with n_runs=30 (minimum) for fast tests."""
    return BenchmarkRunner(n_runs=30)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ==========================================================================
# 1. Core interface tests
# ==========================================================================

class TestBanditAlgorithmInterface:

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BanditAlgorithm(n_arms=5)

    def test_concrete_subclass_must_implement_both_methods(self):
        class Incomplete(BanditAlgorithm):
            def choose_arm(self, t, context=None):
                return 0
            # Missing update

        with pytest.raises(TypeError):
            Incomplete(n_arms=5)

    def test_valid_subclass_instantiates(self):
        class Greedy(BanditAlgorithm):
            def reset(self):
                self.means = np.zeros(self.n_arms)
                self.counts = np.zeros(self.n_arms, dtype=int)

            def choose_arm(self, t, context=None):
                return int(np.argmax(self.means))

            def update(self, arm, reward, t, context=None):
                self.counts[arm] += 1
                n = self.counts[arm]
                self.means[arm] += (reward - self.means[arm]) / n

        alg = Greedy(n_arms=5)
        assert alg.n_arms == 5
        assert alg.context_dim == 0

    def test_invalid_n_arms_raises(self):
        class Dummy(BanditAlgorithm):
            def choose_arm(self, t, context=None): return 0
            def update(self, arm, reward, t, context=None): pass

        with pytest.raises(ValueError):
            Dummy(n_arms=0)


# ==========================================================================
# 2. Environment tests
# ==========================================================================

class TestBernoulliBandit:

    def test_valid_construction(self):
        env = BernoulliBandit(K=10, T=500, seed=0)
        assert env.K == 10
        assert env.T == 500
        assert len(env.means) == 10
        assert all(0 <= m <= 1 for m in env.means)
        assert 0 <= env.a_star < 10

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            BernoulliBandit(K=7, T=500, seed=0)

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            BernoulliBandit(K=10, T=999, seed=0)

    def test_pull_returns_binary(self, rng):
        env = BernoulliBandit(K=10, T=500, seed=0)
        rewards = [env.pull(0, rng) for _ in range(1000)]
        assert all(r in (0.0, 1.0) for r in rewards)

    def test_pull_law_of_large_numbers(self, rng):
        env = BernoulliBandit(K=10, T=10_000, seed=1)
        n = 5000
        empirical_mean = sum(env.pull(0, rng) for _ in range(n)) / n
        assert abs(empirical_mean - env.means[0]) < 0.05

    def test_regret_increment_optimal_arm(self):
        env = BernoulliBandit(K=10, T=500, seed=0)
        assert env.regret_increment(env.a_star) == pytest.approx(0.0)

    def test_regret_increment_suboptimal_arm(self):
        env = BernoulliBandit(K=10, T=500, seed=0)
        for arm in range(env.K):
            assert env.regret_increment(arm) >= 0.0

    def test_invalid_arm_raises(self, rng):
        env = BernoulliBandit(K=10, T=500, seed=0)
        with pytest.raises(IndexError):
            env.pull(10, rng)

    def test_reproducibility(self):
        env1 = BernoulliBandit(K=10, T=500, seed=42)
        env2 = BernoulliBandit(K=10, T=500, seed=42)
        np.testing.assert_array_equal(env1.means, env2.means)

    def test_different_seeds_differ(self):
        env1 = BernoulliBandit(K=10, T=500, seed=0)
        env2 = BernoulliBandit(K=10, T=500, seed=1)
        assert not np.all(env1.means == env2.means)


class TestGaussianBandit:

    def test_valid_construction(self):
        env = GaussianBandit(K=50, T=1000, seed=0)
        assert env.K == 50
        assert env.T == 1000

    def test_pull_is_continuous(self, rng):
        env = GaussianBandit(K=10, T=1000, seed=0)
        rewards = [env.pull(0, rng) for _ in range(100)]
        assert len(set(rewards)) > 1  # not all the same

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            GaussianBandit(K=5, T=1000, seed=0)
        with pytest.raises(ValueError):
            GaussianBandit(K=10, T=500, seed=0)


class TestPiecewiseStationaryBandit:

    def test_construction(self):
        env = PiecewiseStationaryBandit(K=10, T=10_000, seed=0)
        assert env.K == 10
        assert len(env.segment_means) == 11

    def test_segment_changes_at_change_points(self):
        env = PiecewiseStationaryBandit(K=10, T=10_000, seed=0)
        assert env.current_segment(999) == 0
        assert env.current_segment(1000) == 0
        assert env.current_segment(1001) == 1

    def test_mu_star_changes_across_segments(self):
        env = PiecewiseStationaryBandit(K=10, T=10_000, seed=0)
        stars = [env.mu_star(t) for t in [500, 1500, 2500, 3500]]
        # At least one segment should differ from the others
        assert len(set(round(s, 4) for s in stars)) > 1

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            PiecewiseStationaryBandit(K=5, T=10_000, seed=0)


class TestColdStartBandit:

    def test_construction(self):
        env = ColdStartBandit(K=500, T=100, seed=0)
        assert env.K == 500
        assert env.T == 100

    def test_k_greater_than_t(self):
        env = ColdStartBandit(K=500, T=200, seed=0)
        assert env.K > env.T

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            ColdStartBandit(K=100, T=50, seed=0)

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            ColdStartBandit(K=500, T=300, seed=0)


class TestSyntheticContextualBandit:

    def test_construction(self):
        env = SyntheticContextualBandit(K=50, T=10_000, context_dim=10, seed=0)
        assert env.K == 50
        assert env.context_dim == 10

    def test_context_shape(self):
        env = SyntheticContextualBandit(K=50, T=10_000, context_dim=10, seed=0)
        ctx = env.context(1)
        assert ctx.shape == (10,)

    def test_reward_binary(self):
        env = SyntheticContextualBandit(K=5, T=10_000, context_dim=4, seed=0)
        rng = np.random.default_rng(0)
        for t in range(1, 11):
            r = env.pull(0, t, rng)
            assert r in (0.0, 1.0)


# ==========================================================================
# 3. Baseline tests
# ==========================================================================

class TestRandomPolicy:

    def test_arm_in_range(self):
        alg = RandomPolicy(n_arms=10)
        for t in range(1, 101):
            arm = alg.choose_arm(t)
            assert 0 <= arm < 10

    def test_update_does_nothing(self):
        alg = RandomPolicy(n_arms=5)
        alg.update(0, 1.0, 1)  # Should not raise

    def test_reset(self):
        alg = RandomPolicy(n_arms=5, seed=42)
        arms1 = [alg.choose_arm(t) for t in range(1, 20)]
        alg.reset()
        arms2 = [alg.choose_arm(t) for t in range(1, 20)]
        assert arms1 == arms2  # same seed → same sequence


class TestEpsilonGreedy:

    def test_construction(self):
        alg = EpsilonGreedy(n_arms=5, epsilon=0.1)
        assert alg.epsilon == 0.1

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            EpsilonGreedy(n_arms=5, epsilon=1.5)

    def test_converges_to_best_arm(self):
        """After many updates, should mostly select arm 0 (best)."""
        alg = EpsilonGreedy(n_arms=3, epsilon=0.1)
        rng = np.random.default_rng(0)
        means = [0.9, 0.3, 0.1]

        for t in range(1, 2001):
            arm    = alg.choose_arm(t)
            reward = float(rng.uniform() < means[arm])
            alg.update(arm, reward, t)

        # After 2000 rounds, empirical estimates should be reasonable
        assert alg.estimates[0] > alg.estimates[1]
        assert alg.estimates[0] > alg.estimates[2]

    def test_reset_clears_state(self):
        alg = EpsilonGreedy(n_arms=3)
        alg.update(0, 1.0, 1)
        alg.reset()
        np.testing.assert_array_equal(alg.counts,    np.zeros(3))
        np.testing.assert_array_equal(alg.estimates, np.zeros(3))


class TestUCB1:

    def test_initialisation_pulls_each_arm_once(self):
        """UCB1 pulls each arm once when counts[arm]==0, sequentially."""
        alg = UCB1(n_arms=5)
        arms = []
        for t in range(1, 6):
            arm = alg.choose_arm(t)
            arms.append(arm)
            alg.update(arm, 0.5, t)  # update so each arm is marked seen
        assert set(arms) == {0, 1, 2, 3, 4}

    def test_arm_in_range(self):
        alg = UCB1(n_arms=5)
        rng = np.random.default_rng(0)
        for t in range(1, 101):
            arm = alg.choose_arm(t)
            assert 0 <= arm < 5
            alg.update(arm, float(rng.uniform()), t)

    def test_converges_on_simple_bandit(self):
        alg  = UCB1(n_arms=3)
        rng  = np.random.default_rng(0)
        means = [0.9, 0.3, 0.1]

        for t in range(1, 1001):
            arm    = alg.choose_arm(t)
            reward = float(rng.uniform() < means[arm])
            alg.update(arm, reward, t)

        # Best arm (0) should have been pulled most
        assert alg.counts[0] > alg.counts[1]
        assert alg.counts[0] > alg.counts[2]


class TestThompsonSampling:

    def test_bernoulli_mode(self):
        alg = ThompsonSampling(n_arms=5, reward_type="bernoulli")
        assert alg.reward_type == "bernoulli"
        arm = alg.choose_arm(1)
        assert 0 <= arm < 5

    def test_gaussian_mode(self):
        alg = ThompsonSampling(n_arms=5, reward_type="gaussian")
        arm = alg.choose_arm(1)
        assert 0 <= arm < 5

    def test_invalid_reward_type(self):
        with pytest.raises(ValueError):
            ThompsonSampling(n_arms=5, reward_type="poisson")

    def test_beta_posterior_updates(self):
        alg = ThompsonSampling(n_arms=2, reward_type="bernoulli")
        alg.update(0, 1.0, 1)
        alg.update(0, 1.0, 2)
        assert alg.alpha[0] == pytest.approx(3.0)  # 1 + 2 successes
        assert alg.beta[0]  == pytest.approx(1.0)  # 1 + 0 failures


class TestLinUCB:

    def test_construction(self):
        alg = LinUCB(n_arms=5, context_dim=4, alpha=1.0)
        assert alg.n_arms == 5
        assert alg.context_dim == 4

    def test_requires_context(self):
        alg = LinUCB(n_arms=5, context_dim=4)
        with pytest.raises(ValueError):
            alg.choose_arm(1, context=None)

    def test_requires_context_dim_gt_0(self):
        with pytest.raises(ValueError):
            LinUCB(n_arms=5, context_dim=0)

    def test_choose_arm_with_context(self):
        alg = LinUCB(n_arms=5, context_dim=4)
        ctx = np.random.randn(4)
        arm = alg.choose_arm(1, ctx)
        assert 0 <= arm < 5

    def test_update_changes_state(self):
        alg = LinUCB(n_arms=3, context_dim=2)
        A_before = alg.A[0].copy()
        ctx = np.array([1.0, 0.5])
        alg.update(0, 1.0, 1, ctx)
        assert not np.allclose(alg.A[0], A_before)


# ==========================================================================
# 4. Runner tests
# ==========================================================================

class TestBenchmarkRunner:

    def test_min_runs_enforced(self):
        with pytest.raises(ValueError):
            BenchmarkRunner(n_runs=29)

    def test_run_setting_returns_correct_shape(self, small_runner):
        regrets = small_runner.run_setting(
            UCB1, {},
            BernoulliBandit, {"K": 10, "T": 500},
        )
        assert regrets.shape == (30,)

    def test_all_regrets_nonnegative(self, small_runner):
        regrets = small_runner.run_setting(
            UCB1, {},
            BernoulliBandit, {"K": 10, "T": 500},
        )
        assert np.all(regrets >= 0)

    def test_ucb1_beats_random_on_s1(self, small_runner):
        regrets_ucb = small_runner.run_setting(
            UCB1, {},
            BernoulliBandit, {"K": 10, "T": 2000},
        )
        regrets_rnd = small_runner.run_setting(
            RandomPolicy, {},
            BernoulliBandit, {"K": 10, "T": 2000},
        )
        assert regrets_ucb.mean() < regrets_rnd.mean()

    def test_seed_table_length(self):
        assert len(SEED_TABLE) >= 200

    def test_full_suite_returns_all_keys(self, small_runner):
        results = small_runner.full_suite(RandomPolicy, {})
        # S1: 3K x 3T = 9
        s1_keys = [k for k in results if k.startswith("S1")]
        assert len(s1_keys) == 9
        # S2: 3K x 2T = 6
        s2_keys = [k for k in results if k.startswith("S2")]
        assert len(s2_keys) == 6
        # S3: 1
        assert "S3_K10_T10000" in results
        # S5: 3
        s5_keys = [k for k in results if k.startswith("S5")]
        assert len(s5_keys) == 3


# ==========================================================================
# 5. StatisticalAnalyser tests
# ==========================================================================

class TestStatisticalAnalyser:

    @pytest.fixture
    def sample_results(self):
        """Two algorithms with a clear difference."""
        rng  = np.random.default_rng(42)
        good = rng.normal(100, 10, size=30)   # lower regret
        bad  = rng.normal(200, 10, size=30)   # higher regret
        return {"GoodAlg": good, "BadAlg": bad}

    def test_requires_30_runs(self):
        analyser = StatisticalAnalyser()
        with pytest.raises(ValueError):
            analyser.analyse({"A": np.zeros(29), "B": np.zeros(29)})

    def test_returns_all_required_keys(self, sample_results):
        report = StatisticalAnalyser().analyse(sample_results)
        assert "summary" in report
        assert "pairwise" in report
        assert "effect_sizes" in report
        assert "ranking" in report

    def test_summary_contains_ci(self, sample_results):
        report = StatisticalAnalyser().analyse(sample_results)
        for name in sample_results:
            s = report["summary"][name]
            assert "ci_lower" in s
            assert "ci_upper" in s
            assert s["ci_lower"] < s["mean"] < s["ci_upper"]

    def test_significant_difference_detected(self, sample_results):
        report = StatisticalAnalyser().analyse(sample_results)
        pair   = ("GoodAlg", "BadAlg")
        assert report["pairwise"][pair]["significant"] is True
        assert report["pairwise"][pair]["better"] == "GoodAlg"

    def test_large_effect_size_detected(self, sample_results):
        report = StatisticalAnalyser().analyse(sample_results)
        eff = report["effect_sizes"][("GoodAlg", "BadAlg")]
        assert eff["magnitude"] == "large"
        assert abs(eff["cohens_d"]) > 0.8

    def test_negligible_effect_labelled(self):
        rng  = np.random.default_rng(0)
        # Nearly identical distributions
        a = rng.normal(100, 10, size=30)
        b = a + rng.normal(0, 0.01, size=30)  # tiny difference
        report = StatisticalAnalyser().analyse({"A": a, "B": b})
        eff = report["effect_sizes"][("A", "B")]
        assert eff["magnitude"] == "negligible"

    def test_ranking_order(self, sample_results):
        report = StatisticalAnalyser().analyse(sample_results)
        assert report["ranking"][0] == "GoodAlg"  # lower regret first

    def test_format_report_returns_string(self, sample_results):
        report  = StatisticalAnalyser().analyse(sample_results)
        text    = StatisticalAnalyser().format_report(report)
        assert isinstance(text, str)
        assert "REQUIREMENT 1" in text
        assert "REQUIREMENT 4" in text


# ==========================================================================
# 6. BCSCalculator tests
# ==========================================================================

class TestBCSCalculator:

    def test_all_conditions_pass(self):
        result = BCSCalculator.compute(
            rho_s=0.85, p_indep=0.95, K=10, T=10_000
        )
        assert result["BCS"] >= 0.9  # PASS band: all conditions met
        assert result["band"] == "PASS"
        assert result["failing_conditions"] == []

    def test_condition3_fails_at_large_K_small_T(self):
        # K=270, T=10000 → e = 270*log(10000)/10000 ≈ 0.249 > 0.20
        result = BCSCalculator.compute(
            rho_s=0.85, p_indep=0.95, K=270, T=10_000
        )
        assert result["c3_sufficient_horizon"] == pytest.approx(0.0)
        assert any("Condition 3" in f for f in result["failing_conditions"])

    def test_all_conditions_fail(self):
        result = BCSCalculator.compute(
            rho_s=-0.5, p_indep=0.5, K=1000, T=100
        )
        assert result["BCS"] < 0.3  # FAIL band clearly
        assert result["band"] == "FAIL"
        assert len(result["failing_conditions"]) == 3

    def test_partial_band(self):
        # rho_s just above 0.7, p_indep ok, c3 fails
        result = BCSCalculator.compute(
            rho_s=0.75, p_indep=0.92, K=270, T=10_000
        )
        assert 0.6 <= result["BCS"] < 0.9
        assert result["band"] == "PARTIAL"

    def test_clipping(self):
        # rho_s > 0.7 should be clipped to c1=1
        result = BCSCalculator.compute(
            rho_s=1.0, p_indep=1.0, K=5, T=10_000
        )
        assert result["c1_reward_proxy"] == pytest.approx(1.0)
        # rho_s < 0 should be clipped to c1=0
        result2 = BCSCalculator.compute(
            rho_s=-1.0, p_indep=1.0, K=5, T=10_000
        )
        assert result2["c1_reward_proxy"] == pytest.approx(0.0)

    def test_bcs_in_unit_interval(self):
        for rho_s in [-1, 0, 0.5, 0.7, 1.0]:
            for p in [0, 0.5, 0.9, 1.0]:
                result = BCSCalculator.compute(rho_s, p, K=50, T=2000)
                assert 0.0 <= result["BCS"] <= 1.0

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            BCSCalculator.compute(0.8, 0.95, K=10, T=0)

    def test_format_result_returns_string(self):
        result = BCSCalculator.compute(0.8, 0.95, K=10, T=10_000)
        text   = BCSCalculator().format_result(result)
        assert "BCS" in text
        assert "BRIDGE COMPLIANCE SCORE" in text


# ==========================================================================
# 7. LeaderboardSubmitter tests
# ==========================================================================

class TestLeaderboardSubmitter:

    @pytest.fixture
    def submitter(self):
        return LeaderboardSubmitter(
            algorithm_name="TestAlgorithm",
            paper_url="https://example.com/paper",
            code_url="https://github.com/example/repo",
        )

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(0)
        results = {
            "GoodAlg": rng.normal(100, 10, size=30),
            "BadAlg":  rng.normal(200, 10, size=30),
        }
        report = StatisticalAnalyser().analyse(results)
        bcs    = BCSCalculator.compute(0.85, 0.95, K=10, T=500)
        return results, report, bcs

    def test_generate_returns_valid_json(self, submitter, sample_data):
        import json
        results, report, bcs = sample_data
        json_str = submitter.generate(results, report, bcs)
        data = json.loads(json_str)  # must not raise
        assert "algorithm" in data
        assert "certification" in data
        assert "settings" in data

    def test_certification_in_json(self, submitter, sample_data):
        import json
        results, report, bcs = sample_data
        json_str = submitter.generate(results, report, bcs)
        data = json.loads(json_str)
        assert "C1" in data["certification"]

    def test_summary_returns_string(self, submitter, sample_data):
        results, report, bcs = sample_data
        json_str = submitter.generate(results, report, bcs)
        summary  = submitter.summary(json_str)
        assert "TestAlgorithm" in summary
        assert "BCS" in summary

    def test_missing_stats_report_raises(self, submitter):
        with pytest.raises(ValueError):
            submitter.generate(
                {"A": np.zeros(30)},
                {"summary": {}, "n_runs": 30},  # missing pairwise
                None,
            )

    def test_too_few_runs_raises(self, submitter):
        rng = np.random.default_rng(0)
        results = {"A": rng.normal(0, 1, 29), "B": rng.normal(1, 1, 29)}
        with pytest.raises(ValueError):
            StatisticalAnalyser().analyse(results)


# ==========================================================================
# 8. Integration test: full pipeline
# ==========================================================================

class TestFullPipeline:

    def test_ucb1_vs_random_full_pipeline(self):
        """
        End-to-end: run two algorithms, analyse, compute BCS, generate JSON.
        """
        runner   = BenchmarkRunner(n_runs=30)
        analyser = StatisticalAnalyser()

        # Run S1 K=10 T=500 only (fast)
        regrets_ucb = runner.run_setting(
            UCB1, {},
            BernoulliBandit, {"K": 10, "T": 500}
        )
        regrets_rnd = runner.run_setting(
            RandomPolicy, {},
            BernoulliBandit, {"K": 10, "T": 500}
        )

        results = {"UCB1": regrets_ucb, "Random": regrets_rnd}
        report  = analyser.analyse(results)
        bcs     = BCSCalculator.compute(
            rho_s=0.85, p_indep=0.95, K=10, T=500
        )

        submitter = LeaderboardSubmitter(
            algorithm_name="UCB1",
            paper_url="https://example.com",
            code_url="https://github.com",
        )
        json_str = submitter.generate(results, report, bcs)

        import json
        data = json.loads(json_str)

        # UCB1 should have lower regret
        # UCB1 should achieve lower regret than random
        s1_key = "S1_K10_T500"
        if s1_key in data["settings"]:
            assert data["settings"][s1_key]["mean_regret"] < 500 * 0.6
        else:
            # results passed directly, check via report
            assert report["ranking"][0] == "UCB1"

        # Should be certified
        assert "C1" in data["certification"]
