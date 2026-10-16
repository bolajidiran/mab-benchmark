"""
Regression tests: seeding and stream independence, S4 context clock, LinUCB in
the runner, A12 with ties, Normal-Inverse-Gamma posterior, FG-TS construction,
learning curves, and the effect-size rule of the Certifier.
"""
import numpy as np
import pytest

from mab_benchmark.runner import BenchmarkRunner, SEED_TABLE, run_streams
from mab_benchmark.baselines import (RandomPolicy, EpsilonGreedy, UCB1,
                                     ThompsonSampling, LinUCB)
from mab_benchmark.environments import BernoulliBandit, SyntheticContextualBandit
from mab_benchmark.analysis.statistics import StatisticalAnalyser


@pytest.mark.parametrize("Alg", [RandomPolicy, EpsilonGreedy, ThompsonSampling, UCB1])
def test_runs_are_reproducible(Alg):
    r = BenchmarkRunner(30)
    a = r.run_setting(Alg, {}, BernoulliBandit, {"K": 10, "T": 500})
    b = r.run_setting(Alg, {}, BernoulliBandit, {"K": 10, "T": 500})
    assert np.array_equal(a, b)


def test_reward_stream_independent_of_environment():
    for seed in SEED_TABLE[:30]:
        env = BernoulliBandit(10, 500, seed)
        rng, _ = run_streams(seed)
        assert not np.allclose(rng.uniform(size=10), env.means)


def test_ucb1_initial_pulls_not_forced_to_zero():
    hits = 0
    for seed in SEED_TABLE[:30]:
        env = BernoulliBandit(10, 500, seed)
        rng, _ = run_streams(seed)
        hits += sum(env.pull(a, rng) for a in range(10)) > 0
    assert hits >= 25


def test_synthetic_contextual_reward_follows_round_context():
    env = SyntheticContextualBandit(K=50, T=10_000, context_dim=10, seed=SEED_TABLE[0])
    rng = np.random.default_rng(0)
    mu = np.array([env.expected_reward(0, env.context(t)) for t in range(1, 3001)])
    r = np.array([env.pull(0, t, rng) for t in range(1, 3001)])
    assert np.corrcoef(mu, r)[0, 1] > 0.3
    env2 = SyntheticContextualBandit(K=50, T=10_000, context_dim=10, seed=SEED_TABLE[0])
    r2 = np.array([env2.pull(0, np.random.default_rng(0)) for _ in range(5)])
    assert env2._t == 5


def test_linucb_runs_through_runner_and_beats_random_on_s4():
    r = BenchmarkRunner(30)
    lin = r.run_setting(LinUCB, {}, SyntheticContextualBandit,
                        {"K": 10, "T": 10_000, "context_dim": 5}, use_context=True)
    rnd = r.run_setting(RandomPolicy, {}, SyntheticContextualBandit,
                        {"K": 10, "T": 10_000, "context_dim": 5}, use_context=True)
    assert lin.mean() < 0.5 * rnd.mean()


def test_a12_counts_ties_as_half():
    a = np.ones(30); b = np.ones(30)
    b[0] = 1.0 + 1e-9
    rep = StatisticalAnalyser().analyse({"A": a, "B": b})
    assert abs(rep["effect_sizes"][("A", "B")]["a12"] - 0.5) < 0.02


def test_feel_good_module_constructs():
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    FG = importlib.import_module("FeelGoodThompsonSampling").FeelGoodThompsonSampling
    alg = FG(n_arms=5, lam=0.5)
    assert alg.alpha.shape == (5,)
    alg.update(0, 1.7, 1); alg.update(0, -0.3, 2)
    assert alg.beta[0] > 0


def test_curves_end_at_run_setting_values():
    r = BenchmarkRunner(30)
    reg = r.run_setting(ThompsonSampling, {}, BernoulliBandit, {"K": 10, "T": 500})
    cur = r.run_setting_curves(ThompsonSampling, {}, BernoulliBandit, {"K": 10, "T": 500})
    assert cur.shape == (30, 500) and np.allclose(cur[:, -1], reg)


def test_certification_requires_effect_on_a_significant_pair():
    # a significant pair with a negligible effect is not a claim
    from mab_benchmark.analysis.certification import Certifier
    rng = np.random.default_rng(3)
    cells = ["S1_K10_T500", "S2_K10_T1000", "S3_K10_T10000", "S4_SYNTHETIC_K50_T10000", "S5_K500_T50"]
    base = rng.normal(100, 10, 30)
    suite = {a: {c: base + off + rng.normal(0, 0.01, 30) for c in cells}
             for a, off in (("X", 0.0), ("B0_Random", 0.5), ("B1_EpsGreedy", 0.5),
                            ("B2_UCB1", 0.5), ("B3_TS", 0.5))}
    cert = Certifier(require_bcs_for_c1=False).certify("X", suite)
    assert cert["level"] == "UNCERTIFIED"
    assert all(c["verdict"] in ("negligible", "not significant") for c in cert["tier1_claims"])
