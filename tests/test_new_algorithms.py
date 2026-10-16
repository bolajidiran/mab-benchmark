"""Item 5: KL-UCB, sliding-window Thompson sampling, LinTS and the greedy linear model."""
import math

import numpy as np
import pytest

from mab_benchmark.algorithms import KLUCB, Lin, LinTS, SlidingWindowTS
from mab_benchmark.algorithms.kl_ucb import kl_bernoulli, kl_ucb_index
from mab_benchmark.baselines import RandomPolicy, ThompsonSampling, UCB1
from mab_benchmark.environments import BernoulliBandit, PiecewiseStationaryBandit
from mab_benchmark.runner import BenchmarkRunner


def test_kl_ucb_index_properties():
    assert kl_bernoulli(0.3, 0.3) == pytest.approx(0.0, abs=1e-12)
    assert kl_bernoulli(0.3, 0.6) > 0
    i = kl_ucb_index(0.4, 20, math.log(100))
    assert 0.4 <= i <= 1.0
    assert kl_bernoulli(0.4, i) * 20 == pytest.approx(math.log(100), abs=1e-3)   # boundary of the set
    assert kl_ucb_index(0.4, 200, math.log(100)) < i                               # more data, tighter
    assert kl_ucb_index(0.4, 20, math.log(1000)) > i                               # higher level, wider
    assert kl_ucb_index(0.4, 0, 1.0) == 1.0


def test_kl_ucb_beats_ucb1_and_random_on_s1():
    run = BenchmarkRunner(n_runs=30)
    env = {"K": 10, "T": 2000}
    kl = run.run_setting(KLUCB, {}, BernoulliBandit, env).mean()
    ucb = run.run_setting(UCB1, {}, BernoulliBandit, env).mean()
    rnd = run.run_setting(RandomPolicy, {}, BernoulliBandit, env).mean()
    assert kl < ucb < rnd


def test_sliding_window_bookkeeping():
    a = SlidingWindowTS(3, window=4)
    for t, (arm, r) in enumerate([(0, 1), (1, 0), (0, 1), (2, 1), (0, 0), (1, 1)], start=1):
        a.update(arm, r, t)
    # last four updates: (0,1), (2,1), (0,0), (1,1)
    assert a.succ.tolist() == [1, 1, 1] and a.fail.tolist() == [1, 0, 0] and len(a.hist) == 4
    with pytest.raises(ValueError):
        SlidingWindowTS(3, window=0)


def test_sliding_window_ts_tracks_change_points_better_than_ts():
    run = BenchmarkRunner(n_runs=10, enforce_min_runs=False)
    env = {"K": 10, "T": 10_000}
    sw = run.run_setting(SlidingWindowTS, {"window": 1000}, PiecewiseStationaryBandit, env, is_stationary=False)
    ts = run.run_setting(ThompsonSampling, {}, PiecewiseStationaryBandit, env, is_stationary=False)
    assert sw.mean() < ts.mean()


class _LinearEnv:
    """K arms, d-dimensional Gaussian contexts, Bernoulli(sigmoid(x . theta_a)) rewards."""

    def __init__(self, K=5, d=3, T=1500, seed=0):
        g = np.random.default_rng(seed)
        self.K, self.d, self.T = K, d, T
        self.theta = g.normal(size=(K, d)) * 1.5
        self.X = g.normal(size=(T, d))

    def mu(self, t):
        return 1 / (1 + np.exp(-(self.theta @ self.X[t])))


def _play(Alg, kw, seed):
    env = _LinearEnv(seed=seed); np.random.seed(seed + 1)
    alg = Alg(env.K, context_dim=env.d, **kw) if Alg is not RandomPolicy else Alg(env.K)
    rng = np.random.default_rng(seed + 2); reg = 0.0
    for t in range(env.T):
        x = env.X[t]; m = env.mu(t)
        a = alg.choose_arm(t + 1, x)
        r = float(rng.random() < m[a]); reg += m.max() - m[a]
        alg.update(a, r, t + 1, x)
    return reg


def test_linear_algorithms_learn_and_sherman_morrison_is_exact():
    lt = np.mean([_play(LinTS, {"v": 0.5}, s) for s in range(5)])
    gr = np.mean([_play(Lin, {}, s) for s in range(5)])
    rd = np.mean([_play(RandomPolicy, {}, s) for s in range(5)])
    assert lt < 0.6 * rd and gr < 0.6 * rd
    alg = Lin(2, context_dim=3)
    xs = np.random.default_rng(3).normal(size=(20, 3))
    for i, x in enumerate(xs):
        alg.update(0, 1.0, i + 1, x)
    assert np.allclose(alg.A_inv[0], np.linalg.inv(np.eye(3) + xs.T @ xs))
    for C in (Lin, LinTS):
        with pytest.raises(ValueError):
            C(3, context_dim=0)
