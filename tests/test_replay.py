"""Tests for mab_benchmark.replay and the bias-free S4 environments."""
import numpy as np
import pytest

from mab_benchmark import replay as R
from mab_benchmark.baselines import RandomPolicy, UCB1, ThompsonSampling
from mab_benchmark.environments import OBDCalibratedBandit


def test_replay_matches_one_in_k_on_iid_uniform_log():
    rng = np.random.default_rng(0); K, n = 10, 50_000
    arms = rng.integers(0, K, n); rew = (rng.random(n) < 0.3).astype(float)
    for alg in (RandomPolicy(K, seed=1), UCB1(K)):
        out = R.replay(alg, arms, rew)
        assert abs(out["accepted"] / n - 1 / K) < 0.01


def test_bursty_log_starves_a_deterministic_explorer_and_shuffle_restores_it():
    K = 20
    arms, sess = R.bursty_uniform_log(60_000, K, mean_repeats=6, rng=np.random.default_rng(1))
    assert abs(np.bincount(arms, minlength=K) / arms.size - 1 / K).max() < 0.01      # marginal uniform
    prof = R.burst_profile(arms, np.zeros(arms.size), K)
    assert prof["sweep_over_random"] < 0.6 and prof["median_gap"] < prof["iid_median_gap"]
    rew = np.zeros(arms.size)
    a_log = R.replay(UCB1(K), arms, rew)["accepted"]
    r_log = R.replay(RandomPolicy(K, seed=2), arms, rew)["accepted"]
    windows = np.arange(arms.size) // 2_000
    p = R.shuffle_within_windows(windows, np.random.default_rng(3))
    assert np.array_equal(np.sort(windows[p]), np.sort(windows))                     # stays in its window
    a_sh = R.replay(UCB1(K), arms[p], rew)["accepted"]
    assert a_log / r_log < 0.6 and a_sh / r_log > 0.8


def test_iid_log_profile_is_neutral():
    rng = np.random.default_rng(4); K = 20
    arms = rng.integers(0, K, 40_000)
    prof = R.burst_profile(arms, (rng.random(arms.size) < 0.2).astype(float), K)
    assert 0.8 < prof["sweep_over_random"] < 1.2 and abs(prof["click_lift_after_click"] - 1) < 0.25


def test_integrity_and_ratios():
    assert R.replay_integrity(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
    assert R.replay_integrity(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)
    m = R.matched_ratios({"B0_Random": [100, 200], "X": [50, 100]})
    assert m["X"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        R.replay_integrity(["a"], ["b"])


def test_obd_calibrated_environment_with_explicit_means():
    env = OBDCalibratedBandit(K=3, T=100, means=[0.001, 0.004, 0.007])
    assert env.a_star == 2 and env.expected_random_regret() == pytest.approx(100 * (0.007 - 0.004))
    with pytest.raises(ValueError):
        OBDCalibratedBandit(K=4, T=10, means=[0.1, 0.2])


def _kuairec_present():
    from mab_benchmark.environments.s4_kuairec import DEFAULT_ROOT
    return DEFAULT_ROOT.exists() and any(DEFAULT_ROOT.glob("*.npz")) or any(DEFAULT_ROOT.rglob("small_matrix.csv"))


@pytest.mark.skipif(not _kuairec_present(), reason="KuaiRec data not downloaded (docs/datasets.md)")
def test_kuairec_environment_matches_the_thesis_instance():
    from mab_benchmark.environments import KuaiRecBandit
    env = KuaiRecBandit()
    assert (env.K, env.T, env.n_users) == (80, 14_315, 1411)
    assert env.mu_star == pytest.approx(0.1467, abs=5e-4) and env.means.mean() == pytest.approx(0.0402, abs=5e-4)
    assert env.missing_share < 0.001
    rng = np.random.default_rng(0)
    draws = np.array([env.pull(env.a_star, rng) for _ in range(20_000)])
    assert abs(draws.mean() - env.mu_star) < 0.01
    assert env.regret_increment(env.a_star) == 0.0
    assert env.expected_random_regret() == pytest.approx(env.T * (env.mu_star - env.means.mean()))
