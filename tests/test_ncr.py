"""Tests for NCR (analysis.ncr) and the environments' Random-regret normaliser."""
import numpy as np
import pytest

from mab_benchmark.analysis import ncr
from mab_benchmark.baselines import RandomPolicy, ThompsonSampling
from mab_benchmark.environments import (BernoulliBandit, ColdStartBandit, GaussianBandit,
                                        PiecewiseStationaryBandit, SyntheticContextualBandit)
from mab_benchmark.runner import BenchmarkRunner, SEED_TABLE


def test_random_regret_closed_form_stationary():
    env = BernoulliBandit(10, 2000, seed=1000)
    assert env.expected_random_regret() == pytest.approx(2000 * (env.means.max() - env.means.mean()))


@pytest.mark.parametrize("Env,kw,flags", [
    (BernoulliBandit, {"K": 10, "T": 2000}, {}),
    (PiecewiseStationaryBandit, {"K": 10, "T": 10_000}, {"is_stationary": False}),
    (SyntheticContextualBandit, {"K": 20, "T": 10_000, "context_dim": 5}, {"use_context": True}),
])
def test_uniform_random_has_ncr_close_to_one(Env, kw, flags):
    r = BenchmarkRunner(30)
    reg = r.run_setting(RandomPolicy, {}, Env, kw, **flags)
    z = ncr.ncr_for_setting(reg, Env, kw, SEED_TABLE[:30])
    assert abs(np.nanmean(z) - 1.0) < 0.05


def test_ncr_is_scale_free_and_ranks_like_regret():
    r = BenchmarkRunner(30)
    kw = {"K": 10, "T": 2000}
    ts = ncr.ncr_for_setting(r.run_setting(ThompsonSampling, {}, BernoulliBandit, kw), BernoulliBandit, kw, SEED_TABLE[:30])
    rnd = ncr.ncr_for_setting(r.run_setting(RandomPolicy, {}, BernoulliBandit, kw), BernoulliBandit, kw, SEED_TABLE[:30])
    assert 0 <= np.nanmean(ts) < 0.2 < np.nanmean(rnd)


def test_other_environments_have_positive_normaliser():
    for env in (GaussianBandit(10, 1000, seed=1001), ColdStartBandit(500, 100, seed=1002)):
        assert env.expected_random_regret() > 0


def test_ncr_pairing_and_degenerate_instances():
    with pytest.raises(ValueError):
        ncr.ncr([1, 2], [1])
    out = ncr.ncr([1.0, 2.0], [0.0, 4.0])
    assert np.isnan(out[0]) and out[1] == 0.5


def test_runner_accepts_explicit_seeds():
    a = BenchmarkRunner(2, seeds=[5000, 5001], enforce_min_runs=False).run_setting(
        RandomPolicy, {}, BernoulliBandit, {"K": 10, "T": 500})
    b = BenchmarkRunner(2, seeds=[5000, 5001], enforce_min_runs=False).run_setting(
        RandomPolicy, {}, BernoulliBandit, {"K": 10, "T": 500})
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        BenchmarkRunner(5)
