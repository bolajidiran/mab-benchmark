"""Tests for analysis.ope: estimator correctness on known policies, and exact
agreement of the native backend with the Open Bandit Pipeline when obp is installed."""
import numpy as np
import pytest

from mab_benchmark.analysis import ope


def _log(n=200_000, K=6, L=3, seed=0, uniform=False):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.2, size=(K, L))
    if uniform:
        pb = np.full(K, 1 / K)
    else:
        pb = rng.dirichlet(np.ones(K) * 2)
    a = rng.choice(K, n, p=pb); k = rng.integers(0, L, n)
    r = (rng.random(n) < mu[a, k]).astype(float)
    return mu, pb, a, k, r


def test_unbiased_estimators_recover_true_value_under_non_uniform_logging():
    mu, pb, a, k, r = _log()
    K, L = mu.shape; n = a.size
    pe = np.zeros(K); pe[[1, 4]] = 0.5                    # target: half on item 1, half on item 4
    ad = np.broadcast_to(pe[None, :, None], (n, K, L)).copy()
    truth = float(np.mean([pe @ mu[:, kk] for kk in k]))
    q = ope.cross_fitted_item_position_model(a, k, r, K, L, seed=1)
    out = ope.estimate(r, a, pb[a], ad, position=k, estimated_rewards=q, backend="native")
    for e in ("IPW", "SNIPW", "DM", "DR", "SNDR", "Switch-DR", "DRos"):
        assert out["values"][e] == pytest.approx(truth, abs=0.006), e


def test_ipw_equals_replay_value_on_uniform_log_for_a_deterministic_policy():
    mu, pb, a, k, r = _log(uniform=True)
    K, L = mu.shape; n = a.size
    ad = np.zeros((n, K, L)); ad[:, 2, :] = 1.0
    v = ope.estimate(r, a, pb[a], ad, position=k, backend="native")["values"]["IPW"]
    assert v == pytest.approx(r[a == 2].mean(), rel=0.03)


def test_input_validation():
    with pytest.raises(ValueError):
        ope.estimate([1, 0], [0, 1], [0.5, 0.0], np.full((2, 2, 1), 0.5))
    with pytest.raises(ValueError):
        ope.estimate([1, 0], [0, 1], [0.5, 0.5], np.full((2, 2, 1), 0.7))
    with pytest.raises(ValueError):
        ope.estimate([1, 0], [0, 1], [0.5, 0.5], np.full((2, 2, 1), 0.5), estimators=("DR",))


def test_bootstrap_interval_contains_point_estimate():
    mu, pb, a, k, r = _log(n=20_000)
    K, L = mu.shape; n = a.size
    ad = np.full((n, K, L), 1 / K)
    q = ope.cross_fitted_item_position_model(a, k, r, K, L)
    pt = ope.estimate(r, a, pb[a], ad, position=k, estimated_rewards=q, backend="native")["values"]
    ci = ope.bootstrap_ci(r, a, pb[a], ad, position=k, estimated_rewards=q, n_boot=200)
    for e in ("IPW", "SNIPW", "DR"):
        assert ci[e][0] <= pt[e] <= ci[e][1]


@pytest.mark.skipif(not ope.obp_available(), reason="obp not installed (needs Python < 3.11)")
def test_native_backend_equals_obp():
    mu, pb, a, k, r = _log(n=30_000, seed=3)
    K, L = mu.shape; n = a.size
    rng = np.random.default_rng(4)
    ad = rng.dirichlet(np.ones(K), size=(n, L)).transpose(0, 2, 1)       # (n, K, L), rows sum to 1 over K
    q = ope.cross_fitted_item_position_model(a, k, r, K, L)
    nat = ope.estimate(r, a, pb[a], ad, position=k, estimated_rewards=q, backend="native")
    lib = ope.estimate(r, a, pb[a], ad, position=k, estimated_rewards=q, backend="obp",
                       lam_switch=nat["lambdas"]["Switch-DR"], lam_dros=nat["lambdas"]["DRos"])
    for e in ope.ESTIMATORS:
        assert lib["values"][e] == pytest.approx(nat["values"][e], rel=1e-10, abs=1e-12), e
