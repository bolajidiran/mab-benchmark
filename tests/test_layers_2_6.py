"""Tests for the Meta-Policy Orchestrator (Layer 2) and the counterfactual explainer (Layer 6)."""

import numpy as np

from mab_benchmark.baselines import EpsilonGreedy, RandomPolicy, ThompsonSampling, UCB1
from mab_benchmark.environments import BernoulliBandit
from mab_benchmark.explain import TEMPLATE_TIE, CounterfactualExplainer, choice_propensity
from mab_benchmark.orchestration import MetaPolicyOrchestrator, SystemPerformanceMonitor, with_mpo
from mab_benchmark.runner import BenchmarkRunner


def drive(mpo, rewards):
    """Feed arm 0's rewards; the base algorithm always plays arm 0 here."""
    for t, r in enumerate(rewards, 1):
        mpo.choose_arm(t)
        mpo.update(0, r, t)


def test_drop_in_reward_flags_a_change_point_and_resets_thompson_sampling():
    ts = ThompsonSampling(5)
    mpo = MetaPolicyOrchestrator(ts, window=50, measure_latency=False)
    drive(mpo, [1.0] * 400 + [0.0] * 200)
    flags = mpo.log["change_point_flags"]
    assert flags and 400 < flags[0] <= 460
    assert mpo.log["resets"] == flags                       # rule A fires for TS family
    assert ts.alpha[0] + ts.beta[0] < 400                   # posterior was restarted


def test_non_ts_algorithms_are_monitored_but_not_reset():
    ucb = UCB1(5)
    mpo = MetaPolicyOrchestrator(ucb, window=50, measure_latency=False)
    drive(mpo, [1.0] * 400 + [0.0] * 200)
    assert mpo.log["change_point_flags"] and mpo.log["resets"] == []
    assert ucb.counts[0] == 600                             # UCB1 state kept


def test_no_flag_on_a_stationary_stream():
    mpo = MetaPolicyOrchestrator(ThompsonSampling(5), window=100, measure_latency=False)
    rng = np.random.default_rng(0)
    drive(mpo, (rng.random(5000) < 0.6).astype(float))
    assert mpo.log["change_point_flags"] == []


def test_threshold_is_ucb_shaped_and_lambda_makes_it_respond_faster():
    a = MetaPolicyOrchestrator(ThompsonSampling(3), window=100)
    b = MetaPolicyOrchestrator(ThompsonSampling(3), window=100, lam=0.5)
    assert np.isclose(a.threshold(1000), np.sqrt(2 * np.log(1000) / 100))
    assert b.threshold(1000) < a.threshold(1000)


def test_wrapping_without_flags_changes_nothing():
    r = BenchmarkRunner(30)
    plain = r.run_setting(UCB1, {}, BernoulliBandit, {"K": 10, "T": 500})
    wrapped = r.run_setting(with_mpo(UCB1, measure_latency=False), {}, BernoulliBandit, {"K": 10, "T": 500})
    assert np.allclose(plain, wrapped)
    rnd = r.run_setting(with_mpo(RandomPolicy, measure_latency=False), {}, BernoulliBandit, {"K": 10, "T": 500})
    assert np.allclose(rnd, r.run_setting(RandomPolicy, {}, BernoulliBandit, {"K": 10, "T": 500}))


def test_ccm_rule_shifts_weights_and_keeps_them_valid():
    mpo = MetaPolicyOrchestrator(ThompsonSampling(4), regime="2G", measure_latency=False,
                                 latency_fn=lambda t: 260.0 if t > 300 else 10.0)
    drive(mpo, [1.0] * 800)
    assert mpo.log["ccm_flags"] and mpo.log["ccm_flags"][0] > 300
    assert np.isclose(sum(mpo.weights), 1.0) and mpo.weights[0] >= 0.2 - 1e-9
    assert mpo.weights[1] > 0.2 and mpo.weights[2] > 0.2


def test_ccm_rule_excludes_over_budget_arms_when_costs_are_known():
    cost = np.array([[300.0, 0.0], [50.0, 0.0]])            # arm 0 breaks the 2G budget (200 ms)
    ts = ThompsonSampling(2)
    mpo = MetaPolicyOrchestrator(ts, regime="2G", measure_latency=False, arm_cost=cost)
    np.random.seed(0)
    picks = []
    for t in range(1, 1501):
        a = mpo.choose_arm(t)
        mpo.update(a, 1.0 if a == 0 else 0.5, t)            # arm 0 is the better arm
        picks.append(a)
    first = mpo.log["ccm_flags"][0]
    assert np.mean(np.array(picks[first:]) == 0) < np.mean(np.array(picks[:first]) == 0)


def test_performance_monitor_ccm_and_regime_flags():
    m = SystemPerformanceMonitor("4G")                       # delta = 100 ms, B = 500 KB
    f = m.record(150.0, 1000.0)
    assert f["over_100ms"] and f["over_delta"] and f["over_B"]
    assert np.isclose(m.ccm(), 0.5 * 0.5 + 0.5 * 1.0)


def test_explainer_branches():
    ex = CounterfactualExplainer(3, tolerance="absolute")
    for a, r in ((0, 1.0), (1, 1.0), (2, 0.0)):
        ex.observe(a, r, 1 / 3)
    e = ex.explain(10, 0)
    assert e.kind == "indistinguishable" and e.text == TEMPLATE_TIE and abs(e.delta) <= 0.01
    ex.observe(0, 1.0, 1 / 3)
    assert ex.explain(11, 0).kind == "margin"
    assert ex.explain(11, 2).kind == "exploration"


def test_explainer_ips_is_unbiased_under_uniform_logging():
    rng = np.random.default_rng(1)
    mu = np.array([0.2, 0.5, 0.7]); ex = CounterfactualExplainer(3)
    for _ in range(60_000):
        a = int(rng.integers(3)); ex.observe(a, float(rng.random() < mu[a]), 1 / 3)
    assert np.allclose(ex.values(), mu, atol=0.02)


def test_choice_propensities():
    eg = EpsilonGreedy(4, epsilon=0.2); eg.estimates[:] = [0.1, 0.9, 0.3, 0.2]
    assert np.allclose(choice_propensity(eg, 5), [0.05, 0.85, 0.05, 0.05])
    u = UCB1(3); u.counts[:] = [10, 10, 10]; u.estimates[:] = [0.1, 0.8, 0.2]
    assert np.allclose(choice_propensity(u, 31), [0, 1, 0])
    ts = ThompsonSampling(3); ts.alpha[:] = [50, 2, 2]; ts.beta[:] = [2, 50, 50]
    p = choice_propensity(ts, 10, n_mc=500, rng=np.random.default_rng(0))
    assert np.isclose(p.sum(), 1.0) and p[0] > 0.95
    assert np.allclose(choice_propensity(RandomPolicy(4, seed=0), 3), 0.25)


def _low_ctr_log(n, mu, seed):
    rng = np.random.default_rng(seed); K = mu.size
    a = rng.integers(0, K, n); r = (rng.random(n) < mu[a]).astype(float)
    return a, r


def test_scale_aware_tolerance_grades_explanations_below_one_percent_ctr():
    mu = np.array([0.002, 0.003, 0.0045, 0.007])         # click-through rates below 1%
    a, r = _low_ctr_log(400_000, mu, 7)
    for tol in ("statistical", "relative"):
        ex = CounterfactualExplainer(4, tolerance=tol)
        for ai, ri in zip(a, r):
            ex.observe(int(ai), ri, 0.25)
        assert ex.explain(1, 3).kind == "margin"          # best arm, 56% above the next
        assert ex.explain(1, 0).kind == "exploration"
    ab = CounterfactualExplainer(4, tolerance="absolute")
    for ai, ri in zip(a, r):
        ab.observe(int(ai), ri, 0.25)
    assert ab.explain(1, 3).kind == "indistinguishable"  # eps = 0.01 is larger than every gap


def test_statistical_tolerance_calls_equal_arms_indistinguishable():
    mu = np.array([0.004, 0.004, 0.001])
    hits = 0
    for seed in range(40):
        a, r = _low_ctr_log(60_000, mu, seed)
        ex = CounterfactualExplainer(3)
        for ai, ri in zip(a, r):
            ex.observe(int(ai), ri, 1 / 3)
        hits += ex.explain(1, 0).kind == "indistinguishable"
    assert hits >= 34                                     # about 95% expected at z = 1.96


def test_gap_standard_error_matches_direct_computation():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 3, 5000); r = rng.random(5000); p = rng.uniform(0.2, 0.5, 5000)
    ex = CounterfactualExplainer(3)
    for ai, ri, pi in zip(a, r, p):
        ex.observe(int(ai), ri, pi)
    x0 = np.where(a == 0, r / p, 0.0); x1 = np.where(a == 1, r / p, 0.0)
    assert np.isclose(ex.gap_se(0, 1), (x0 - x1).std() / np.sqrt(a.size))
