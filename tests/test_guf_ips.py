"""Tests for the GUF, CCM, Pareto and IPS modules (Chapter Three, Sections 3.5, 3.6, 3.9)."""
import numpy as np
import pytest

from mab_benchmark.analysis import guf, ips, BCSCalculator


def test_pareto_empty_pool():
    assert guf.compute_pareto_front({}) == []


def test_pareto_single_policy():
    assert guf.compute_pareto_front({"A": (1, 0, 0)}) == ["A"]


def test_pareto_strict_dominance():
    assert guf.compute_pareto_front({"A": (1, 0, 0), "B": (2, 1, 0)}) == ["A"]


def test_pareto_incomparable():
    assert guf.compute_pareto_front({"A": (1, 2, 0), "B": (2, 1, 0)}) == ["A", "B"]


def test_pareto_equal_policies_both_kept():
    assert guf.compute_pareto_front({"A": (1, 1, 0), "B": (1, 1, 0)}) == ["A", "B"]


def test_dominance_not_symmetric():
    assert guf.is_dominated((2, 1, 0), (1, 1, 0)) and not guf.is_dominated((1, 1, 0), (2, 1, 0))


def test_dominator_diagnostics():
    pts = {"A": (1, 0, 0), "B": (2, 0, 0), "C": (3, 0, 0)}
    assert guf.dominators(pts, "C") == ["A", "B"]


def test_ccm_non_negative_and_zero_iff_within_budget():
    assert guf.ccm([10, 20], [1, 2], delta_ms=20, B_kb=50) == 0.0
    assert guf.ccm([30, 20], [1, 2], delta_ms=20, B_kb=50) > 0.0
    assert guf.ccm([10], [100], delta_ms=20, B_kb=50) > 0.0


def test_guf_reduces_to_ncr_when_constraints_inactive():
    pts = {"A": (0.5, 0.0, 0.0), "B": (0.3, 0.0, 0.0)}
    s = guf.weight_sweep(pts)
    assert s["stable"] and s["by_w1"][0.6]["ranking"] == ["B", "A"]
    assert guf.guf_value(0.5, 0, 0) == pytest.approx(0.6 * 0.5)
    with pytest.raises(ValueError):
        guf.guf_value(-0.1, 0, 0)


def test_normalised_guf_lets_latency_change_the_ranking():
    # same NCR gap as in the primary cell order of magnitude; a large latency
    # violation must be able to reverse the order once both terms are normalised
    pts = {"fast": (0.10, 0.0, 0.0), "slow": (0.05, 0.15, 0.0)}
    s = guf.weight_sweep(pts)
    assert s["by_w1"][0.8]["ranking"] == ["slow", "fast"]
    assert s["by_w1"][0.4]["ranking"] == ["fast", "slow"]
    assert not s["stable"]


def test_bcs_gate():
    assert guf.bcs_gate(0.911)["certifiable"] is True
    assert guf.bcs_gate(0.877)["certifiable"] is False
    assert guf.bcs_gate(None)["certifiable"] is False
    out = guf.run_gap5_pipeline({"A": (0.2, 0, 0), "B": (0.4, 0, 0)}, bcs=0.95)
    assert out["gate"]["certifiable"] and out["sweep"]["by_w1"][0.6]["ranking"] == ["A", "B"]


def test_extended_bcs():
    b = BCSCalculator.compute(1.0, 1.0, 10, 10_000)
    e = guf.bcs_extended(b, 0.0)
    assert e["c4_ccm"] == 1.0 and e["BCS_ext"] == pytest.approx((b["c1_reward_proxy"] + b["c2_arm_independence"] + b["c3_sufficient_horizon"] + 1) / 4, abs=1e-4)


def test_ips_unbiased_on_uniform_log():
    rng = np.random.default_rng(0); K, T = 5, 400_000
    mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    a = rng.integers(0, K, T); r = (rng.random(T) < mu[a]).astype(float)
    est = ips.arm_value_ips(r, a, np.full(T, 1 / K), K)
    assert np.allclose(est, mu, atol=0.01)
    target = (a == 4).astype(float)        # deterministic policy "always arm 4"
    assert ips.policy_value_ips(r, np.full(T, 1 / K), target) == pytest.approx(0.5, abs=0.01)
