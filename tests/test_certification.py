"""Tests for the Certifier (C1/C3/C4) and the repository leaderboard."""

import json

import numpy as np
import pytest

from mab_benchmark.analysis.bcs import BCSCalculator
from mab_benchmark.analysis.certification import Certifier
from mab_benchmark.analysis.leaderboard import Leaderboard, LeaderboardSubmitter

CELLS = ["S1_K10_T10000", "S2_K10_T1000", "S3_K10_T10000", "S4_SYNTHETIC_K50_T10000", "S5_K500_T100"]
GOOD_BCS = BCSCalculator.compute(0.95, 0.95, 10, 10_000)      # PASS, rho >= 0.7, e <= 0.2
BAD_BCS = BCSCalculator.compute(0.30, 1.00, 80, 1_706)        # FAIL


def make_suite(test_mean=50.0, seed=0, cells=CELLS, drop=None):
    rng = np.random.default_rng(seed)
    means = {"T": test_mean, "B0_Random": 200, "B1_EpsGreedy": 90, "B2_UCB1": 80, "B3_TS": 60}
    suite = {a: {c: rng.normal(m, 5, 30) for c in cells} for a, m in means.items()}
    if drop:
        suite.pop(drop)
    return suite


def tier2(test_ctr=0.0050, seed=1):
    rng = np.random.default_rng(seed)
    base = {"B0_Random": 0.0035, "B1_EpsGreedy": 0.0040, "B2_UCB1": 0.0036, "B3_TS": 0.0039}
    out = {b: rng.normal(m, 0.0002, 30) for b, m in base.items()}
    out["T"] = rng.normal(test_ctr, 0.0002, 30)
    return out


def test_c1_requires_all_conditions_and_names_the_claims():
    cert = Certifier().certify("T", make_suite(), bcs=GOOD_BCS)
    assert cert["level"] == "C1"
    assert cert["tier1_summary"]["B3_TS"]["better"] == len(CELLS)
    assert all(c["verdict"] == "better" for c in cert["tier1_claims"])


def test_missing_setting_blocks_certification():
    cert = Certifier().certify("T", make_suite(cells=CELLS[:4]), bcs=GOOD_BCS)
    assert cert["level"] == "UNCERTIFIED"
    assert any("missing settings: S5" in r for r in cert["reasons_not_higher"])


def test_missing_baseline_blocks_certification():
    cert = Certifier().certify("T", make_suite(drop="B2_UCB1"), bcs=GOOD_BCS)
    assert cert["level"] == "UNCERTIFIED"
    assert any("baseline incomplete" in r for r in cert["reasons_not_higher"])


def test_a_significant_pair_between_baselines_does_not_certify_the_test_algorithm():
    # B0 vs B3 is significant here, but the test algorithm loses to everyone
    cert = Certifier(require_bcs_for_c1=False).certify("T", make_suite(test_mean=400.0))
    assert cert["level"] == "UNCERTIFIED"
    assert cert["tier1_summary"]["B0_Random"]["worse"] == len(CELLS)


def test_bcs_rule_follows_the_chosen_reading():
    suite = make_suite()
    assert Certifier().certify("T", suite, bcs=BAD_BCS)["level"] == "UNCERTIFIED"
    assert Certifier().certify("T", suite)["level"] == "UNCERTIFIED"           # BCS not computed
    assert Certifier(require_bcs_for_c1=False).certify("T", suite, bcs=BAD_BCS)["level"] == "C1"


def test_c3_needs_tier2_win_on_a_tier1_baseline_and_bridge_conditions():
    suite = make_suite()
    assert Certifier().certify("T", suite, bcs=GOOD_BCS, tier2=tier2())["level"] == "C3"
    # Tier 2 not better: stays C1
    assert Certifier().certify("T", suite, bcs=GOOD_BCS, tier2=tier2(test_ctr=0.0030))["level"] == "C1"
    # bridge condition 1 fails (rho 0.5) even though BCS >= 0.9 is not reachable then: use TwoTier reading
    weak = BCSCalculator.compute(0.5, 1.0, 10, 100_000)
    c = Certifier(require_bcs_for_c1=False).certify("T", suite, bcs=weak, tier2=tier2())
    assert c["level"] == "C1" and any("bridge condition" in r for r in c["reasons_not_higher"])


def test_c4_requires_a_valid_live_ab_test():
    suite, t2 = make_suite(), tier2()
    ab = {"baseline": "B3_TS", "users": 5000, "days": 14, "protocol_disclosed": True,
          "clicks_test": 260, "impressions_test": 50_000, "clicks_baseline": 195, "impressions_baseline": 50_000}
    assert Certifier().certify("T", suite, bcs=GOOD_BCS, tier2=t2, ab_test=ab)["level"] == "C4"
    short = dict(ab, days=3)
    c = Certifier().certify("T", suite, bcs=GOOD_BCS, tier2=t2, ab_test=short)
    assert c["level"] == "C3" and c["ab_test"]["checks"]["days >= 7"] is False


def test_leaderboard_round_trip(tmp_path):
    suite = make_suite()
    cert = Certifier().certify("T", suite, bcs=GOOD_BCS)
    sub = LeaderboardSubmitter("T", "https://example.org/paper", "https://example.org/code").build(suite, cert)
    assert LeaderboardSubmitter.validate(sub) == []
    json.loads(LeaderboardSubmitter.to_json(sub))
    lb = Leaderboard(tmp_path)
    lb.add(sub)
    worse = make_suite(test_mean=70.0, seed=5)
    worse["U"] = worse.pop("T")
    lb.add(LeaderboardSubmitter("U", "", "https://example.org/u").build(
        worse, Certifier(require_bcs_for_c1=False).certify("U", worse)))
    rows = lb.build()
    assert [r["algorithm"] for r in rows] == ["T", "U"]
    assert (tmp_path / "LEADERBOARD.md").read_text().count("| C1 |") == 2


def test_validate_rejects_inconsistent_certificates():
    suite = make_suite()
    cert = Certifier(require_bcs_for_c1=False).certify("T", make_suite(test_mean=400.0))
    sub = LeaderboardSubmitter("T", "", "x").build(suite, cert)
    sub["certification"]["level"] = "C1"                      # forged level
    assert any("C1 checks is false" in p for p in LeaderboardSubmitter.validate(sub))
    with pytest.raises(ValueError):
        Leaderboard("unused").add(sub)


def test_replay_bcs_adds_integrity_condition_and_starvation_rule():
    ok = BCSCalculator.compute_replay(0.94, 1.0, 80, 14_315, rho_shuffle=0.94,
                                      matched_ratios={"B0_Random": 1.0, "B3_TS": 0.99})
    assert ok["c_R_replay_integrity"] == 1.0 and ok["band_R"] == ok["band"]
    bad = BCSCalculator.compute_replay(0.94, 1.0, 80, 14_315, rho_shuffle=0.2,
                                       matched_ratios={"B0_Random": 1.0, "B2_UCB1": 0.18})
    assert bad["BCS_R"] < ok["BCS_R"] and bad["starved_algorithms"] == ["B2_UCB1"]
    assert any("Replay integrity" in f for f in bad["failing_conditions_R"])
    # averaging alone would pass an order-dependent replay; the hard requirement does not
    avg_only = BCSCalculator.compute_replay(0.94, 1.0, 80, 14_315, rho_shuffle=0.657,
                                            matched_ratios={"B0_Random": 1.0})
    assert avg_only["BCS_R"] >= 0.9 and avg_only["band_R"] == "PASS" and not avg_only["certifiable"]
    und = BCSCalculator.compute_replay(0.94, 1.0, 80, 14_315, 0.94, {}, tier1_bias_free=False)
    assert und["band_R"] == "UNDEFINED" and not und["certifiable"]


def test_certifier_uses_bcs_r_and_the_starvation_rule():
    import numpy as np
    from mab_benchmark.analysis.certification import Certifier
    rng = np.random.default_rng(0)
    cells = [f"{s}_K10_T500" for s in ("S1", "S2", "S3", "S4", "S5")]
    suite = {b: {c: rng.normal(100, 5, 30) for c in cells} for b in ("B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS")}
    suite["X"] = {c: rng.normal(50, 5, 30) for c in cells}
    cert = Certifier(require_bcs_for_c1=True)
    passing = {"BCS": 0.95, "BCS_R": 0.95, "c1_reward_proxy": 1.0, "exploration_cost": 0.05, "starved_algorithms": []}
    failing = dict(passing, BCS_R=0.80)
    assert cert.certify("X", suite, bcs=passing)["level"] == "C1"
    r = cert.certify("X", suite, bcs=failing)
    assert r["level"] == "UNCERTIFIED" and any("0.8000" in x for x in r["reasons_not_higher"])
    tier2 = {"X": rng.normal(0.02, 0.001, 30), "B0_Random": rng.normal(0.01, 0.001, 30)}
    ok = cert.certify("X", suite, bcs=passing, tier2=tier2)
    starved = cert.certify("X", suite, bcs=dict(passing, starved_algorithms=["X"]), tier2=tier2)
    assert ok["level"] == "C3" and starved["level"] == "C1"
    order_dep = cert.certify("X", suite, bcs=dict(passing, rho_shuffle=0.657))
    assert order_dep["level"] == "UNCERTIFIED" and any("replay integrity" in x for x in order_dep["reasons_not_higher"])
    assert any("starved" in x for x in starved["reasons_not_higher"])
