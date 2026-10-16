"""
mab_benchmark.analysis.bcs
===========================

BCSCalculator — Bridge Compliance Score.

Source: TwoTierProtocol_GAP5.docx, Section E (three-condition checklist);
        MAB_BenchmarkSuite.pdf, Section 4 (BCS derivation).

What the BCS is
---------------
The BCS collapses the three bridge conditions from TwoTierProtocol
Section E into one continuous number in [0, 1].  It answers the
question: how reliably does a Tier 1 (offline regret) result predict
Tier 2 (deployment) performance?

    BCS = 1  →  all three conditions fully satisfied;
              Tier 1 is a reliable proxy for deployment.
    BCS = 0  →  all three conditions failed;
              Tier 1 and Tier 2 results are likely to diverge.

Formula
-------
    c1 = clip(rho_S / 0.7,   0, 1)   # reward proxy alignment
    c2 = clip(p_indep / 0.9, 0, 1)   # arm independence
    c3 = clip(1 - e / 0.20,  0, 1)   # sufficient horizon
    BCS = (c1 + c2 + c3) / 3

where clip(x, lo, hi) = max(lo, min(hi, x)).

Interpretation bands
--------------------
    BCS >= 0.9  →  PASS:    Tier 1 reliable proxy. Tier 2 recommended.
    0.6 <= BCS  →  PARTIAL: Tier 2 required for deployment claims.
    BCS < 0.6   →  FAIL:    Tiers likely to diverge. Flag in paper.

Replay settings (S4 on logged data, mandatory since 1.3.0)
----------------------------------------------------------
When Tier 2 is a replay of logged data, the replayer's i.i.d. assumption
(Li et al., 2011) must itself be checked, because a bursty log favours
exploitation (Pires et al., 2025; Chapter Four, Section 4.6). compute_replay()
adds a fourth condition and a starvation rule:

    cR = clip(rho_shuffle / 0.7, 0, 1)   # replay integrity: Tier 2 ranking
                                         # with the log as recorded vs with
                                         # rows permuted inside each hour
    BCS_R = (c1 + c2 + c3 + cR) / 4
    starved(a) = m_a / m_rand < 0.5      # matched events of a vs B0 Random

A setting is certifiable only if BCS_R >= 0.9 AND rho_shuffle >= 0.7: the
integrity condition is a hard requirement as well as a term of the average,
because averaging alone lets an order-dependent replay pass (on KuaiRec, a
bursty log with rho_shuffle = 0.657 would still average to 0.918 if c1 were
met; Chapter Four). The scheme also
requires Tier 1 to be a bias-free environment (a simulator calibrated on
the same domain, or a fully observed dataset), not another replay. A starved
algorithm's Tier 2 result is not certifiable, whatever the BCS.
"""

from __future__ import annotations
import math


class BCSCalculator:
    """
    Compute the Bridge Compliance Score (BCS) from the three conditions
    defined in TwoTierProtocol_GAP5.docx Section E.

    All thresholds are fixed by the specification and must not be changed
    without updating the benchmark version number.

    Thresholds
    ----------
    Condition 1 (reward proxy):    rho_S >= 0.7
    Condition 2 (arm independence): p_indep >= 0.9
    Condition 3 (sufficient horizon): e <= 0.20
    """

    # Fixed thresholds from TwoTierProtocol_GAP5.docx Section E
    THRESHOLD_C1 = 0.7    # Spearman rho between Tier 1 and Tier 2 rankings
    THRESHOLD_C2 = 0.9    # proportion of arm pairs with |rho_ij| <= 0.3
    THRESHOLD_C3 = 0.20   # UCB1 exploration cost ratio O(K log T) / T

    # BCS interpretation bands
    BAND_PASS    = 0.9
    BAND_PARTIAL = 0.6

    @staticmethod
    def _clip(x: float) -> float:
        """Clip x to [0, 1]."""
        return max(0.0, min(1.0, float(x)))

    @classmethod
    def compute(
        cls,
        rho_s:   float,
        p_indep: float,
        K:       int,
        T:       int,
    ) -> dict:
        """
        Compute the BCS from the three raw condition statistics.

        Parameters
        ----------
        rho_s : float in [-1, 1]
            Spearman rank correlation between the Tier 1 ranking of the
            mandatory baselines (by cumulative regret) and the Tier 2
            ranking of the same baselines (by CTR or nDCG).
            Compute using scipy.stats.spearmanr on the two ranking lists.

        p_indep : float in [0, 1]
            Proportion of arm pairs (i, j) in the real-world dataset
            where the Spearman rank correlation of their reward sequences
            satisfies |rho_ij| <= 0.3.
            If using a synthetic environment, set p_indep=1.0.

        K : int
            Number of arms in the benchmark setting.
            Used to compute the UCB1 exploration cost ratio.

        T : int
            Time horizon of the benchmark setting.
            Used to compute the UCB1 exploration cost ratio.

        Returns
        -------
        dict with keys:
            BCS                  : float in [0,1]  — the score
            c1_reward_proxy      : float in [0,1]  — normalised Condition 1
            c2_arm_independence  : float in [0,1]  — normalised Condition 2
            c3_sufficient_horizon: float in [0,1]  — normalised Condition 3
            exploration_cost     : float            — O(K log T) / T
            band                 : str             — "PASS", "PARTIAL", "FAIL"
            flag                 : str             — interpretive message
            failing_conditions   : list[str]       — which conditions failed
            action_required      : str             — what to do
        """
        # ----- Condition 1: reward proxy alignment -----
        c1 = cls._clip(rho_s / cls.THRESHOLD_C1)

        # ----- Condition 2: arm independence -----
        c2 = cls._clip(p_indep / cls.THRESHOLD_C2)

        # ----- Condition 3: sufficient horizon -----
        # UCB1 exploration cost: O(K log T) / T
        # The constant in O(K log T) is taken as 1 for the benchmark.
        # For very small T (e.g. S5), this will be large — by design.
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}.")
        e  = (K * math.log(max(T, 2))) / T
        c3 = cls._clip(1.0 - e / cls.THRESHOLD_C3)

        # ----- BCS -----
        bcs = (c1 + c2 + c3) / 3.0

        # ----- Band and flag -----
        if bcs >= cls.BAND_PASS:
            band   = "PASS"
            flag   = (
                "All three bridge conditions satisfied. "
                "Tier 1 result is a reliable proxy for deployment. "
                "Tier 2 validation is recommended but not required "
                "for C1 algorithmic claims."
            )
            action = "None required for C1 claims."
        elif bcs >= cls.BAND_PARTIAL:
            band   = "PARTIAL"
            flag   = (
                "One or more bridge conditions partially failed. "
                "Tier 2 (deployment) validation is required before "
                "making any deployment performance claims."
            )
            action = "Run Tier 2 evaluation. Address failing conditions."
        else:
            band   = "FAIL"
            flag   = (
                "Bridge conditions failed. Tier 1 and Tier 2 results "
                "are likely to diverge. Report BCS in the paper. "
                "C3 (general superiority) certification is not possible "
                "without addressing the failing conditions."
            )
            action = (
                "Mandatory: Tier 2 required. Address failing conditions. "
                "For Condition 3 failure: extend T or reduce K."
            )

        # ----- Identify failing conditions -----
        failures = []
        if rho_s < cls.THRESHOLD_C1:
            failures.append(
                f"Condition 1 (reward proxy): "
                f"rho_S={rho_s:.3f} < threshold {cls.THRESHOLD_C1}. "
                f"c1={c1:.3f}. "
                "Flag domain as 'high deployment risk'."
            )
        if p_indep < cls.THRESHOLD_C2:
            failures.append(
                f"Condition 2 (arm independence): "
                f"p_indep={p_indep:.3f} < threshold {cls.THRESHOLD_C2}. "
                f"c2={c2:.3f}. "
                "Add correlated-arm algorithm (graph bandit / GP-UCB)."
            )
        if e > cls.THRESHOLD_C3:
            failures.append(
                f"Condition 3 (sufficient horizon): "
                f"e={e:.4f} > threshold {cls.THRESHOLD_C3}. "
                f"c3={c3:.3f}. "
                "Pre-asymptotic regime: extend T or reduce K."
            )

        return {
            "BCS":                    round(bcs, 4),
            "c1_reward_proxy":        round(c1,  4),
            "c2_arm_independence":    round(c2,  4),
            "c3_sufficient_horizon":  round(c3,  4),
            "exploration_cost":       round(e,   4),
            "band":                   band,
            "flag":                   flag,
            "failing_conditions":     failures,
            "action_required":        action,
        }

    THRESHOLD_CR = 0.7          # Spearman rho, logged vs within-hour-shuffled replay
    STARVATION_FLOOR = 0.5      # matched events relative to B0 Random

    @classmethod
    def compute_replay(cls, rho_s: float, p_indep: float, K: int, T: int, rho_shuffle: float,
                       matched_ratios: dict, tier1_bias_free: bool = True) -> dict:
        """BCS for a replay-based Tier 2 (see the module docstring)."""
        base = cls.compute(rho_s, p_indep, K, T)
        cR = cls._clip(rho_shuffle / cls.THRESHOLD_CR)
        bcs_r = (base["c1_reward_proxy"] + base["c2_arm_independence"] + base["c3_sufficient_horizon"] + cR) / 4.0
        starved = sorted(a for a, r in matched_ratios.items() if r < cls.STARVATION_FLOOR)
        failures = list(base["failing_conditions"])
        if rho_shuffle < cls.THRESHOLD_CR:
            failures.append(f"Replay integrity: rho_shuffle={rho_shuffle:.3f} < {cls.THRESHOLD_CR}; "
                            "the replay ranking depends on the order of the log.")
        if starved:
            failures.append("Starved in replay (matched events < 0.5 x Random): " + ", ".join(starved))
        if not tier1_bias_free:
            failures.append("Tier 1 is not a bias-free environment; the BCS is undefined.")
        band = ("PASS" if bcs_r >= cls.BAND_PASS else "PARTIAL" if bcs_r >= cls.BAND_PARTIAL else "FAIL")
        if not tier1_bias_free:
            band = "UNDEFINED"
        return base | {"c_R_replay_integrity": round(cR, 4), "rho_shuffle": float(rho_shuffle),
                       "BCS_R": round(bcs_r, 4), "band_R": band,
                       "starved_algorithms": starved, "matched_ratios": {k: float(v) for k, v in matched_ratios.items()},
                       "failing_conditions_R": failures,
                       "replay_integrity_met": bool(rho_shuffle >= cls.THRESHOLD_CR),
                       "certifiable": band == "PASS" and rho_shuffle >= cls.THRESHOLD_CR}

    @classmethod
    def from_ranking_lists(
        cls,
        tier1_ranking: list[str],
        tier2_ranking: list[str],
        p_indep: float,
        K: int,
        T: int,
    ) -> dict:
        """
        Convenience method: compute BCS from ranking lists directly.

        Parameters
        ----------
        tier1_ranking : list[str]
            Algorithm names ordered best-to-worst by cumulative regret
            (lowest regret first).  Must contain the same names as
            tier2_ranking.

        tier2_ranking : list[str]
            Algorithm names ordered best-to-worst by domain metric
            (highest CTR / nDCG first).

        p_indep, K, T : same as compute()

        Example
        -------
        >>> tier1 = ["UCB1", "TS", "LinUCB", "EpsGreedy", "Random"]
        >>> tier2 = ["LinUCB", "UCB1", "TS", "EpsGreedy", "Random"]
        >>> BCSCalculator.from_ranking_lists(tier1, tier2, 0.93, 270, 10000)
        """
        from scipy.stats import spearmanr

        if set(tier1_ranking) != set(tier2_ranking):
            raise ValueError(
                "tier1_ranking and tier2_ranking must contain "
                "the same algorithm names."
            )

        # Convert to rank vectors aligned by name
        names     = tier1_ranking
        rank1     = list(range(1, len(names) + 1))
        rank2_map = {name: i + 1 for i, name in enumerate(tier2_ranking)}
        rank2     = [rank2_map[name] for name in names]

        rho_s, _ = spearmanr(rank1, rank2)
        return cls.compute(float(rho_s), p_indep, K, T)

    def format_result(self, result: dict) -> str:
        """Return a human-readable string for a BCS result dict."""
        lines = [
            "=" * 60,
            "BRIDGE COMPLIANCE SCORE (BCS)",
            "=" * 60,
            f"  BCS           : {result['BCS']:.4f}  [{result['band']}]",
            f"  c1 (proxy)    : {result['c1_reward_proxy']:.4f}",
            f"  c2 (indep)    : {result['c2_arm_independence']:.4f}",
            f"  c3 (horizon)  : {result['c3_sufficient_horizon']:.4f}",
            f"  Expl. cost    : {result['exploration_cost']:.4f}",
            "",
            f"  Flag   : {result['flag']}",
            f"  Action : {result['action_required']}",
        ]
        if result["failing_conditions"]:
            lines.append("")
            lines.append("  Failing conditions:")
            for fc in result["failing_conditions"]:
                lines.append(f"    • {fc}")
        lines.append("=" * 60)
        return "\n".join(lines)
