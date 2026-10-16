"""
mab_benchmark.analysis.certification
=====================================

Certifier: issues the C1, C3 and C4 certification levels of the MAB Unified
Benchmark Suite (thesis Section 3.9.7; TwoTierProtocol_GAP5.docx, Section G).

A certificate is a claim about the submitted algorithm against the mandatory
baselines B0 to B3 on all five settings, so the Certifier evaluates exactly
that, one (cell, baseline) claim at a time.

Levels and their conditions
---------------------------
C1  Algorithmic Efficiency (Tier 1)
    a. results for every one of the five settings S1 to S5;
    b. the four mandatory baselines B0 to B3 in every cell;
    c. n >= 30 paired runs in every cell;
    d. in at least one cell, the submitted algorithm has significantly lower
       cumulative regret than a mandatory baseline (two-sided Wilcoxon,
       Holm-Bonferroni within the cell, alpha = 0.05) with |Cohen's d| >= 0.2;
    e. BCS >= 0.9 (thesis Sections 3.9.5 and 3.9.7). TwoTierProtocol Section G
       does not ask for e.; pass require_bcs_for_c1=False to apply that reading.

C3  General Superiority (Tier 1 and Tier 2)
    C1, and for the SAME baseline B: significantly higher Tier 2 metric (CTR on
    the replay) at alpha = 0.05 with a relative improvement of at least 1%,
    and bridge conditions 1 (rho_S >= 0.7) and 3 (e <= 0.20) satisfied.

C4  Deployment Validated
    C3, and a live A/B test against the same baseline with >= 1,000 users,
    >= 7 days, the protocol disclosed, and a significantly higher CTR
    (one-sided two-proportion z-test, p < 0.05).

Every certificate lists the individual claims it rests on, so a reader can
see which (setting, baseline) pairs carry the claim and which do not.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from scipy import stats as scipy_stats

from mab_benchmark.analysis.statistics import StatisticalAnalyser

MANDATORY_BASELINES = ("B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS")
SETTINGS = ("S1", "S2", "S3", "S4", "S5")
LEVELS = ("UNCERTIFIED", "C1", "C3", "C4")


class Certifier:
    """Issue C1 / C3 / C4 for one submitted algorithm.

    Parameters
    ----------
    alpha : family-wise error rate per cell (Holm-Bonferroni).
    d_min : smallest |Cohen's d| that may be claimed (0.2, BenchmarkSpec S-3).
    tier2_min_improvement : minimum relative Tier 2 improvement (0.01 = 1%).
    require_bcs_for_c1 : thesis reading (True) or TwoTierProtocol reading (False).
    bcs_threshold : BCS needed for C1 when required (0.9).
    ab_min_users, ab_min_days : C4 sample requirements (1,000 users, 7 days).
    """

    def __init__(self, alpha: float = 0.05, d_min: float = 0.2,
                 tier2_min_improvement: float = 0.01, require_bcs_for_c1: bool = True,
                 bcs_threshold: float = 0.9, ab_min_users: int = 1000,
                 ab_min_days: int = 7) -> None:
        self.alpha = alpha
        self.d_min = d_min
        self.tier2_min_improvement = tier2_min_improvement
        self.require_bcs_for_c1 = require_bcs_for_c1
        self.bcs_threshold = bcs_threshold
        self.ab_min_users = ab_min_users
        self.ab_min_days = ab_min_days

    # ------------------------------------------------------------------
    # Tier 1
    # ------------------------------------------------------------------
    def tier1_claims(self, test: str, suite: dict[str, dict[str, np.ndarray]]) -> list[dict]:
        """One claim per (cell, mandatory baseline) present in `suite`.

        suite : {algorithm: {cell_key: regret array (n_runs,)}}, the format of
                BenchmarkRunner.baseline_sweep() plus the test algorithm's
                BenchmarkRunner.full_suite() under its own name.
        """
        claims = []
        for cell in suite[test]:
            algs = [test] + [b for b in MANDATORY_BASELINES if b in suite and cell in suite[b]]
            if len(algs) < 2:
                continue
            res = {a: np.asarray(suite[a][cell], float) for a in algs}
            rep = StatisticalAnalyser(self.alpha).analyse(res)
            for b in algs[1:]:
                key = (test, b) if (test, b) in rep["pairwise"] else (b, test)
                pw, es = rep["pairwise"][key], rep["effect_sizes"][key]
                d_test_minus_b = es["cohens_d"] if key[0] == test else -es["cohens_d"]
                mt, mb = float(res[test].mean()), float(res[b].mean())
                if not pw["significant"]:
                    verdict = "not significant"
                elif abs(d_test_minus_b) < self.d_min:
                    verdict = "negligible"
                else:
                    verdict = "better" if mt < mb else "worse"
                claims.append(OrderedDict(
                    setting=cell, baseline=b, test_mean=round(mt, 4), baseline_mean=round(mb, 4),
                    p_raw=float(pw["p_raw"]), holm_threshold=float(pw["p_threshold"]),
                    significant=bool(pw["significant"]), cohens_d=round(float(d_test_minus_b), 4),
                    magnitude=es["magnitude"], verdict=verdict))
        return claims

    # ------------------------------------------------------------------
    # Tier 2 (replay CTR, higher is better)
    # ------------------------------------------------------------------
    def tier2_claims(self, test: str, tier2: dict[str, np.ndarray]) -> list[dict]:
        algs = [test] + [b for b in MANDATORY_BASELINES if b in tier2]
        res = {a: np.asarray(tier2[a], float) for a in algs}
        rep = StatisticalAnalyser(self.alpha).analyse({a: -v for a, v in res.items()})
        out = []
        for b in algs[1:]:
            key = (test, b) if (test, b) in rep["pairwise"] else (b, test)
            pw = rep["pairwise"][key]
            mt, mb = float(res[test].mean()), float(res[b].mean())
            rel = (mt - mb) / mb if mb else float("inf")
            ok = pw["significant"] and mt > mb and rel >= self.tier2_min_improvement
            out.append(OrderedDict(
                baseline=b, test_ctr=mt, baseline_ctr=mb, relative_improvement=round(rel, 4),
                p_raw=float(pw["p_raw"]), holm_threshold=float(pw["p_threshold"]),
                significant=bool(pw["significant"]),
                verdict="better" if ok else ("worse" if pw["significant"] and mt < mb else "not established")))
        return out

    # ------------------------------------------------------------------
    # Live A/B test
    # ------------------------------------------------------------------
    def ab_test(self, ab: dict) -> dict:
        """ab = {baseline, users, days, protocol_disclosed,
                 clicks_test, impressions_test, clicks_baseline, impressions_baseline}"""
        n1, n2 = ab["impressions_test"], ab["impressions_baseline"]
        p1, p2 = ab["clicks_test"] / n1, ab["clicks_baseline"] / n2
        p = (ab["clicks_test"] + ab["clicks_baseline"]) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else float("inf")
        z = (p1 - p2) / se if se > 0 else 0.0
        pval = float(scipy_stats.norm.sf(z))
        checks = {"users >= %d" % self.ab_min_users: ab["users"] >= self.ab_min_users,
                  "days >= %d" % self.ab_min_days: ab["days"] >= self.ab_min_days,
                  "protocol disclosed": bool(ab.get("protocol_disclosed", False)),
                  "CTR higher, one-sided p < %.2f" % self.alpha: p1 > p2 and pval < self.alpha}
        return {"baseline": ab["baseline"], "ctr_test": p1, "ctr_baseline": p2,
                "relative_lift": (p1 - p2) / p2 if p2 else float("inf"), "z": z, "p_one_sided": pval,
                "checks": checks, "passed": all(checks.values())}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def certify(self, test: str, suite: dict, bcs: dict | None = None,
                tier2: dict | None = None, ab_test: dict | None = None) -> dict:
        if test not in suite:
            raise ValueError(f"{test!r} is not in the suite results.")
        cells = list(suite[test])
        checks = OrderedDict()
        checks["all five settings S1 to S5"] = all(any(c.startswith(s + "_") for c in cells) for s in SETTINGS)
        missing = sorted({b for b in MANDATORY_BASELINES for c in cells if b not in suite or c not in suite[b]})
        checks["mandatory baselines B0 to B3 in every cell"] = not missing
        n_min = min(len(np.asarray(v)) for a in suite for v in suite[a].values())
        checks["n >= 30 runs in every cell"] = n_min >= 30

        reasons = []
        if not checks["all five settings S1 to S5"]:
            reasons.append("missing settings: " + ", ".join(s for s in SETTINGS
                                                              if not any(c.startswith(s + "_") for c in cells)))
        if missing:
            reasons.append("baseline incomplete: " + ", ".join(missing))
        if n_min < 30:
            reasons.append(f"only {n_min} runs in some cell")

        claims = self.tier1_claims(test, suite) if n_min >= 30 else []
        better = [c for c in claims if c["verdict"] == "better"]
        checks["significant, non-negligible regret advantage over a baseline"] = bool(better)
        if not better:
            reasons.append("no significant, non-negligible advantage over any mandatory baseline")

        # A replay-based Tier 2 is gated by BCS_R (replay integrity included), since 1.3.0.
        bcs_val = None if bcs is None else float(bcs.get("BCS_R", bcs.get("BCS", float("nan"))))
        starved = bool(bcs) and test in (bcs.get("starved_algorithms") or [])
        if self.require_bcs_for_c1:
            checks[f"BCS >= {self.bcs_threshold}"] = bcs_val is not None and bcs_val >= self.bcs_threshold
            if not checks[f"BCS >= {self.bcs_threshold}"]:
                reasons.append("BCS not computed" if bcs_val is None
                               else f"BCS = {bcs_val:.4f} < {self.bcs_threshold}")
            if bcs and "rho_shuffle" in bcs:
                ok_r = float(bcs["rho_shuffle"]) >= 0.7
                checks["replay integrity rho_shuffle >= 0.7"] = ok_r
                if not ok_r:
                    reasons.append(f"replay integrity failed: rho_shuffle = {float(bcs['rho_shuffle']):.3f} < 0.7")
        c1 = all(checks.values())

        # ---- C3 ----
        t2 = self.tier2_claims(test, tier2) if tier2 else []
        tier1_beaten = {c["baseline"] for c in better}
        c3_pairs = [c["baseline"] for c in t2 if c["verdict"] == "better" and c["baseline"] in tier1_beaten]
        bridge13 = bool(bcs) and bcs.get("c1_reward_proxy", 0) >= 1.0 and bcs.get("exploration_cost", 1) <= 0.20
        c3 = c1 and bool(c3_pairs) and bridge13 and not starved
        c3_reasons = []
        if c1 and starved:
            c3_reasons.append("starved in the Tier 2 replay (matched events < 0.5 x Random); Tier 2 not certifiable")
        if c1 and not c3 and not starved:
            if not tier2:
                c3_reasons.append("no Tier 2 results supplied")
            elif not c3_pairs:
                c3_reasons.append("no baseline beaten on both tiers")
            if not bridge13:
                c3_reasons.append("bridge condition 1 (rho_S >= 0.7) or 3 (e <= 0.20) not satisfied")

        # ---- C4 ----
        ab = self.ab_test(ab_test) if ab_test else None
        c4 = c3 and ab is not None and ab["passed"] and ab["baseline"] in c3_pairs
        level = "C4" if c4 else "C3" if c3 else "C1" if c1 else "UNCERTIFIED"
        return OrderedDict(
            algorithm=test, level=level, label=_LABEL[level],
            checks=checks, reasons_not_higher=reasons + c3_reasons,
            tier1_summary=_summarise(claims), tier1_claims=claims,
            tier2_claims=t2, ab_test=ab, bcs=bcs,
            rules={"alpha": self.alpha, "d_min": self.d_min,
                   "tier2_min_improvement": self.tier2_min_improvement,
                   "require_bcs_for_c1": self.require_bcs_for_c1, "bcs_threshold": self.bcs_threshold})


_LABEL = {"UNCERTIFIED": "Not certified",
          "C1": "C1 Algorithmic Efficiency Certified",
          "C3": "C3 General Superiority Certified",
          "C4": "C4 Deployment Validated"}


def _summarise(claims: list[dict]) -> dict:
    out = OrderedDict()
    for b in MANDATORY_BASELINES:
        cb = [c for c in claims if c["baseline"] == b]
        if cb:
            out[b] = {v: sum(c["verdict"] == v for c in cb)
                      for v in ("better", "worse", "negligible", "not significant")}
            out[b]["cells"] = len(cb)
    return out
