"""
mab_benchmark.analysis.leaderboard
====================================

LeaderboardSubmitter — generates the JSON file required for submission
to the MAB Unified Benchmark Suite leaderboard on Papers With Code.

Leaderboard URL
---------------
https://paperswithcode.com/benchmark/mab-unified-benchmark-suite

Submission rules (from MAB_BenchmarkSuite.pdf, Section 5)
----------------------------------------------------------
1. Run the full suite with BenchmarkRunner (n_runs >= 30).
2. Pass StatisticalAnalyser (all four requirements).
3. Compute BCS for Settings S4 / S5.
4. Generate this JSON file and submit to Papers With Code.
5. Link the leaderboard entry from the paper.

Certification levels (TwoTierProtocol_GAP5.docx, Section G)
-------------------------------------------------------------
C1  Tier 1 only    — algorithmic efficiency claim
C2  Tier 2 only    — deployment performance claim (not issued here)
C3  Both tiers     — general superiority (requires BCS >= 0.9 + Tier 2)
UNCERTIFIED        — requirements not met
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np


# Current benchmark specification version
BENCHMARK_VERSION = "1.0.0"

# Leaderboard URL
LEADERBOARD_URL = (
    "https://paperswithcode.com/benchmark/mab-unified-benchmark-suite"
)


class LeaderboardSubmitter:
    """
    Generate a Papers With Code leaderboard submission JSON.

    Usage
    -----
    >>> submitter = LeaderboardSubmitter(
    ...     algorithm_name="MyAlgorithm",
    ...     paper_url="https://arxiv.org/abs/XXXX.XXXXX",
    ...     code_url="https://github.com/user/repo",
    ... )
    >>> json_str = submitter.generate(results, stats_report, bcs_result)
    >>> submitter.save(json_str, "submission.json")
    >>> print(submitter.summary(json_str))
    """

    def __init__(
        self,
        algorithm_name: str,
        paper_url: str,
        code_url: str,
        institution: str = "",
        notes: str = "",
    ) -> None:
        self.algorithm_name = algorithm_name
        self.paper_url      = paper_url
        self.code_url       = code_url
        self.institution    = institution
        self.notes          = notes

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def generate(
        self,
        results:      dict[str, np.ndarray],
        stats_report: dict,
        bcs_result:   dict | None = None,
    ) -> str:
        """
        Generate the leaderboard submission JSON string.

        Parameters
        ----------
        results : dict
            Output of BenchmarkRunner.full_suite() or .run_setting().
            Maps setting_key -> np.ndarray(n_runs,) of cumulative regrets.

        stats_report : dict
            Output of StatisticalAnalyser.analyse() on results.
            Must contain "summary", "pairwise", "effect_sizes", "n_runs".

        bcs_result : dict or None
            Output of BCSCalculator.compute().
            Required for C3 certification and strongly recommended for
            all submissions involving S4 or S5.

        Returns
        -------
        str : JSON-formatted submission file content.
        """
        self._validate_inputs(results, stats_report)

        n_runs = stats_report["n_runs"]
        cert   = self._certification_level(stats_report, bcs_result)

        submission = {
            # ---- Metadata ----
            "benchmark":          f"MAB Unified Benchmark Suite v{BENCHMARK_VERSION}",
            "leaderboard_url":    LEADERBOARD_URL,
            "submission_date":    datetime.now(timezone.utc).isoformat(),

            # ---- Algorithm info ----
            "algorithm":   self.algorithm_name,
            "paper_url":   self.paper_url,
            "code_url":    self.code_url,
            "institution": self.institution,
            "notes":       self.notes,

            # ---- Run parameters ----
            "n_runs":    n_runs,
            "alpha":     stats_report.get("alpha", 0.05),

            # ---- Per-setting results ----
            "settings":  self._setting_results(results, stats_report),

            # ---- Ranking ----
            "algorithm_ranking_by_S1": stats_report.get("ranking", []),

            # ---- Pairwise significance ----
            "pairwise_tests": self._format_pairwise(
                stats_report["pairwise"]
            ),

            # ---- Effect sizes ----
            "effect_sizes": self._format_effect_sizes(
                stats_report["effect_sizes"]
            ),

            # ---- Bridge Compliance Score ----
            "bridge_compliance_score": (
                bcs_result if bcs_result is not None
                else {"note": "BCS not computed for this submission."}
            ),

            # ---- Certification ----
            "certification": cert,
        }

        return json.dumps(submission, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # Save and summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save(json_str: str, path: str) -> None:
        """Write the JSON string to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Submission saved to: {path}")

    @staticmethod
    def summary(json_str: str) -> str:
        """Return a one-page human-readable summary of the submission."""
        data   = json.loads(json_str)
        lines  = [
            "=" * 65,
            "LEADERBOARD SUBMISSION SUMMARY",
            "=" * 65,
            f"Algorithm   : {data['algorithm']}",
            f"Paper       : {data['paper_url']}",
            f"Code        : {data['code_url']}",
            f"Submitted   : {data['submission_date']}",
            f"n_runs      : {data['n_runs']}",
            f"Cert. level : {data['certification']}",
            "",
            "Primary metric: mean cumulative regret on S1 (lower = better)",
            "-" * 65,
        ]

        # Show S1 results
        for key, val in data["settings"].items():
            if key.startswith("S1"):
                lines.append(
                    f"  {key:<25s}  "
                    f"mean={val['mean_regret']:>10.2f}  "
                    f"95%CI=[{val['ci_lower']:>10.2f}, "
                    f"{val['ci_upper']:>10.2f}]"
                )

        # BCS
        bcs = data.get("bridge_compliance_score", {})
        if "BCS" in bcs:
            lines += [
                "",
                f"Bridge Compliance Score (BCS): {bcs['BCS']:.4f}  [{bcs['band']}]",
                f"  {bcs['flag']}",
            ]

        lines.append("=" * 65)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(results: dict, stats_report: dict) -> None:
        required_keys = {"summary", "pairwise", "effect_sizes", "n_runs"}
        missing = required_keys - set(stats_report.keys())
        if missing:
            raise ValueError(
                f"stats_report is missing keys: {missing}. "
                "Run StatisticalAnalyser.analyse() first."
            )
        if stats_report["n_runs"] < 30:
            raise ValueError(
                f"n_runs={stats_report['n_runs']} < 30. "
                "Certification requires at least 30 independent runs."
            )

    @staticmethod
    def _setting_results(
        results:      dict[str, np.ndarray],
        stats_report: dict,
    ) -> dict:
        """Format per-setting mean, CI, and std."""
        out = {}
        for key, arr in results.items():
            arr   = np.asarray(arr, dtype=float)
            n     = len(arr)
            m     = float(arr.mean())
            s     = float(arr.std(ddof=1))
            se    = s / n ** 0.5
            out[key] = {
                "mean_regret": round(m,            2),
                "std":         round(s,            2),
                "ci_lower":    round(m - 1.96 * se, 2),
                "ci_upper":    round(m + 1.96 * se, 2),
                "n_runs":      n,
            }
        return out

    @staticmethod
    def _format_pairwise(pairwise: dict) -> dict:
        """Serialise pairwise dict (tuple keys → string keys)."""
        return {
            f"{a}_vs_{b}": {
                "p_raw":       round(v["p_raw"],       5),
                "p_threshold": round(v["p_threshold"], 5),
                "significant": v["significant"],
                "better":      v["better"],
            }
            for (a, b), v in pairwise.items()
        }

    @staticmethod
    def _format_effect_sizes(effect_sizes: dict) -> dict:
        """Serialise effect sizes dict (tuple keys → string keys)."""
        return {
            f"{a}_vs_{b}": {
                "cohens_d":  v["cohens_d"],
                "a12":       v["a12"],
                "magnitude": v["magnitude"],
                "claim":     v["claim"],
            }
            for (a, b), v in effect_sizes.items()
        }

    @staticmethod
    def _certification_level(
        stats_report: dict,
        bcs_result:   dict | None,
    ) -> str:
        """
        Determine certification level from TwoTierProtocol Section G.

        C1  — Tier 1 only: algorithmic efficiency certified.
        C3  — Both tiers: general superiority (BCS >= 0.9 required).
        UNCERTIFIED — requirements not met.
        """
        n = stats_report.get("n_runs", 0)
        if n < 30:
            return (
                "UNCERTIFIED: n_runs < 30. "
                "Minimum 30 independent runs required."
            )

        # Check that at least one significant result exists
        pairwise   = stats_report.get("pairwise", {})
        any_sig    = any(v["significant"] for v in pairwise.values())
        effect_ok  = any(
            v["magnitude"] != "negligible"
            for v in stats_report.get("effect_sizes", {}).values()
        )

        if not any_sig:
            return (
                "UNCERTIFIED: No statistically significant pairwise "
                "difference found after Holm-Bonferroni correction."
            )

        if not effect_ok:
            return (
                "C1_NEGLIGIBLE: Statistically significant differences "
                "found but all have |d| < 0.2 (negligible effect). "
                "Do not claim practical superiority."
            )

        bcs_val = bcs_result.get("BCS", 0.0) if bcs_result else None

        if bcs_val is not None and bcs_val >= 0.9:
            return (
                f"C1: Algorithmic Efficiency Certified. "
                f"BCS={bcs_val:.4f} [PASS] — Tier 1 is a reliable "
                f"proxy for deployment. Tier 2 required for C3."
            )
        elif bcs_val is not None and bcs_val >= 0.6:
            return (
                f"C1: Algorithmic Efficiency Certified. "
                f"BCS={bcs_val:.4f} [PARTIAL] — Tier 2 required "
                f"before making any deployment performance claims."
            )
        elif bcs_val is not None:
            return (
                f"C1_FLAGGED: Algorithmic efficiency result certified "
                f"but BCS={bcs_val:.4f} [FAIL]. Tier 1 and Tier 2 "
                f"likely to diverge. Address failing BCS conditions."
            )
        else:
            return (
                "C1: Algorithmic Efficiency Certified. "
                "BCS not computed — required for deployment claims."
            )


# ------------------------------------------------------------------
# JSON serialisation helper for numpy types
# ------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
