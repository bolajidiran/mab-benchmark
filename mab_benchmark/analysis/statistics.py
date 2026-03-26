"""
mab_benchmark.analysis.statistics
==================================

StatisticalAnalyser — applies all four mandatory statistical requirements
to benchmark results in a single call.

Four requirements (GAP5 Pillar 3; BenchmarkSpec S-1 through S-5)
-----------------------------------------------------------------
1. Summary statistics: mean ± 95% CI across >= 30 independent runs.
2. Wilcoxon signed-rank test for every pairwise algorithm comparison.
3. Holm-Bonferroni correction across all C(N,2) pairwise comparisons.
4. Effect sizes: Cohen's d and A12 statistic for every significant pair.

Usage
-----
>>> from mab_benchmark.analysis import StatisticalAnalyser
>>> analyser = StatisticalAnalyser()
>>> report   = analyser.analyse(results)      # results: {alg: regret_array}
>>> print(report["summary"]["UCB1"]["mean"])
>>> print(report["pairwise"][("UCB1","TS")]["significant"])
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy import stats as scipy_stats


class StatisticalAnalyser:
    """
    Apply all four statistical requirements to benchmark results.

    Parameters
    ----------
    alpha : float
        Family-wise error rate for Holm-Bonferroni correction.
        Default: 0.05 (as specified in BenchmarkSpec S-5).

    Methods
    -------
    analyse(results)
        Main entry point. Takes a dict {alg_name: regret_array} and
        returns a nested dict with all four requirement outputs.

    format_report(report)
        Return a human-readable string summary of the report.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}.")
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def analyse(
        self,
        results: dict[str, np.ndarray],
    ) -> dict:
        """
        Apply all four statistical requirements.

        Parameters
        ----------
        results : dict
            Maps algorithm name (str) to np.ndarray of shape (n_runs,)
            containing cumulative regret per run.
            Lower regret = better.

        Returns
        -------
        dict with keys:
            "summary"      — Requirement 1: summary statistics per algorithm
            "pairwise"     — Requirements 2 & 3: significance tests
            "effect_sizes" — Requirement 4: Cohen's d and A12
            "ranking"      — Algorithms ranked by mean regret (best first)
            "n_runs"       — Number of runs used
            "alpha"        — Significance threshold used
        """
        if not results:
            raise ValueError("results dict is empty.")

        n_runs_list = [len(v) for v in results.values()]
        if len(set(n_runs_list)) > 1:
            raise ValueError(
                "All algorithms must have the same number of runs. "
                f"Got run counts: { {k: len(v) for k,v in results.items()} }"
            )
        n = n_runs_list[0]
        if n < 30:
            raise ValueError(
                f"Requirement 1 violated: n_runs={n} < 30. "
                "Run at least 30 independent replications."
            )

        report = {
            "n_runs": n,
            "alpha":  self.alpha,
        }

        # Requirement 1 ------------------------------------------------
        report["summary"] = self._summary_stats(results, n)

        # Requirements 2 & 3 ------------------------------------------
        report["pairwise"] = self._pairwise_tests(results)

        # Requirement 4 ------------------------------------------------
        report["effect_sizes"] = self._effect_sizes(results,
                                                      report["pairwise"])

        # Ranking -------------------------------------------------------
        report["ranking"] = sorted(
            results.keys(),
            key=lambda k: report["summary"][k]["mean"],
        )

        return report

    # ------------------------------------------------------------------
    # Requirement 1: Summary statistics
    # ------------------------------------------------------------------

    def _summary_stats(
        self,
        results: dict[str, np.ndarray],
        n: int,
    ) -> dict:
        """
        Compute mean, std, 95% CI for each algorithm.

        Formula: mean ± z_{0.025} * std / sqrt(n)
        where z_{0.025} = 1.96 for n >= 30.
        """
        summary = {}
        z = 1.96   # z-score for 95% CI (valid for n >= 30)

        for name, arr in results.items():
            arr  = np.asarray(arr, dtype=float)
            m    = float(arr.mean())
            s    = float(arr.std(ddof=1))
            se   = s / np.sqrt(n)
            summary[name] = {
                "mean":     m,
                "std":      s,
                "se":       se,
                "ci_lower": m - z * se,
                "ci_upper": m + z * se,
                "n_runs":   n,
                "min":      float(arr.min()),
                "max":      float(arr.max()),
                "median":   float(np.median(arr)),
            }
        return summary

    # ------------------------------------------------------------------
    # Requirements 2 & 3: Wilcoxon + Holm-Bonferroni
    # ------------------------------------------------------------------

    def _pairwise_tests(
        self,
        results: dict[str, np.ndarray],
    ) -> dict:
        """
        Requirement 2: Wilcoxon signed-rank test for every pair.
        Requirement 3: Holm-Bonferroni correction across all pairs.

        For each pair (A, B):
          d_i = regret_A[i] - regret_B[i]
          H0: median(d_i) = 0  (no difference)
          Two-sided test.

        Returns
        -------
        dict mapping (alg_a, alg_b) -> {
            "stat"         : float  Wilcoxon test statistic
            "p_raw"        : float  uncorrected p-value
            "p_threshold"  : float  Holm-Bonferroni threshold
            "significant"  : bool   True if p_raw <= p_threshold
            "better"       : str    name of better algorithm (lower regret)
                                    or "equal" if not significant
        }
        """
        names = list(results.keys())
        pairs = list(combinations(names, 2))
        m     = len(pairs)

        if m == 0:
            return {}

        # Step 1: compute raw p-values for all pairs
        raw = {}
        for (a, b) in pairs:
            arr_a = np.asarray(results[a], dtype=float)
            arr_b = np.asarray(results[b], dtype=float)
            diff  = arr_a - arr_b

            # If all differences are zero, Wilcoxon raises an error
            if np.all(diff == 0):
                raw[(a, b)] = (0.0, 1.0)
                continue

            stat, p = scipy_stats.wilcoxon(
                arr_a, arr_b, alternative="two-sided"
            )
            raw[(a, b)] = (float(stat), float(p))

        # Step 2: Holm-Bonferroni correction
        # Sort pairs by ascending p-value
        sorted_pairs = sorted(pairs, key=lambda pair: raw[pair][1])

        pairwise = {}
        stop      = False   # once we fail to reject, stop rejecting

        for j, (a, b) in enumerate(sorted_pairs):
            stat, p = raw[(a, b)]
            # Holm threshold for rank j (0-indexed): alpha / (m - j)
            threshold = self.alpha / (m - j)

            if not stop and p <= threshold:
                significant = True
            else:
                stop        = True
                significant = False

            arr_a = np.asarray(results[a], dtype=float)
            arr_b = np.asarray(results[b], dtype=float)

            if significant:
                better = a if arr_a.mean() < arr_b.mean() else b
            else:
                better = "equal (not significant)"

            pairwise[(a, b)] = {
                "stat":        stat,
                "p_raw":       p,
                "p_threshold": threshold,
                "significant": significant,
                "better":      better,
            }

        return pairwise

    # ------------------------------------------------------------------
    # Requirement 4: Effect sizes
    # ------------------------------------------------------------------

    def _effect_sizes(
        self,
        results: dict[str, np.ndarray],
        pairwise: dict,
    ) -> dict:
        """
        Requirement 4: Cohen's d and A12 statistic for every pair.

        Cohen's d = (mean_A - mean_B) / pooled_std
        A12 = P(regret_A < regret_B)  (lower regret is better)

        Magnitude labels (Cohen's d):
            |d| < 0.2  → negligible
            |d| < 0.5  → small
            |d| < 0.8  → medium
            |d| >= 0.8 → large

        The label "negligible" is assigned regardless of significance —
        a difference with |d| < 0.2 must not be claimed as meaningful
        even if p < alpha (BenchmarkSpec S-3).
        """
        effect_sizes = {}

        for (a, b) in pairwise.keys():
            arr_a = np.asarray(results[a], dtype=float)
            arr_b = np.asarray(results[b], dtype=float)
            n     = len(arr_a)

            # Cohen's d (pooled standard deviation)
            s_a    = arr_a.std(ddof=1)
            s_b    = arr_b.std(ddof=1)
            s_pool = np.sqrt(((n - 1) * s_a**2 + (n - 1) * s_b**2)
                             / (2 * n - 2))
            d = ((arr_a.mean() - arr_b.mean()) / s_pool
                 if s_pool > 0 else 0.0)

            # A12 statistic: P(regret_A < regret_B)
            # (lower regret = better for algorithm A)
            wins = float(np.sum(
                arr_a[:, None] < arr_b[None, :]
            ))
            a12 = wins / (n * n)

            # Magnitude label
            abs_d = abs(d)
            if abs_d < 0.2:
                magnitude = "negligible"
            elif abs_d < 0.5:
                magnitude = "small"
            elif abs_d < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"

            sig = pairwise[(a, b)]["significant"]

            # Mandatory: flag negligible effects even when significant
            claim = "negligible — do not claim as meaningful"
            if sig and magnitude != "negligible":
                better = pairwise[(a, b)]["better"]
                claim  = f"significant and {magnitude} — {better} is better"
            elif not sig:
                claim = f"not significant (magnitude: {magnitude})"

            effect_sizes[(a, b)] = {
                "cohens_d":   round(float(d), 4),
                "a12":        round(float(a12), 4),
                "magnitude":  magnitude,
                "claim":      claim,
            }

        return effect_sizes

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    def format_report(self, report: dict) -> str:
        """
        Return a human-readable string summary of the statistical report.
        """
        lines = [
            "=" * 70,
            "MAB BENCHMARK — STATISTICAL ANALYSIS REPORT",
            f"n_runs={report['n_runs']}  alpha={report['alpha']}",
            "=" * 70,
            "",
            "REQUIREMENT 1 — Summary Statistics (mean cumulative regret)",
            "-" * 60,
        ]

        for name in report["ranking"]:
            s = report["summary"][name]
            lines.append(
                f"  {name:<25s}  "
                f"mean={s['mean']:>10.2f}  "
                f"std={s['std']:>8.2f}  "
                f"95%CI=[{s['ci_lower']:>10.2f}, {s['ci_upper']:>10.2f}]"
            )

        lines += [
            "",
            "REQUIREMENTS 2 & 3 — Wilcoxon Tests + Holm-Bonferroni",
            "-" * 60,
        ]

        for (a, b), res in report["pairwise"].items():
            sig_str = "* SIGNIFICANT" if res["significant"] else "  not sig."
            lines.append(
                f"  {a} vs {b}:  "
                f"p_raw={res['p_raw']:.4f}  "
                f"thresh={res['p_threshold']:.4f}  "
                f"{sig_str}  → better: {res['better']}"
            )

        lines += [
            "",
            "REQUIREMENT 4 — Effect Sizes",
            "-" * 60,
        ]

        for (a, b), eff in report["effect_sizes"].items():
            lines.append(
                f"  {a} vs {b}:  "
                f"d={eff['cohens_d']:>+.4f}  "
                f"A12={eff['a12']:.4f}  "
                f"[{eff['magnitude']}]  → {eff['claim']}"
            )

        lines += ["", "=" * 70]
        return "\n".join(lines)
