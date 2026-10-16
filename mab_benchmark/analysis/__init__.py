"""
mab_benchmark.analysis
=======================

Three modules:

statistics   — StatisticalAnalyser (all four requirements)
bcs          — BCSCalculator       (bridge compliance score)
leaderboard  — LeaderboardSubmitter (JSON submission)
guf          — GUF, CCM, Pareto front, weight sweep, extended BCS
ips          — IPS and clipped IPS estimators
certification — Certifier, C1/C3/C4
"""

from mab_benchmark.analysis.statistics import StatisticalAnalyser
from mab_benchmark.analysis.bcs import BCSCalculator
from mab_benchmark.analysis.leaderboard import LeaderboardSubmitter, Leaderboard
from mab_benchmark.analysis.certification import Certifier
from mab_benchmark.analysis import guf, ips

__all__ = [
    "StatisticalAnalyser",
    "BCSCalculator",
    "LeaderboardSubmitter",
    "Leaderboard",
    "Certifier",
]
