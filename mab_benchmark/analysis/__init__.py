"""
mab_benchmark.analysis
=======================

Three modules:

statistics   — StatisticalAnalyser (all four requirements)
bcs          — BCSCalculator       (bridge compliance score)
leaderboard  — LeaderboardSubmitter (Papers With Code JSON)
"""

from mab_benchmark.analysis.statistics import StatisticalAnalyser
from mab_benchmark.analysis.bcs import BCSCalculator
from mab_benchmark.analysis.leaderboard import LeaderboardSubmitter

__all__ = [
    "StatisticalAnalyser",
    "BCSCalculator",
    "LeaderboardSubmitter",
]
