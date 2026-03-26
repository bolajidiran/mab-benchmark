"""
mab_benchmark — MAB Unified Benchmark Suite v1.0
=================================================

Gap 5 Research Programme · March 2026

Quick start
-----------
>>> from mab_benchmark import BenchmarkRunner, StatisticalAnalyser
>>> from mab_benchmark import BCSCalculator, LeaderboardSubmitter
>>> from mab_benchmark.baselines import UCB1, ThompsonSampling
>>>
>>> runner  = BenchmarkRunner(n_runs=30)
>>> results = runner.full_suite(UCB1, {})
>>> report  = StatisticalAnalyser().analyse(results)

Sources (all in project repository)
------------------------------------
GAP5_EvalFramework.docx       — Pillars 3 & 4
TwoTierProtocol_GAP5.docx     — Section E bridge conditions
BenchmarkSpec_Vermorel2005    — 28-dimension specification
BenchmarkMatrix_Silva2022     — Priority matrix
"""

from mab_benchmark.core import BanditAlgorithm
from mab_benchmark.runner import BenchmarkRunner
from mab_benchmark.analysis.statistics import StatisticalAnalyser
from mab_benchmark.analysis.bcs import BCSCalculator
from mab_benchmark.analysis.leaderboard import LeaderboardSubmitter

__version__ = "1.0.0"
__all__ = [
    "BanditAlgorithm",
    "BenchmarkRunner",
    "StatisticalAnalyser",
    "BCSCalculator",
    "LeaderboardSubmitter",
]
