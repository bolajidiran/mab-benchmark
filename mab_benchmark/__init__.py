"""
mab_benchmark — MAB Unified Benchmark Suite v1.3
=================================================

Five benchmark settings (S1 to S5), four mandatory baselines (B0 to B3), the
statistical protocol, the GUF, CCM and BCS, the Certifier (C1, C3, C4) and the
repository leaderboard, the Meta-Policy Orchestrator (Layer 2) and the
counterfactual explainer and system performance monitor (Layer 6).

Gap 5 Research Programme · March 2026

Quick start
-----------
>>> from mab_benchmark import BenchmarkRunner, StatisticalAnalyser
>>> from mab_benchmark import BCSCalculator, LeaderboardSubmitter
>>> from mab_benchmark.baselines import UCB1, ThompsonSampling
>>>
>>> runner  = BenchmarkRunner(n_runs=30)
>>> results = runner.full_suite(UCB1, {})
>>> suite = runner.baseline_sweep(); suite["UCB1*"] = results   # {algorithm: {cell: regrets}}
>>> report  = StatisticalAnalyser().analyse({a: suite[a]["S1_K10_T10000"] for a in suite})
>>> from mab_benchmark import Certifier
>>> cert = Certifier().certify("UCB1*", suite, bcs=bcs_result)

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
from mab_benchmark.analysis.leaderboard import LeaderboardSubmitter, Leaderboard
from mab_benchmark.analysis.certification import Certifier
from mab_benchmark.orchestration import MetaPolicyOrchestrator, SystemPerformanceMonitor, with_mpo
from mab_benchmark.explain import CounterfactualExplainer

__version__ = "1.3.0"
__all__ = [
    "BanditAlgorithm",
    "BenchmarkRunner",
    "StatisticalAnalyser",
    "BCSCalculator",
    "LeaderboardSubmitter",
    "Leaderboard",
    "Certifier",
    "MetaPolicyOrchestrator",
    "SystemPerformanceMonitor",
    "with_mpo",
    "CounterfactualExplainer",
]
