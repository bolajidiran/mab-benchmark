"""
mab_benchmark.baselines
=======================

Four mandatory baseline algorithms as specified in GAP5 Contested Zone B
and BenchmarkSpec_Vermorel2005.docx, Gaps A-1, A-2.

Every submission must compare against all four.  A submission omitting
any one is automatically marked 'Baseline Incomplete' by the runner.

Baselines
---------
B0  RandomPolicy         Uniform random arm selection
B1  EpsilonGreedy        Epsilon-greedy (epsilon=0.1)
B2  UCB1                 Auer et al. (2002), O(log T) regret bound
B3  ThompsonSampling     Beta-Bernoulli (S1,S3,S5) or Normal-InvGamma (S2)
B4  LinUCB               Li et al. (2010), contextual baseline (S4 only)
"""

from mab_benchmark.baselines.b0_random import RandomPolicy
from mab_benchmark.baselines.b1_epsilon_greedy import EpsilonGreedy
from mab_benchmark.baselines.b2_ucb1 import UCB1
from mab_benchmark.baselines.b3_thompson_sampling import ThompsonSampling
from mab_benchmark.baselines.b4_linucb import LinUCB

MANDATORY_BASELINES = {
    "B0_Random":    RandomPolicy,
    "B1_EpsGreedy": EpsilonGreedy,
    "B2_UCB1":      UCB1,
    "B3_TS":        ThompsonSampling,
}

CONTEXTUAL_BASELINE = {
    "B4_LinUCB": LinUCB,
}

__all__ = [
    "RandomPolicy",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonSampling",
    "LinUCB",
    "MANDATORY_BASELINES",
    "CONTEXTUAL_BASELINE",
]
