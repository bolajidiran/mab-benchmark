"""
mab_benchmark.algorithms
========================

Algorithms beyond the mandatory baselines.

FeelGoodThompsonSampling   FG-TS, the thesis test algorithm (lambda = 0.5)
KLUCB                      KL-UCB for Bernoulli rewards (Garivier & Cappe, 2011)
SlidingWindowTS            sliding-window Thompson sampling (Trovo et al., 2020)
LinTS                      linear Thompson sampling (Agrawal & Goyal, 2013)
Lin                        greedy linear model, epsilon-greedy with epsilon = 0
"""
from mab_benchmark.algorithms.fg_ts import FeelGoodThompsonSampling
from mab_benchmark.algorithms.kl_ucb import KLUCB
from mab_benchmark.algorithms.linear import Lin, LinTS
from mab_benchmark.algorithms.sw_ts import SlidingWindowTS

__all__ = ["FeelGoodThompsonSampling", "KLUCB", "SlidingWindowTS", "LinTS", "Lin"]
