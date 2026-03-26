"""
mab_benchmark.environments
==========================

Five benchmark settings as specified in GAP5_EvalFramework.docx Pillar 4
and BenchmarkSpec_Vermorel2005.docx.

Settings
--------
S1  BernoulliBandit          Stationary Bernoulli arms
S2  GaussianBandit           Stationary Gaussian arms
S3  PiecewiseStationaryBandit  10 abrupt change-points
S4  YahooReplayer            Contextual, Yahoo! R6B or MIND
S5  ColdStartBandit          K=500 arms, T in {50,100,200}
"""

from mab_benchmark.environments.s1_bernoulli import BernoulliBandit
from mab_benchmark.environments.s2_gaussian import GaussianBandit
from mab_benchmark.environments.s3_nonstationary import PiecewiseStationaryBandit
from mab_benchmark.environments.s4_contextual import YahooReplayer, SyntheticContextualBandit
from mab_benchmark.environments.s5_coldstart import ColdStartBandit

__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "PiecewiseStationaryBandit",
    "YahooReplayer",
    "SyntheticContextualBandit",
    "ColdStartBandit",
]
