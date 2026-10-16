"""
mab_benchmark.environments
==========================

Benchmark settings.

S1     BernoulliBandit            stationary Bernoulli arms
S2     GaussianBandit             stationary Gaussian arms
S3     PiecewiseStationaryBandit  10 abrupt change-points
S4     SyntheticContextualBandit  linear-logistic contexts (synthetic)
S4     OBD replay                 obd_preprocessing/ (Tier 2, logged data) with
                                  the mandatory replay diagnostics of mab_benchmark.replay
S4-SIM OBDCalibratedBandit        bias-free Tier 1 calibrated on OBD item CTRs
S4-KR  KuaiRecBandit              bias-free real data (fully observed KuaiRec matrix)
S5     ColdStartBandit            K = 500 arms, T in {50, 100, 200}
YahooReplayer                     deprecated (Yahoo! R6B was never made available)
"""

from mab_benchmark.environments.s1_bernoulli import BernoulliBandit
from mab_benchmark.environments.s2_gaussian import GaussianBandit
from mab_benchmark.environments.s3_nonstationary import PiecewiseStationaryBandit
from mab_benchmark.environments.s4_contextual import YahooReplayer, SyntheticContextualBandit
from mab_benchmark.environments.s4_obd_sim import OBDCalibratedBandit
from mab_benchmark.environments.s4_kuairec import KuaiRecBandit
from mab_benchmark.environments.s5_coldstart import ColdStartBandit

__all__ = [
    "BernoulliBandit",
    "GaussianBandit",
    "PiecewiseStationaryBandit",
    "YahooReplayer",
    "SyntheticContextualBandit",
    "OBDCalibratedBandit",
    "KuaiRecBandit",
    "ColdStartBandit",
]
