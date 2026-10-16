"""
exp_common.py: shared definitions for the Chapter Four and Five synthetic
experiments.

Every result file written by the scripts in this folder records the package
version, the random-stream layout and the seeds, so each table in the thesis
can be traced to a raw array.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import mab_benchmark                                            # noqa: E402
from mab_benchmark import BanditAlgorithm                       # noqa: E402
from mab_benchmark.runner import SEED_TABLE, STREAM_LAYOUT      # noqa: E402
from mab_benchmark.baselines import (RandomPolicy, EpsilonGreedy, UCB1,  # noqa: E402
                                     ThompsonSampling, LinUCB)
from mab_benchmark.environments import (BernoulliBandit, GaussianBandit,  # noqa: E402
                                        PiecewiseStationaryBandit,
                                        SyntheticContextualBandit, ColdStartBandit)
from FeelGoodThompsonSampling import FeelGoodThompsonSampling   # noqa: E402

OUT = HERE / "results"
RAW = OUT / "raw"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (OUT, RAW, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

N_RUNS = 30


class BernoulliTSUniform(BanditAlgorithm):
    """TEST-1 of Table 4.11: OBP-style Bernoulli TS with uniform Beta(1, 1)
    priors (identical in law to B3; kept as a separate class, as in Track 0)."""

    def reset(self):
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def choose_arm(self, t, context=None):
        return int(np.argmax(np.random.beta(self.alpha, self.beta)))

    def update(self, arm, reward, t, context=None):
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r


# name -> (class, kwargs). Order = order in every table and figure.
ALGS = {
    "FG-TS": (FeelGoodThompsonSampling, {"lam": 0.5}),
    "B3_TS": (ThompsonSampling, {"reward_type": "bernoulli"}),
    "B2_UCB1": (UCB1, {}),
    "B1_EpsGreedy": (EpsilonGreedy, {"epsilon": 0.1}),
    "B0_Random": (RandomPolicy, {}),
    "BernoulliTS_uniform": (BernoulliTSUniform, {}),
    "B4_LinUCB": (LinUCB, {}),            # S4 only
}
LABEL = {"FG-TS": "FG-TS (λ = 0.5)", "B3_TS": "B3 Thompson Sampling", "B2_UCB1": "B2 UCB1",
         "B1_EpsGreedy": "B1 ε-Greedy", "B0_Random": "B0 Random",
         "BernoulliTS_uniform": "Bernoulli TS (uniform priors)", "B4_LinUCB": "B4 LinUCB"}

# setting key -> (env class, env kwargs, runner kwargs) ; same cells as BenchmarkRunner.full_suite
CELLS = {}
for K in (10, 50, 200):
    for T in (500, 2_000, 10_000):
        CELLS[f"S1_K{K}_T{T}"] = (BernoulliBandit, {"K": K, "T": T}, {})
for K in (10, 50, 200):
    for T in (1_000, 10_000):
        CELLS[f"S2_K{K}_T{T}"] = (GaussianBandit, {"K": K, "T": T}, {})
CELLS["S3_K10_T10000"] = (PiecewiseStationaryBandit, {"K": 10, "T": 10_000}, {"is_stationary": False})
for T in (10_000, 50_000):
    CELLS[f"S4_SYNTHETIC_K50_T{T}"] = (SyntheticContextualBandit, {"K": 50, "T": T, "context_dim": 10},
                                       {"use_context": True})
for T in (50, 100, 200):
    CELLS[f"S5_K500_T{T}"] = (ColdStartBandit, {"K": 500, "T": T}, {})


def algs_for(cell: str) -> list[str]:
    names = [a for a in ALGS if a != "B4_LinUCB"]
    if cell.startswith("S4"):
        names.append("B4_LinUCB")
    return names


def provenance() -> dict:
    import scipy
    return {"mab_benchmark": mab_benchmark.__version__, "stream_layout": STREAM_LAYOUT,
            "seeds": SEED_TABLE[:N_RUNS], "n_runs": N_RUNS, "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}


def write_json(obj, path):
    def d(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, tuple): return list(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=d), encoding="utf-8")
