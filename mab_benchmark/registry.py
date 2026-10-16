"""
mab_benchmark.registry
======================

Plug-in registries for environments (datasets and simulators), algorithms
and metrics. A component is addressed by a short name in a YAML study
configuration or on the command line.

Built-in components are registered when this module is imported. Third-party
packages add their own without editing mab_benchmark, through Python entry
points in their pyproject.toml:

    [project.entry-points."mab_benchmark.environments"]
    my_logs = "my_package.envs:MyLoggedBandit"

    [project.entry-points."mab_benchmark.algorithms"]
    my_ucb = "my_package.algs:MyUCB"

    [project.entry-points."mab_benchmark.metrics"]
    my_metric = "my_package.metrics:my_metric"

or at run time with register_environment / register_algorithm /
register_metric.

Contracts
---------
Environment  a class constructed as Env(**params, seed=seed) with attributes
             K and T and methods pull(...) and regret_increment(...) as in
             mab_benchmark.environments. Registration declares whether pull
             takes the round t (kind="nonstationary" or "contextual") and
             whether the environment exposes context(t) (kind="contextual").
             expected_random_regret() is needed for the NCR metric.
Algorithm    a BanditAlgorithm subclass, constructed as Alg(n_arms=K, **params).
Metric       a function f(regrets, info) -> float, where regrets is the array of
             per-run cumulative regrets and info a dict with keys "K", "T",
             "random_regret" (array of per-run expected random regret, or None)
             and "n_runs".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points
from typing import Callable

import numpy as np

ENV_GROUP, ALG_GROUP, METRIC_GROUP = ("mab_benchmark.environments", "mab_benchmark.algorithms",
                                      "mab_benchmark.metrics")
KINDS = ("stationary", "nonstationary", "contextual")


@dataclass
class EnvironmentSpec:
    name: str
    target: object                 # class, or "module:attr" string resolved on first use
    kind: str = "stationary"
    defaults: dict = field(default_factory=dict)
    description: str = ""

    def load(self):
        if isinstance(self.target, str):
            self.target = _resolve(self.target)
        return self.target


@dataclass
class AlgorithmSpec:
    name: str
    target: object
    defaults: dict = field(default_factory=dict)
    description: str = ""

    def load(self):
        if isinstance(self.target, str):
            self.target = _resolve(self.target)
        return self.target


_ENVS: dict[str, EnvironmentSpec] = {}
_ALGS: dict[str, AlgorithmSpec] = {}
_METRICS: dict[str, Callable] = {}
_DISCOVERED = False


def _resolve(path: str):
    mod, _, attr = path.partition(":")
    obj = import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------- registration
def register_environment(name: str, target, kind: str = "stationary", defaults: dict | None = None,
                         description: str = "", overwrite: bool = False) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if name in _ENVS and not overwrite:
        raise ValueError(f"environment {name!r} is already registered")
    _ENVS[name] = EnvironmentSpec(name, target, kind, dict(defaults or {}), description)


def register_algorithm(name: str, target, defaults: dict | None = None, description: str = "",
                       overwrite: bool = False) -> None:
    if name in _ALGS and not overwrite:
        raise ValueError(f"algorithm {name!r} is already registered")
    _ALGS[name] = AlgorithmSpec(name, target, dict(defaults or {}), description)


def register_metric(name: str, fn: Callable, overwrite: bool = False) -> None:
    if name in _METRICS and not overwrite:
        raise ValueError(f"metric {name!r} is already registered")
    if not callable(fn):
        raise TypeError("a metric must be callable as f(regrets, info) -> float")
    _METRICS[name] = fn


def _discover() -> None:
    """Load third-party components from entry points (once)."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for group, reg in ((ENV_GROUP, _ENVS), (ALG_GROUP, _ALGS), (METRIC_GROUP, _METRICS)):
        for ep in entry_points(group=group):
            if ep.name in reg:
                continue                                  # built-ins win; never silently replaced
            if group == ENV_GROUP:
                obj = ep.load()
                register_environment(ep.name, obj, kind=getattr(obj, "BENCHMARK_KIND", "stationary"),
                                     description=f"entry point {ep.value}")
            elif group == ALG_GROUP:
                register_algorithm(ep.name, ep.value, description=f"entry point {ep.value}")
            else:
                register_metric(ep.name, ep.load())


def get_environment(name: str) -> EnvironmentSpec:
    _discover()
    try:
        return _ENVS[name]
    except KeyError:
        raise KeyError(f"unknown environment {name!r}; known: {sorted(_ENVS)}") from None


def get_algorithm(name: str) -> AlgorithmSpec:
    _discover()
    try:
        return _ALGS[name]
    except KeyError:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(_ALGS)}") from None


def get_metric(name: str) -> Callable:
    _discover()
    try:
        return _METRICS[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(_METRICS)}") from None


def list_components() -> dict:
    _discover()
    return {"environments": {k: (v.kind, v.description) for k, v in sorted(_ENVS.items())},
            "algorithms": {k: v.description for k, v in sorted(_ALGS.items())},
            "metrics": sorted(_METRICS)}


# --------------------------------------------------------------------------- built-in metrics
def mean_regret(regrets, info):
    return float(np.mean(regrets))


def median_regret(regrets, info):
    return float(np.median(regrets))


def sd_regret(regrets, info):
    return float(np.std(regrets, ddof=1)) if len(regrets) > 1 else float("nan")


def ncr(regrets, info):
    """Normalised causal regret (Chapter Three): mean of R / R_rand over runs."""
    rr = info.get("random_regret")
    if rr is None:
        raise ValueError("the NCR metric needs an environment with expected_random_regret()")
    rr = np.asarray(rr, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(rr > 0, np.asarray(regrets, float) / rr, np.nan)
    return float(np.nanmean(v))


def per_round_regret(regrets, info):
    return float(np.mean(regrets) / info["T"])


# --------------------------------------------------------------------------- built-ins
def _builtins() -> None:
    E = "mab_benchmark.environments"
    register_environment("S1_bernoulli", f"{E}.s1_bernoulli:BernoulliBandit", "stationary",
                         {"K": 10, "T": 10_000}, "S1 stationary Bernoulli")
    register_environment("S2_gaussian", f"{E}.s2_gaussian:GaussianBandit", "stationary",
                         {"K": 10, "T": 10_000}, "S2 stationary Gaussian")
    register_environment("S3_piecewise", f"{E}.s3_nonstationary:PiecewiseStationaryBandit", "nonstationary",
                         {"K": 10, "T": 10_000}, "S3 piecewise-stationary, 10 change-points")
    register_environment("S4_synthetic", f"{E}.s4_contextual:SyntheticContextualBandit", "contextual",
                         {"K": 50, "T": 10_000, "context_dim": 10}, "S4 synthetic linear-logistic contexts")
    register_environment("S4_obd_sim", f"{E}.s4_obd_sim:OBDCalibratedBandit", "stationary",
                         {"K": 80, "T": 14_315}, "S4 Tier 1: Bernoulli arms with OBD item CTRs")
    register_environment("S4_kuairec", f"{E}.s4_kuairec:KuaiRecBandit", "stationary",
                         {"K": 80, "T": 14_315}, "S4 Tier 1: fully observed KuaiRec matrix (data not bundled)")
    register_environment("S5_coldstart", f"{E}.s5_coldstart:ColdStartBandit", "stationary",
                         {"K": 500, "T": 100}, "S5 cold start, K > T")
    B = "mab_benchmark.baselines"
    register_algorithm("B0_Random", f"{B}.b0_random:RandomPolicy", description="mandatory baseline")
    register_algorithm("B1_EpsGreedy", f"{B}.b1_epsilon_greedy:EpsilonGreedy", description="mandatory baseline")
    register_algorithm("B2_UCB1", f"{B}.b2_ucb1:UCB1", description="mandatory baseline")
    register_algorithm("B3_TS", f"{B}.b3_thompson_sampling:ThompsonSampling", description="mandatory baseline")
    register_algorithm("B4_LinUCB", f"{B}.b4_linucb:LinUCB", description="contextual baseline (S4)")
    register_algorithm("FG-TS", "mab_benchmark.algorithms.fg_ts:FeelGoodThompsonSampling", {"lam": 0.5},
                       "Feel-Good Thompson sampling (thesis test algorithm)")
    A = "mab_benchmark.algorithms"
    register_algorithm("KL-UCB", f"{A}.kl_ucb:KLUCB", {"c": 0.0}, "KL-UCB, Bernoulli (Garivier & Cappe, 2011)")
    register_algorithm("SW-TS", f"{A}.sw_ts:SlidingWindowTS", {"window": 1000},
                       "sliding-window Thompson sampling (Trovo et al., 2020)")
    register_algorithm("LinTS", f"{A}.linear:LinTS", {"v": 0.5}, "linear Thompson sampling (Agrawal & Goyal, 2013)")
    register_algorithm("Lin", f"{A}.linear:Lin", description="greedy linear model (epsilon = 0)")
    for n, f in (("mean_regret", mean_regret), ("median_regret", median_regret), ("sd_regret", sd_regret),
                 ("ncr", ncr), ("per_round_regret", per_round_regret)):
        register_metric(n, f)


_builtins()
