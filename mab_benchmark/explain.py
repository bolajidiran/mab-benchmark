"""
mab_benchmark.explain
=====================

Layer 6 of the architecture: the counterfactual explainer, from Chapter Three,
"The Counterfactual Explainer".

For the arm a_t chosen at round t, the explainer computes the clipped IPS
counterfactual value of every arm from the history up to t (Layer 1,
mab_benchmark.analysis.ips, c_min = 0.01),

    mu_CF(a) = (1/n) sum_s X_s(a),   X_s(a) = r_s 1{a_s = a} / max(pi_s(a_s), c_min),

and the distinguishability gap

    Delta_t = mu_CF(a_t) - max_{a != a_t} mu_CF(a).

Explanations
    |Delta_t| <= eps_t  "We showed this because we could not distinguish its
                        value from your top choice."
    Delta_t  >  eps_t   the estimated margin over the best alternative.
    Delta_t  < -eps_t   the chosen arm is estimated below the best alternative,
                        which happens on exploration rounds; the explainer says so.

Tolerance eps_t (argument `tolerance`)
    "statistical" (default, scale-aware): eps_t = z * SE(Delta_t), z = 1.96.
        "Could not distinguish" then means exactly that the two IPS estimates
        are not statistically different at the 5% level, whatever the reward
        scale. Because a_s takes one value per round, X_s(a) X_s(j) = 0 and
            SE(Delta_t)^2 = ( m2(a) + m2(j) - Delta_t^2 ) / n,
        with m2(a) = (1/n) sum_s X_s(a)^2.
    "relative" (scale-aware): eps_t = rel * max(|mu_CF(a_t)|, |mu_CF(j)|),
        a gap smaller than the share `rel` (default 10%) of the larger value.
    "absolute": eps_t = eps (default c_min = 0.01), the constant of Chapter
        Three. On click-through rates below 1% every gap falls below 0.01, so
        this setting suits rewards on a 0 to 1 scale with large gaps only.

Propensities pi_s(a_s): logged data supply them (OBD pscore). For an online
algorithm, `choice_propensity` estimates the probability that the algorithm
picks each arm from its current state: exact for Random, epsilon-Greedy and
UCB1, Monte Carlo (n_mc posterior draws) for Thompson-Sampling-family
algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mab_benchmark.analysis.ips import C_MIN

TEMPLATE_TIE = "We showed this because we could not distinguish its value from your top choice."


@dataclass
class Explanation:
    t: int
    arm: int
    delta: float
    mu_chosen: float
    best_alternative: int
    mu_best_alternative: float
    kind: str                 # "indistinguishable" | "margin" | "exploration"
    text: str
    eps: float = float("nan")  # tolerance used at this round


TOLERANCES = ("statistical", "relative", "absolute")


def _fmt(v: float) -> str:
    return f"{v:.3g}"


@dataclass
class CounterfactualExplainer:
    n_arms: int
    tolerance: str = "statistical"
    eps: float = C_MIN          # used by tolerance="absolute"
    rel: float = 0.10           # used by tolerance="relative"
    z: float = 1.96             # used by tolerance="statistical"
    c_min: float = C_MIN
    _num: np.ndarray = field(init=False, repr=False)
    _sq: np.ndarray = field(init=False, repr=False)
    _n: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        if self.tolerance not in TOLERANCES:
            raise ValueError(f"tolerance must be one of {TOLERANCES}")
        self._num = np.zeros(self.n_arms)
        self._sq = np.zeros(self.n_arms)

    def observe(self, arm: int, reward: float, propensity: float) -> None:
        x = float(reward) / max(float(propensity), self.c_min)
        self._num[arm] += x
        self._sq[arm] += x * x
        self._n += 1

    def values(self) -> np.ndarray:
        return self._num / self._n if self._n else np.zeros(self.n_arms)

    def gap_se(self, arm: int, other: int) -> float:
        if self._n < 2:
            return float("inf")
        mu = self.values()
        d = mu[arm] - mu[other]
        v = (self._sq[arm] + self._sq[other]) / self._n - d * d
        return float(np.sqrt(max(v, 0.0) / self._n))

    def tolerance_at(self, arm: int, other: int) -> float:
        mu = self.values()
        if self.tolerance == "absolute":
            return float(self.eps)
        if self.tolerance == "relative":
            return float(self.rel * max(abs(mu[arm]), abs(mu[other])))
        return float(self.z * self.gap_se(arm, other))

    def explain(self, t: int, arm: int) -> Explanation:
        mu = self.values()
        others = np.delete(np.arange(self.n_arms), arm)
        j = int(others[np.argmax(mu[others])])
        d = float(mu[arm] - mu[j])
        eps = self.tolerance_at(arm, j)
        rel = f" ({abs(d) / abs(mu[j]):.0%})" if mu[j] else ""
        if abs(d) <= eps:
            kind, text = "indistinguishable", TEMPLATE_TIE
        elif d > 0:
            kind = "margin"
            text = (f"We showed this because its estimated value ({_fmt(mu[arm])}) is {_fmt(d)}{rel} above "
                    f"the next best option ({_fmt(mu[j])}).")
        else:
            kind = "exploration"
            text = (f"We showed this to learn more about it: its estimated value ({_fmt(mu[arm])}) is "
                    f"{_fmt(-d)}{rel} below the current best option ({_fmt(mu[j])}).")
        return Explanation(t, int(arm), d, float(mu[arm]), j, float(mu[j]), kind, text, eps)


def choice_propensity(alg, t: int, n_mc: int = 200, rng: np.random.Generator | None = None) -> np.ndarray:
    """Probability that `alg` picks each arm at round t from its current state."""
    K = alg.n_arms
    name = type(alg).__name__
    if hasattr(alg, "base"):                                   # MetaPolicyOrchestrator
        return choice_propensity(alg.base, t, n_mc, rng)
    if name == "RandomPolicy":
        return np.full(K, 1.0 / K)
    if hasattr(alg, "epsilon") and hasattr(alg, "estimates"):
        p = np.full(K, alg.epsilon / K)
        best = np.flatnonzero(alg.estimates == alg.estimates.max())
        p[int(best[0])] += 1 - alg.epsilon                       # np.argmax takes the first maximum
        return p
    if name == "UCB1":
        p = np.zeros(K)
        if (alg.counts == 0).any():
            p[int(np.flatnonzero(alg.counts == 0)[0])] = 1.0
            return p
        idx = alg.estimates + np.sqrt(2.0 * np.log(t) / alg.counts)
        p[int(np.argmax(idx))] = 1.0
        return p
    if hasattr(alg, "alpha") and hasattr(alg, "beta"):        # Beta-Bernoulli TS family
        rng = rng or np.random.default_rng(t)
        draws = rng.beta(np.maximum(alg.alpha, 1e-6), np.maximum(alg.beta, 1e-6), size=(n_mc, K))
        if hasattr(alg, "_feel_good_bonus"):
            draws = draws + np.array([alg._feel_good_bonus(a) for a in range(K)])
        return np.bincount(draws.argmax(1), minlength=K) / n_mc
    raise TypeError(f"no propensity model for {name}")
