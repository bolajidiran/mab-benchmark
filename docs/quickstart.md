# Quick start

## Install

```
pip install "mab-benchmark[yaml]"
```

Python 3.10 to 3.13. The core needs NumPy, SciPy and pandas only.

## Reproduce the primary cell of the thesis

```
mab-benchmark run examples/study_s1.yaml
```

This runs FG-TS and the four mandatory baselines on S1 (K = 10, T = 10,000) for 30
seeded runs and writes `results/s1-primary/runs.csv` and `summary.json`. The mean
regrets are 49.03 (B3 Thompson sampling), 195.21 (FG-TS), 348.94 (B2 UCB1),
446.59 (B1 epsilon-Greedy) and 4,037.47 (B0 Random); if you get different numbers,
your installation does not match the release.

## From Python

```python
from mab_benchmark import BenchmarkRunner, StatisticalAnalyser
from mab_benchmark.algorithms import FeelGoodThompsonSampling
from mab_benchmark.baselines import UCB1
from mab_benchmark.environments import BernoulliBandit

runner = BenchmarkRunner(n_runs=30)
res = {name: runner.run_setting(cls, kw, BernoulliBandit, {"K": 10, "T": 10_000})
       for name, cls, kw in [("FG-TS", FeelGoodThompsonSampling, {"lam": 0.5}), ("UCB1", UCB1, {})]}
report = StatisticalAnalyser().analyse(res)
print(report["ranking"], report["pairwise"])
```

## Your own algorithm

Subclass `mab_benchmark.BanditAlgorithm` (methods `choose_arm(t, context)`,
`update(arm, reward, t, context)` and `reset()`), then either pass the class to
`BenchmarkRunner` or register it and name it in a study file ([plug-ins](plugins.md)).
