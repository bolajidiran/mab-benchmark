# mab-benchmark: Unified Benchmark Suite for Multi-Armed Bandit Recommender Systems (1.3.0)

**Ojediran Alaba Bolaji**, PhD Computer Science, Lagos State University (LASU), Ojo, Nigeria, 2026.
Supervisors: Professor O.A. Enikuomehin and Professor A. Rahman.

[![CI](https://github.com/bolajidiran/mab-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/bolajidiran/mab-benchmark/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mab-benchmark)](https://pypi.org/project/mab-benchmark/)
[![Python](https://img.shields.io/badge/python-3.10%20to%203.13-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository holds the benchmark of the thesis *Developing a Unified Performance Model of
Multi-Armed Bandit Recommender Systems*. It defines five benchmark settings (S1 to S5), four
mandatory baselines (B0 to B3), a fixed statistical protocol, the Grand Unified Optimisation
Function (GUF) with its Contextual Constraint Metric (CCM), the Bridge Compliance Score (BCS), the
certification levels C1, C3 and C4 with a repository leaderboard, the Meta-Policy Orchestrator
(architecture Layer 2) and the counterfactual explainer (Layer 6). Since 1.3.0 the GUF is
normalised (NCR), the BCS is a setting-level gate, every replay result carries mandatory
replay-bias diagnostics, off-policy evaluation uses the Open Bandit Pipeline's estimators, and
studies run from YAML files through a command line. Documentation: [`docs/`](docs/index.md).

The state of every component, with two verdicts each (up to date? validated?), is in
**[`FRAMEWORK_STATUS.md`](FRAMEWORK_STATUS.md)** and the scorecard
`experiments/results/figures/fig14_validation_scorecard.png`.

## Quick start

```bash
pip install "mab-benchmark[yaml]"          # from PyPI; or, in a clone: pip install -e ".[dev]"
mab-benchmark run examples/study_s1.yaml   # FG-TS and B0 to B3 on S1, 30 seeded runs
pytest -q                                  # the test suite, in a clone
```

```python
from mab_benchmark import BenchmarkRunner, StatisticalAnalyser, Certifier
from mab_benchmark.algorithms import FeelGoodThompsonSampling

runner = BenchmarkRunner(n_runs=30)
suite  = runner.baseline_sweep()                        # {B0..B3: {cell: regrets}}
suite["FG-TS"] = runner.full_suite(FeelGoodThompsonSampling, {"lam": 0.5})

report = StatisticalAnalyser().analyse({a: suite[a]["S1_K10_T10000"] for a in suite})
print(StatisticalAnalyser().format_report(report))
```

Command line: `mab-benchmark version | list | validate STUDY | run STUDY | bcs ...`
([docs/cli.md](docs/cli.md)). New environments, algorithms and metrics plug in through a registry
or entry points ([docs/plugins.md](docs/plugins.md)). External submissions: [docs/submitting.md](docs/submitting.md).

## The benchmark in brief

| Setting | What it tests |
|---|---|
| S1 stationary Bernoulli | basic learning efficiency; K in {10, 50, 200}, T in {500, 2,000, 10,000} |
| S2 stationary Gaussian | continuous rewards |
| S3 piecewise-stationary | recovery after ten change-points |
| S4 contextual | synthetic linear contexts (K = 50, d = 10), and replay on the Open Bandit Dataset |
| S5 cold start | more arms than rounds (K = 500, T in {50, 100, 200}) |

Baselines: B0 Random, B1 ε-Greedy (ε = 0.1), B2 UCB1 (Auer et al., 2002), B3 Thompson Sampling
(Agrawal & Goyal, 2013); B4 LinUCB is added in S4. Further algorithms in `mab_benchmark.algorithms`:
FG-TS, KL-UCB, sliding-window TS, LinTS and the greedy linear model Lin. Every comparison uses 30 runs from the
pre-registered `SEED_TABLE`, the Wilcoxon signed-rank test with Holm-Bonferroni correction,
Cohen's d and the Vargha-Delaney A12.

## Results

Every table of Chapters Four and Five comes from `experiments/` (21 synthetic cells, 30 runs each);
the raw arrays, tables and figures are in `experiments/results/`, and `experiments/README.md`
explains how to regenerate them.

Primary metric, S1 with K = 10 and T = 10,000 (mean cumulative regret, 30 runs):

| Rank | Algorithm | Mean | SD |
|---|---|---|---|
| 1 | B3 Thompson Sampling | 49.03 | 27.44 |
| 2 | FG-TS (λ = 0.5) | 195.21 | 38.57 |
| 3 | B2 UCB1 | 348.94 | 64.33 |
| 4 | B1 ε-Greedy | 446.59 | 90.02 |
| 5 | B0 Random | 4,037.47 | 762.62 |

All ten pairs are significant after Holm-Bonferroni and all ten effects are large. On S3, B2 UCB1
recovers best after change-points (946.5), ahead of FG-TS (1,741.9) and B3 TS (2,136.0). On the
synthetic S4, LinUCB reaches 1,100.0 at T = 10,000, while every context-free algorithm stays at
Random level (about 4,544). On S5, ε-Greedy has the lowest regret in all three cells.

S4 on the Open Bandit Dataset (`obd_preprocessing/`): the replay ratio CTR(Bernoulli TS) /
CTR(Random) is 1.407 (95% bootstrap CI 1.235 to 1.601), against 1.427 in the logged data and 1.43 in
Saito et al. (2021).

## Bridge Compliance Score on S4

`experiments/s4_bcs_bridge.py` measures each bridge condition on the Open Bandit Dataset. Each run
replays the whole log, as the replayer method of Li et al. (2011) does, which gives each policy
14,315 decisions on the uniform random log (c3 = 0.73). Arm independence, measured on hourly item
CTRs of the random log, is p_indep = 0.9997 (c2 = 1.00). A Tier 1 simulator matched to the S4 domain
(K = 80, the 80 item CTRs of the random log, the same horizon) ranks the six context-free algorithms
almost exactly as the real replay does (ρ = 0.94, 95% bootstrap interval 0.66 to 0.94).

| Variant | ρ | c1 | c2 | c3 | BCS |
|---|---|---|---|---|---|
| **random log, full horizon, domain-matched Tier 1 (primary)** | **0.94** | **1.00** | **1.00** | **0.73** | **0.911 PASS** |
| random log, full horizon, S1 Tier 1 | 0.20 | 0.29 | 1.00 | 0.73 | 0.673 PARTIAL |
| pooled logs, full horizon, domain-matched Tier 1 | 0.49 | 0.69 | 1.00 | 0.94 | 0.877 PARTIAL |
| pooled logs, full horizon, S1 Tier 1 | -0.60 | 0.00 | 1.00 | 0.94 | 0.645 PARTIAL |

The pass is narrow. At the lower bound of ρ the score is 0.890, and with raw six-hour bins for c2 it
is 0.8997 (PARTIAL); the pooled logs, whose replay is biased by truncation (Saito Test 2),
do not pass. A generic S1 Tier 1 does not pass with either stream: an offline result certifies
deployment only when the offline benchmark is matched to the deployment domain.

## Certification and leaderboard

`experiments/build_leaderboard.py` certifies the two test algorithms of the thesis with the S4 BCS.
FG-TS receives **C1**: it beats B0 Random in 16 of 21 cells, B2 UCB1 in 8, B1 ε-Greedy in 2 and
B3 TS in 1. It does not reach C3, because on the S4 replay it is not significantly better than any
baseline it beats on Tier 1. The table is in `Leaderboard/LEADERBOARD.md`; new entries are added by
pull request and validated by `Leaderboard.add()`.

## Layers 2 and 6

`experiments/run_layers_2_6.py` runs Steps 3 and 5 of the thesis protocol. On S3, wrapping B3 TS in
the Meta-Policy Orchestrator lowers its regret from 2,136 to 912 (-57%) and FG-TS from 1,742 to
1,043 (-40%); MPO-wrapped TS then ties with UCB1 (946, Wilcoxon p = 0.44). On stationary S1 the
monitor raises no flag.

The counterfactual explainer has a scale-aware tolerance: by default two arms are "indistinguishable"
when their inverse-propensity estimates differ by less than 1.96 standard errors. On S1, the
explanations it grades have the sign of the true margin in 94% of sampled rounds for B3 TS, 94% for
UCB1 and 81% for ε-Greedy. On the real ZOZOTOWN log, with click-through rates of 0.1% to 0.7%, it
grades 72% of sampled rounds and 98% of those have the right sign; the fixed tolerance of 0.01
grades 0.2% of them, because every true item margin there is smaller than 0.01.

## Validity check against Saito et al. (2021)

`validation/saito2020/` reproduces four published results of the Open Bandit Dataset paper. It is a
validity check, not part of the framework; nothing in `mab_benchmark/` imports it.

| Test | Result |
|---|---|
| 1. On-policy CTR of the six logs (Table 1) | reproduced; one row count differs (Bernoulli TS, ALL) |
| 2. Replay recovers the BTS / Random ratio | ALL and Women's pass; Men's fails (replay truncation bias) |
| 3. OPE estimator RMSE (Tables 4 and 9) | 42 of 42 inside the 95% CI; identical to obp 0.5.0 |
| 4. Best estimator changes with sample size (Tables 6 and 10) | reproduced; Table 6 prints one column twice |

## Repository layout

```
mab_benchmark/            the package
  core.py, runner.py      BanditAlgorithm, BenchmarkRunner, SEED_TABLE, run_streams
  environments/, baselines/  S1 to S5, B0 to B4
  analysis/               statistics, BCS, GUF/CCM/Pareto, IPS, Certifier, leaderboard
  orchestration.py        Meta-Policy Orchestrator, system performance monitor (Layers 2, 6)
  explain.py              counterfactual explainer (Layer 6)
  algorithms/             FG-TS, KL-UCB, SW-TS, LinTS, Lin
  replay.py, analysis/ope.py   replayer and replay-bias diagnostics; off-policy evaluation
  registry.py, study.py, cli.py  plug-ins, YAML studies, the mab-benchmark command
tests/                    the test suite
docs/                     documentation (mkdocs), including the SLR screening log in docs/slr/
examples/                 study files
Leaderboard/              submissions, leaderboard.json, LEADERBOARD.md
experiments/              every table and figure, the leaderboard builder, the status checker
obd_preprocessing/        S4 Open Bandit Dataset pipeline (Fixes 1 to 5) and replay
validation/saito2020/     validity check against Saito et al. (2021)
FeelGoodThompsonSampling.py (shim to mab_benchmark.algorithms), s4_obd_experiment.py, s4_bernoulli_ts_experiment.py
FG_TS_Benchmark_Final.ipynb  FG-TS evaluation on the saved results (open from the repository root)
FRAMEWORK_STATUS.md       component-by-component status
```

## Pre-registered seeds

`runner.SEED_TABLE` holds the 200 seeds 1000 to 1199. Run i uses `SEED_TABLE[i]`, which feeds
`SeedSequence([seed, 1])`; its two children drive the rewards and the algorithm (`STREAM_LAYOUT =
"1.2.0"`), and the environment draws its arm means from `default_rng(seed)`.

## Citation

```bibtex
@phdthesis{ojediran2026unified,
  title   = {Developing a Unified Performance Model of Multi-Armed Bandit Recommender Systems},
  author  = {Ojediran, Alaba Bolaji},
  school  = {Lagos State University},
  year    = {2026},
  address = {Ojo, Lagos, Nigeria},
  url     = {https://github.com/bolajidiran/mab-benchmark}
}
```

## References

Agrawal, S., & Goyal, N. (2013). Further optimal regret bounds for Thompson sampling. *AISTATS 2013*.
Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning, 47*, 235-256.
Li, L., Chu, W., Langford, J., & Wang, X. (2011). Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms. *WSDM 2011*.
Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2021). Open Bandit Dataset and Pipeline: Towards realistic and reproducible off-policy evaluation. *NeurIPS 2021 Datasets and Benchmarks Track*. arXiv:2008.07146.
Zhang, T. (2022). Feel-Good Thompson Sampling for contextual bandits and reinforcement learning. *SIAM Journal on Mathematics of Data Science, 4*(2), 834-857. https://doi.org/10.1137/21M140924X

## Licence

MIT, 2026, Ojediran Alaba Bolaji, Lagos State University.
