# mab_benchmark — MAB Unified Benchmark Suite v1.0

**OJEDIRAN ALABA BOLAJI · March 2026**

A self-contained Python package that implements the full MAB Unified
Benchmark Suite: five standard problem settings, four mandatory baselines,
four statistical requirements, the Bridge Compliance Score, and a
Papers With Code leaderboard submission formatter.

---

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas` (pandas only required for
`YahooReplayer`).

---

## Quick Start

```python
from mab_benchmark import BenchmarkRunner, StatisticalAnalyser
from mab_benchmark import BCSCalculator, LeaderboardSubmitter
from mab_benchmark.baselines import UCB1, RandomPolicy

# 1. Run your algorithm on the full suite
runner  = BenchmarkRunner(n_runs=30)
results = runner.full_suite(UCB1, {}, verbose=True)

# 2. Statistical analysis (all four requirements)
report  = StatisticalAnalyser().analyse(results)
print(StatisticalAnalyser().format_report(report))

# 3. Bridge Compliance Score
bcs = BCSCalculator.compute(
    rho_s=0.85,     # Spearman rho between Tier 1 and Tier 2 rankings
    p_indep=0.95,   # proportion of independent arm pairs
    K=10,           # arms (for the setting you are checking)
    T=10_000,       # horizon
)
print(BCSCalculator().format_result(bcs))

# 4. Generate leaderboard JSON
submitter = LeaderboardSubmitter(
    algorithm_name="UCB1",
    paper_url="https://arxiv.org/abs/XXXX.XXXXX",
    code_url="https://github.com/user/repo",
)
json_str = submitter.generate(results, report, bcs)
LeaderboardSubmitter.save(json_str, "submission.json")
print(submitter.summary(json_str))
```

---

## Implementing Your Own Algorithm

Subclass `BanditAlgorithm` and implement exactly two methods:

```python
from mab_benchmark import BanditAlgorithm
import numpy as np

class MyAlgorithm(BanditAlgorithm):

    def reset(self):
        self.counts    = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)

    def choose_arm(self, t, context=None):
        # your arm selection logic here
        return int(np.argmax(self.estimates))

    def update(self, arm, reward, t, context=None):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (reward - self.estimates[arm]) / n

# Evaluate immediately
runner  = BenchmarkRunner(n_runs=30)
results = runner.full_suite(MyAlgorithm, {})
```

---

## Package Structure

```
mab_benchmark/
├── mab_benchmark/
│   ├── core.py                    BanditAlgorithm base class
│   ├── runner.py                  BenchmarkRunner (all five settings)
│   ├── environments/
│   │   ├── s1_bernoulli.py        S1: Stationary Bernoulli
│   │   ├── s2_gaussian.py         S2: Stationary Gaussian
│   │   ├── s3_nonstationary.py    S3: Piecewise-stationary (10 change-points)
│   │   ├── s4_contextual.py       S4: Yahoo! R6B replayer + synthetic fallback
│   │   └── s5_coldstart.py        S5: Cold-start (K=500 > T)
│   ├── baselines/
│   │   ├── b0_random.py           B0: Random Policy
│   │   ├── b1_epsilon_greedy.py   B1: ε-Greedy (ε=0.1)
│   │   ├── b2_ucb1.py             B2: UCB1 (Auer et al., 2002)
│   │   ├── b3_thompson_sampling.py B3: Thompson Sampling
│   │   └── b4_linucb.py           B4: LinUCB (contextual, S4 only)
│   └── analysis/
│       ├── statistics.py          StatisticalAnalyser (4 requirements)
│       ├── bcs.py                 BCSCalculator (bridge compliance score)
│       └── leaderboard.py         LeaderboardSubmitter (Papers With Code)
└── tests/
    └── test_mab_benchmark.py
```

---

## The Five Benchmark Settings

| ID | Name | K | T | Reward | Key feature |
|----|------|---|---|--------|-------------|
| S1 | Stationary Bernoulli  | 10/50/200 | 500–10k  | Binary     | Fixed means |
| S2 | Stationary Gaussian   | 10/50/200 | 1k–10k   | Continuous | Fixed means |
| S3 | Piecewise-Stationary  | 10        | 10k      | Binary     | 10 abrupt changes |
| S4 | Contextual (Yahoo!)   | ~270      | 10k–50k  | Binary CTR | Real data, replayer |
| S5 | Cold-Start (K > T)    | 500       | 50–200   | Binary     | Many arms, few rounds |

---

## The Four Mandatory Baselines

| ID | Algorithm | Required in |
|----|-----------|------------|
| B0 | Random Policy         | All settings |
| B1 | ε-Greedy (ε=0.1)      | All settings |
| B2 | UCB1                  | All settings |
| B3 | Thompson Sampling     | All settings |
| B4 | LinUCB                | S4 only      |

---

## The Bridge Compliance Score (BCS)

```
BCS = (c1 + c2 + c3) / 3

c1 = clip(rho_S   / 0.7,  0, 1)   # reward proxy alignment
c2 = clip(p_indep / 0.9,  0, 1)   # arm independence
c3 = clip(1 - e   / 0.20, 0, 1)   # sufficient horizon
```

| BCS | Band | Meaning |
|-----|------|---------|
| ≥ 0.9 | PASS | Tier 1 is reliable proxy for deployment |
| 0.6–0.9 | PARTIAL | Tier 2 required for deployment claims |
| < 0.6 | FAIL | Tiers likely to diverge |

---

## Setting S4: Yahoo! R6B Access

The full S4 setting requires the Yahoo! Front Page dataset (R6B),
available from [Yahoo! Webscope](https://webscope.sandbox.yahoo.com/).

If unavailable, use the built-in `SyntheticContextualBandit` fallback
(already included in `runner.full_suite()`). Results from the synthetic
fallback are labelled `S4_SYNTHETIC` in leaderboard submissions.

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/ -v --cov=mab_benchmark --cov-report=term-missing
```

---

## Sources

All design decisions are drawn from:
- `GAP5_EvalFramework.docx` — Pillars 3 & 4
- `TwoTierProtocol_GAP5.docx` — Section E bridge conditions
- `BenchmarkSpec_Vermorel2005.docx` — 28-dimension specification
- `BenchmarkMatrix_Silva2022.docx` — Priority matrix
- `MAB_BenchmarkSuite.pdf` — Full derivation and specification

---

## Leaderboard

Submit results to:
**https://paperswithcode.com/benchmark/mab-unified-benchmark-suite**

Ranking: mean cumulative regret on S1 (primary). S2–S5 reported as secondary.

---

## Licence

MIT
