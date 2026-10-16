# mab-benchmark

`mab-benchmark` is the unified performance model for multi-armed bandit (MAB)
recommender systems developed in the PhD thesis *Developing a Unified Performance
Model of Multi-Armed Bandit Recommender Systems* (Ojediran, Lagos State University,
2026). It fixes the parts of an evaluation that published studies usually choose
for themselves, so that two studies of the same algorithms can reach the same
conclusion.

What it fixes:

| Element | What the benchmark prescribes |
|---|---|
| Settings | S1 stationary Bernoulli, S2 Gaussian, S3 piecewise-stationary, S4 real data (Open Bandit Dataset) with a bias-free Tier 1, S5 cold start |
| Baselines | B0 Random, B1 epsilon-Greedy, B2 UCB1, B3 Thompson sampling in every cell |
| Seeds | `SEED_TABLE` = 1000 to 1199, stream layout 1.2.0 |
| Statistics | at least 30 runs, Wilcoxon signed-rank, Holm correction, Cohen's d and A12 |
| Objective | the normalised GUF, Gamma = w1 NCR + w2 L + w3 CCM |
| Certification | the Bridge Compliance Score gate (BCS >= 0.9; BCS_R for replay settings) and levels C1, C3, C4 |

```
pip install mab-benchmark            # core
pip install "mab-benchmark[yaml]"    # YAML study files
mab-benchmark version
```

Start with the [quick start](quickstart.md); the [protocol](protocol.md) page states
every rule with its justification, and [submitting a result](submitting.md) explains
how an external algorithm joins the leaderboard.
