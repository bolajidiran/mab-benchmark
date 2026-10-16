# Studies and the command line

A study file names the algorithms, the settings and the metrics; the protocol
supplies everything else.

```yaml
name: s1-primary
n_runs: 30                       # at least 30; seeds are SEED_TABLE[0:n_runs]
algorithms:
  - name: FG-TS
    params: {lam: 0.5}
settings:
  - environment: S1_bernoulli
    params: {K: 10, T: 10000}
    label: S1_K10_T10000
metrics: [mean_regret, sd_regret, ncr]
include_mandatory_baselines: true   # B0 to B3 are added if absent
output: results/s1-primary
```

| Command | What it does |
|---|---|
| `mab-benchmark version` | package version, stream layout, seed table |
| `mab-benchmark list` | registered environments, algorithms and metrics |
| `mab-benchmark validate STUDY.yaml` | checks the file, builds each environment once |
| `mab-benchmark run STUDY.yaml [-o DIR]` | runs the study, writes `runs.csv` and `summary.json` |
| `mab-benchmark bcs ...` | Bridge Compliance Score, with the replay condition if `--rho-shuffle` is given |

`validate` rejects fewer than 30 runs, explicit seeds, unknown keys, duplicate labels,
unknown components and parameters an environment refuses (for example a horizon S1
does not allow). `summary.json` holds, per setting, every metric, the full statistical
report (Wilcoxon, Holm, effect sizes), the ranking, the configuration and a
provenance block (package version, stream layout, seeds, Python, NumPy, platform).
