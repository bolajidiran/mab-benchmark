# Changelog

## 1.3.0

- Normalised GUF: the regret term is NCR = R / R_rand (`mab_benchmark.analysis.ncr`);
  every environment provides `expected_random_regret()`.
- The BCS is a setting-level gate (`guf.bcs_gate`), not a constraint on a policy.
- Replay settings: `BCSCalculator.compute_replay` adds the replay-integrity condition
  c_R, the starvation rule and the bias-free Tier 1 requirement; `mab_benchmark.replay`
  provides the replayer and the diagnostics.
- Bias-free S4 Tier 1 environments: `OBDCalibratedBandit` and `KuaiRecBandit`.
- Off-policy evaluation with the seven Open Bandit Pipeline estimators
  (`mab_benchmark.analysis.ope`), native backend equal to obp 0.5.7.
- The Certifier gates on BCS_R and refuses C3 to a starved algorithm.
- Plug-in registry with entry points, YAML/JSON studies (`mab_benchmark.study`) and
  the `mab-benchmark` command line.
- New algorithms: KL-UCB, sliding-window Thompson sampling, LinTS, Lin; FG-TS moved
  into `mab_benchmark.algorithms` (identical code; the old module is a shim).
- Packaging for PyPI as `mab-benchmark`; CI on Python 3.10 to 3.13, an obp-equality
  job on Python 3.10, distribution and documentation builds.
- Unchanged: environments, baselines, SEED_TABLE and stream layout 1.2.0, so 1.2.0
  results reproduce exactly.

## 1.2.0

Independent random streams per run (stream layout 1.2.0), the S4 bridge on the Open
Bandit Dataset, Layers 2 and 6, certification and the leaderboard.
