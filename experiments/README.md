# experiments: every table and figure of Chapters Four and Five (mab_benchmark 1.2.0)

These scripts generate, from raw runs, every number, table and figure that Chapters Four and
Five report, and then check the whole framework component by component. Every value traces to
a saved raw array. Run them from this folder, in this order (times on 2 cores):

| Step | Script | Output | Time |
|---|---|---|---|
| 1 | `run_synthetic_suite.py --jobs 2` | `results/raw/<cell>__<algorithm>.npy` (30 values each), `results/suite_runs.csv`, `results/suite_meta.json` | about 45 min (resumable) |
| 2 | `run_curves_lambda.py --jobs 2` | `results/curves_<cell>.npz` (30 x T learning curves for S1, S3, S4 synthetic), `results/lambda_sweep.csv`, `results/lambda_curves_S1.npz` | about 15 min |
| 3 | `run_latency.py` | `results/latency_profile.csv`, `results/latency_meta.json` (measured per-round latency, L and CCM) | about 2 min |
| 4 | `run_layers_2_6.py --jobs 2` | `results/layers26_*` (MPO on S3 and S1, CCM stress tests, explainer on S1 and OBD) | about 6 min |
| 5 | `s4_bcs_bridge.py --jobs 2` (needs `obd_preprocessing/cache/`) | `results/s4_bcs_bridge.json`, `s4_bcs_bridge_runs.csv`, `s4_bcs_tier1_*.csv`, `s4_bcs_validation.csv` | about 95 min |
| 6 | `pytest ../tests -q --junitxml=results/pytest_v120.xml` | test record read by step 10 | about 4 min |
| 7 | `build_tables.py` | `results/tables/*.csv`, `results/tables/CHAPTER4_5_TABLES.md` | seconds |
| 8 | `build_leaderboard.py` | `../Leaderboard/`, `../fgts_submission.json`, `results/certificates.json` | seconds |
| 9 | `make_notebook.py --execute` | `../FG_TS_Benchmark_Final.ipynb`, executed | about 2 min |
| 10 | `framework_status.py` | `../FRAMEWORK_STATUS.md`, `../FRAMEWORK_STATUS.json` | seconds |
| 11 | `make_figures.py` | `results/figures/fig01` to `fig14` | about 1 min |

Set `OMP_NUM_THREADS=1` when running steps 1 to 5 with `--jobs 2`; otherwise BLAS threads compete
with the worker processes and the run slows down by an order of magnitude.

`exp_common.py` defines the 21 cells (the `full_suite` grid), the algorithm set and the seed table
(`SEED_TABLE[0:30]`, stream layout 1.2.0).

## Tables

| File | Thesis table |
|---|---|
| `table_4_2_S1_all_cells.csv` | Table 4.2 (nine S1 cells) |
| `table_4_3_primary_summary.csv` | Table 4.3 (mean, SD, z and t29 confidence intervals) |
| `table_4_4_all_algorithms_all_settings.csv`, `table_4_4_ranks.csv` | Table 4.4, every algorithm and cell |
| `table_4_5_wilcoxon_holm_primary.csv` | Table 4.5 |
| `table_4_6_effect_sizes_primary.csv` | Table 4.6 (Cohen's d, A12 with ties) |
| `table_pairwise_summary_all_settings.csv`, `pairwise_<cell>.json` | pairwise tests in every cell |
| `table_4_9_latency_measured.csv` | Table 4.9 (measured latency) |
| `table_4_11_two_test_algorithms_S1.csv` | Table 4.11 |
| `table_4_14a_pareto.csv` | Table 4.14a (Pareto characterisation) |
| `table_4_15_4_17_4_19_guf_values.csv`, `table_4_16_4_18_4_20_guf_rankings.csv` | Tables 4.15 to 4.20 (GUF weight sweep) |
| `figure_5_1_lambda_sweep_table.csv` | data of Figure 5.1 |
| `table_5_1_design_expectations.csv` | Table 5.1 (design expectations for the optimism bonus) |
| `table_layer2_mpo.csv`, `table_layer2_mpo_vs_ucb1.csv`, `table_layer2_ccm_rule.csv` | Layer 2 (Meta-Policy Orchestrator) |
| `table_layer6_explainer.csv`, `table_layer6_ips_vs_model.csv` | Layer 6 (counterfactual explainer, three tolerance settings) |
| `table_bcs_s4_bridge.csv`, `table_bcs_s4_sensitivity.csv`, `table_bcs_s4_arm_independence.csv`, `table_bcs_s4_replay_validation.csv` | Bridge Compliance Score on S4 |

## Figures

| File | Content |
|---|---|
| `fig01_s1_all_cells.png` | S1, nine cells: means with 95% CI |
| `fig02_s1_learning_curves.png` | Figure 5.2: cumulative regret curves, S1 primary |
| `fig03_lambda_sensitivity.png` | Figure 5.1: FG-TS bonus weight on S1, S3 and S5 |
| `fig04_primary_pairwise_a12.png` | Tables 4.5 and 4.6 as an A12 matrix with significance marks |
| `fig05_rank_heatmap.png` | rank of every algorithm in every cell |
| `fig06_s3_curves.png` | S3 curves with the change-points |
| `fig07_s4_synthetic.png` | S4 synthetic: curves and final regret |
| `fig08_pareto_fronts.png` | Figure 4.1: Pareto fronts over regret and measured latency |
| `fig09_guf_weight_sweep.png` | GUF rank across the w1 sweep |
| `fig10_latency_profile.png` | measured latency against the CCM budgets |
| `fig11_mpo_s3.png` | Layer 2: S3 regret with and without the MPO; detection rate by size of the drop |
| `fig12_explainer.png` | Layer 6: sign agreement, Δt against the truth, OBD by tolerance setting |
| `fig13_bcs_s4_bridge.png` | BCS on S4 with every condition measured |
| `fig14_validation_scorecard.png` | the component scorecard of `FRAMEWORK_STATUS.md` |

The palette is fixed per algorithm, so an algorithm keeps its colour in every figure.

## Notes on the results

Bernoulli TS with uniform priors and B3 Thompson Sampling draw the same numbers under the
benchmark's seeding, so their rows are equal run by run; FG-TS with λ = 0 is equal to B3 run by run
as well. On S5 (K = 500, T = 100) every λ > 0 gives the same regret: with λ = 0.1 or λ = 2, FG-TS
pulls a never-pulled arm in every round, and the bonus is the same for every never-pulled arm, so
the choice reduces to the largest Beta(1, 1) sample among them, which does not depend on λ.

`s4_bcs_bridge.py` replays the whole log in each run with a block-vectorised replay that visits only
the rows whose acceptance coin passes; every visited row still gets its own independent choice. It is
checked against the per-row replay of `obd_preprocessing/step5_run_s4_replay.py` on the 30 pooled
slices: UCB1, which is deterministic, gives identical accepted events and clicks in all 30 runs; the
stochastic algorithms agree in distribution (`table_bcs_s4_replay_validation.csv`). The primary BCS
variant (random log, full horizon, domain-matched Tier 1) is fixed in `build_tables.py`,
`build_leaderboard.py` and `make_notebook.py`.

The counterfactual explainer (`mab_benchmark/explain.py`) has three tolerance settings. The default,
`statistical`, calls two arms indistinguishable when their IPS gap is within 1.96 standard errors; it
and `relative` (10% of the larger estimate) work on any reward scale, including click-through rates
below 1%. `absolute` uses the constant 0.01 of Chapter Three.
