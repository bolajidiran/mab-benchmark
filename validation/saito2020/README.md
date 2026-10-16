# Validity check against Saito et al. (2020/2021)

This folder checks the thesis framework against the published results of the Open
Bandit Dataset (OBD) and Open Bandit Pipeline paper (Saito et al., 2021,
arXiv:2008.07146v5). It is a **validity check, not part of the framework**: nothing
in `mab_benchmark/` or `obd_preprocessing/` imports it, and it adds no benchmark
setting, metric or BCS input. It reads the raw OBD logs and, for Test 2 only, the
pre-registered seed table of `mab_benchmark.runner`.

The findings and their meaning for the thesis are in **`VALIDATION_REPORT.md`**.

## The four tests

| Test | Published result | What is reproduced | Script |
|---|---|---|---|
| 1 | Table 1: rows, items and on-policy CTR of the six logs | Same statistics on the raw and the cleaned (Fix 5) logs | `test1_on_policy_ctr.py` |
| 2 | Table 1: CTR(Bernoulli TS) / CTR(Random) in each campaign | The thesis' S4 replay protocol recovering that ratio, ALL, Men's, Women's | `test2_external_validity.py` |
| 3 | Tables 4 and 9: RMSE of IPW, SNIPW, DM, DR, SNDR, Switch-DR, DRos at n = 300,000, both directions | Saito's OPE benchmark protocol, 200 bootstrap seeds per configuration | `test3_test4_ope_benchmark.py` |
| 4 | Tables 6 and 10: the same at n = 10,000 vs 300,000 | The change of the most accurate estimator with sample size; Table 10 also locates a column of Table 6 that the paper prints twice | `test3_test4_ope_benchmark.py` |

## How to run (from the repository root)

```
pip install -r validation/saito2020/requirements.txt
validation\saito2020\run_all.bat        (Windows)
bash validation/saito2020/run_all.sh    (Linux / macOS)
```

Steps, in order:

| Step | Script | Time (2 cores) |
|---|---|---|
| 0 | `s0_prepare_logs.py`: reads the six logs once into `cache/` (git-ignored) | about 4 min |
| 1 | `test1_on_policy_ctr.py` | seconds |
| 2 | `test2_external_validity.py` | about 20 min |
| 3, 4 | `test3_test4_ope_benchmark.py --all`: 12 configurations x 200 seeds (resumable: results are appended seed by seed) | about 5 h |
| - | `summarise_validation.py`: verdicts of Tests 3 and 4 and the overall summary | seconds |
| - | `make_figures.py`: the five figures in `outputs/figures/` | seconds |
| opt. | `check_equivalence_obp050.py`: runs the official obp 0.5.0 and pyieoe 0.1.1 code next to `ope_native.py` (needs Python 3.9; see its docstring) | about 1 h |

Data: the scripts look for `open_bandit_dataset/<policy>/<campaign>/<campaign>.csv`
(or `.csv.gz`), then `validation/saito2020/_transfer/<policy>_<campaign>.csv.gz`; set
`OBD_DIR` to use another folder.

## Why a reimplementation, and how it was verified

The official benchmark (`zr-obp/benchmark/ope`) is locked to Python 3.9, obp 0.5.0,
pyieoe 0.1.1 and scikit-learn 0.24.2, and obp imports PyTorch at module level.
`ope_native.py` reimplements the protocol with numpy, pandas and scikit-learn only.
`check_equivalence_obp050.py` ran the official classes on identical inputs in a
Python 3.9 environment with the locked versions (`outputs/equivalence_obp050.json`):

- the Bernoulli TS evaluation-policy distribution (100,000 Monte Carlo slates from the
  ZOZOTOWN priors) is identical in all three campaigns;
- the seven estimates are identical to the last bit (maximum absolute difference 0.0)
  for every seed checked, at n = 10,000 and at n = 300,000, in both directions,
  including the tuned Switch-DR and DRos hyperparameters.

Three details of the official code are reproduced on purpose because they change the
numbers: (1) obp 0.5.0 tunes Switch-DR and DRos by estimated MSE with a bias upper
bound (delta = 0.05), not by SLOPE as in later obp versions; (2) it stores the tuned
value on the estimator object, and pyieoe reuses that object, so the value tuned on
seed 0 is used for all 200 seeds; (3) the bootstrap resamples the first n rows of the
log in time order.

## Deviations from the published protocol (all documented in the outputs)

1. The 200-seed runs used scikit-learn 1.9.1, numpy 2.5.3 and Python 3.13. On the seeds
   shared with the equivalence check they differ from the official scikit-learn 0.24.2
   values by at most 1e-16 (`outputs/version_drift.json`), so the version change has
   no measurable effect.
2. Row order: obp sorts each log by the timestamp string with an unstable sort; here
   the file order is kept. All six files are already in time order, so the two orders
   can differ only among rows that share a timestamp (verified on Random Men's:
   4,034 rows moved, all within equal timestamps; `outputs/equivalence_loader_random_men.json`).
3. The Bernoulli TS log of the ALL campaign has 12,357,200 rows here and 12,168,084 in
   Table 1; its CTR still matches. The other five logs match the published row counts
   exactly.

## Outputs (`outputs/`)

`s0_logs_report.json`; `test1_on_policy_ctr.json/.csv/.md`;
`test2_external_validity.json/.md`, `test2_runs.csv`, `test2_curves.csv`;
`ope/<behaviour>_to_<evaluation>_<campaign>_n<n>_raw_seeds.csv` (one row per seed:
the seven estimates, tuned lambdas, timing), `ope/ope_summary_raw.json/.csv`;
`test3_test4_verdicts.json/.md`; `validation_summary.json`;
`equivalence_obp050.json`, `equivalence_loader_*.json`, `version_drift.json`;
`figures/fig1` to `fig5`.

## References

Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2021). Open Bandit Dataset and
Pipeline: Towards realistic and reproducible off-policy evaluation. NeurIPS 2021
Datasets and Benchmarks Track. arXiv:2008.07146.
Li, L., Chu, W., Langford, J., & Wang, X. (2011). Unbiased offline evaluation of
contextual-bandit-based news article recommendation algorithms. WSDM 2011.
Su, Y., Dimakopoulou, M., Krishnamurthy, A., & Dudík, M. (2020). Doubly robust
off-policy evaluation with shrinkage. ICML 2020.
