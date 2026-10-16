# OBD preprocessing for Setting S4 (Fixes 1 to 5)

Reproducible preprocessing of the Open Bandit Dataset (Saito et al., 2020; CC BY 4.0)
and the S4 replay for the thesis *Developing a Performance Model of
Multi-Armed-Bandit Recommender Systems* (Ojediran, LASU 2026).

## How to run

From the `mab-benchmark` repository root, with `open_bandit_dataset/random/all/all.csv`
and `open_bandit_dataset/bts/all/all.csv` in place (`.csv.gz` copies also work):

```
pip install -r obd_preprocessing/requirements.txt
obd_preprocessing\run_all.bat          (Windows)
bash obd_preprocessing/run_all.sh      (Linux / macOS)
```

Or run the steps one by one, in this order (each reads the previous step's `cache/`):

| Step | Script | Fix | What it does |
|---|---|---|---|
| 1 | `step1_fix5_integrity.py` | Fix 5 | Validates both logs, drops and logs exact duplicates, checks time order, builds a compact cache |
| 2 | `step2_fix3_positions.py` | Fix 3 | Tests position bias; builds one time-ordered stream of both logs with position kept |
| 3 | `step3_fix4_context.py` | Fix 4 | One-hot user features (with placeholder matching), item features, affinity |
| 4 | `step4_fix1_fix2_slices.py` | Fixes 1 and 2 | Full log cut into 30 disjoint, consecutive time slices |
| 5 | `step5_run_s4_replay.py` | all | Rejection-sampling replay of 8 algorithms x 30 paired runs, statistics, external-validity check, learning curves |
| - | `compare_variants.py` | - | Main analysis vs keep-duplicates sensitivity, conclusion by conclusion |

`common.py` holds the shared paths, constants, seeds, the fast LinUCB classes and the
ZOZOTOWN Bernoulli TS class; `zozotown_prior_bts_all.json` holds its production priors
(copied from obp 0.4.1, `obp/policy/conf/prior_bts.yaml`, Apache 2.0); `figures.py` draws
the thesis figures.

Every step accepts `--variant main|keep_duplicates` (default `main`). Step 1 with
`--keep-duplicates` writes to the `keep_duplicates` variant automatically; its cache and
outputs go to `cache/keep_duplicates/` and `outputs/keep_duplicates/`, so the main results
are never overwritten. Step 5 also accepts `--jobs N`, `--quick`, `--skip-selftest` and `--reports-only`.
Full run: about 3 minutes for steps 1 to 4 and about 48 minutes for step 5 on 2 cores.

## Chapter Four scripts (repository root)

`s4_obd_experiment.py` and `s4_bernoulli_ts_experiment.py` run on this pipeline (they start
it if its outputs are missing) and write the files Chapter Four uses:
`s4_obd_results.json`, `s4_obd_summary.txt`, `s4_obd_learning.png`,
`s4_bernoulli_ts_results.json`, `s4_bernoulli_ts_summary.txt`, `s4_bernoulli_ts_learning.png`.
`s4_bernoulli_ts_experiment.py` also runs a seeded synthetic Track 0 (S1, S3, S5). The
Bridge Compliance Score of S4 is computed in `experiments/s4_bcs_bridge.py`.

## Decisions taken

| Topic | Decision | Reason |
|---|---|---|
| Data | Random **and** Bernoulli TS logs of the "all" campaign | Random log alone gives about 60 clicks per algorithm in total under replay |
| Replay | Rejection sampling, accept if matched and `u < min(1, c / propensity)`, `c = 1/80` | Valid for a non-uniform logging policy; reduces to the standard replayer on the random log |
| Duplicates | 162 (random) and 2,188 (bts) exact duplicates dropped and listed in `outputs/` | Chosen by BJ; note that 97% of them are clicks, see Findings |
| Positions | Pooled for the main analysis, CTR per position reported from the same runs | No significant position bias (chi-square p = 0.39 random, 0.69 bts) |
| Runs | 30 consecutive, non-overlapping time slices; run i uses seed `SEED_TABLE[i]` from `mab_benchmark.runner` (1000 to 1029) | Independent, paired runs (Requirement 1) |
| Context | User one-hot + position (d = 35) and item features + log1p affinity (k = 42) | Chosen by BJ: user + item + affinity |
| User hashes | The ~35.3% "placeholder" category, hashed differently in the two logs, is matched by an explicit three-part rule (step 3 docstring) | Without it, a third of the rows would split into two columns |
| Algorithm clock | `t` passed to an algorithm = its accepted events + 1 | Its real decision horizon |
| External validity | Bernoulli TS with ZOZOTOWN priors replayed like every other algorithm; replay ratio vs CTR(bts log) / CTR(random log) with a 10,000-sample paired bootstrap CI | Its on-policy CTR is known, so replay can be checked against the truth |

## Algorithms

B0 to B3 are the unchanged classes from `mab_benchmark.baselines`. FG-TS uses lambda = 0.5
(as in `fgts_submission.json`), with the bonus `lambda * sqrt(log(t+1) / (n+1))` used by the
package. B4 is disjoint LinUCB (alpha = 1) on the user context;
B4h is hybrid LinUCB (Li et al., 2010, Algorithm 2) on user + item + affinity.
`LinUCBFast` and `HybridLinUCB` are fast but equivalent versions: step 5 self-tests them
against the package `LinUCB` and a direct transcription of Algorithm 2 (`outputs/s4_selftest.json`:
identical choices, apart from one exact numerical tie).

## Findings

1. **Fix 5.** Both logs pass every integrity check (no missing values, binary clicks,
   80 items, 3 positions, random propensity exactly 1/80, time-ordered). 161 of the 162
   random-log duplicates and 2,100 of the 2,188 bts duplicates are **clicks**, logged at the
   same microsecond. Dropping them lowers CTR from 0.347% to 0.335% (random) and from 0.495%
   to 0.478% (bts), further from the published 0.35% and 0.50%. They look like repeat clicks,
   not corrupt rows; run step 1 with `--keep-duplicates` to show the conclusions do not depend on it.
2. **Fix 3.** No significant position bias in this campaign, so pooling is justified.
3. **Fixes 1 and 2.** 13,729,177 rows, each used in exactly one of 30 runs (about 457,640 rows
   per run).
4. **S4 result (8 algorithms).** Bernoulli TS with ZOZOTOWN priors ranks first
   (mean CTR 0.492%, 95% CI 0.437% to 0.546%). After Holm-Bonferroni correction the only
   significant pair is Bernoulli TS (ZOZOTOWN) > B0 Random (p = 0.00008, d = 1.04). **No pair
   among the seven algorithms that learn from a uniform prior or from scratch differs
   significantly.** Accepted events differ strongly by algorithm (B2 UCB1: 6,630 in total and
   24 clicks; Bernoulli TS (ZOZOTOWN): 114,974), because rejection replay accepts few events for
   a policy that settles on items the logging policy rarely showed.
5. **External validity holds.** Replay recovers the known advantage of the logging policy:
   ratio 1.407x (95% CI 1.235 to 1.601) against the ground truth 1.427x on the same data
   (published 1.43x). H1, H2, H3a and H3b are supported on the pooled stream; on the
   random-log-only stream as well (ratio 1.499x, CI 1.089 to 2.119).
6. **Truncation bias.** The B0 Random replay CTR (0.349%) is 4.0% above the random-log CTR
   (0.335%): the size of the bias that the bts rows introduce.
7. **Keeping the duplicates** (2,350 rows, 0.017% of the data, 96% of them clicks;
   `outputs/variant_comparison.md`). Unchanged: Bernoulli TS (ZOZOTOWN) ranks first; no
   significant pair among the other learners; external validity (H2, H3a, H3b)
   holds; H1 on the pooled stream. Changed: H1 on the random-log-only stream (p = 0.017 ->
   0.057), the number of baselines Bernoulli TS (ZOZOTOWN) beats significantly (1 -> 4), and the
   order of the non-significant learners. The order of those learners is therefore not a
   robust result and should not be reported as a ranking.

The reading for the thesis: on OBD "all", offline replay is valid (it recovers a known, large
effect at the right size) but has too little power to separate algorithms that learn from
scratch within a 1/30 slice of the log. `experiments/s4_bcs_bridge.py` replays the whole log
per run and measures every bridge condition.

## Outputs (`outputs/`)

`fix5_integrity_report.*`, `fix5_duplicates_removed_*.csv`, `fix3_position_report.*`,
`fix4_context_report.*`, `fix1_fix2_slices.*`, `s4_selftest.json`, `s4_replay_runs.csv`
(one row per stream, run and algorithm), `s4_replay_results.json`, `s4_replay_summary.md/.txt`,
`s4_external_validity.json/.md`, `s4_learning_curves.csv`, `variant_comparison.json/.md`;
the same set for the sensitivity run in `outputs/keep_duplicates/`. `cache/` (about 200 MB) is regenerated by steps 1 to 4 and is git-ignored.

## Reference

Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2020). A large-scale open dataset for
bandit algorithms. arXiv:2008.07146.
Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to
personalized news article recommendation. WWW 2010.
Li, L., Chu, W., Langford, J., & Wang, X. (2011). Unbiased offline evaluation of
contextual-bandit-based news article recommendation algorithms. WSDM 2011.
