# Validity check of the thesis framework against Saito et al. (2021)

Ojediran, A. B. (LASU 2026). Prepared 11 October 2026. Supporting material for
Chapter Four. This is a validity check; it is not part of the evaluation framework.

## 1. Purpose and scope

This check asks one question: when I apply my framework's data handling, replay
protocol and statistical machinery to the Open Bandit Dataset (OBD), do I recover the
results that Saito et al. (2021) published for the same data? A framework that cannot
reproduce known results on its own reference dataset cannot be trusted to certify new
ones. The check uses four published results: the dataset statistics (Table 1), the
on-policy advantage of Bernoulli Thompson Sampling (TS) over Random (Table 1), the
accuracy ranking of seven off-policy evaluation (OPE) estimators at n = 300,000 in both
directions (Tables 4 and 9), and the change of that ranking with sample size (Tables 6
and 10). Nothing here feeds the Bridge Compliance Score (BCS), adds a benchmark setting
or changes S1 to S5; the code lives in `validation/saito2020/` and no framework module
imports it. Saito et al. are a fixed point to calibrate against, not a permanent
component of the thesis.

All pass criteria were written into the scripts before each test ran. Every number
below comes from a file in `outputs/`.

## 2. Was the comparison fair? Equivalence with the official code

My reimplementation of Saito et al.'s OPE benchmark returns exactly what their code
returns. Their benchmark (`zr-obp/benchmark/ope`) is locked to Python 3.9, obp 0.5.0,
pyieoe 0.1.1 and scikit-learn 0.24.2 (Saito et al., 2021). I rebuilt that environment
and ran the official classes next to `ope_native.py` on identical inputs. The Bernoulli
TS evaluation-policy distribution matched in all three campaigns, and all seven
estimates matched to the last bit (maximum absolute difference 0.0) in six
configurations: ALL in both directions at n = 10,000 and n = 300,000, Men's
(Bernoulli TS log) and Women's (Random log) at n = 10,000. The official loader, applied
to the raw Random Men's file, produced the same rows, actions, positions, rewards,
propensities and one-hot context as my cache once its row order was applied; its
unstable timestamp sort moved 4,034 of 452,949 rows, every one within a group of
identical timestamps. Finally, the 200-seed runs used scikit-learn 1.9.1; on the
seeds the equivalence check shares with them, the estimates differ from the official
scikit-learn 0.24.2 values by at most 1 x 10^-16 (`version_drift.json`). The comparison
that follows therefore tests the data and the protocol, not a translation of the code.

Three details of the official code change the numbers, and I reproduced them on
purpose. obp 0.5.0 tunes the Switch-DR and DRos hyperparameter by estimated mean squared
error with a high-probability bias bound (delta = 0.05), not by the SLOPE rule of later
obp versions. It also stores the tuned value on the estimator object, and pyieoe reuses
that object, so the value tuned on bootstrap seed 0 serves all 200 seeds. The bootstrap
resamples the first n rows of each log in time order. Section 6 returns to the second
detail, because it decides one of the published rankings.

## 3. Test 1: dataset statistics (Table 1)

The raw OBD logs reproduce every published click-through rate (CTR). Five of the six
row counts match exactly, all six item counts match, all six CTRs round to the
published two decimals, and all six relative CTRs round to the published values
(`test1_on_policy_ctr.md`, Figure 1).

| Log | Rows (this study) | Rows (published) | CTR % (95% CI) | Published CTR % |
|---|---|---|---|---|
| Random, ALL | 1,374,327 | 1,374,327 | 0.347 (±0.010) | 0.35 (±0.010) |
| Bernoulli TS, ALL | 12,357,200 | 12,168,084 | 0.495 (±0.004) | 0.50 (±0.004) |
| Random, Men's | 452,949 | 452,949 | 0.512 (±0.021) | 0.51 (±0.021) |
| Bernoulli TS, Men's | 4,077,727 | 4,077,727 | 0.674 (±0.008) | 0.67 (±0.008) |
| Random, Women's | 864,585 | 864,585 | 0.482 (±0.015) | 0.48 (±0.014) |
| Bernoulli TS, Women's | 7,765,497 | 7,765,497 | 0.638 (±0.006) | 0.64 (±0.056) |

Two discrepancies sit in the paper, not in the data. The Bernoulli TS ALL log holds
189,116 more rows than Table 1 states, yet its CTR and every OPE result built on it
reproduce, so the published count is most probably from an earlier release of the
file. The published interval for Bernoulli TS Women's (±0.056) is ten times the
normal-approximation value (±0.006) that reproduces the other five rows, which points
to a misplaced decimal.

The cleaned logs (Fix 5, exact duplicates removed) do not reproduce Table 1: every CTR
falls by 0.012 to 0.022 percentage points, below the published rounding interval. That
result is informative rather than embarrassing. It shows that Saito et al. computed
Table 1, and the ground truths of their OPE benchmark, on the raw logs with the
duplicate rows included. For the thesis this confirms two statements at once: the
dataset description in Section 3.4.2 (1,374,327 rows, CTR 0.35% for random/all) is
correct, and Fix 5 is a deliberate departure from the published preprocessing, which
Chapter Four must state as such.

## 4. Test 2: does the S4 replay recover the known advantage of Bernoulli TS?

The S4 replay protocol recovers the direction of the Bernoulli TS advantage in every
campaign and its size in two of three. I replayed Bernoulli TS with the ZOZOTOWN
production priors against Random under the S4 protocol (rejection sampling with
c = 1/K, 30 disjoint time slices, seeds 1000 to 1029, one-sided paired Wilcoxon test)
and compared the replay ratio with the on-policy ratio CTR(Bernoulli TS log) /
CTR(Random log) (Li et al., 2011; Saito et al., 2021). The primary analysis uses the
cleaned logs, as S4 does; the raw logs are reported alongside (`test2_external_validity.md`,
Figure 2).

| Campaign | Stream | Replay ratio (95% CI) | Ground truth | H1 p (one-sided) | H3b: truth inside CI | Random replay vs its log |
|---|---|---|---|---|---|---|
| ALL | both logs | 1.361 (1.212 to 1.550) | 1.427 | 0.00002 | pass | +8.4% |
| ALL | Random log only | 1.182 (0.820 to 1.704) | 1.427 | 0.23 | pass | +4.1% |
| Men's | both logs | 1.104 (0.971 to 1.272) | 1.320 | 0.19 | **fail** | +20.0% |
| Men's | Random log only | 1.509 (1.055 to 2.172) | 1.320 | 0.009 | pass | -16.7% |
| Women's | both logs | 1.254 (1.095 to 1.458) | 1.329 | 0.004 | pass | +19.4% |
| Women's | Random log only | 1.130 (0.865 to 1.473) | 1.329 | 0.19 | pass | +6.6% |

The failure in Men's has a mechanism, and the mechanism matters more than the verdict.
With both logs pooled, a Bernoulli TS log row counts for an algorithm with probability
min(1, c / propensity). For items the production policy rarely showed (propensity
below c) the correct weight exceeds 1 but the rule truncates it at 1, so those items
count too little; they are exactly the items Bernoulli TS learned to avoid because
their CTR is low. Random therefore earns a replay CTR 8% (ALL) to 20% (Men's, Women's)
above its true on-policy CTR, and the ratio is squeezed towards 1. On the Random log
alone the replayer is exact, because every propensity equals 1/K, and the interval
contains the truth in every campaign; the price is power, since each run keeps only
about 430 to 630 events per algorithm and two of three campaigns lose significance.
Think of two scales: one is accurate but reads to the nearest kilogram, the other reads
to the gram but adds a fixed weight. The pooled stream is the second scale.

Duplicates sharpen the same point. On the raw logs the Random replay CTR in ALL rises
to 18% above its log, against 8% on the cleaned logs, because 96% of the duplicate rows
in ALL are clicks and many of them fall on low-propensity rows that the replay weights
heavily. Fix 5 is therefore not cosmetic for S4: it halves the replay bias in the
campaign the thesis uses.

## 5. Test 3: the OPE estimator benchmark at n = 300,000 (Tables 4 and 9)

The framework reproduces the published OPE benchmark cell by cell. Over 200 bootstrap
seeds per configuration, all 42 RMSE values at n = 300,000 lie inside the 95% bootstrap
interval of my estimate, the largest relative difference is 1.3%, the Spearman rank
correlation with the published ranking is at least 0.97 in all six configurations,
and the published best estimator is the reproduced best in all six (DRos for the
Bernoulli TS log; SNIPW, SNIPW and DM for the Random log) (`test3_test4_verdicts.md`,
Figures 3 and 5).

| Configuration | Best reproduced | Best published | Spearman rho | Published inside 95% CI | Median difference |
|---|---|---|---|---|---|
| Bernoulli TS log -> Random, ALL | DRos (0.319) | DRos (0.316) | 0.97 | 7/7 | +0.8% |
| Bernoulli TS log -> Random, Men's | DRos (0.459) | DRos (0.459) | 1.00 | 7/7 | +0.1% |
| Bernoulli TS log -> Random, Women's | DRos (0.560) | DRos (0.561) | 1.00 | 7/7 | -0.3% |
| Random log -> Bernoulli TS, ALL | SNIPW (0.473) | SNIPW (0.468) | 1.00 | 7/7 | +1.2% |
| Random log -> Bernoulli TS, Men's | SNIPW (1.141) | SNIPW (1.133) | 1.00 | 7/7 | +0.8% |
| Random log -> Bernoulli TS, Women's | DM (0.580) | DM (0.584) | 0.99 | 7/7 | +0.1% |

RMSE x 10^3; the full 7 x 6 table with bias and standard deviation per estimator is
in `test3_test4_verdicts.md`.

The Random-log direction tests Section 3.9 of the thesis directly, and it qualifies
that section. Inverse propensity weighting (IPW) on the Random log uses exact
propensities (1/K), so the estimator is unbiased for the value of the evaluation policy
on the logged contexts (Li et al., 2011). Yet its RMSE is dominated by bias, not by
variance: in Men's the mean error is -1.28 x 10^-3 against an RMSE of 1.31 x 10^-3, and
in Women's +1.60 against 1.65. I argue that this bias is not a fault of IPW. The ground
truth is the CTR of the whole seven-day Bernoulli TS log, collected by a policy whose
parameters changed during collection, while the estimate describes a fixed Beta-prior
distribution applied to the first 300,000 rows. Unbiasedness under exact propensities
is a property of the estimator; it does not remove the gap between a time window and
the week, or between a static policy and an adaptive one. Section 3.9 should keep its
claim and add that condition.

## 6. Test 4: sample size changes the most accurate estimator (Tables 6 and 10)

Sample size changes which estimator wins, exactly as published. On the Bernoulli TS
log, the Direct Method (DM) is the most accurate estimator at n = 10,000 in Men's and
Women's, and DRos takes over at n = 300,000; in ALL, DRos leads at both sizes. All three
reproduced best estimators at n = 10,000 match the published ones, the switch is
reproduced in every campaign, and the direction of change from 10,000 to 300,000 agrees
with the published direction for 20 of 21 estimator-campaign pairs (Figure 4). Men's at
n = 10,000 reproduces all seven published RMSE values to within 0.001.

The one disagreement exposes a typesetting error in the paper. The Women's column of
Table 6 (3.156, 3.038, 2.665, 3.055, 3.069, 3.055, 3.051) is identical, digit for digit,
to the Men's column of Table 10, which reports the opposite direction. I ran both
configurations. The Random log -> Bernoulli TS run for Men's returns 3.156, 3.039, 2.667,
3.056, 3.069, 3.056, 3.051 (median difference 0.02%); the Bernoulli TS log -> Random run
for Women's returns 0.80 to 2.14, 61% below the printed column. The printed Women's
values in Table 6 therefore belong to Table 10, and the true Women's values are the ones
in `test3_test4_verdicts.md`. The conclusion of Table 6 survives the correction: DM is
still the best Women's estimator at n = 10,000 (0.804) and DRos at n = 300,000 (0.560).
No reader of the paper could have found this without re-running the benchmark.

One published ranking rests on an implementation detail. Because obp 0.5.0 reuses the
hyperparameter tuned on seed 0, DRos in ALL (Bernoulli TS log) uses lambda = 10,000 at
n = 300,000 and lambda = 5 at n = 10,000 for all 200 seeds. When each seed tunes its own
lambda, the DRos RMSE in ALL rises from 0.319 to 0.459 at n = 300,000 (still the best,
narrowly ahead of DR at 0.486) and from 0.791 to 1.165 at n = 10,000, where DM (0.816)
then becomes the best estimator. In the other campaigns the per-seed choice coincides
with the seed-0 choice and the RMSE changes by at most 0.001. The published "DRos is best in ALL at both
sizes" is therefore partly a property of the tuning protocol, not only of the
estimator.

## 7. What this means for the thesis framework

| Framework component | Test | Outcome | Consequence for the thesis |
|---|---|---|---|
| Dataset description (Section 3.4.2) and data integrity (Fix 5) | 1 | Raw logs reproduce Table 1; cleaned logs do not | The description stands. Chapter Four must state that Fix 5 departs from the published preprocessing and why (duplicates are 96% clicks) |
| S4 replay protocol and external validity (H1 to H3) | 2 | Direction recovered in all campaigns; size recovered in ALL and Women's, not Men's | Pooled replay with c = 1/K is biased towards 1 for low-CTR policies; S4 results on pooled streams are lower bounds on differences, and random-only replay is the unbiased but low-power reference |
| IPS estimator (Section 3.9) | 3 (Random log) | Reproduced; RMSE bias-dominated despite exact propensities | Keep the unbiasedness claim and add its condition: same contexts, same period, static evaluation policy |
| Statistical framework: reporting the full set-up (sample size, tuning, seeds) | 4 | Best estimator changes with n; one ranking changes with the tuning protocol | Direct evidence for Gap 5: a ranking is a property of estimator plus set-up, so a standard must fix the set-up |
| Reproducibility (pre-registered seeds, pinned software, released code) | 2 and all | Bit-identical to the official code; a copied table column found only by re-running | Re-running, not reading, is what validates a benchmark |

These results bear on the thesis in three registers. First, for the practitioner: the
framework's tools reproduce a published industrial benchmark to within 1.3% on every
correctly printed cell, so a practitioner can trust the S4 pipeline's arithmetic, but
should read a pooled-replay ratio as a conservative estimate of a real difference.
Second, for the researcher: two published conclusions changed under scrutiny (one
column belongs to another table; one ranking depends on how the code caches a tuned
value), which is exactly the kind of fragility my 62-paper SLR attributes to
unstandardised evaluation set-ups. Third, for replication: the check succeeded only
because the dataset, the code and the software versions were all public and pinned;
the framework already requires the first two, and this check argues for adding the
third, a locked environment file, to the certification requirements.

The overall verdict is that the framework passes its validity check. It reproduces the
published facts it should reproduce, it fails where a known statistical mechanism says
it should fail, and it detects two errors that the published record could not reveal
on its own.

## 8. Limitations of the check

The check covers one dataset and one paper by design. It validates the data pipeline,
the S4 replay and the OPE machinery; it does not validate S1 to S3 or S5, the GUF, the
CCM or the BCS threshold, which have no published OBD counterpart. Test 2 uses two
policies only, because Saito et al. publish on-policy values for Bernoulli TS and
Random alone. The OPE runs use the raw logs, as the paper did; repeating Tests 3 and 4
on the cleaned logs (`--variant clean`) is possible but answers a different question.

## References

Dudík, M., Erhan, D., Langford, J., & Li, L. (2014). Doubly robust policy evaluation
and optimization. *Statistical Science, 29*(4), 485-511. [Verified against the obp
0.5.0 source; not yet in the thesis reference list]

Li, L., Chu, W., Langford, J., & Wang, X. (2011). Unbiased offline evaluation of
contextual-bandit-based news article recommendation algorithms. *Proceedings of WSDM
2011*, 297-306.

Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2021). Open Bandit Dataset and
Pipeline: Towards realistic and reproducible off-policy evaluation. *NeurIPS 2021
Datasets and Benchmarks Track*. arXiv:2008.07146v5.

Su, Y., Dimakopoulou, M., Krishnamurthy, A., & Dudík, M. (2020). Doubly robust
off-policy evaluation with shrinkage. *Proceedings of ICML 2020*. [Verified against
the obp 0.5.0 source; not yet in the thesis reference list]

Wang, Y.-X., Agarwal, A., & Dudík, M. (2017). Optimal and adaptive off-policy
evaluation in contextual bandits. *Proceedings of ICML 2017*. [Verified against the
obp 0.5.0 source; not yet in the thesis reference list]
