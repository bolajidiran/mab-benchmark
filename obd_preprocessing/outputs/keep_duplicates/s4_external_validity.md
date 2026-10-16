# S4 external validity (variant: keep_duplicates)

## External validity: BernoulliTS with ZOZOTOWN priors vs ground truth

| Stream | Replay CTR BTS-Z | Replay CTR Random | Replay ratio (95% CI) | Ground-truth ratio (this data) | Published ratio | Rank of BTS-Z |
|---|---|---|---|---|---|---|
| pooled | 0.5299% | 0.4011% | 1.321x (1.114 to 1.568) | 1.428x | 1.43x | 1 of 8 |
| random_only | 0.5474% | 0.4600% | 1.190x (0.902 to 1.593) | 1.428x | 1.43x | 1 of 8 |

| Hypothesis | pooled | random_only |
|---|---|---|
| H1: BTS-Z CTR > Random (one-sided paired Wilcoxon) | p = 0.0001, supported | p = 0.0571, not supported |
| H2: BTS-Z ranks in the top two | rank 1, supported | rank 1, supported |
| H3a: direction matches ground truth (ratio > 1) | supported | supported |
| H3b: ground-truth ratio inside the 95% CI | supported | supported |

