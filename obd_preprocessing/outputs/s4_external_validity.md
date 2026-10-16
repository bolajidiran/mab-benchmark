# S4 external validity (variant: main)

## External validity: BernoulliTS with ZOZOTOWN priors vs ground truth

| Stream | Replay CTR BTS-Z | Replay CTR Random | Replay ratio (95% CI) | Ground-truth ratio (this data) | Published ratio | Rank of BTS-Z |
|---|---|---|---|---|---|---|
| pooled | 0.4915% | 0.3494% | 1.407x (1.235 to 1.601) | 1.427x | 1.43x | 1 of 8 |
| random_only | 0.4386% | 0.2925% | 1.499x (1.089 to 2.119) | 1.427x | 1.43x | 1 of 8 |

| Hypothesis | pooled | random_only |
|---|---|---|
| H1: BTS-Z CTR > Random (one-sided paired Wilcoxon) | p = 0.0000, supported | p = 0.0173, supported |
| H2: BTS-Z ranks in the top two | rank 1, supported | rank 1, supported |
| H3a: direction matches ground truth (ratio > 1) | supported | supported |
| H3b: ground-truth ratio inside the 95% CI | supported | supported |

