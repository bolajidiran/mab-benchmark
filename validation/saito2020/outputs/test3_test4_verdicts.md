# Tests 3 and 4: OPE estimator benchmark against Saito et al.

RMSE x 10^3 over 200 bootstrap seeds; published values in brackets.

## bts log -> random, ALL, n = 300,000 (200 seeds; tuned lambda Switch-DR 500, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 0.494 | 0.493 | +0.213 | 0.447 |
| SNIPW | 0.508 | 0.507 | +0.240 | 0.448 |
| DM | 1.032 | 1.026 | +1.024 | 0.125 |
| DR | 0.486 | 0.482 | +0.228 | 0.430 |
| SNDR | 0.486 | 0.482 | +0.222 | 0.433 |
| Switch-DR | 0.486 | 0.482 | +0.228 | 0.430 |
| DRos | 0.319 | 0.316 | +0.116 | 0.298 |

Spearman rho = 0.972; best reproduced DRos, best published DRos (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.8%.

## bts log -> random, Men's, n = 300,000 (200 seeds; tuned lambda Switch-DR 500, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 0.789 | 0.789 | -0.740 | 0.276 |
| SNIPW | 0.645 | 0.644 | -0.578 | 0.287 |
| DM | 0.779 | 0.773 | +0.764 | 0.153 |
| DR | 0.614 | 0.613 | -0.547 | 0.279 |
| SNDR | 0.660 | 0.659 | -0.595 | 0.286 |
| Switch-DR | 0.614 | 0.613 | -0.547 | 0.279 |
| DRos | 0.459 | 0.459 | -0.372 | 0.270 |

Spearman rho = 1.000; best reproduced DRos, best published DRos (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.1%.

## bts log -> random, Women's, n = 300,000 (200 seeds; tuned lambda Switch-DR 500, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 0.776 | 0.776 | -0.701 | 0.333 |
| SNIPW | 0.804 | 0.804 | -0.734 | 0.331 |
| DM | 0.812 | 0.816 | +0.793 | 0.175 |
| DR | 0.801 | 0.803 | -0.733 | 0.323 |
| SNDR | 0.788 | 0.791 | -0.721 | 0.320 |
| Switch-DR | 0.801 | 0.803 | -0.733 | 0.323 |
| DRos | 0.560 | 0.561 | -0.468 | 0.308 |

Spearman rho = 1.000; best reproduced DRos, best published DRos (agree or tied: yes); published inside 95% CI 7/7; median relative difference -0.3%.

## random log -> bts, ALL, n = 300,000 (200 seeds; tuned lambda Switch-DR 50, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 0.478 | 0.474 | -0.353 | 0.323 |
| SNIPW | 0.473 | 0.468 | -0.344 | 0.325 |
| DM | 1.501 | 1.499 | -1.493 | 0.147 |
| DR | 0.507 | 0.501 | -0.396 | 0.318 |
| SNDR | 0.506 | 0.500 | -0.394 | 0.319 |
| Switch-DR | 0.507 | 0.501 | -0.396 | 0.318 |
| DRos | 0.514 | 0.508 | -0.408 | 0.313 |

Spearman rho = 1.000; best reproduced SNIPW, best published SNIPW (agree or tied: yes); published inside 95% CI 7/7; median relative difference +1.2%.

## random log -> bts, Men's, n = 300,000 (200 seeds; tuned lambda Switch-DR 50, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 1.306 | 1.298 | -1.278 | 0.268 |
| SNIPW | 1.141 | 1.133 | -1.108 | 0.273 |
| DM | 1.757 | 1.758 | -1.749 | 0.164 |
| DR | 1.178 | 1.169 | -1.147 | 0.270 |
| SNDR | 1.161 | 1.151 | -1.128 | 0.274 |
| Switch-DR | 1.178 | 1.169 | -1.147 | 0.270 |
| DRos | 1.180 | 1.171 | -1.149 | 0.269 |

Spearman rho = 1.000; best reproduced SNIPW, best published SNIPW (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.8%.

## random log -> bts, Women's, n = 300,000 (200 seeds; tuned lambda Switch-DR 50, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 1.649 | 1.645 | +1.596 | 0.416 |
| SNIPW | 1.588 | 1.585 | +1.534 | 0.411 |
| DM | 0.580 | 0.584 | -0.515 | 0.266 |
| DR | 1.541 | 1.539 | +1.490 | 0.395 |
| SNDR | 1.526 | 1.524 | +1.474 | 0.394 |
| Switch-DR | 1.541 | 1.539 | +1.490 | 0.395 |
| DRos | 1.525 | 1.524 | +1.474 | 0.394 |

Spearman rho = 0.991; best reproduced DM, best published DM (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.1%.

## bts log -> random, ALL, n = 10,000 (200 seeds; tuned lambda Switch-DR 500, DRos 5)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 1.918 | 1.899 | +1.162 | 1.529 |
| SNIPW | 1.661 | 1.641 | +0.827 | 1.444 |
| DM | 0.816 | 0.797 | +0.393 | 0.717 |
| DR | 1.202 | 1.203 | +0.228 | 1.184 |
| SNDR | 1.160 | 1.159 | +0.244 | 1.137 |
| Switch-DR | 1.202 | 1.203 | +0.228 | 1.184 |
| DRos | 0.791 | 0.765 | +0.269 | 0.746 |

Spearman rho = 1.000; best reproduced DRos, best published DRos (agree or tied: yes); published inside 95% CI 7/7; median relative difference +1.0%.

## bts log -> random, Men's, n = 10,000 (200 seeds; tuned lambda Switch-DR 100, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 3.683 | 3.683 | -3.666 | 0.363 |
| SNIPW | 3.662 | 3.661 | -3.643 | 0.369 |
| DM | 3.041 | 3.041 | -3.015 | 0.399 |
| DR | 3.748 | 3.747 | -3.732 | 0.337 |
| SNDR | 3.758 | 3.757 | -3.743 | 0.336 |
| Switch-DR | 3.748 | 3.747 | -3.732 | 0.337 |
| DRos | 3.727 | 3.727 | -3.712 | 0.340 |

Spearman rho = 1.000; best reproduced DM, best published DM (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.0%.

## bts log -> random, Women's, n = 10,000 (200 seeds; tuned lambda Switch-DR 50, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 2.138 | 3.156 | -2.076 | 0.512 |
| SNIPW | 1.282 | 3.038 | -1.075 | 0.699 |
| DM | 0.804 | 2.665 | +0.283 | 0.754 |
| DR | 1.195 | 3.055 | -1.030 | 0.607 |
| SNDR | 1.621 | 3.069 | -1.508 | 0.595 |
| Switch-DR | 1.195 | 3.055 | -1.030 | 0.607 |
| DRos | 1.179 | 3.051 | -1.010 | 0.609 |

Spearman rho = 0.782; best reproduced DM, best published DM (agree or tied: yes); published inside 95% CI 0/7; median relative difference -60.9%.

## random log -> bts, ALL, n = 10,000 (200 seeds; tuned lambda Switch-DR 50, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 1.104 | 1.104 | -0.500 | 0.987 |
| SNIPW | 1.101 | 1.100 | +0.042 | 1.103 |
| DM | 1.128 | 1.128 | -0.937 | 0.630 |
| DR | 1.141 | 1.133 | -0.750 | 0.862 |
| SNDR | 1.158 | 1.148 | -0.727 | 0.903 |
| Switch-DR | 1.141 | 1.133 | -0.750 | 0.862 |
| DRos | 1.133 | 1.125 | -0.738 | 0.862 |

Spearman rho = 0.964; best reproduced SNIPW, best published SNIPW (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.7%.

## random log -> bts, Men's, n = 10,000 (200 seeds; tuned lambda Switch-DR 50, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 3.156 | 3.156 | -3.091 | 0.639 |
| SNIPW | 3.039 | 3.038 | -2.965 | 0.666 |
| DM | 2.667 | 2.665 | -2.590 | 0.635 |
| DR | 3.056 | 3.055 | -2.986 | 0.651 |
| SNDR | 3.069 | 3.069 | -2.999 | 0.653 |
| Switch-DR | 3.056 | 3.055 | -2.986 | 0.651 |
| DRos | 3.051 | 3.051 | -2.981 | 0.652 |

Spearman rho = 1.000; best reproduced DM, best published DM (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.0%.

## random log -> bts, Women's, n = 10,000 (200 seeds; tuned lambda Switch-DR 50, DRos 10000)

| Estimator | Reproduced | Published | Bias | SD |
|---|---|---|---|---|
| IPW | 6.788 | 6.789 | +6.066 | 3.055 |
| SNIPW | 5.676 | 5.685 | +4.971 | 2.748 |
| DM | 2.982 | 2.986 | +2.513 | 1.609 |
| DR | 5.731 | 5.724 | +5.011 | 2.790 |
| SNDR | 5.480 | 5.476 | +4.790 | 2.668 |
| Switch-DR | 5.731 | 5.724 | +5.011 | 2.790 |
| DRos | 5.701 | 5.694 | +4.983 | 2.776 |

Spearman rho = 1.000; best reproduced DM, best published DM (agree or tied: yes); published inside 95% CI 7/7; median relative difference +0.1%.

- ALL: best at 10,000 reproduced DRos vs published DRos (T4a pass); best at 300,000 reproduced DRos vs published DRos; direction of change agrees for 7/7; switch reproduced: yes
- Men's: best at 10,000 reproduced DM vs published DM (T4a pass); best at 300,000 reproduced DRos vs published DRos; direction of change agrees for 7/7; switch reproduced: yes
- Women's: best at 10,000 reproduced DM vs published DM (T4a pass); best at 300,000 reproduced DRos vs published DRos; direction of change agrees for 6/7; switch reproduced: yes

## Which table holds the duplicated column?

| Estimator | Shared published value | Reproduced, Table 6 Women (BTS -> Random) | Reproduced, Table 10 Men (Random -> BTS) |
|---|---|---|---|
| IPW | 3.156 | 2.138 | 3.156 |
| SNIPW | 3.038 | 1.282 | 3.039 |
| DM | 2.665 | 0.804 | 2.667 |
| DR | 3.055 | 1.195 | 3.056 |
| SNDR | 3.069 | 1.621 | 3.069 |
| Switch-DR | 3.055 | 1.195 | 3.056 |
| DRos | 3.051 | 1.179 | 3.051 |

Median absolute relative difference: Table 6 Women 60.9%, Table 10 Men 0.0%. The shared column reproduces as Table 10, Men (Random -> BTS).
