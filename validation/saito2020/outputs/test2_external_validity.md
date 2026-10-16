# Test 2: external validity of the S4 replay protocol

| Data | Campaign | Stream | Replay CTR BTS-Z % | Replay CTR Random % | Replay ratio (95% CI) | Ground truth | Published | H1 p | H3b | Random replay vs log |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | ALL | pooled | 0.495 | 0.363 | 1.361 (1.212 to 1.550) | 1.427 | 1.429 | 2.5e-05 | pass | +8.4% |
| clean | ALL | random_only | 0.412 | 0.349 | 1.182 (0.820 to 1.704) | 1.427 | 1.429 | 0.23 | pass | +4.1% |
| clean | Men's | pooled | 0.655 | 0.593 | 1.104 (0.971 to 1.272) | 1.320 | 1.314 | 0.19 | fail | +20.0% |
| clean | Men's | random_only | 0.622 | 0.412 | 1.509 (1.055 to 2.172) | 1.320 | 1.314 | 0.0087 | pass | -16.7% |
| clean | Women's | pooled | 0.697 | 0.556 | 1.254 (1.095 to 1.458) | 1.329 | 1.333 | 0.0044 | pass | +19.4% |
| clean | Women's | random_only | 0.561 | 0.496 | 1.130 (0.865 to 1.473) | 1.329 | 1.333 | 0.19 | pass | +6.6% |
| raw | ALL | pooled | 0.511 | 0.410 | 1.246 (1.064 to 1.461) | 1.428 | 1.429 | 0.0029 | pass | +18.3% |
| raw | ALL | random_only | 0.427 | 0.301 | 1.418 (0.908 to 2.255) | 1.428 | 1.429 | 0.19 | pass | -13.3% |
| raw | Men's | pooled | 0.651 | 0.606 | 1.075 (0.949 to 1.217) | 1.316 | 1.314 | 0.19 | fail | +18.3% |
| raw | Men's | random_only | 0.556 | 0.390 | 1.426 (1.007 to 2.023) | 1.316 | 1.314 | 0.022 | pass | -24.0% |
| raw | Women's | pooled | 0.670 | 0.554 | 1.210 (1.080 to 1.370) | 1.325 | 1.333 | 0.0023 | pass | +15.0% |
| raw | Women's | random_only | 0.644 | 0.491 | 1.312 (0.984 to 1.816) | 1.325 | 1.333 | 0.076 | pass | +2.0% |
