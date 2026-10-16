# S4 replay: Open Bandit Dataset (random + bts, campaign all)

## Main result: both logs, all positions pooled

| Rank | Algorithm | Mean CTR | SD | 95% CI | Mean accepted events | Mean clicks |
|---|---|---|---|---|---|---|
| 1 | BernoulliTS_Zozotown | 0.4915% | 0.1522% | 0.4371% to 0.5460% | 3,832 | 18.7 |
| 2 | B1_EpsGreedy | 0.4151% | 0.2065% | 0.3411% to 0.4890% | 3,519 | 13.9 |
| 3 | B4h_HybridLinUCB | 0.4020% | 0.3494% | 0.2770% to 0.5270% | 479 | 1.9 |
| 4 | FG-TS | 0.3878% | 0.2398% | 0.3020% to 0.4736% | 1,319 | 5.0 |
| 5 | B3_TS | 0.3868% | 0.1645% | 0.3279% to 0.4457% | 1,470 | 5.7 |
| 6 | B2_UCB1 | 0.3561% | 0.3359% | 0.2358% to 0.4763% | 221 | 0.8 |
| 7 | B0_Random | 0.3494% | 0.1177% | 0.3073% to 0.3915% | 2,371 | 8.3 |
| 8 | B4_LinUCB | 0.3401% | 0.2846% | 0.2382% to 0.4419% | 438 | 1.5 |

Pairwise tests (two-sided paired Wilcoxon over 30 runs, Holm-Bonferroni, alpha = 0.05; d and A12 oriented so that positive d / A12 > 0.5 favour the first algorithm):

| Pair | p (raw) | Holm threshold | Significant | Higher CTR | Cohen's d | A12 |
|---|---|---|---|---|---|---|
| B0_Random vs BernoulliTS_Zozotown | 7.911e-05 | 0.001786 | yes | BernoulliTS_Zozotown | -1.045 | 0.241 |
| FG-TS vs BernoulliTS_Zozotown | 0.0113 | 0.001852 | no | not significant | -0.516 | 0.319 |
| B3_TS vs BernoulliTS_Zozotown | 0.01283 | 0.001923 | no | not significant | -0.661 | 0.317 |
| BernoulliTS_Zozotown vs B4_LinUCB | 0.02623 | 0.002 | no | not significant | 0.664 | 0.699 |
| B2_UCB1 vs BernoulliTS_Zozotown | 0.06356 | 0.002083 | no | not significant | -0.520 | 0.384 |
| B1_EpsGreedy vs BernoulliTS_Zozotown | 0.07324 | 0.002174 | no | not significant | -0.422 | 0.372 |
| B1_EpsGreedy vs B4_LinUCB | 0.1347 | 0.002273 | no | not significant | 0.301 | 0.593 |
| BernoulliTS_Zozotown vs B4h_HybridLinUCB | 0.1519 | 0.002381 | no | not significant | 0.332 | 0.606 |
| B0_Random vs B1_EpsGreedy | 0.2621 | 0.0025 | no | not significant | -0.390 | 0.416 |
| B3_TS vs B4_LinUCB | 0.2621 | 0.002632 | no | not significant | 0.201 | 0.572 |
| B1_EpsGreedy vs B2_UCB1 | 0.3707 | 0.002778 | no | not significant | 0.212 | 0.558 |
| B0_Random vs B3_TS | 0.3931 | 0.002941 | no | not significant | -0.261 | 0.434 |
| B0_Random vs B4h_HybridLinUCB | 0.428 | 0.003125 | no | not significant | -0.202 | 0.471 |
| B2_UCB1 vs FG-TS | 0.4958 | 0.003333 | no | not significant | -0.109 | 0.467 |
| B0_Random vs B4_LinUCB | 0.5978 | 0.003571 | no | not significant | 0.043 | 0.517 |
| FG-TS vs B4_LinUCB | 0.612 | 0.003846 | no | not significant | 0.181 | 0.555 |
| B0_Random vs FG-TS | 0.6408 | 0.004167 | no | not significant | -0.203 | 0.467 |
| B1_EpsGreedy vs B3_TS | 0.6554 | 0.004545 | no | not significant | 0.151 | 0.536 |
| B2_UCB1 vs B3_TS | 0.6554 | 0.005 | no | not significant | -0.116 | 0.473 |
| B1_EpsGreedy vs FG-TS | 0.7303 | 0.005556 | no | not significant | 0.122 | 0.546 |
| B0_Random vs B2_UCB1 | 0.7457 | 0.00625 | no | not significant | -0.026 | 0.483 |
| B2_UCB1 vs B4h_HybridLinUCB | 0.7509 | 0.007143 | no | not significant | -0.134 | 0.466 |
| B3_TS vs B4h_HybridLinUCB | 0.8078 | 0.008333 | no | not significant | -0.056 | 0.502 |
| B3_TS vs FG-TS | 0.8394 | 0.01 | no | not significant | -0.005 | 0.521 |
| B4_LinUCB vs B4h_HybridLinUCB | 0.8712 | 0.0125 | no | not significant | -0.194 | 0.447 |
| B1_EpsGreedy vs B4h_HybridLinUCB | 0.9032 | 0.01667 | no | not significant | 0.045 | 0.539 |
| B2_UCB1 vs B4_LinUCB | 0.9494 | 0.025 | no | not significant | 0.051 | 0.520 |
| FG-TS vs B4h_HybridLinUCB | 0.9569 | 0.05 | no | not significant | -0.047 | 0.502 |

Per-position CTR from the same runs (clicks / accepted events, summed over runs):

| Algorithm | Position 1 | Position 2 | Position 3 |
|---|---|---|---|
| BernoulliTS_Zozotown | 0.5071% | 0.5353% | 0.4222% |
| B1_EpsGreedy | 0.4025% | 0.3784% | 0.4072% |
| B4h_HybridLinUCB | 0.4724% | 0.3809% | 0.3067% |
| FG-TS | 0.3057% | 0.4080% | 0.4233% |
| B3_TS | 0.4150% | 0.3727% | 0.3731% |
| B2_UCB1 | 0.3833% | 0.3775% | 0.3236% |
| B0_Random | 0.3846% | 0.2856% | 0.3743% |
| B4_LinUCB | 0.4115% | 0.4339% | 0.1825% |

## Sensitivity: random log only (no truncation bias, lower power)

| Rank | Algorithm | Mean CTR | SD | 95% CI | Mean accepted events | Mean clicks |
|---|---|---|---|---|---|---|
| 1 | BernoulliTS_Zozotown | 0.4386% | 0.2472% | 0.3501% to 0.5270% | 570 | 2.5 |
| 2 | FG-TS | 0.3506% | 0.2640% | 0.2562% to 0.4451% | 522 | 1.8 |
| 3 | B4h_HybridLinUCB | 0.3072% | 0.4421% | 0.1490% to 0.4654% | 141 | 0.5 |
| 4 | B3_TS | 0.2935% | 0.2036% | 0.2206% to 0.3663% | 541 | 1.6 |
| 5 | B0_Random | 0.2925% | 0.1832% | 0.2269% to 0.3580% | 573 | 1.7 |
| 6 | B1_EpsGreedy | 0.2676% | 0.3289% | 0.1499% to 0.3853% | 590 | 1.6 |
| 7 | B2_UCB1 | 0.2158% | 0.4023% | 0.0718% to 0.3597% | 103 | 0.2 |
| 8 | B4_LinUCB | 0.1985% | 0.3755% | 0.0641% to 0.3328% | 121 | 0.3 |

Pairwise tests (two-sided paired Wilcoxon over 30 runs, Holm-Bonferroni, alpha = 0.05; d and A12 oriented so that positive d / A12 > 0.5 favour the first algorithm):

| Pair | p (raw) | Holm threshold | Significant | Higher CTR | Cohen's d | A12 |
|---|---|---|---|---|---|---|
| BernoulliTS_Zozotown vs B4_LinUCB | 0.002744 | 0.001786 | no | not significant | 0.755 | 0.778 |
| B1_EpsGreedy vs BernoulliTS_Zozotown | 0.005107 | 0.001852 | no | not significant | -0.588 | 0.257 |
| B2_UCB1 vs BernoulliTS_Zozotown | 0.01328 | 0.001923 | no | not significant | -0.667 | 0.235 |
| B3_TS vs BernoulliTS_Zozotown | 0.03272 | 0.002 | no | not significant | -0.641 | 0.374 |
| B0_Random vs BernoulliTS_Zozotown | 0.03454 | 0.002083 | no | not significant | -0.671 | 0.334 |
| B1_EpsGreedy vs FG-TS | 0.1026 | 0.002174 | no | not significant | -0.279 | 0.338 |
| BernoulliTS_Zozotown vs B4h_HybridLinUCB | 0.1504 | 0.002273 | no | not significant | 0.367 | 0.673 |
| FG-TS vs B4_LinUCB | 0.1718 | 0.002381 | no | not significant | 0.469 | 0.731 |
| B0_Random vs B4_LinUCB | 0.1909 | 0.0025 | no | not significant | 0.318 | 0.703 |
| B2_UCB1 vs FG-TS | 0.1943 | 0.002632 | no | not significant | -0.396 | 0.274 |
| B4_LinUCB vs B4h_HybridLinUCB | 0.2347 | 0.002778 | no | not significant | -0.265 | 0.438 |
| FG-TS vs BernoulliTS_Zozotown | 0.2534 | 0.002941 | no | not significant | -0.344 | 0.437 |
| B3_TS vs B4_LinUCB | 0.2588 | 0.003125 | no | not significant | 0.315 | 0.706 |
| B0_Random vs B1_EpsGreedy | 0.3359 | 0.003333 | no | not significant | 0.093 | 0.602 |
| B3_TS vs FG-TS | 0.3931 | 0.003571 | no | not significant | -0.242 | 0.411 |
| B2_UCB1 vs B4h_HybridLinUCB | 0.3967 | 0.003846 | no | not significant | -0.216 | 0.452 |
| B0_Random vs B2_UCB1 | 0.4004 | 0.004167 | no | not significant | 0.245 | 0.703 |
| B2_UCB1 vs B3_TS | 0.4091 | 0.004545 | no | not significant | -0.244 | 0.297 |
| FG-TS vs B4h_HybridLinUCB | 0.4388 | 0.005 | no | not significant | 0.119 | 0.624 |
| B1_EpsGreedy vs B4_LinUCB | 0.4651 | 0.005556 | no | not significant | 0.196 | 0.663 |
| B1_EpsGreedy vs B3_TS | 0.4958 | 0.00625 | no | not significant | -0.095 | 0.398 |
| B1_EpsGreedy vs B2_UCB1 | 0.543 | 0.007143 | no | not significant | 0.141 | 0.659 |
| B1_EpsGreedy vs B4h_HybridLinUCB | 0.5677 | 0.008333 | no | not significant | -0.102 | 0.557 |
| B2_UCB1 vs B4_LinUCB | 0.5754 | 0.01 | no | not significant | 0.044 | 0.505 |
| B0_Random vs FG-TS | 0.5838 | 0.0125 | no | not significant | -0.256 | 0.406 |
| B0_Random vs B4h_HybridLinUCB | 0.7731 | 0.01667 | no | not significant | -0.044 | 0.586 |
| B3_TS vs B4h_HybridLinUCB | 0.8101 | 0.025 | no | not significant | -0.040 | 0.592 |
| B0_Random vs B3_TS | 0.9741 | 0.05 | no | not significant | -0.005 | 0.451 |

Per-position CTR from the same runs (clicks / accepted events, summed over runs):

| Algorithm | Position 1 | Position 2 | Position 3 |
|---|---|---|---|
| BernoulliTS_Zozotown | 0.5077% | 0.2662% | 0.5393% |
| FG-TS | 0.3277% | 0.3080% | 0.4181% |
| B4h_HybridLinUCB | 0.4332% | 0.2168% | 0.3411% |
| B3_TS | 0.3195% | 0.2883% | 0.2612% |
| B0_Random | 0.2602% | 0.3479% | 0.2639% |
| B1_EpsGreedy | 0.3145% | 0.2636% | 0.2142% |
| B2_UCB1 | 0.3906% | 0.1914% | 0.0978% |
| B4_LinUCB | 0.3331% | 0.1660% | 0.1638% |

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

## Checks

- B0 Random replay CTR (pooled stream) = 0.3487% vs random-log CTR 0.3353%: relative difference +4.0% (size of the truncation bias from the bts rows).
- Self-tests: fast LinUCB classes made identical choices to the references.
- Runtime: 244.2 minutes on 2 process(es).
