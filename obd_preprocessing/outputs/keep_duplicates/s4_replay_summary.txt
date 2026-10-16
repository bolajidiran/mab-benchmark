# S4 replay: Open Bandit Dataset (random + bts, campaign all)

## Main result: both logs, all positions pooled

| Rank | Algorithm | Mean CTR | SD | 95% CI | Mean accepted events | Mean clicks |
|---|---|---|---|---|---|---|
| 1 | BernoulliTS_Zozotown | 0.5299% | 0.1147% | 0.4889% to 0.5710% | 3,830 | 20.2 |
| 2 | B4_LinUCB | 0.4749% | 0.4359% | 0.3190% to 0.6309% | 438 | 2.0 |
| 3 | B0_Random | 0.4011% | 0.1646% | 0.3422% to 0.4600% | 2,354 | 9.4 |
| 4 | B4h_HybridLinUCB | 0.3639% | 0.3015% | 0.2560% to 0.4718% | 466 | 1.7 |
| 5 | FG-TS | 0.3565% | 0.1865% | 0.2898% to 0.4232% | 1,327 | 4.7 |
| 6 | B2_UCB1 | 0.3400% | 0.3644% | 0.2096% to 0.4704% | 219 | 0.7 |
| 7 | B1_EpsGreedy | 0.3308% | 0.1683% | 0.2706% to 0.3910% | 3,323 | 10.4 |
| 8 | B3_TS | 0.3173% | 0.1526% | 0.2626% to 0.3719% | 1,472 | 4.8 |

Pairwise tests (two-sided paired Wilcoxon over 30 runs, Holm-Bonferroni, alpha = 0.05; d and A12 oriented so that positive d / A12 > 0.5 favour the first algorithm):

| Pair | p (raw) | Holm threshold | Significant | Higher CTR | Cohen's d | A12 |
|---|---|---|---|---|---|---|
| B1_EpsGreedy vs BernoulliTS_Zozotown | 6.918e-06 | 0.001786 | yes | BernoulliTS_Zozotown | -1.382 | 0.167 |
| B3_TS vs BernoulliTS_Zozotown | 1.218e-05 | 0.001852 | yes | BernoulliTS_Zozotown | -1.575 | 0.110 |
| FG-TS vs BernoulliTS_Zozotown | 9.903e-05 | 0.001923 | yes | BernoulliTS_Zozotown | -1.120 | 0.237 |
| B0_Random vs BernoulliTS_Zozotown | 0.0002316 | 0.002 | yes | BernoulliTS_Zozotown | -0.908 | 0.228 |
| BernoulliTS_Zozotown vs B4h_HybridLinUCB | 0.003744 | 0.002083 | no | not significant | 0.728 | 0.728 |
| B2_UCB1 vs BernoulliTS_Zozotown | 0.009932 | 0.002174 | no | not significant | -0.703 | 0.281 |
| B2_UCB1 vs B4_LinUCB | 0.08352 | 0.002273 | no | not significant | -0.336 | 0.417 |
| B0_Random vs B3_TS | 0.08407 | 0.002381 | no | not significant | 0.528 | 0.678 |
| B0_Random vs B1_EpsGreedy | 0.1241 | 0.0025 | no | not significant | 0.422 | 0.623 |
| B1_EpsGreedy vs B4_LinUCB | 0.1294 | 0.002632 | no | not significant | -0.436 | 0.427 |
| B3_TS vs B4_LinUCB | 0.1294 | 0.002778 | no | not significant | -0.483 | 0.419 |
| BernoulliTS_Zozotown vs B4_LinUCB | 0.1294 | 0.002941 | no | not significant | 0.172 | 0.659 |
| B0_Random vs FG-TS | 0.2129 | 0.003125 | no | not significant | 0.254 | 0.569 |
| B0_Random vs B2_UCB1 | 0.2206 | 0.003333 | no | not significant | 0.216 | 0.581 |
| B3_TS vs FG-TS | 0.2449 | 0.003571 | no | not significant | -0.230 | 0.434 |
| B4_LinUCB vs B4h_HybridLinUCB | 0.3389 | 0.003846 | no | not significant | 0.296 | 0.556 |
| B1_EpsGreedy vs B3_TS | 0.428 | 0.004167 | no | not significant | 0.084 | 0.531 |
| B0_Random vs B4h_HybridLinUCB | 0.5028 | 0.004545 | no | not significant | 0.153 | 0.567 |
| B1_EpsGreedy vs FG-TS | 0.5158 | 0.005 | no | not significant | -0.145 | 0.462 |
| FG-TS vs B4_LinUCB | 0.5425 | 0.005556 | no | not significant | -0.353 | 0.467 |
| B3_TS vs B4h_HybridLinUCB | 0.7303 | 0.00625 | no | not significant | -0.195 | 0.460 |
| B2_UCB1 vs FG-TS | 0.7375 | 0.007143 | no | not significant | -0.057 | 0.438 |
| B2_UCB1 vs B4h_HybridLinUCB | 0.7702 | 0.008333 | no | not significant | -0.071 | 0.463 |
| B1_EpsGreedy vs B4h_HybridLinUCB | 0.7922 | 0.01 | no | not significant | -0.136 | 0.487 |
| B2_UCB1 vs B3_TS | 0.7922 | 0.0125 | no | not significant | 0.081 | 0.494 |
| B0_Random vs B4_LinUCB | 0.8236 | 0.01667 | no | not significant | -0.224 | 0.516 |
| B1_EpsGreedy vs B2_UCB1 | 0.8553 | 0.025 | no | not significant | -0.032 | 0.529 |
| FG-TS vs B4h_HybridLinUCB | 0.8712 | 0.05 | no | not significant | -0.029 | 0.515 |

Per-position CTR from the same runs (clicks / accepted events, summed over runs):

| Algorithm | Position 1 | Position 2 | Position 3 |
|---|---|---|---|
| BernoulliTS_Zozotown | 0.5310% | 0.5130% | 0.5359% |
| B4_LinUCB | 0.6190% | 0.4725% | 0.2599% |
| B0_Random | 0.3849% | 0.3783% | 0.4309% |
| B4h_HybridLinUCB | 0.4060% | 0.3919% | 0.2923% |
| FG-TS | 0.3942% | 0.3237% | 0.3328% |
| B2_UCB1 | 0.3939% | 0.3756% | 0.1847% |
| B1_EpsGreedy | 0.3154% | 0.3205% | 0.3032% |
| B3_TS | 0.3378% | 0.3436% | 0.2883% |

## Sensitivity: random log only (no truncation bias, lower power)

| Rank | Algorithm | Mean CTR | SD | 95% CI | Mean accepted events | Mean clicks |
|---|---|---|---|---|---|---|
| 1 | BernoulliTS_Zozotown | 0.5474% | 0.2480% | 0.4586% to 0.6361% | 573 | 3.1 |
| 2 | B0_Random | 0.4600% | 0.3227% | 0.3446% to 0.5755% | 568 | 2.6 |
| 3 | B3_TS | 0.3448% | 0.2277% | 0.2633% to 0.4263% | 550 | 1.9 |
| 4 | FG-TS | 0.3107% | 0.2634% | 0.2164% to 0.4049% | 516 | 1.6 |
| 5 | B4h_HybridLinUCB | 0.3072% | 0.4421% | 0.1490% to 0.4654% | 141 | 0.5 |
| 6 | B1_EpsGreedy | 0.2573% | 0.2637% | 0.1629% to 0.3517% | 588 | 1.5 |
| 7 | B2_UCB1 | 0.2158% | 0.4023% | 0.0718% to 0.3597% | 103 | 0.2 |
| 8 | B4_LinUCB | 0.1985% | 0.3755% | 0.0641% to 0.3328% | 121 | 0.3 |

Pairwise tests (two-sided paired Wilcoxon over 30 runs, Holm-Bonferroni, alpha = 0.05; d and A12 oriented so that positive d / A12 > 0.5 favour the first algorithm):

| Pair | p (raw) | Holm threshold | Significant | Higher CTR | Cohen's d | A12 |
|---|---|---|---|---|---|---|
| BernoulliTS_Zozotown vs B4_LinUCB | 0.0002833 | 0.001786 | yes | BernoulliTS_Zozotown | 1.096 | 0.798 |
| B1_EpsGreedy vs BernoulliTS_Zozotown | 0.0003801 | 0.001852 | yes | BernoulliTS_Zozotown | -1.133 | 0.189 |
| FG-TS vs BernoulliTS_Zozotown | 0.002766 | 0.001923 | no | not significant | -0.925 | 0.297 |
| B2_UCB1 vs BernoulliTS_Zozotown | 0.004032 | 0.002 | no | not significant | -0.992 | 0.220 |
| B3_TS vs BernoulliTS_Zozotown | 0.004032 | 0.002083 | no | not significant | -0.851 | 0.271 |
| B0_Random vs B4_LinUCB | 0.00586 | 0.002174 | no | not significant | 0.747 | 0.758 |
| B0_Random vs B2_UCB1 | 0.01224 | 0.002273 | no | not significant | 0.670 | 0.746 |
| B0_Random vs B1_EpsGreedy | 0.01656 | 0.002381 | no | not significant | 0.688 | 0.713 |
| BernoulliTS_Zozotown vs B4h_HybridLinUCB | 0.01966 | 0.0025 | no | not significant | 0.670 | 0.702 |
| B0_Random vs B4h_HybridLinUCB | 0.05577 | 0.002632 | no | not significant | 0.395 | 0.657 |
| FG-TS vs B4_LinUCB | 0.07272 | 0.002778 | no | not significant | 0.346 | 0.691 |
| B3_TS vs B4_LinUCB | 0.07953 | 0.002941 | no | not significant | 0.471 | 0.738 |
| B0_Random vs FG-TS | 0.08794 | 0.003125 | no | not significant | 0.507 | 0.595 |
| B0_Random vs B3_TS | 0.0961 | 0.003333 | no | not significant | 0.413 | 0.593 |
| B0_Random vs BernoulliTS_Zozotown | 0.1142 | 0.003571 | no | not significant | -0.304 | 0.382 |
| B2_UCB1 vs FG-TS | 0.1161 | 0.003846 | no | not significant | -0.279 | 0.321 |
| B1_EpsGreedy vs B3_TS | 0.1642 | 0.004167 | no | not significant | -0.355 | 0.327 |
| B2_UCB1 vs B3_TS | 0.1864 | 0.004545 | no | not significant | -0.395 | 0.266 |
| B1_EpsGreedy vs FG-TS | 0.2286 | 0.005 | no | not significant | -0.202 | 0.390 |
| B4_LinUCB vs B4h_HybridLinUCB | 0.2347 | 0.005556 | no | not significant | -0.265 | 0.438 |
| B1_EpsGreedy vs B2_UCB1 | 0.3458 | 0.00625 | no | not significant | 0.122 | 0.684 |
| B1_EpsGreedy vs B4_LinUCB | 0.3458 | 0.007143 | no | not significant | 0.181 | 0.692 |
| B2_UCB1 vs B4h_HybridLinUCB | 0.3967 | 0.008333 | no | not significant | -0.216 | 0.452 |
| B3_TS vs B4h_HybridLinUCB | 0.4123 | 0.01 | no | not significant | 0.107 | 0.625 |
| B2_UCB1 vs B4_LinUCB | 0.5754 | 0.0125 | no | not significant | 0.044 | 0.505 |
| B1_EpsGreedy vs B4h_HybridLinUCB | 0.7775 | 0.01667 | no | not significant | -0.137 | 0.586 |
| B3_TS vs FG-TS | 0.8542 | 0.025 | no | not significant | 0.139 | 0.509 |
| FG-TS vs B4h_HybridLinUCB | 0.8552 | 0.05 | no | not significant | 0.009 | 0.591 |

Per-position CTR from the same runs (clicks / accepted events, summed over runs):

| Algorithm | Position 1 | Position 2 | Position 3 |
|---|---|---|---|
| BernoulliTS_Zozotown | 0.6804% | 0.4565% | 0.5030% |
| B0_Random | 0.4860% | 0.5167% | 0.3707% |
| B3_TS | 0.3607% | 0.3491% | 0.3263% |
| FG-TS | 0.3135% | 0.3094% | 0.3068% |
| B4h_HybridLinUCB | 0.4332% | 0.2168% | 0.3411% |
| B1_EpsGreedy | 0.2987% | 0.2174% | 0.2305% |
| B2_UCB1 | 0.3918% | 0.1908% | 0.0978% |
| B4_LinUCB | 0.3359% | 0.1652% | 0.1645% |

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

## Checks

- B0 Random replay CTR (pooled stream) = 0.3978% vs random-log CTR 0.3469%: relative difference +14.7% (size of the truncation bias from the bts rows).
- Self-tests: fast LinUCB classes made identical choices to the references.
- Runtime: 53.4 minutes on 2 process(es).
