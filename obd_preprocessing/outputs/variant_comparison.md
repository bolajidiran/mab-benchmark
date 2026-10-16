# Sensitivity: duplicates dropped (main) vs kept

Strict check: CHANGED: at least one significant pair or hypothesis outcome differs; see the 'same' flags.

## Conclusion by conclusion

| Conclusion | Stream | Main | Kept | Same |
|---|---|---|---|---|
| K1 BernoulliTS_Zozotown ranks first | pooled | yes | yes | yes |
| K2 No significant pair among the other learners | pooled | yes | yes | yes |
| K3 External validity (H2, H3a, H3b) holds | pooled | yes | yes | yes |
| K4 H1: BernoulliTS_Zozotown > B0 Random | pooled | yes (p = 0.0000) | yes (p = 0.0001) | yes |
| K5 Order of the other (non-significant) learners | pooled | see table below | see table below | **no** |
| K1 BernoulliTS_Zozotown ranks first | random_only | yes | yes | yes |
| K2 No significant pair among the other learners | random_only | yes | yes | yes |
| K3 External validity (H2, H3a, H3b) holds | random_only | yes | yes | yes |
| K4 H1: BernoulliTS_Zozotown > B0 Random | random_only | yes (p = 0.0173) | no (p = 0.0571) | **no** |
| K5 Order of the other (non-significant) learners | random_only | see table below | see table below | **no** |

| Log | Rows main | Rows kept | Clicks main | Clicks kept | CTR main | CTR kept |
|---|---|---|---|---|---|---|
| random | 1,374,165 | 1,374,327 | 4,607 | 4,768 | 0.3353% | 0.3469% |
| bts | 12,355,012 | 12,357,200 | 59,108 | 61,208 | 0.4784% | 0.4953% |

## Stream: pooled

| Algorithm | Mean CTR main | Mean CTR kept | Rank main | Rank kept |
|---|---|---|---|---|
| BernoulliTS_Zozotown | 0.4915% | 0.5299% | 1 | 1 |
| B1_EpsGreedy | 0.4151% | 0.3308% | 2 | 7 |
| B4h_HybridLinUCB | 0.4020% | 0.3639% | 3 | 4 |
| FG-TS | 0.3878% | 0.3565% | 4 | 5 |
| B3_TS | 0.3868% | 0.3173% | 5 | 8 |
| B2_UCB1 | 0.3561% | 0.3400% | 6 | 6 |
| B0_Random | 0.3494% | 0.4011% | 7 | 3 |
| B4_LinUCB | 0.3401% | 0.4749% | 8 | 2 |

| Quantity | Main | Kept | Same conclusion |
|---|---|---|---|
| Significant pairs (Holm) | 1 | 4 | no |
| Smallest raw p | 7.911e-05 | 6.918e-06 | |
| Replay ratio BTS-Z / Random | 1.407x | 1.321x | |
| Ground-truth ratio | 1.427x | 1.428x | |
| H1_bts_beats_random | supported | supported | yes |
| H2_rank_top_two | supported | supported | yes |
| H3a_direction_matches | supported | supported | yes |
| H3b_ratio_matches | supported | supported | yes |

## Stream: random_only

| Algorithm | Mean CTR main | Mean CTR kept | Rank main | Rank kept |
|---|---|---|---|---|
| BernoulliTS_Zozotown | 0.4386% | 0.5474% | 1 | 1 |
| FG-TS | 0.3506% | 0.3107% | 2 | 4 |
| B4h_HybridLinUCB | 0.3072% | 0.3072% | 3 | 5 |
| B3_TS | 0.2935% | 0.3448% | 4 | 3 |
| B0_Random | 0.2925% | 0.4600% | 5 | 2 |
| B1_EpsGreedy | 0.2676% | 0.2573% | 6 | 6 |
| B2_UCB1 | 0.2158% | 0.2158% | 7 | 7 |
| B4_LinUCB | 0.1985% | 0.1985% | 8 | 8 |

| Quantity | Main | Kept | Same conclusion |
|---|---|---|---|
| Significant pairs (Holm) | 0 | 2 | no |
| Smallest raw p | 0.002744 | 0.0002833 | |
| Replay ratio BTS-Z / Random | 1.499x | 1.190x | |
| Ground-truth ratio | 1.427x | 1.428x | |
| H1_bts_beats_random | supported | not supported | no |
| H2_rank_top_two | supported | supported | yes |
| H3a_direction_matches | supported | supported | yes |
| H3b_ratio_matches | supported | supported | yes |
