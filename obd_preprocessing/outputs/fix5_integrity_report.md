# Fix 5: integrity checks on the Open Bandit Dataset (campaign: all)

| Check | random | bts |
|---|---|---|
| File MD5 | 2eb4265914edd7345eb3ae78ca3fa1e9 | 0619c83803613d59a3c1b47a4b058519 |
| Rows (raw) | 1,374,327 | 12,357,200 |
| Missing values | 0 | 0 |
| Click values | [0, 1] | [0, 1] |
| item_id range | 0-79 | 0-79 |
| Positions | [1, 2, 3] | [1, 2, 3] |
| Propensity min / max | 0.0125 / 0.0125 | 1e-06 / 0.95424 |
| Timestamps in order | True | True |
| Period (UTC) | 2019-11-24 00:00:00 to 2019-11-30 23:59:59 | 2019-11-24 00:00:00 to 2019-11-30 23:59:59 |
| Exact duplicates found | 162 | 2188 |
| Clicks among duplicates | 161 | 2100 |
| Duplicates removed | 162 | 2188 |
| Rows (clean) | 1,374,165 | 12,355,012 |
| Clicks (clean) | 4,607 | 59,108 |
| CTR (clean) | 0.3353% | 0.4784% |
| Published CTR | 0.35% | 0.50% |
| User feature cardinalities | {'user_feature_0': 4, 'user_feature_1': 6, 'user_feature_2': 10, 'user_feature_3': 10} | {'user_feature_0': 4, 'user_feature_1': 6, 'user_feature_2': 11, 'user_feature_3': 10} |
| Affinity non-zero share | 0.0417% | 0.0422% |

All checks passed: True
