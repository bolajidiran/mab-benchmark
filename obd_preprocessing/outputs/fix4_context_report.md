# Fix 4: user + item + affinity context

User context x (dimension d = 35): bias, one-hot user features, one-hot position.
Shared item context z (dimension k = 42): standardised item_feature_0, one-hot item_feature_1 to 3, log1p(user-item affinity).

| User feature | Categories random | Categories bts | Identical hashes | Placeholder matched | One-hot columns | Rows still unmatched |
|---|---|---|---|---|---|---|
| user_feature_0 | 4 | 4 | 3 | yes | 4 | 0 (0.000%) |
| user_feature_1 | 6 | 6 | 5 | yes | 6 | 0 (0.000%) |
| user_feature_2 | 10 | 11 | 9 | yes | 11 | 30 (0.000%) |
| user_feature_3 | 10 | 10 | 9 | yes | 10 | 0 (0.000%) |

Placeholder category: rows carrying a log-specific hash on all four user features: random 35.363%, bts 35.304% (same rows on all four features: True). These two hashes are matched as one category under rules (a) to (c).

Item features: category partition identical across the two item_context files: True; item_feature_0 identical: True.
