# Test 1: on-policy CTR against Saito et al. (Table 1)

| Log | Variant | Rows | Published rows | CTR % (95% CI) | Published CTR % | Relative CTR | Published |
|---|---|---|---|---|---|---|---|
| random/all | raw | 1,374,327 | 1,374,327 | 0.3469 (±0.0098) | 0.35 (±0.010) | 1.000 | 1.00 |
| bts/all | raw | 12,357,200 | 12,168,084 | 0.4953 (±0.0039) | 0.50 (±0.004) | 1.428 | 1.43 |
| random/men | raw | 452,949 | 452,949 | 0.5124 (±0.0208) | 0.51 (±0.021) | 1.477 | 1.48 |
| bts/men | raw | 4,077,727 | 4,077,727 | 0.6743 (±0.0079) | 0.67 (±0.008) | 1.944 | 1.94 |
| random/women | raw | 864,585 | 864,585 | 0.4815 (±0.0146) | 0.48 (±0.014) | 1.388 | 1.39 |
| bts/women | raw | 7,765,497 | 7,765,497 | 0.6379 (±0.0056) | 0.64 (±0.056) | 1.839 | 1.84 |
| random/all | clean | 1,374,165 | 1,374,327 | 0.3353 (±0.0097) | 0.35 (±0.010) | 1.000 | 1.00 |
| bts/all | clean | 12,355,012 | 12,168,084 | 0.4784 (±0.0038) | 0.50 (±0.004) | 1.427 | 1.43 |
| random/men | clean | 452,868 | 452,949 | 0.4946 (±0.0204) | 0.51 (±0.021) | 1.475 | 1.48 |
| bts/men | clean | 4,076,843 | 4,077,727 | 0.6528 (±0.0078) | 0.67 (±0.008) | 1.947 | 1.94 |
| random/women | clean | 864,448 | 864,585 | 0.4657 (±0.0144) | 0.48 (±0.014) | 1.389 | 1.39 |
| bts/women | clean | 7,764,005 | 7,765,497 | 0.6191 (±0.0055) | 0.64 (±0.056) | 1.847 | 1.84 |

**raw**: T1a rows 5/6, items 6/6; T1b CTR pass (within rounding 6/6); T1c relative CTR 6/6; T1d ordering pass.

**clean**: T1a rows 0/6, items 6/6; T1b CTR fail (within rounding 0/6); T1c relative CTR 4/6; T1d ordering pass.
