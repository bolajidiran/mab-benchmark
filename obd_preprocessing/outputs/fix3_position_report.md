# Fix 3: position-bias test and pooled replay stream

## Position bias (chi-square test of CTR by position)

| Log | CTR pos 1 | CTR pos 2 | CTR pos 3 | chi-square (dof 2) | p-value |
|---|---|---|---|---|---|
| random | 0.3428% | 0.3366% | 0.3263% | 1.90 | 0.387 |
| bts | 0.4776% | 0.4808% | 0.4769% | 0.74 | 0.69 |

Decision: no significant position bias, so the main S4 analysis pools
the three positions; per-position CTR is reported from the same runs.

## Pooled replay stream (random + bts, all positions, time-ordered)

| Stream | Rows | Clicks | Rows random | Rows bts | CTR random | CTR bts |
|---|---|---|---|---|---|---|
| pooled | 13,729,177 | 63,715 | 1,374,165 | 12,355,012 | 0.3353% | 0.4784% |

| Position | Rows | Clicks |
|---|---|---|
| 1 | 4,575,701 | 21,236 |
| 2 | 4,576,910 | 21,344 |
| 3 | 4,576,566 | 21,135 |
