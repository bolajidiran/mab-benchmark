# Fix 3: position-bias test and pooled replay stream

## Position bias (chi-square test of CTR by position)

| Log | CTR pos 1 | CTR pos 2 | CTR pos 3 | chi-square (dof 2) | p-value |
|---|---|---|---|---|---|
| random | 0.3541% | 0.3470% | 0.3396% | 1.40 | 0.497 |
| bts | 0.4943% | 0.5000% | 0.4917% | 2.99 | 0.224 |

Decision: no significant position bias, so the main S4 analysis pools
the three positions; per-position CTR is reported from the same runs.

## Pooled replay stream (random + bts, all positions, time-ordered)

| Stream | Rows | Clicks | Rows random | Rows bts | CTR random | CTR bts |
|---|---|---|---|---|---|---|
| pooled | 13,731,527 | 65,976 | 1,374,327 | 12,357,200 | 0.3469% | 0.4953% |

| Position | Rows | Clicks |
|---|---|---|
| 1 | 4,576,486 | 21,979 |
| 2 | 4,577,776 | 22,187 |
| 3 | 4,577,265 | 21,810 |
