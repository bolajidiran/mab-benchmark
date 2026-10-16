# Replay settings and the BCS

The replayer method (Li et al., 2011) is unbiased when the logged rows are
independent and identically distributed. Real logs often are not: on the Open Bandit
Dataset the same items reappear in bursts, and a deterministic explorer such as UCB1
then waits for an item that the log does not show. Since 1.3.0, every replay-based
Tier 2 result carries two mandatory diagnostics.

**Replay integrity.** Replay the log as recorded and again with its rows shuffled
within time windows (one hour on OBD). `rho_shuffle` is the Spearman correlation of
the two Tier 2 rankings, and `c_R = clip(rho_shuffle / 0.7, 0, 1)`.

**Starvation.** `m_a / m_rand`, the matched events of algorithm a divided by those of
B0 Random in the same run. Below 0.5 the algorithm is starved and its Tier 2 result
cannot be certified.

The gate for a replay setting is `BCS_R = (c1 + c2 + c3 + c_R) / 4 >= 0.9`, with a
Tier 1 that is bias-free: a simulator calibrated on the domain (`S4_obd_sim`) or a
fully observed dataset (`S4_kuairec`). In code:

```python
from mab_benchmark.analysis.bcs import BCSCalculator
from mab_benchmark import replay

ratios = replay.matched_ratios({"B0_Random": m_random, "B2_UCB1": m_ucb})
r = BCSCalculator.compute_replay(rho_s, p_indep, K, T, rho_shuffle, ratios)
print(r["BCS_R"], r["band_R"], r["starved_algorithms"])
```

or `mab-benchmark bcs --rho-s ... --p-indep ... -K 80 -T 14315 --rho-shuffle ... --ratio B2_UCB1=0.18`.

Off-policy evaluation of a learned policy uses `mab_benchmark.analysis.ope`, which
calls the Open Bandit Pipeline's estimators when `obp` is installed and otherwise
uses native implementations that match them to a relative tolerance of 1e-10.
