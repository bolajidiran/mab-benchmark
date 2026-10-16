# The protocol

Every rule below exists because the systematic review behind the thesis found that
its absence made published results incomparable (62 papers, March 2024, extended by
31 papers identified through citation searching in October 2026).

1. **Settings S1 to S5.** A claim is reported per cell (setting, K, T); a claim made
   in one setting does not transfer to another. S1 accepts T in {500, 2,000, 10,000};
   S2 accepts T in {1,000, 10,000}; S3 runs K = 10, T = 10,000 with 10 change-points;
   S5 runs K = 500, T in {50, 100, 200}.
2. **Mandatory baselines.** B0 to B3 run in every cell. A study without all four is
   marked "baseline incomplete" and cannot be certified.
3. **Seeds.** Run i uses `SEED_TABLE[i]`. The environment draws its instance from
   `default_rng(seed)`; rewards and the algorithm's randomness come from two child
   streams of `SeedSequence([seed, 1])` (stream layout 1.2.0). Studies cannot choose
   their own seeds.
4. **Statistics.** At least 30 runs per cell; two-sided Wilcoxon signed-rank tests on
   paired runs; Holm correction over all pairs in the cell; Cohen's d and the
   Vargha-Delaney A12 for every comparison, with |d| < 0.2 labelled negligible.
5. **Normalised GUF.** NCR = R / R_rand, where R_rand is the expected regret of the
   uniform random policy on the same instance (NCR = 1 for Random, 0 for an oracle);
   L is the mean relative latency violation; CCM adds bandwidth. Gamma =
   w1 NCR + w2 L + w3 CCM with default weights (0.6, 0.2, 0.2), subject to
   CCM <= tau_CCM.
6. **The BCS gate.** The ranking of a setting is certified only if its Bridge
   Compliance Score is at least 0.9. The BCS belongs to the setting, not to an
   algorithm. Settings without a real-data Tier 2 (S1 to S3, S5) have no BCS and
   their rankings are Tier 1 results only. See [replay settings](replay.md).
7. **Certification levels.** C1: all settings, all baselines, 30 runs, a significant,
   non-negligible regret advantage over a baseline, and the gate passed. C3: in
   addition, a baseline beaten on both tiers. C4: in addition, a live A/B test with at
   least 1,000 users over 7 days.
