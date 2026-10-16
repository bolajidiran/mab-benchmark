# Submitting a result

The leaderboard lives in the repository (`Leaderboard/`), and every entry is a JSON
file that anyone can re-check. To submit an algorithm:

1. **Implement** it as a `BanditAlgorithm` subclass in a public repository.
2. **Run the full suite with the baselines**, unchanged seeds and at least 30 runs:

   ```python
   from mab_benchmark import BenchmarkRunner, Certifier, LeaderboardSubmitter
   runner = BenchmarkRunner(n_runs=30)
   suite = runner.baseline_sweep()                 # B0 to B3, every cell
   suite["MyAlg"] = runner.full_suite(MyAlg, {})
   ```

3. **Compute the S4 gate.** Replay the Open Bandit Dataset uniform log with your
   algorithm and the six reference algorithms (`experiments/s4_bcs_bridge.py` and
   `experiments/s4_replay_bias.py` show how), and compute `BCS_R` with
   `BCSCalculator.compute_replay`. Report `rho_shuffle` and the matched-event ratios
   even when they are unfavourable.
4. **Certify**: `cert = Certifier().certify("MyAlg", suite, bcs=bcs_r, tier2=tier2)`.
   The certifier gates C1 on BCS_R and refuses C3 to an algorithm the replay starves.
5. **Build and validate the submission**:

   ```python
   sub = LeaderboardSubmitter("MyAlg", paper_url, code_url, institution).build(suite, cert)
   assert not LeaderboardSubmitter.validate(sub)
   LeaderboardSubmitter.save(sub, "Leaderboard/submissions/myalg.json")
   ```

6. **Open a pull request** adding that one file, with the exact command that produced
   it and the output of `mab-benchmark version`. A maintainer re-runs it; the entry is
   merged when the numbers reproduce, and `Leaderboard.build()` regenerates the table.

What makes a submission ineligible: fewer than 30 runs in any cell, a missing
baseline, changed seeds or stream layout, a level claimed with a failed check, or
numbers that do not reproduce from the stated code.
