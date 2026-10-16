"""
build_leaderboard.py: certifies the thesis' two test algorithms with the v1.2.0
Certifier and rebuilds the repository leaderboard in ../Leaderboard/.

Inputs: results/raw/ (Tier 1, 21 cells), results/s4_bcs_bridge.json and
results/s4_bcs_bridge_runs.csv (BCS and Tier 2, primary variant: random_only,
full-log replay, domain-matched Tier 1). Writes ../Leaderboard/submissions/*.json,
../Leaderboard/leaderboard.json, ../Leaderboard/LEADERBOARD.md,
../fgts_submission.json and results/certificates.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import exp_common as E
from mab_benchmark.analysis.certification import Certifier
from mab_benchmark.analysis.leaderboard import Leaderboard, LeaderboardSubmitter

PRIMARY_BCS = "random_only: full-log Tier 2, domain-matched Tier 1"
BASE = ["B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS"]
TESTS = {"FG-TS": "FG-TS (lambda = 0.5), optimism-bonus Thompson Sampling",
         "BernoulliTS_uniform": "Bernoulli TS, uniform priors (TEST-1 of Table 4.11)"}


def main():
    suite = {a: {c: np.load(E.RAW / f"{c}__{a}.npy") for c in E.CELLS} for a in BASE + list(TESTS)}
    bcs, tier2 = None, {}
    f = E.OUT / "s4_bcs_bridge.json"
    if f.exists():
        bcs = json.loads(f.read_text())["bcs_variants"][PRIMARY_BCS]
        runs = pd.read_csv(E.OUT / "s4_bcs_bridge_runs.csv")
        r = runs[runs.stream == "random_only"]
        tier2 = {a: r[r.algorithm == a].sort_values("run").ctr.to_numpy() for a in BASE + ["FG-TS"]}
    lb = Leaderboard(E.REPO / "Leaderboard")
    certs = {}
    for name, desc in TESTS.items():
        t2 = tier2 if name in tier2 else None
        thesis = Certifier().certify(name, suite, bcs=bcs, tier2=t2)
        twotier = Certifier(require_bcs_for_c1=False).certify(name, suite, bcs=bcs, tier2=t2)
        certs[name] = {"thesis_reading": thesis, "twotier_reading_level": twotier["level"],
                       "twotier_reading_reasons": twotier["reasons_not_higher"]}
        sub = LeaderboardSubmitter(
            name, paper_url="https://doi.org/10.1137/21M140924X" if name == "FG-TS" else "",
            code_url="https://github.com/bolajidiran/mab-benchmark", institution="Lagos State University",
            notes=f"{desc}. Level under the TwoTierProtocol reading (no BCS requirement for C1): "
                  f"{twotier['level']}.").build(suite, thesis)
        lb.add(sub)
        if name == "FG-TS":
            LeaderboardSubmitter.save(sub, str(E.REPO / "fgts_submission.json"))
        print(name, thesis["level"], thesis["reasons_not_higher"], "| TwoTier reading:", twotier["level"])
    rows = lb.build()
    E.write_json(certs, E.OUT / "certificates.json")
    print(pd.DataFrame(rows)[["algorithm", "mean_regret", "level", "bcs"]])


if __name__ == "__main__":
    main()
