"""
make_notebook.py: writes ../FG_TS_Benchmark_Final.ipynb.

The notebook uses the package and the FG-TS implementation in
FeelGoodThompsonSampling.py, reads the raw arrays of experiments/results/, and
checks that a fresh run of the primary cell reproduces those arrays bit for bit.

Run: python make_notebook.py            (writes the notebook)
     python make_notebook.py --execute  (writes and executes it in place)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "FG_TS_Benchmark_Final.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# FG-TS on the MAB Unified Benchmark Suite (mab_benchmark 1.2.0)

**Ojediran Alaba Bolaji**, PhD Computer Science, Lagos State University.

This notebook evaluates the optimism-bonus Thompson Sampling algorithm of the thesis (FG-TS, λ = 0.5) against the four mandatory baselines on all five benchmark settings, and reproduces the Chapter Four and Five results.

- Every run uses `run_streams(SEED_TABLE[i])`: independent reward and algorithm streams, so every number is fixed by the seed table; section 2 checks this bit for bit.
- FG-TS comes from one implementation, `FeelGoodThompsonSampling.py`. It adds the count-based bonus λ√(ln(t+1)/(n_a+1)) to each Beta posterior sample; Zhang (2022) defines the feel-good term inside the loss of a contextual model, so the algorithm here is an adaptation inspired by that paper.
- The design expectations tested in section 9 are this thesis' own.
- Certification uses the `Certifier` and the repository leaderboard (section 11), with the S4 Bridge Compliance Score measured on the Open Bandit Dataset (section 10).

**Reference.** Zhang, T. (2022). Feel-Good Thompson Sampling for contextual bandits and reinforcement learning. *SIAM Journal on Mathematics of Data Science, 4*(2), 834-857. https://doi.org/10.1137/21M140924X

Run the cells in order. `MODE = "load"` (default) reads the saved raw arrays in seconds; `MODE = "rerun"` regenerates every cell with the runner (about 45 minutes on two cores)."""))

cells.append(code("""import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

REPO = Path.cwd()
if not (REPO / "mab_benchmark").is_dir():
    raise SystemExit("Open this notebook from the repository root (the folder that contains mab_benchmark/).")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))

import mab_benchmark
from mab_benchmark import BenchmarkRunner, StatisticalAnalyser, BCSCalculator
from mab_benchmark.runner import SEED_TABLE, STREAM_LAYOUT
from mab_benchmark.analysis.certification import Certifier
from mab_benchmark.analysis.leaderboard import LeaderboardSubmitter, Leaderboard
from FeelGoodThompsonSampling import FeelGoodThompsonSampling
import exp_common as E

print("mab_benchmark", mab_benchmark.__version__, "| stream layout", STREAM_LAYOUT, "| seeds", SEED_TABLE[:3], "...")
assert mab_benchmark.__version__ >= "1.2.0", "update the package: pip install -e ."
FIVE = ["FG-TS", "B3_TS", "B2_UCB1", "B1_EpsGreedy", "B0_Random"]
LABEL = E.LABEL
MODE = "load"          # "load" or "rerun\""""))

cells.append(md("## 1. Results of all 21 cells (30 runs each)\nIn `load` mode the raw arrays come from `experiments/results/raw/`, written by `experiments/run_synthetic_suite.py`."))
cells.append(code("""def run_cell(name, cell):
    cls, kw = E.ALGS[name]
    env, env_kw, run_kw = E.CELLS[cell]
    return BenchmarkRunner(30).run_setting(cls, kw, env, env_kw, **run_kw)

suite = {}
for a in FIVE + ["B4_LinUCB"]:
    suite[a] = {}
    for c in E.CELLS:
        if a == "B4_LinUCB" and not c.startswith("S4"):
            continue
        f = E.RAW / f"{c}__{a}.npy"
        suite[a][c] = run_cell(a, c) if MODE == "rerun" or not f.exists() else np.load(f)
print("cells:", len(suite["FG-TS"]), "| runs per cell:", {len(v) for v in suite["FG-TS"].values()})"""))

cells.append(md("## 2. Seeding check: a fresh run reproduces the saved arrays exactly\nThe primary cell (S1, K = 10, T = 10,000) is re-run now for all five algorithms. Identical numbers prove that every result is fixed by `SEED_TABLE` alone."))
cells.append(code("""for a in FIVE:
    fresh = run_cell(a, "S1_K10_T10000")
    same = np.array_equal(fresh, suite[a]["S1_K10_T10000"])
    print(f"{LABEL[a]:<26s} mean {fresh.mean():9.2f}   identical to saved arrays: {same}")
    assert same"""))

cells.append(md("## 3. The optimism bonus\nFG-TS adds `λ √(ln(t+1)/(n_a+1))` to each Beta posterior sample. With λ = 0 it is exactly B3 Thompson Sampling (checked in section 6)."))
cells.append(code("""t = 1000
print(f"bonus at t = {t}, lambda = 0.5")
for n in (0, 1, 5, 20, 100, 500):
    print(f"  n_a = {n:>4d}   bonus = {0.5*np.sqrt(np.log(t+1)/(n+1)):.4f}")"""))

cells.append(md("## 4. Mean cumulative regret in every cell (lower is better)"))
cells.append(code("""rows = []
for c in E.CELLS:
    algs = FIVE + (["B4_LinUCB"] if c.startswith("S4") else [])
    m = {a: suite[a][c].mean() for a in algs}
    order = sorted(algs, key=m.get)
    rows.append({"cell": c, **{a: round(m[a], 2) for a in algs}, "FG-TS rank": order.index("FG-TS") + 1})
table = pd.DataFrame(rows).set_index("cell")
table"""))

cells.append(md("## 5. The four statistical requirements on the primary cell"))
cells.append(code("""an = StatisticalAnalyser(alpha=0.05)
report = an.analyse({a: suite[a]["S1_K10_T10000"] for a in FIVE})
print(an.format_report(report))"""))

cells.append(md("## 6. Learning curves and the λ sweep"))
cells.append(code("""cur = np.load(E.OUT / "curves_S1_K10_T10000.npz")
lam = pd.read_csv(E.OUT / "lambda_sweep.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
for a in FIVE[:-1]:
    m = cur[a].mean(0); h = 1.96 * cur[a].std(0, ddof=1) / np.sqrt(30); x = np.arange(1, m.size + 1)
    a1.plot(x, m, label=LABEL[a]); a1.fill_between(x, m - h, m + h, alpha=0.15)
a1.set(title="S1, K = 10, T = 10,000 (B0 Random omitted)", xlabel="round t", ylabel="mean cumulative regret"); a1.legend()
g = lam[lam.cell == "S1_K10_T10000"].groupby("lambda").cumulative_regret.agg(["mean", "std"])
a2.errorbar(g.index, g["mean"], yerr=1.96 * g["std"] / np.sqrt(30), marker="o")
a2.axhline(suite["B3_TS"]["S1_K10_T10000"].mean(), ls="--", c="grey", label="B3 TS")
a2.set(title="FG-TS bonus weight on S1", xlabel="λ", ylabel="mean cumulative regret"); a2.legend()
plt.tight_layout(); plt.show()
l0 = lam[(lam.cell == "S1_K10_T10000") & (lam["lambda"] == 0)].sort_values("run").cumulative_regret.to_numpy()
print("λ = 0 identical to B3 run by run:", np.allclose(l0, suite["B3_TS"]["S1_K10_T10000"]))"""))

cells.append(md("## 7. S3: change-points, and the Meta-Policy Orchestrator (Layer 2)\nThe MPO resets the posterior of Thompson-Sampling-family algorithms when it flags a sustained drop in realised reward (`experiments/run_layers_2_6.py`)."))
cells.append(code("""f = E.OUT / "layers26_mpo_summary.csv"
if f.exists():
    s = pd.read_csv(f)
    display(s[s.cell == "S3_K10_T10000"][["algorithm", "regret_plain", "regret_mpo", "change_pct", "wilcoxon_p",
                                         "flags_per_run", "resets_per_run", "detected_all_cp", "false_alarms_per_run"]].round(3))
    display(s[s.cell == "S1_K10_T10000"][["algorithm", "regret_plain", "regret_mpo", "flags_per_run"]].round(3))
else:
    print("run experiments/run_layers_2_6.py first")"""))

cells.append(md("## 8. FG-TS against each mandatory baseline in every cell\nOne claim per (cell, baseline): `better` means significantly lower regret after Holm-Bonferroni with |d| ≥ 0.2."))
cells.append(code("""claims = Certifier(require_bcs_for_c1=False).tier1_claims("FG-TS", {a: suite[a] for a in FIVE})
cl = pd.DataFrame(claims)
display(cl.pivot(index="setting", columns="baseline", values="verdict"))
print(cl.groupby(["baseline", "verdict"]).size().unstack(fill_value=0))"""))

cells.append(md("## 9. Design expectations for the optimism bonus (Table 5.1)"))
cells.append(code("""pd.read_csv(E.TAB / "table_5_1_design_expectations.csv")"""))

cells.append(md("## 10. Bridge Compliance Score on S4\nThe S4 bridge test (`experiments/s4_bcs_bridge.py`) measures each condition: the full-log replay horizon (c3), arm independence on the real log (c2), and a Tier 1 simulator matched to the S4 domain (c1). The certificate uses the variant fixed in advance as primary: the unbiased random-log replay with the domain-matched Tier 1."))
cells.append(code("""f = E.OUT / "s4_bcs_bridge.json"
PRIMARY_BCS = "random_only: full-log Tier 2, domain-matched Tier 1"
if f.exists():
    v = json.loads(f.read_text())["bcs_variants"]
    rows = []
    for k, b in v.items():
        rows.append({"variant": k, "rho_S": b["rho_S"], "c1": b["c1_reward_proxy"], "c2": b["c2_arm_independence"],
                     "c3": b["c3_sufficient_horizon"], "BCS": b["BCS"], "band": b["band"],
                     "rho 95% CI": b.get("rho_boot_ci95", "")})
    display(pd.DataFrame(rows).round(4))
    bcs_s4 = v[PRIMARY_BCS]
else:
    bcs_s4 = None; print("run experiments/s4_bcs_bridge.py first")"""))

cells.append(md("## 11. Certification and leaderboard entry"))
cells.append(code("""tier2 = None
f2 = E.OUT / "s4_bcs_bridge_runs.csv"
if f2.exists():                       # Tier 2: CTR in the full-log replay of the uniform random log
    r2 = pd.read_csv(f2); r2 = r2[r2.stream == "random_only"]
    tier2 = {a: r2[r2.algorithm == a].sort_values("run").ctr.to_numpy() for a in FIVE}
cert = Certifier().certify("FG-TS", {a: suite[a] for a in FIVE}, bcs=bcs_s4, tier2=tier2)
print(cert["label"]); [print("  -", r) for r in cert["reasons_not_higher"]]
print(json.dumps(cert["tier1_summary"], indent=1))
sub = LeaderboardSubmitter("FG-TS", paper_url="https://doi.org/10.1137/21M140924X",
                           code_url="https://github.com/bolajidiran/mab-benchmark",
                           institution="Lagos State University").build({a: suite[a] for a in FIVE}, cert)
print(LeaderboardSubmitter.summary(sub))
display(pd.DataFrame(cert["tier2_claims"]))
# experiments/build_leaderboard.py writes this submission to Leaderboard/ and fgts_submission.json"""))

cells.append(md("""## 12. How to read these results

1. **Stationary S1.** B3 Thompson Sampling has the lowest mean regret in 7 of the 9 S1 cells (ε-Greedy in the other two, both with K = 200). FG-TS has lower mean regret than UCB1 in 8 of the 9 cells; the bonus costs it regret against B3 on every stationary cell.
2. **λ.** On S1, regret rises with λ at every step; on S3 it falls up to λ = 1 and rises again at λ = 2. λ = 0 reproduces B3 run by run.
3. **S3.** Of the plain algorithms, UCB1 recovers best after change-points. Section 7 shows how far the MPO's posterior reset moves B3 and FG-TS, with the paired test.
4. **S4 synthetic.** Only LinUCB uses the context; every context-free algorithm stays at Random level, as the setting intends.
5. **S5.** With K = 500 and T ≤ 200, ε-Greedy has the lowest regret in all three cells; with λ > 0, FG-TS pulls a never-pulled arm in every round.
6. **Certification.** The level follows from the checks printed in section 11; read the listed reasons before quoting a level."""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    nbf.write(nb, NB)
    if a.execute:
        from nbclient import NotebookClient
        n = nbf.read(NB, as_version=4)
        NotebookClient(n, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
        nbf.write(n, NB)
    print("written", NB)


if __name__ == "__main__":
    main()
