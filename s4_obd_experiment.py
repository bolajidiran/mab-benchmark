"""
=============================================================================
S4 OPEN BANDIT DATASET EXPERIMENT - Chapter Four Section 4.4
Runs the S4 pipeline in obd_preprocessing/ (Steps 1 to 5)
=============================================================================
PhD Thesis : Developing a Performance Model of Multi-Armed-Bandit
             Recommender Systems (LASU 2026)
Student    : Ojediran Alaba Bolaji
Supervisors: Professor O.A. Enikuomehin | Professor A. Rahman

The experiment is carried out by obd_preprocessing/:
    Fix 1  full log, both behaviour policies (13.7M cleaned rows)
    Fix 2  30 disjoint, consecutive time slices = 30 independent paired runs
    Fix 3  position-bias test; positions pooled, per-position CTR reported
    Fix 4  user + item + affinity context (B4 LinUCB, B4h hybrid LinUCB)
    Fix 5  integrity checks, duplicate handling, time order
and this script writes the files Chapter Four uses:
    s4_obd_results.json, s4_obd_summary.txt, s4_obd_learning.png
The Bridge Compliance Score of S4 is computed in experiments/s4_bcs_bridge.py.

HOW TO RUN (Windows Command Prompt, from the repository root):
    python s4_obd_experiment.py                      main analysis
    python s4_obd_experiment.py --variant keep_duplicates
    python s4_obd_experiment.py --rerun              force Steps 1 to 5 again
If the pipeline outputs are missing, Steps 1 to 5 run automatically
(about 50 minutes on 2 cores).

Citation: Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2020).
          A large-scale open dataset for bandit algorithms. arXiv:2008.07146.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
PIPE = REPO / "obd_preprocessing"
sys.path.insert(0, str(PIPE))
import common as C          # noqa: E402
import figures as F         # noqa: E402


def run_pipeline(variant: str) -> None:
    steps = ["step1_fix5_integrity.py", "step2_fix3_positions.py", "step3_fix4_context.py",
             "step4_fix1_fix2_slices.py", "step5_run_s4_replay.py"]
    for step in steps:
        cmd = [sys.executable, str(PIPE / step), "--variant", variant]
        if step.startswith("step1") and variant == "keep_duplicates":
            cmd.append("--keep-duplicates")
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    C.add_variant_arg(ap)
    ap.add_argument("--rerun", action="store_true", help="run Steps 1 to 5 again")
    args = ap.parse_args()
    variant = args.variant or "main"
    C.set_variant(variant)
    if args.rerun or not (C.OUT_DIR / "s4_replay_results.json").exists():
        run_pipeline(variant)
    res = json.loads((C.OUT_DIR / "s4_replay_results.json").read_text())
    curves = pd.read_csv(C.OUT_DIR / "s4_learning_curves.csv")
    A = res["analyses"]["pooled"]
    out_dir = REPO if variant == "main" else C.OUT_DIR

    table = {a: {**A["summary"][a], "rank": A["ranking"].index(a) + 1} for a in A["ranking"]}
    payload = {
        "experiment": "S4 Open Bandit Dataset, rejection-sampling replay",
        "variant": variant,
        "dataset": "OBD campaign all, random + bts logs, rejection-sampling replay (c = 1/80)",
        "n_runs": 30, "metric": "CTR on accepted events (higher is better)",
        "ranking": A["ranking"], "table_4_8": table,
        "pairwise": A["pairwise"],
        "per_position_ctr": res["positions"]["pooled"],
        "sensitivity_random_log_only": res["analyses"]["random_only"],
        "checks": res["checks"],
        "objective_3_status": (f"{sum(p['significant'] for p in A['pairwise'])} of {len(A['pairwise'])} pairs "
                               "differ significantly after Holm-Bonferroni correction; the BCS of S4 is "
                               "reported by experiments/s4_bcs_bridge.py."),
        "source_files": {"pipeline_results": str((C.OUT_DIR / "s4_replay_results.json").relative_to(REPO)),
                         "pipeline_summary": str((C.OUT_DIR / "s4_replay_summary.md").relative_to(REPO))},
    }
    C.write_json(payload, out_dir / "s4_obd_results.json")

    sig = [p for p in A["pairwise"] if p["significant"]]
    L = ["S4 OBD EXPERIMENT - Ojediran, A.B. - LASU 2026",
         "=" * 78, "",
         f"Variant  : {variant}",
         "Dataset  : Open Bandit Dataset, campaign all, random + bts logs (Saito et al., 2020)",
         "Replay   : rejection sampling, accept if matched and u < min(1, (1/80) / propensity)",
         "Runs     : 30 disjoint time slices, seeds 1000 to 1029 (mab_benchmark.runner)", "",
         "TABLE 4.8 - S4 CTR RESULTS (higher = better)",
         f"  {'Rank':<5}{'Algorithm':<24}{'Mean CTR':>10}{'SD':>9}{'95% CI':>22}{'Accepted':>10}{'Clicks':>8}",
         "  " + "-" * 86]
    for i, a in enumerate(A["ranking"], 1):
        s = A["summary"][a]
        L.append(f"  {i:<5}{a:<24}{s['mean_ctr']:>9.4%}{s['sd_ctr']:>9.4%}"
                 f"   [{s['ci95_low']:.4%}, {s['ci95_high']:.4%}]{s['mean_accepted']:>10,.0f}{s['mean_clicks']:>8.1f}")
    L += ["", f"Significant pairs after Holm-Bonferroni: {len(sig)} of {len(A['pairwise'])}",
          *[f"  {p['a']} vs {p['b']}: p = {p['p_raw']:.4g}, higher CTR = {p['higher_ctr']}" for p in sig],
          "", "Objective 3: " + payload["objective_3_status"],
          "", f"Full tables: {payload['source_files']['pipeline_summary']}"]
    (out_dir / "s4_obd_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    F.learning_figure(curves, A["summary"], "pooled", out_dir / "s4_obd_learning.png",
                      "S4: replay on the Open Bandit Dataset (random + bts, 30 runs)")
    print("\n".join(L))
    print(f"\nWritten: {out_dir / 's4_obd_results.json'}, s4_obd_summary.txt, s4_obd_learning.png")


if __name__ == "__main__":
    main()
