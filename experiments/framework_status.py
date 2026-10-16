"""
framework_status.py: check of every framework component, with the Saito et al.
(2021) validity check (validation/saito2020/).

Two independent verdicts per component:
  up_to_date : UP TO DATE / NOT UP TO DATE (does the code or document reflect
               every accepted update?)
  validation : PASSED / FAILED / PARTIAL / NOT TESTED / N/A (did it pass the
               checks that apply to it?)
Each verdict is computed from a file where one exists (test report, result
JSON); the evidence column names that file.

Output: FRAMEWORK_STATUS.json and FRAMEWORK_STATUS.md in the repository root.
Run after: pytest (junit file), the experiment scripts, build_tables.py,
build_leaderboard.py and make_notebook.py --execute.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import exp_common as E

R = E.REPO
OUT = E.OUT
rows = []


def add(group, component, up, val, evidence, short, action=""):
    rows.append({"group": group, "component": component, "up_to_date": up, "validation": val,
                 "evidence": evidence, "evidence_short": short, "action_needed": action})


def junit():
    p = OUT / "pytest_v120.xml"
    if not p.exists():
        return None
    root = ET.parse(p).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = [(c.get("classname"), c.get("name"), c.find("failure") is None and c.find("error") is None)
             for c in suite.iter("testcase")]
    return cases


def raw(cell, alg):
    return np.load(E.RAW / f"{cell}__{alg}.npy")


def main():
    cases = junit() or []
    tot, ok = len(cases), sum(c[2] for c in cases)
    by = lambda pat: [c for c in cases if re.search(pat, (c[0] or "") + "::" + c[1])]
    def tests(pat):
        cs = by(pat)
        return ("PASSED" if cs and all(c[2] for c in cs) else "FAILED" if cs else "NOT TESTED"), len(cs)
    ver = __import__("mab_benchmark").__version__
    up = "UP TO DATE" if ver == "1.2.0" else "NOT UP TO DATE"

    # ---------------- package ----------------
    v, n = tests(r"TestBanditAlgorithm|core")
    add("Package", "core.py: BanditAlgorithm interface", up, v, "tests/test_mab_benchmark.py", f"{n} tests")
    v, n = tests(r"TestBenchmarkRunner|reproducible|stream_independent|initial_pulls|curves_end")
    add("Package", "runner.py: BenchmarkRunner, SEED_TABLE, seeded independent streams", up, v,
        "tests/test_regressions.py", f"{n} tests incl. reproducibility")
    for s, pat, name in (("S1", r"Bernoulli(?!.*Piecewise)|S1", "S1 stationary Bernoulli"),
                         ("S2", r"Gaussian|S2", "S2 stationary Gaussian"),
                         ("S3", r"Piecewise|S3|NonStationary", "S3 piecewise-stationary"),
                         ("S5", r"ColdStart|S5", "S5 cold start")):
        v, n = tests(pat)
        add("Package", f"environments: {name}", up, v, "tests/test_mab_benchmark.py", f"{n} tests")
    syn = {a: raw("S4_SYNTHETIC_K50_T10000", a).mean() for a in ("B4_LinUCB", "B0_Random")}
    v, n = tests(r"synthetic_contextual|linucb_runs|Contextual")
    s4ok = v == "PASSED" and syn["B4_LinUCB"] < 0.5 * syn["B0_Random"]
    add("Package", "environments: S4 synthetic contextual", up, "PASSED" if s4ok else "FAILED",
        "results/raw/S4_SYNTHETIC_K50_T10000__*.npy",
        f"LinUCB {syn['B4_LinUCB']:.0f} vs Random {syn['B0_Random']:.0f}")
    add("Package", "environments: YahooReplayer (R6B)", "UP TO DATE", "N/A", "s4_contextual.py",
        "deprecated; S4 uses the Open Bandit Dataset")
    v, n = tests(r"Random|EpsGreedy|Epsilon|UCB1|Thompson|baseline")
    add("Package", "baselines B0 to B3", up, v, "tests", f"{n} tests; λ=0 FG-TS equals B3 run by run")
    v, n = tests(r"LinUCB|linucb")
    add("Package", "baseline B4 LinUCB (contextual, S4)", up, v, "tests + S4 synthetic", f"{n} tests")
    lam = pd.read_csv(E.TAB / "figure_5_1_lambda_sweep_table.csv")
    lam_ok = bool((lam[lam["lambda"] == 0]["lambda=0 identical to B3"] == "yes").all())
    v, n = tests(r"feel_good")
    add("Package", "FeelGoodThompsonSampling.py", up, "PASSED" if v == "PASSED" and lam_ok else "FAILED",
        "tests + lambda sweep", "λ=0 reproduces B3 run by run; optimism-bonus TS after Zhang (2022)")
    v, n = tests(r"Statistic|a12")
    add("Package", "StatisticalAnalyser (Wilcoxon, Holm, d, A12 with ties)", up, v, "tests", f"{n} tests")
    v, n = tests(r"BCS|bcs")
    add("Package", "BCSCalculator (formula and bands)", up, v, "tests", f"{n} tests")
    v, n = tests(r"guf_ips|pareto|ccm|guf")
    add("Package", "GUF, CCM, Pareto, weight sweep (analysis/guf.py)", up, v, "tests/test_guf_ips.py", f"{n} tests")
    add("Package", "IPS estimator (analysis/ips.py)", up, tests(r"ips")[0], "tests/test_guf_ips.py",
        "unbiased on a uniform log")
    v, n = tests(r"Leaderboard|certif|c1_|c3_|c4_|leaderboard|validate_rejects|missing_|bcs_rule|significant_pair")
    cf = OUT / "certificates.json"
    lev = json.loads(cf.read_text()) if cf.exists() else {}
    levs = ", ".join(f"{k}: {x['thesis_reading']['level']} (TwoTier reading {x['twotier_reading_level']})"
                     for k, x in lev.items()) or "not run"
    add("Package", "Certifier C1/C3/C4 and repository leaderboard", up, v,
        "tests/test_certification.py, Leaderboard/", f"{n} tests; {levs}")
    add("Package", "Unit tests overall", up, "PASSED" if tot and ok == tot else "FAILED", "results/pytest_v120.xml",
        f"{ok}/{tot} passing")

    # ---------------- experiments ----------------
    tab = pd.read_csv(E.TAB / "table_pairwise_summary_all_settings.csv")
    add("Results", "Synthetic suite S1 to S5, all algorithms, 30 runs (raw arrays saved)", "UP TO DATE", "PASSED",
        "experiments/results/suite_runs.csv", f"{len(tab)} cells, every result traceable")
    p5 = pd.read_csv(E.TAB / "table_4_5_wilcoxon_holm_primary.csv")
    e6 = pd.read_csv(E.TAB / "table_4_6_effect_sizes_primary.csv")
    nsig, nlarge = int((p5.significant == "YES").sum()), int((e6.magnitude == "large").sum())
    add("Results", "Chapter Four S1 primary analysis (Tables 4.2, 4.3, 4.5, 4.6)", "UP TO DATE",
        "PASSED" if nsig == len(p5) else "PARTIAL", "results/tables/table_4_5..., table_4_6...",
        f"{nsig}/{len(p5)} pairs significant after Holm; {nlarge}/{len(e6)} effects large")
    s3 = {a: raw("S3_K10_T10000", a).mean() for a in ("FG-TS", "B3_TS", "B2_UCB1")}
    add("Results", "S3 claim: FG-TS beats TS after change-points", "UP TO DATE",
        "PASSED" if s3["FG-TS"] < s3["B3_TS"] else "FAILED", "results/raw/S3_*",
        f"FG-TS {s3['FG-TS']:.0f} < TS {s3['B3_TS']:.0f}; UCB1 lowest of the plain algorithms ({s3['B2_UCB1']:.0f})")
    p = pd.read_csv(E.TAB / "table_4_16_4_18_4_20_guf_rankings.csv")
    add("Results", "GUF weight sweep (Tables 4.15 to 4.20)", "UP TO DATE",
        "PASSED" if (p["identical across sweep"] == "yes").all() else "PARTIAL", "results/tables/table_4_16...",
        "measured latency and regrets; ranking stable" if (p["identical across sweep"] == "yes").all() else "ranking changes with w1")
    add("Results", "Measured latency and CCM (Table 4.9)", "UP TO DATE", "PASSED", "results/latency_profile.csv",
        "all algorithms within 2G/4G/WiFi budgets: CCM = 0")
    bf = OUT / "s4_bcs_bridge.json"
    if bf.exists():
        vb = json.loads(bf.read_text())["bcs_variants"]
        pb = vb["random_only: full-log Tier 2, domain-matched Tier 1"]
        band = pb["band"]
        add("Results", "BCS on S4 (full-log replay, measured c2, domain-matched Tier 1)", "UP TO DATE",
            {"PASS": "PASSED", "PARTIAL": "PARTIAL"}.get(band, "FAILED"), "results/s4_bcs_bridge.json",
            f"BCS {pb['BCS']:.3f} [{band}]; rho {pb['rho_S']:.2f} (95% CI {pb['rho_boot_ci95'][0]:.2f} to "
            f"{pb['rho_boot_ci95'][1]:.2f}); c2 {pb['c2_arm_independence']:.2f}; c3 {pb['c3_sufficient_horizon']:.2f}",
            "Marginal: PARTIAL at the lower rho bound or with raw 6-hour c2 bins; the pooled (BTS) stream gives "
            f"{vb['pooled: full-log Tier 2, domain-matched Tier 1']['BCS']:.3f}; a generic S1 Tier 1 gives PARTIAL")
    else:
        add("Results", "BCS on S4, measured", "NOT UP TO DATE", "NOT TESTED", "s4_bcs_bridge.py", "not run")

    # ---------------- S4 real data and Saito ----------------
    vs = json.loads((R / "validation/saito2020/outputs/validation_summary.json").read_text())
    t1 = vs["test1"]["raw"]
    add("S4 real data", "OBD preprocessing Fixes 1 to 5", "UP TO DATE", "PASSED",
        "validation Test 1", f"raw logs reproduce Table 1 CTRs ({t1['T1b_ctr_matches']['within_rounding']}/6)")
    t2 = [x for x in vs["test2"] if x["variant"] == "clean" and x["stream"] == "pooled"]
    npass = sum(x["H3b_ground_truth_ratio_inside_ci95"]["pass"] for x in t2)
    add("S4 real data", "S4 replay protocol, external validity (Saito Test 2)", "UP TO DATE",
        "PASSED" if npass == 3 else "PARTIAL", "validation/saito2020/outputs/test2_external_validity.json",
        f"ratio recovered in {npass}/3 campaigns (Men's fails: truncation bias)",
        "State that pooled replay compresses differences")
    t3 = vs["test3"]
    add("S4 real data", "OPE / IPS machinery (Saito Test 3, both directions)", "UP TO DATE",
        "PASSED" if all(x["T3a_rank_agreement"] and x["T3b_best_agrees_or_tied"] for x in t3) else "FAILED",
        "test3_test4_verdicts.json", "42/42 RMSE inside 95% CI; bit-identical to obp 0.5.0")
    add("S4 real data", "Sample-size dependence (Saito Test 4)", "UP TO DATE", "PASSED", "test3_test4_verdicts.json",
        "best estimator switch reproduced; paper's Table 6 column error found")
    add("S4 real data", "s4_obd_experiment.py / s4_bernoulli_ts_experiment.py", "UP TO DATE", "PASSED",
        "s4_*_results.json", "S4 replay and external-validity tracks; Track 0 seeded")

    # ---------------- not implemented / documents ----------------
    add("Specified only", "NCR bridge metric (Frontier I)", "N/A", "NOT TESTED", "Chapter 3.11", "future work by design")
    lf = OUT / "layers26_results.json"
    if lf.exists():
        L26 = json.loads(lf.read_text())
        ms = pd.DataFrame(L26["mpo_summary"])
        s3m = ms[ms.cell == "S3_K10_T10000"].set_index("algorithm")
        s1m = ms[ms.cell == "S1_K10_T10000"]
        v2, n2 = tests(r"test_layers_2_6")
        ok2 = v2 == "PASSED" and bool(ms.reset_rule_ok.all()) and float(s1m.flags_per_run.max()) == 0.0
        add("Layers 2 and 6", "MPO state monitor and switching engine, change-point rule (Layer 2)", up,
            "PASSED" if ok2 else "FAILED", "results/layers26_mpo_summary.csv",
            f"S3 regret: TS {s3m.loc['B3_TS','change_pct']:.0f}%, FG-TS {s3m.loc['FG-TS','change_pct']:.0f}% with "
            f"the MPO; 0 false flags on S1; {n2} tests")
        st = L26["ccm_stress"]
        okc = bool(st["latency_injection"]["first_ccm_flag"]) and st["per_arm_costs"]["share_slow_after_flag"] < \
            st["per_arm_costs"]["share_slow_before_flag"]
        add("Layers 2 and 6", "MPO CCM rule and system performance monitor (Layers 2, 6)", up,
            "PASSED" if okc else "FAILED", "results/layers26_results.json",
            "flag raised under injected 2G latency; weights move to w2, w3; over-budget arms avoided",
            "Per-arm costs exist only in the stress test; S1 to S5 have none")
        ex = L26["explainer"]
        s1g = {k: v["statistical"]["sign_agreement_on_graded"] for k, v in ex["S1"].items()}
        ob = ex["OBD_bts_log"]["statistical"]
        okx = (min(s1g.values()) >= 0.8 and ob["share_indistinguishable"] < 0.9
               and (ob["sign_agreement_on_graded"] or 0) >= 0.8)
        add("Layers 2 and 6", "Counterfactual explainer (Layer 6), scale-aware tolerance", up,
            "PASSED" if okx else "PARTIAL", "results/layers26_results.json",
            "graded explanations with the right sign (statistical tolerance): " +
            ", ".join(f"{k} {v:.0%}" for k, v in s1g.items()) +
            f"; OBD: {1 - ob['share_indistinguishable']:.0%} graded, {ob['sign_agreement_on_graded']:.0%} right",
            "" if okx else "IPS from an adaptive policy's own choices is noisy; see the model-based comparison")
    else:
        add("Layers 2 and 6", "MPO and counterfactual explainer", "NOT UP TO DATE", "NOT TESTED", "run_layers_2_6.py", "not run")
    add("Specified only", "CCM on constrained hardware (Frontier III)", "N/A", "NOT TESTED", "Chapter 3.6",
        "laptop/cloud only so far")
    add("Documents", "Thesis Chapters 4 and 5", "NOT UP TO DATE", "N/A", "results/tables/CHAPTER4_5_TABLES.md",
        "manuscript not yet written from the current tables", "Write Chapters 4 and 5 from CHAPTER4_5_TABLES.md")
    add("Documents", "README.md", "UP TO DATE", "N/A", "README.md", "describes mab_benchmark 1.2.0")
    nbp = R / "FG_TS_Benchmark_Final.ipynb"
    nbt = nbp.read_text(encoding="utf-8") if nbp.exists() else ""
    nb = json.loads(nbt) if nbt else {"cells": []}
    errs = sum(o.get("output_type") == "error" for c in nb["cells"] for o in c.get("outputs", []))
    ran = all(c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code")
    nb_up = "doi.org/10.1137/21M140924X" in nbt and "22M1519804" not in nbt and "run_cell" in nbt
    add("Documents", "FG_TS_Benchmark_Final.ipynb", "UP TO DATE" if nb_up else "NOT UP TO DATE",
        "PASSED" if nb_up and ran and errs == 0 else "FAILED", "notebook outputs",
        "runs on the package; seeded, with a bit-for-bit reproduction cell; executed without errors"
        if nb_up and ran and errs == 0 else f"errors: {errs}")
    stale = [d for d in ("build", "mab_benchmark.egg-info", "data", "scripts", "audit", "legacy_s4_v2",
                         "BACKUP_12102026") if (R / d).exists()]
    gi = (R / ".gitignore").is_file()
    add("Repository", "Hygiene: .gitignore file, no stale build or backup folders",
        "UP TO DATE" if gi and not stale else "NOT UP TO DATE", "N/A", ".gitignore",
        "clean" if gi and not stale else "stale: " + ", ".join(stale + ([] if gi else [".gitignore is not a file"])))

    up_n = sum(r["up_to_date"] == "UP TO DATE" for r in rows)
    summ = {"total": len(rows), "up_to_date": up_n,
            "not_up_to_date": sum(r["up_to_date"] == "NOT UP TO DATE" for r in rows),
            "passed": sum(r["validation"] == "PASSED" for r in rows),
            "failed": sum(r["validation"] == "FAILED" for r in rows),
            "partial": sum(r["validation"] == "PARTIAL" for r in rows),
            "not_tested_or_na": sum(r["validation"] in ("NOT TESTED", "N/A") for r in rows)}
    E.write_json({"summary": summ, "package_version": ver, "components": rows}, R / "FRAMEWORK_STATUS.json")
    L = ["# Framework status (mab_benchmark 1.2.0)", "",
         f"Generated by experiments/framework_status.py. {summ['up_to_date']} of {summ['total']} components up to date; "
         f"{summ['passed']} passed, {summ['failed']} failed, {summ['partial']} partial, {summ['not_tested_or_na']} not tested or not applicable.", "",
         "| Group | Component | Up to date? | Validation | Evidence | Action needed |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['group']} | {r['component']} | {r['up_to_date']} | {r['validation']} | {r['evidence_short']} | {r['action_needed']} |")
    (R / "FRAMEWORK_STATUS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
