"""
summarise_validation.py: computes the verdicts of Tests 3 and 4 and gathers
all four tests into one summary.

Pass criteria for Tests 3 and 4, written before the 200-seed runs finished:
  per configuration (direction x campaign at n = 300,000; Test 3)
    T3a  Spearman rank correlation between reproduced and published RMSE
         over the seven estimators is at least 0.6;
    T3b  the published best estimator is the reproduced best, or its
         reproduced 95% bootstrap interval overlaps that of the reproduced
         best (a statistical tie);
    T3c  (descriptive) number of estimators whose published RMSE lies inside
         the reproduced 95% interval; median relative difference.
  per campaign (Bernoulli TS log -> Random; Test 4)
    T4a  the published best estimator at n = 10,000 is the reproduced best
         or statistically tied with it (criterion of T3b);
    T4b  for each estimator, the direction of the RMSE change from
         n = 10,000 to n = 300,000 agrees with the published direction;
    T4c  the reproduced best estimator changes between n = 10,000 and
         n = 300,000 in the campaigns where the published best changes.

Output: outputs/test3_test4_verdicts.json / .md, outputs/validation_summary.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import saito_common as C


def _cfg(s, beh, camp, n):
    r = s[(s.behaviour == beh) & (s.campaign == camp) & (s.n == n)]
    return None if r.empty else r.iloc[0]


def _best(r, key="rmse"):
    vals = {e: r[f"{key}_{e}"] for e in C.ESTIMATORS}
    return min(vals, key=vals.get), vals


def _tied(r, a, b):
    la, ha = r[f"rmse_ci95_{a}"]; lb, hb = r[f"rmse_ci95_{b}"]
    return a == b or (la <= hb and lb <= ha)


def config_verdict(r):
    rep = np.array([r[f"rmse_{e}"] for e in C.ESTIMATORS])
    pub = np.array([r[f"published_{e}"] for e in C.ESTIMATORS])
    rho, p = stats.spearmanr(rep, pub)
    best_rep, _ = _best(r)
    best_pub = C.ESTIMATORS[int(np.argmin(pub))]
    inside = [e for e in C.ESTIMATORS if r[f"rmse_ci95_{e}"][0] <= r[f"published_{e}"] <= r[f"rmse_ci95_{e}"][1]]
    rel = (rep - pub) / pub
    return {"behaviour": r.behaviour, "evaluation": r.evaluation, "campaign": r.campaign, "n": int(r.n),
            "n_seeds": int(r.n_seeds), "ground_truth": r.ground_truth,
            "lambda_Switch-DR": r["lambda_Switch-DR"], "lambda_DRos": r["lambda_DRos"],
            "rmse_reproduced": dict(zip(C.ESTIMATORS, rep.round(4).tolist())),
            "rmse_published": dict(zip(C.ESTIMATORS, pub.tolist())),
            "spearman_rho": float(rho), "spearman_p": float(p),
            "T3a_rank_agreement": bool(rho >= 0.6),
            "best_reproduced": best_rep, "best_published": best_pub,
            "T3b_best_agrees_or_tied": bool(_tied(r, best_pub, best_rep)),
            "published_inside_ci95": inside, "n_published_inside_ci95": len(inside),
            "median_relative_difference": float(np.median(rel)),
            "max_abs_relative_difference": float(np.abs(rel).max()),
            "bias_x1e3": {e: round(r[f"bias_{e}"], 4) for e in C.ESTIMATORS},
            "sd_x1e3": {e: round(r[f"sd_{e}"], 4) for e in C.ESTIMATORS}}


def main():
    s = pd.DataFrame(json.loads((C.OUT_DIR / "ope" / "ope_summary_raw.json").read_text()))
    t3 = [config_verdict(r) for beh in ("bts", "random") for camp in C.CAMPAIGNS
          if (r := _cfg(s, beh, camp, 300_000)) is not None]
    t4 = []
    pub10 = C.PUBLISHED["table6_bts_to_random_n10000"]; pub300 = C.PUBLISHED["table4_bts_to_random_n300000"]
    for camp in C.CAMPAIGNS:
        a, b = _cfg(s, "bts", camp, 10_000), _cfg(s, "bts", camp, 300_000)
        if a is None:
            continue
        v10 = config_verdict(a)
        best_pub10 = min(C.ESTIMATORS, key=lambda e: pub10[e][camp])
        rec = {"campaign": camp, "n10000": v10,
               "T4a_best_at_10000_agrees_or_tied": bool(_tied(a, best_pub10, v10["best_reproduced"])),
               "best_published_10000": best_pub10, "best_reproduced_10000": v10["best_reproduced"]}
        if b is not None:
            best_pub300 = min(C.ESTIMATORS, key=lambda e: pub300[e][camp])
            best_rep300 = _best(b)[0]
            agree = {e: bool(np.sign(b[f"rmse_{e}"] - a[f"rmse_{e}"]) == np.sign(pub300[e][camp] - pub10[e][camp]))
                     for e in C.ESTIMATORS}
            rec.update({"best_published_300000": best_pub300, "best_reproduced_300000": best_rep300,
                        "direction_agreement": agree, "T4b_directions_agree": f"{sum(agree.values())}/7",
                        "published_best_changes": best_pub10 != best_pub300,
                        "reproduced_best_changes": v10["best_reproduced"] != best_rep300,
                        "T4c_switch_reproduced": (v10["best_reproduced"] != best_rep300) == (best_pub10 != best_pub300)})
        t4.append(rec)
    t10 = [config_verdict(r) for camp in C.CAMPAIGNS if (r := _cfg(s, "random", camp, 10_000)) is not None]
    dup = None
    a, b = _cfg(s, "bts", "women", 10_000), _cfg(s, "random", "men", 10_000)
    if a is not None and b is not None:
        shared = np.array([pub10[e]["women"] for e in C.ESTIMATORS])
        ra = np.array([a[f"rmse_{e}"] for e in C.ESTIMATORS]); rb = np.array([b[f"rmse_{e}"] for e in C.ESTIMATORS])
        dup = {"shared_published_column": dict(zip(C.ESTIMATORS, shared.tolist())),
               "reproduced_table6_women_bts_to_random": dict(zip(C.ESTIMATORS, ra.round(4).tolist())),
               "reproduced_table10_men_random_to_bts": dict(zip(C.ESTIMATORS, rb.round(4).tolist())),
               "median_abs_rel_diff_table6_women": float(np.median(np.abs(ra - shared) / shared)),
               "median_abs_rel_diff_table10_men": float(np.median(np.abs(rb - shared) / shared))}
        dup["column_belongs_to"] = ("Table 10, Men (Random -> BTS)" if dup["median_abs_rel_diff_table10_men"]
                                    < dup["median_abs_rel_diff_table6_women"] else "Table 6, Women (BTS -> Random)")
    out = {"test3": t3, "test4": t4, "table10_random_to_bts_n10000": t10, "duplicated_column_check": dup}
    C.write_json(out, C.OUT_DIR / "test3_test4_verdicts.json")

    L = ["# Tests 3 and 4: OPE estimator benchmark against Saito et al.", "",
         "RMSE x 10^3 over 200 bootstrap seeds; published values in brackets.", ""]
    for v in t3 + [x["n10000"] for x in t4] + t10:
        L += [f"## {v['behaviour']} log -> {v['evaluation']}, {C.CAMPAIGN_LABEL[v['campaign']]}, n = {v['n']:,} "
              f"({v['n_seeds']} seeds; tuned lambda Switch-DR {v['lambda_Switch-DR']:g}, DRos {v['lambda_DRos']:g})", "",
              "| Estimator | Reproduced | Published | Bias | SD |", "|---|---|---|---|---|"]
        for e in C.ESTIMATORS:
            L.append(f"| {e} | {v['rmse_reproduced'][e]:.3f} | {v['rmse_published'][e]:.3f} | "
                     f"{v['bias_x1e3'][e]:+.3f} | {v['sd_x1e3'][e]:.3f} |")
        L += ["", f"Spearman rho = {v['spearman_rho']:.3f}; best reproduced {v['best_reproduced']}, best published "
                  f"{v['best_published']} (agree or tied: {'yes' if v['T3b_best_agrees_or_tied'] else 'no'}); "
                  f"published inside 95% CI {v['n_published_inside_ci95']}/7; median relative difference "
                  f"{100 * v['median_relative_difference']:+.1f}%.", ""]
    for x in t4:
        L.append(f"- {C.CAMPAIGN_LABEL[x['campaign']]}: best at 10,000 reproduced {x['best_reproduced_10000']} vs "
                 f"published {x['best_published_10000']} (T4a {'pass' if x['T4a_best_at_10000_agrees_or_tied'] else 'fail'})"
                 + (f"; best at 300,000 reproduced {x['best_reproduced_300000']} vs published "
                    f"{x['best_published_300000']}; direction of change agrees for {x['T4b_directions_agree']}; "
                    f"switch reproduced: {'yes' if x['T4c_switch_reproduced'] else 'no'}" if "T4b_directions_agree" in x else ""))
    if dup:
        L += ["", "## Which table holds the duplicated column?", "",
              "| Estimator | Shared published value | Reproduced, Table 6 Women (BTS -> Random) | Reproduced, Table 10 Men (Random -> BTS) |",
              "|---|---|---|---|"]
        for e in C.ESTIMATORS:
            L.append(f"| {e} | {dup['shared_published_column'][e]:.3f} | {dup['reproduced_table6_women_bts_to_random'][e]:.3f} | "
                     f"{dup['reproduced_table10_men_random_to_bts'][e]:.3f} |")
        L += ["", f"Median absolute relative difference: Table 6 Women {100 * dup['median_abs_rel_diff_table6_women']:.1f}%, "
                  f"Table 10 Men {100 * dup['median_abs_rel_diff_table10_men']:.1f}%. The shared column reproduces as "
                  f"{dup['column_belongs_to']}."]
    (C.OUT_DIR / "test3_test4_verdicts.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    summ = {"test1": json.loads((C.OUT_DIR / "test1_on_policy_ctr.json").read_text())["verdict"],
            "test2": [{k: r[k] for k in ("variant", "campaign", "stream", "replay_ratio", "replay_ratio_ci95",
                                         "ground_truth_ratio", "H1_bts_beats_random", "H3a_ratio_above_1",
                                         "H3b_ground_truth_ratio_inside_ci95")}
                      for r in json.loads((C.OUT_DIR / "test2_external_validity.json").read_text())["results"]],
            "test3": [{k: v[k] for k in ("behaviour", "campaign", "spearman_rho", "T3a_rank_agreement", "best_reproduced",
                                         "best_published", "T3b_best_agrees_or_tied", "n_published_inside_ci95",
                                         "median_relative_difference")} for v in t3],
            "test4": [{k: x.get(k) for k in ("campaign", "best_reproduced_10000", "best_published_10000",
                                              "T4a_best_at_10000_agrees_or_tied", "T4b_directions_agree",
                                              "T4c_switch_reproduced")} for x in t4],
            "duplicated_column_check": dup and {k: dup[k] for k in ("median_abs_rel_diff_table6_women",
                                                                  "median_abs_rel_diff_table10_men", "column_belongs_to")}}
    eq = C.OUT_DIR / "equivalence_obp050.json"
    if eq.exists():
        e = json.loads(eq.read_text())
        summ["equivalence_with_official_obp050"] = {
            "action_dist_identical": {c: v["identical"] for c, v in e["A_action_dist"].items()},
            "pipeline_max_abs_diff": {b["config"]: max(x["max_abs_diff"] for x in b["estimators"].values())
                                      for b in e["B_pipeline"]}}
        drift = []
        for b in e["B_pipeline"]:
            beh, rest = b["config"].split("->")
            ev, camp, nn = rest.split()
            f = C.OUT_DIR / "ope" / f"{beh}_to_{ev}_{camp}_n{nn[2:]}_raw_seeds.csv"
            if not f.exists():
                continue
            prod = pd.read_csv(f).set_index("seed")
            for k, sd in enumerate(b["seeds"]):
                if sd not in prod.index:
                    continue
                gt = prod.loc[sd, "ground_truth"]
                for est in C.ESTIMATORS:
                    o = b["estimators"][est]["official"][k]; m = prod.loc[sd, est]
                    drift.append({"config": b["config"], "seed": sd, "estimator": est,
                                  "official_sklearn_0_24_2": o, "this_run": m,
                                  "abs_diff_x1e3": abs(m - o) * 1e3,
                                  "abs_error_official_x1e3": abs(o - gt) * 1e3})
        if drift:
            d = pd.DataFrame(drift)
            C.write_json({"note": "Same seeds and inputs; official obp 0.5.0 under scikit-learn 0.24.2 vs this "
                                  "run under the current environment. IPW and SNIPW do not use the reward model, so "
                                  "any difference there would indicate an implementation error.",
                          "environment_this_run": C.environment_info(),
                          "max_abs_diff_x1e3_by_estimator": d.groupby("estimator").abs_diff_x1e3.max().to_dict(),
                          "median_abs_error_official_x1e3": float(d.abs_error_official_x1e3.median()),
                          "rows": drift}, C.OUT_DIR / "version_drift.json")
            summ["version_drift_max_abs_diff_x1e3"] = d.groupby("estimator").abs_diff_x1e3.max().to_dict()
    C.write_json(summ, C.OUT_DIR / "validation_summary.json")


if __name__ == "__main__":
    main()
