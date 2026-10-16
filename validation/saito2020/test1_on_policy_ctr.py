"""
Test 1: reproduce Table 1 of Saito et al. (on-policy CTR of every log).

For the six logs ({Random, Bernoulli TS} x {ALL, Men's, Women's}) this test
computes the number of rows, the number of items, the CTR with a 95%
confidence interval (normal approximation, as in the paper, and Wilson)
and the CTR relative to Random in the ALL campaign, on the raw log (as
published) and on the cleaned log (exact duplicates removed, Fix 5).

Pass criteria, set before running:
  T1a  row count and item count equal the published values;
  T1b  the published CTR (given to two decimals of a percent) lies within
       its rounding interval (+/- 0.005 percentage points) of the raw CTR,
       or inside the 95% interval of the raw CTR;
  T1c  the relative CTR rounds to the published value (two decimals);
  T1d  the ordering Bernoulli TS > Random holds in every campaign, with
       non-overlapping 95% intervals.

Output: outputs/test1_on_policy_ctr.json / .csv / .md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import saito_common as C


def main():
    rep = {r["policy"] + "/" + r["campaign"]: r
           for r in json.loads((C.OUT_DIR / "s0_logs_report.json").read_text())["logs"]}
    pub = C.PUBLISHED["table1"]["rows"]
    z = 1.959963984540054
    rows = []
    for variant in ("raw", "clean"):
        base = rep["random/all"][f"clicks_{variant}"] / rep["random/all"][f"rows_{variant}"]
        for camp in C.CAMPAIGNS:
            for pol in C.POLICIES:
                key = f"{pol}/{camp}"
                r, p = rep[key], pub[key]
                n, k = r[f"rows_{variant}"], r[f"clicks_{variant}"]
                ctr = k / n
                half = z * np.sqrt(ctr * (1 - ctr) / n)
                wl, wh = C.wilson(k, n)
                rows.append({
                    "variant": variant, "log": key, "policy": pol, "campaign": camp,
                    "rows": n, "published_rows": p["n_data"], "rows_match": n == p["n_data"],
                    "rows_difference": n - p["n_data"],
                    "items": r["n_items_observed"], "published_items": p["n_items"],
                    "items_match": r["n_items_observed"] == p["n_items"],
                    "clicks": k, "ctr_percent": 100 * ctr,
                    "ci95_pm_percent_normal": 100 * half,
                    "ci95_wilson_percent": [100 * wl, 100 * wh],
                    "published_ctr_percent": p["ctr_percent"], "published_ci_pm_percent": p["ci_pm_percent"],
                    "published_within_rounding": abs(100 * ctr - p["ctr_percent"]) <= 0.005 + 1e-12,
                    "published_inside_ci95": bool(100 * (ctr - half) <= p["ctr_percent"] <= 100 * (ctr + half)),
                    "relative_ctr": ctr / base, "published_relative_ctr": p["relative_ctr"],
                    "relative_ctr_rounds_to_published": round(ctr / base, 2) == p["relative_ctr"],
                    "ci_pm_matches_published_3dp": round(100 * half, 3) == p["ci_pm_percent"]})
    df = pd.DataFrame(rows)
    verdict = {}
    for variant in ("raw", "clean"):
        d = df[df.variant == variant]
        order = {}
        for camp in C.CAMPAIGNS:
            b = d[(d.campaign == camp) & (d.policy == "bts")].iloc[0]
            rn = d[(d.campaign == camp) & (d.policy == "random")].iloc[0]
            order[camp] = bool(b.ctr_percent - b.ci95_pm_percent_normal > rn.ctr_percent + rn.ci95_pm_percent_normal)
        verdict[variant] = {
            "T1a_rows_and_items_match": {"logs_matching_rows": int(d.rows_match.sum()),
                                         "logs_matching_items": int(d.items_match.sum()), "of": len(d),
                                         "mismatches": d.loc[~d.rows_match, ["log", "rows", "published_rows",
                                                                             "rows_difference"]].to_dict("records")},
            "T1b_ctr_matches": {"within_rounding": int(d.published_within_rounding.sum()),
                                "inside_ci95": int(d.published_inside_ci95.sum()), "of": len(d),
                                "pass": bool((d.published_within_rounding | d.published_inside_ci95).all())},
            "T1c_relative_ctr_matches": {"count": int(d.relative_ctr_rounds_to_published.sum()), "of": len(d),
                                         "pass": bool(d.relative_ctr_rounds_to_published.all())},
            "T1d_bts_above_random_every_campaign": {"by_campaign": order, "pass": all(order.values())}}
    out = {"test": "Test 1: on-policy CTR (Saito et al., Table 1)", "rows": rows, "verdict": verdict}
    C.write_json(out, C.OUT_DIR / "test1_on_policy_ctr.json")
    df.drop(columns=["ci95_wilson_percent"]).to_csv(C.OUT_DIR / "test1_on_policy_ctr.csv", index=False)

    L = ["# Test 1: on-policy CTR against Saito et al. (Table 1)", "",
         "| Log | Variant | Rows | Published rows | CTR % (95% CI) | Published CTR % | Relative CTR | Published |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['log']} | {r['variant']} | {r['rows']:,} | {r['published_rows']:,} | "
                 f"{r['ctr_percent']:.4f} (±{r['ci95_pm_percent_normal']:.4f}) | {r['published_ctr_percent']:.2f} "
                 f"(±{r['published_ci_pm_percent']:.3f}) | {r['relative_ctr']:.3f} | {r['published_relative_ctr']:.2f} |")
    for v, ver in verdict.items():
        L += ["", f"**{v}**: T1a rows {ver['T1a_rows_and_items_match']['logs_matching_rows']}/6, items "
                  f"{ver['T1a_rows_and_items_match']['logs_matching_items']}/6; T1b CTR "
                  f"{'pass' if ver['T1b_ctr_matches']['pass'] else 'fail'} "
                  f"(within rounding {ver['T1b_ctr_matches']['within_rounding']}/6); T1c relative CTR "
                  f"{ver['T1c_relative_ctr_matches']['count']}/6; T1d ordering "
                  f"{'pass' if ver['T1d_bts_above_random_every_campaign']['pass'] else 'fail'}."]
    (C.OUT_DIR / "test1_on_policy_ctr.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
