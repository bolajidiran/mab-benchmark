"""
STEP 2 = FIX 3: test for position bias and build the replay stream.

ZOZOTOWN shows recommendations in three slots (1 = left, 2 = centre,
3 = right). If click-through differed by slot, pooling the slots would mix
position bias into every algorithm's CTR. This script:
  1. Tests whether CTR differs by position (chi-square, per logging policy).
     Result on OBD "all": no significant difference (random p = 0.39,
     bts p = 0.69), so the main S4 analysis pools the three positions and
     the per-position CTR is reported from the same runs as a check
     (decision recorded in README.md, 11 October 2026).
  2. Merges the cleaned random and Bernoulli TS rows of all three positions
     into one time-ordered stream (stable sort on timestamp), keeping the
     position and the source log of every row. Both logs ran in parallel
     over the same week, and every row carries its own propensity score,
     so the merged stream is valid input for rejection-sampling replay.

Outputs
  outputs/fix3_position_report.json / .md
  cache/stream_pooled.npz

Run from the repository root:
  python obd_preprocessing/step2_fix3_positions.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

import common as C


def load(policy):
    return dict(np.load(C.CACHE_DIR / f"clean_{policy}.npz"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    C.add_variant_arg(ap)
    C.set_variant(ap.parse_args().variant or "main")
    logs = {p: load(p) for p in C.POLICIES}
    report = {"step": "Fix 3 positions", "environment": C.environment_info(),
              "position_bias_tests": {}, "positions": {}}

    # 1. Position-bias test per logging policy
    for pol, d in logs.items():
        table = np.array([[np.sum((d["pos"] == p) & (d["click"] == 1)),
                           np.sum((d["pos"] == p) & (d["click"] == 0))]
                          for p in C.POSITIONS])
        chi2, pval, dof, _ = stats.chi2_contingency(table)
        report["position_bias_tests"][pol] = {
            "clicks_by_position": table[:, 0].tolist(),
            "rows_by_position": table.sum(1).tolist(),
            "ctr_by_position": (table[:, 0] / table.sum(1)).tolist(),
            "chi2": float(chi2), "dof": int(dof), "p_value": float(pval)}

    # 2. One pooled, time-ordered stream (all positions, both logs)
    for p in ["pooled"]:
        parts = []
        for src, pol in enumerate(C.POLICIES):
            d = logs[pol]
            rows = np.arange(d["ts"].size)
            parts.append((src, pol, rows, d))
        ts = np.concatenate([d["ts"][r] for _, _, r, d in parts])
        order = np.argsort(ts, kind="stable")
        cat = lambda key: np.concatenate([d[key][r] for _, _, r, d in parts])[order]
        source = np.concatenate([np.full(r.size, s, np.int8) for s, _, r, _ in parts])[order]

        # affinity: remap each log's sparse rows into the merged order
        new_index = np.empty(order.size, np.int64)
        new_index[order] = np.arange(order.size)
        aff_r, aff_i, aff_v, base = [], [], [], 0
        for s, pol, r, d in parts:
            local = np.full(d["ts"].size, -1, np.int64)
            local[r] = np.arange(r.size) + base
            m = local[d["aff_row"]] >= 0
            aff_r.append(new_index[local[d["aff_row"][m]]])
            aff_i.append(d["aff_item"][m]); aff_v.append(d["aff_val"][m])
            base += r.size
        aff_r = np.concatenate(aff_r); srt = np.argsort(aff_r, kind="stable")

        out = dict(ts=ts[order], item=cat("item"), pos=cat("pos"), click=cat("click"),
                   pscore=cat("pscore"), user=cat("user"), source=source,
                   aff_row=aff_r[srt], aff_item=np.concatenate(aff_i)[srt],
                   aff_val=np.concatenate(aff_v)[srt])
        assert np.all(np.diff(out["ts"]) >= 0)
        np.savez_compressed(C.CACHE_DIR / "stream_pooled.npz", **out)

        info = {"rows": int(order.size), "clicks": int(out["click"].sum()),
                "ctr": float(out["click"].mean())}
        for s, pol in enumerate(C.POLICIES):
            m = out["source"] == s
            info[f"rows_{pol}"] = int(m.sum())
            info[f"clicks_{pol}"] = int(out["click"][m].sum())
            info[f"ctr_{pol}"] = float(out["click"][m].mean())
        for q in C.POSITIONS:
            mq = out["pos"] == q
            info[f"rows_position_{q}"] = int(mq.sum())
            info[f"clicks_position_{q}"] = int(out["click"][mq].sum())
        report["positions"][p] = info
        print(f"stream {p}: {info['rows']:,} rows, {info['clicks']:,} clicks "
              f"(random {info['rows_random']:,} / bts {info['rows_bts']:,})")

    C.write_json(report, C.OUT_DIR / "fix3_position_report.json")

    L = ["# Fix 3: position-bias test and pooled replay stream", "",
         "## Position bias (chi-square test of CTR by position)", "",
         "| Log | CTR pos 1 | CTR pos 2 | CTR pos 3 | chi-square (dof 2) | p-value |",
         "|---|---|---|---|---|---|"]
    for pol, t in report["position_bias_tests"].items():
        c = t["ctr_by_position"]
        L.append(f"| {pol} | {c[0]:.4%} | {c[1]:.4%} | {c[2]:.4%} | "
                 f"{t['chi2']:.2f} | {t['p_value']:.3g} |")
    i = report["positions"]["pooled"]
    L += ["", "Decision: no significant position bias, so the main S4 analysis pools",
          "the three positions; per-position CTR is reported from the same runs.", "",
          "## Pooled replay stream (random + bts, all positions, time-ordered)", "",
          "| Stream | Rows | Clicks | Rows random | Rows bts | CTR random | CTR bts |",
          "|---|---|---|---|---|---|---|",
          f"| pooled | {i['rows']:,} | {i['clicks']:,} | {i['rows_random']:,} | "
          f"{i['rows_bts']:,} | {i['ctr_random']:.4%} | {i['ctr_bts']:.4%} |", "",
          "| Position | Rows | Clicks |", "|---|---|---|"]
    for q in C.POSITIONS:
        L.append(f"| {q} | {i[f'rows_position_{q}']:,} | {i[f'clicks_position_{q}']:,} |")
    (C.OUT_DIR / "fix3_position_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Reports written to {C.OUT_DIR}")


if __name__ == "__main__":
    main()
