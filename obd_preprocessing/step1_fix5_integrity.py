"""
STEP 1 = FIX 5: integrity checks, duplicate removal and a compact cache.

For each logging policy (random, bts) in the "all" campaign this script:
  1. Reads the raw log in chunks (all.csv or all.csv.gz), recording its MD5.
  2. Checks: required columns; no missing values; click in {0, 1};
     item_id in [0, K-1]; position in {1, 2, 3}; propensity in (0, 1];
     random-log propensity equal to 1/K; timestamps in time order.
  3. Finds exact duplicate rows (all fields except the CSV index), keeps
     the first occurrence, drops the rest and writes them to a CSV so the
     removal is auditable.
  4. Sorts by timestamp if (and only if) the log is out of order.
  5. Saves a compact cache: timestamp, item, position, click, propensity,
     integer-coded user features (per-log vocabulary) and the user-item
     affinity matrix in sparse (row, item, value) form.

Outputs
  outputs/fix5_integrity_report.json / .md
  outputs/fix5_duplicates_removed_<policy>.csv
  cache/clean_<policy>.npz, cache/user_vocab_<policy>.json

Run from the repository root:
  python obd_preprocessing/step1_fix5_integrity.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import common as C

REQUIRED = (["timestamp", "item_id", "position", "click", "propensity_score"]
            + C.USER_COLS + C.AFF_COLS)


def process_log(policy: str, keep_duplicates: bool = False) -> dict:
    path = C.raw_log_path(policy)
    t0 = time.time()
    print(f"\n[{policy}] reading {path} ...")
    rep = {"policy": policy, "file": str(path.relative_to(C.REPO)),
           "file_md5": C.md5_of(path)}

    ts, item, pos, click, ps, hashes, orig_idx = [], [], [], [], [], [], []
    user_codes = {c: [] for c in C.USER_COLS}
    vocab = {c: {} for c in C.USER_COLS}
    aff_r, aff_i, aff_v = [], [], []
    n_null = 0
    offset = 0
    for ch in pd.read_csv(path, chunksize=C.CHUNK_ROWS):
        missing = [c for c in REQUIRED if c not in ch.columns]
        if missing:
            raise SystemExit(f"[{policy}] missing columns: {missing}")
        idx_col = ch.columns[0]
        n_null += int(ch[REQUIRED].isna().sum().sum())
        body = ch.drop(columns=[idx_col])
        hashes.append(pd.util.hash_pandas_object(body, index=False).to_numpy())
        orig_idx.append(ch[idx_col].to_numpy(np.int64))
        ts.append(pd.to_datetime(ch["timestamp"], format="ISO8601", utc=True)
                  .dt.tz_convert(None).to_numpy().astype("datetime64[ns]")
                  .astype(np.int64))       # nanoseconds since epoch, UTC
        item.append(ch["item_id"].to_numpy(np.int16))
        pos.append(ch["position"].to_numpy(np.int8))
        click.append(ch["click"].to_numpy(np.int8))
        ps.append(ch["propensity_score"].to_numpy(np.float64))
        for c in C.USER_COLS:
            v = vocab[c]
            col = ch[c].astype(str).to_numpy()
            for h in pd.unique(col):
                v.setdefault(h, len(v))
            user_codes[c].append(np.fromiter((v[h] for h in col), np.int16, len(col)))
        A = ch[C.AFF_COLS].to_numpy(np.float32)
        r, i = np.nonzero(A)
        aff_r.append(r.astype(np.int64) + offset)
        aff_i.append(i.astype(np.int16))
        aff_v.append(A[r, i])
        offset += len(ch)
        print(f"   {offset:>12,} rows", end="\r")
    print()

    ts = np.concatenate(ts); item = np.concatenate(item); pos = np.concatenate(pos)
    click = np.concatenate(click); ps = np.concatenate(ps)
    hashes = np.concatenate(hashes); orig_idx = np.concatenate(orig_idx)
    ucodes = np.stack([np.concatenate(user_codes[c]) for c in C.USER_COLS], 1)
    aff_r = np.concatenate(aff_r); aff_i = np.concatenate(aff_i); aff_v = np.concatenate(aff_v)
    n = len(ts)

    checks = {
        "rows_raw": n,
        "missing_values": n_null,
        "click_values": sorted(np.unique(click).tolist()),
        "item_id_min": int(item.min()), "item_id_max": int(item.max()),
        "n_items": int(np.unique(item).size),
        "position_values": sorted(np.unique(pos).tolist()),
        "propensity_min": float(ps.min()), "propensity_max": float(ps.max()),
        "timestamps_in_order": bool(np.all(np.diff(ts) >= 0)),
        "first_timestamp": str(pd.Timestamp(ts.min(), tz="UTC")),
        "last_timestamp": str(pd.Timestamp(ts.max(), tz="UTC")),
        "affinity_nonzero_share": float(len(aff_v) / (n * C.K)),
        "affinity_max": float(aff_v.max()) if len(aff_v) else 0.0,
        "user_feature_cardinality": {c: len(vocab[c]) for c in C.USER_COLS},
    }
    ok = {
        "no_missing_values": n_null == 0,
        "click_is_binary": set(checks["click_values"]) <= {0, 1},
        "item_id_in_range": checks["item_id_min"] >= 0 and checks["item_id_max"] <= C.K - 1,
        "position_in_1_2_3": set(checks["position_values"]) <= {1, 2, 3},
        "propensity_in_(0,1]": bool(ps.min() > 0 and ps.max() <= 1),
    }
    if policy == "random":
        ok["random_propensity_equals_1/K"] = bool(np.allclose(ps, 1.0 / C.K, atol=1e-9))
    if not all(ok.values()):
        bad = [k for k, v in ok.items() if not v]
        raise SystemExit(f"[{policy}] integrity check failed: {bad}")

    # Exact duplicates: keep the first occurrence of each row hash.
    _, first = np.unique(hashes, return_index=True)
    keep = np.zeros(n, bool); keep[first] = True
    dup_rows = np.flatnonzero(~keep)
    dups = pd.DataFrame({
        "original_row_index": orig_idx[dup_rows],
        "timestamp": pd.to_datetime(ts[dup_rows], utc=True).astype(str),
        "item_id": item[dup_rows], "position": pos[dup_rows],
        "click": click[dup_rows], "propensity_score": ps[dup_rows]})
    dups.to_csv(C.OUT_DIR / f"fix5_duplicates_removed_{policy}.csv", index=False)
    if keep_duplicates:                     # sensitivity run: list, do not drop
        keep[:] = True

    # Apply the duplicate mask and (if needed) a stable time sort.
    order = np.flatnonzero(keep)
    sorted_needed = not checks["timestamps_in_order"]
    if sorted_needed:
        order = order[np.argsort(ts[order], kind="stable")]
    newpos = np.full(n, -1, np.int64); newpos[order] = np.arange(order.size)
    m = newpos[aff_r] >= 0
    aff_r2 = newpos[aff_r[m]]
    srt = np.argsort(aff_r2, kind="stable")

    np.savez_compressed(
        C.CACHE_DIR / f"clean_{policy}.npz",
        ts=ts[order], item=item[order], pos=pos[order], click=click[order],
        pscore=ps[order], user=ucodes[order], orig_idx=orig_idx[order],
        aff_row=aff_r2[srt], aff_item=aff_i[m][srt], aff_val=aff_v[m][srt])
    (C.CACHE_DIR / f"user_vocab_{policy}.json").write_text(
        json.dumps({c: list(vocab[c].keys()) for c in C.USER_COLS}), encoding="utf-8")

    kept_click = click[order]
    rep.update({
        "checks": checks, "checks_passed": ok,
        "duplicates_found": int(dup_rows.size),
        "duplicates_removed": 0 if keep_duplicates else int(dup_rows.size),
        "duplicate_clicks_removed": int(click[dup_rows].sum()),
        "sorted_by_timestamp_in_this_step": sorted_needed,
        "rows_clean": int(order.size),
        "clicks_clean": int(kept_click.sum()),
        "ctr_clean": float(kept_click.mean()),
        "published_ctr": C.PUBLISHED_CTR[policy],
        "ctr_by_position": {int(p): float(kept_click[pos[order] == p].mean())
                            for p in C.POSITIONS},
        "seconds": round(time.time() - t0, 1),
    })
    print(f"[{policy}] rows {n:,} -> {order.size:,} ({'kept' if keep_duplicates else 'dropped'} {dup_rows.size} duplicates); "
          f"CTR {rep['ctr_clean']:.4%} (published {C.PUBLISHED_CTR[policy]:.2%})")
    return rep


def to_markdown(reports: list[dict]) -> str:
    L = ["# Fix 5: integrity checks on the Open Bandit Dataset (campaign: all)", ""]
    L.append("| Check | random | bts |")
    L.append("|---|---|---|")
    r, b = reports
    rows = [
        ("File MD5", r["file_md5"], b["file_md5"]),
        ("Rows (raw)", f"{r['checks']['rows_raw']:,}", f"{b['checks']['rows_raw']:,}"),
        ("Missing values", r["checks"]["missing_values"], b["checks"]["missing_values"]),
        ("Click values", r["checks"]["click_values"], b["checks"]["click_values"]),
        ("item_id range", f"{r['checks']['item_id_min']}-{r['checks']['item_id_max']}",
         f"{b['checks']['item_id_min']}-{b['checks']['item_id_max']}"),
        ("Positions", r["checks"]["position_values"], b["checks"]["position_values"]),
        ("Propensity min / max", f"{r['checks']['propensity_min']:.6g} / {r['checks']['propensity_max']:.6g}",
         f"{b['checks']['propensity_min']:.6g} / {b['checks']['propensity_max']:.6g}"),
        ("Timestamps in order", r["checks"]["timestamps_in_order"], b["checks"]["timestamps_in_order"]),
        ("Period (UTC)", f"{r['checks']['first_timestamp'][:19]} to {r['checks']['last_timestamp'][:19]}",
         f"{b['checks']['first_timestamp'][:19]} to {b['checks']['last_timestamp'][:19]}"),
        ("Exact duplicates found", r["duplicates_found"], b["duplicates_found"]),
        ("Clicks among duplicates", r["duplicate_clicks_removed"], b["duplicate_clicks_removed"]),
        ("Duplicates removed", r["duplicates_removed"], b["duplicates_removed"]),
        ("Rows (clean)", f"{r['rows_clean']:,}", f"{b['rows_clean']:,}"),
        ("Clicks (clean)", f"{r['clicks_clean']:,}", f"{b['clicks_clean']:,}"),
        ("CTR (clean)", f"{r['ctr_clean']:.4%}", f"{b['ctr_clean']:.4%}"),
        ("Published CTR", f"{r['published_ctr']:.2%}", f"{b['published_ctr']:.2%}"),
        ("User feature cardinalities", r["checks"]["user_feature_cardinality"],
         b["checks"]["user_feature_cardinality"]),
        ("Affinity non-zero share", f"{r['checks']['affinity_nonzero_share']:.4%}",
         f"{b['checks']['affinity_nonzero_share']:.4%}"),
    ]
    for name, x, y in rows:
        L.append(f"| {name} | {x} | {y} |")
    L += ["", "All checks passed: " +
          str(all(r["checks_passed"].values()) and all(b["checks_passed"].values())), ""]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="sensitivity run: report duplicates but keep them")
    C.add_variant_arg(ap)
    args = ap.parse_args()
    # --keep-duplicates writes to the keep_duplicates variant unless told otherwise
    C.set_variant(args.variant or ("keep_duplicates" if args.keep_duplicates else "main"))
    print(f"variant: {C.VARIANT}  (cache: {C.CACHE_DIR}, outputs: {C.OUT_DIR})")
    reports = [process_log(p, args.keep_duplicates) for p in C.POLICIES]
    C.write_json({"step": "Fix 5 integrity", "variant": C.VARIANT,
                  "keep_duplicates": args.keep_duplicates, "environment": C.environment_info(),
                  "reports": reports}, C.OUT_DIR / "fix5_integrity_report.json")
    (C.OUT_DIR / "fix5_integrity_report.md").write_text(to_markdown(reports), encoding="utf-8")
    print(f"\nReports written to {C.OUT_DIR}")


if __name__ == "__main__":
    main()
