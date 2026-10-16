"""
s0_prepare_logs.py: read the six raw OBD logs once and save compact arrays.

Logs: {random, bts} x {all, men, women}. For each log this script
  - reads the CSV in chunks (plain or gzip), recording its MD5;
  - keeps the columns obp uses: timestamp, item_id, position, click,
    propensity_score, user_feature_0..3;
  - codes the user features as ranks in the sorted vocabulary of the full
    log, which is what pd.get_dummies(..., drop_first=True) in obp uses;
  - converts position to 0, 1, 2 (obp: rankdata(position, "dense") - 1);
  - flags exact duplicate rows (all fields except the CSV index; the same
    rule as Fix 5) without removing them, so each test can use the raw log
    (as Saito et al. did) or the cleaned log;
  - checks time order. obp sorts by the timestamp string with an unstable
    sort; when the file is already in time order (checked here), file order
    differs from obp order only among rows with an identical timestamp.

Output: cache/log_<policy>_<campaign>.npz and outputs/s0_logs_report.json
Run from the repository root:  python validation/saito2020/s0_prepare_logs.py [--only random/men ...]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import saito_common as C

USE = ["timestamp", "item_id", "position", "click", "propensity_score"] + C.USER_COLS


def prepare(policy: str, campaign: str) -> dict:
    path = C.raw_log_path(policy, campaign)
    t0 = time.time()
    print(f"[{policy}/{campaign}] reading {path}", flush=True)
    ts, item, pos, click, ps, hashes = [], [], [], [], [], []
    vocab = {c: {} for c in C.USER_COLS}
    prov = {c: [] for c in C.USER_COLS}
    n_missing = 0
    for ch in pd.read_csv(path, chunksize=C.CHUNK_ROWS, dtype={c: str for c in C.USER_COLS}):
        body = ch.drop(columns=[ch.columns[0]])
        n_missing += int(ch[USE].isna().sum().sum())
        hashes.append(pd.util.hash_pandas_object(body, index=False).to_numpy())
        ts.append(pd.to_datetime(ch["timestamp"], format="ISO8601", utc=True)
                  .dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype(np.int64))
        item.append(ch["item_id"].to_numpy(np.int16))
        pos.append(ch["position"].to_numpy(np.int16))
        click.append(ch["click"].to_numpy(np.int8))
        ps.append(ch["propensity_score"].to_numpy(np.float64))
        for c in C.USER_COLS:
            v = vocab[c]
            col = ch[c].to_numpy()
            for h in pd.unique(col):
                v.setdefault(h, len(v))
            prov[c].append(np.fromiter((v[h] for h in col), np.int32, len(col)))
    ts = np.concatenate(ts); item = np.concatenate(item); pos = np.concatenate(pos)
    click = np.concatenate(click); ps = np.concatenate(ps); hashes = np.concatenate(hashes)
    n = ts.size
    # user features: provisional codes -> rank in sorted vocabulary
    codes = np.zeros((n, 4), np.int32); sizes = []
    for j, c in enumerate(C.USER_COLS):
        keys = list(vocab[c].keys())
        rank = np.empty(len(keys), np.int32)
        rank[np.argsort(np.array(keys, dtype=object))] = np.arange(len(keys))
        codes[:, j] = rank[np.concatenate(prov[c])]
        sizes.append(len(keys))
    # positions: dense rank - 1, as obp
    upos = np.unique(pos)
    pos0 = np.searchsorted(upos, pos).astype(np.int8)
    # duplicates: later occurrences of an identical row
    order = np.argsort(hashes, kind="stable")
    hs = hashes[order]
    later = np.zeros(n, bool)
    later[order[1:][hs[1:] == hs[:-1]]] = True
    dts = np.diff(ts)
    K = C.N_ITEMS[campaign]
    rep = {"policy": policy, "campaign": campaign, "file": str(path.name), "file_md5": C.md5_of(path),
           "rows_raw": int(n), "clicks_raw": int(click.sum()), "ctr_raw": float(click.mean()),
           "duplicates_flagged": int(later.sum()), "duplicate_clicks": int(click[later].sum()),
           "rows_clean": int(n - later.sum()), "clicks_clean": int(click[~later].sum()),
           "ctr_clean": float(click[~later].mean()),
           "n_items_observed": int(np.unique(item).size), "n_items_expected": K,
           "item_id_range": [int(item.min()), int(item.max())],
           "positions_raw": upos.tolist(), "user_feature_vocab_sizes": sizes,
           "context_dim_one_hot_drop_first": int(sum(s - 1 for s in sizes)),
           "time_ordered_in_file": bool((dts >= 0).all()),
           "rows_sharing_a_timestamp_with_previous_row": int((dts == 0).sum()),
           "propensity_min": float(ps.min()), "propensity_max": float(ps.max()),
           "propensity_equal_1_over_K": bool(np.allclose(ps, 1.0 / K)) if policy == "random" else None,
           "missing_values_in_used_columns": n_missing, "seconds": None}
    if not rep["time_ordered_in_file"]:
        o = np.argsort(ts, kind="stable")
        ts, item, pos0, click, ps, codes, later = (a[o] for a in (ts, item, pos0, click, ps, codes, later))
        rep["sorted_here"] = True
    np.savez(C.cache_path(policy, campaign), ts=ts, item=item, pos=pos0, click=click, pscore=ps,
             user_codes=codes, vocab_sizes=np.array(sizes), dup=later)
    rep["seconds"] = round(time.time() - t0, 1)
    print({k: rep[k] for k in ("rows_raw", "ctr_raw", "duplicates_flagged", "ctr_clean",
                               "time_ordered_in_file", "seconds")}, flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="e.g. random/men bts/all")
    a = ap.parse_args()
    jobs = [tuple(x.split("/")) for x in a.only] if a.only else \
        [(p, c) for c in C.CAMPAIGNS for p in C.POLICIES]
    out = C.OUT_DIR / "s0_logs_report.json"
    reports = {}
    if out.exists():
        reports = {f"{r['policy']}/{r['campaign']}": r for r in __import__("json").loads(out.read_text())["logs"]}
    for p, c in jobs:
        reports[f"{p}/{c}"] = prepare(p, c)
    keys = [f"{p}/{c}" for c in C.CAMPAIGNS for p in C.POLICIES if f"{p}/{c}" in reports]
    C.write_json({"environment": C.environment_info(), "logs": [reports[k] for k in keys]}, out)


if __name__ == "__main__":
    main()
