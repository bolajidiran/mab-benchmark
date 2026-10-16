"""
STEP 3 = FIX 4: encode user, item and affinity features as context.

OBD stores user and item categories as hashed strings. This script turns
them into numeric context:

  User context x_t (per-arm features in hybrid LinUCB, the only features
  in disjoint LinUCB):
      [ bias | one-hot(user_feature_0..3) | one-hot(position) ]
  The hashes are salted differently in the two logs for one category:
  in each log about 35.3% of rows carry a log-specific hash on ALL FOUR
  user features at once (a placeholder, most likely "no user profile").
  Matching rule (checked, not assumed): for each feature, a random-only
  hash and a bts-only hash are treated as the same category when
    (a) each covers more than 1% of its log's rows,
    (b) their row shares differ by less than 1 percentage point, and
    (c) in each log the rows carrying the log-specific hash are exactly
        the same rows on all four features.
  All other hashes are matched only when identical; any remaining
  log-specific category gets its own one-hot column.

  Item context (shared features z_{t,a} in hybrid LinUCB):
      [ item_feature_0 (z-score) | one-hot(item_feature_1..3) | log1p(affinity_{t,a}) ]
  item_context.csv is taken from the random log. The bts file hashes the
  same categories differently, but the partition of items is identical
  (checked here), so the two files describe the same features.

Outputs
  outputs/fix4_context_report.json / .md
  cache/context.npz    (user one-hot indices per row, item feature matrix)

Run from the repository root:
  python obd_preprocessing/step3_fix4_context.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C


def main():
    import argparse
    ap = argparse.ArgumentParser()
    C.add_variant_arg(ap)
    C.set_variant(ap.parse_args().variant or "main")
    s = dict(np.load(C.CACHE_DIR / "stream_pooled.npz"))
    vocab = {pol: json.loads((C.CACHE_DIR / f"user_vocab_{pol}.json").read_text())
             for pol in C.POLICIES}
    report = {"step": "Fix 4 context", "environment": C.environment_info(),
              "user_features": {}, "item_features": {}}

    # ---- placeholder matching (rule (a) to (c) in the docstring) ----------
    n = s["ts"].size
    placeholder = {}
    only_rows = {}
    for src, pol in enumerate(C.POLICIES):
        m = s["source"] == src
        other = C.POLICIES[1 - src]
        masks, cands = [], {}
        for j, f in enumerate(C.USER_COLS):
            sh = set(vocab["random"][f]) & set(vocab["bts"][f])
            codes, cnt = np.unique(s["user"][m, j], return_counts=True)
            big = [(vocab[pol][f][c], cnt_ / m.sum()) for c, cnt_ in zip(codes, cnt)
                   if vocab[pol][f][c] not in sh and cnt_ / m.sum() > 0.01]
            cands[f] = big
            bigset = {h for h, _ in big}
            bad = np.array([h in bigset for h in vocab[pol][f]])
            masks.append(bad[s["user"][m, j]])
        M = np.stack(masks, 1)
        only_rows[pol] = {"any": float(M.any(1).mean()), "all": float(M.all(1).mean())}
        placeholder[pol] = cands
    match = {}
    same_rows = all(abs(only_rows[p]["any"] - only_rows[p]["all"]) < 1e-12 for p in C.POLICIES)
    for f in C.USER_COLS:
        a, b = placeholder["random"][f], placeholder["bts"][f]
        if same_rows and len(a) == 1 and len(b) == 1 and abs(a[0][1] - b[0][1]) < 0.01:
            match[f] = {"random_hash": a[0][0], "bts_hash": b[0][0],
                        "share_random": a[0][1], "share_bts": b[0][1]}
    report["placeholder_matching"] = {"rule_c_same_rows_all_features": same_rows,
                                      "log_specific_rows": only_rows, "matched": match}

    # ---- user features: union vocabulary over both logs -----------------
    active = np.zeros((n, len(C.USER_COLS) + 1), np.int32)   # + position
    offset = 1                                                # column 0 = bias
    names = ["bias"]
    for j, f in enumerate(C.USER_COLS):
        bts_v = [match[f]["random_hash"] if (f in match and h == match[f]["bts_hash"]) else h
                 for h in vocab["bts"][f]]
        union = list(dict.fromkeys(vocab["random"][f] + bts_v))
        idx = {h: i for i, h in enumerate(union)}
        for h_b, h_m in zip(vocab["bts"][f], bts_v):
            idx[h_b] = idx[h_m]
        code = np.empty(n, np.int32)
        for src, pol in enumerate(C.POLICIES):
            lut = np.array([idx[h] for h in vocab[pol][f]], np.int32)
            m = s["source"] == src
            code[m] = lut[s["user"][m, j]]
        active[:, j] = offset + code
        shared = set(vocab["random"][f]) & set(vocab["bts"][f])
        if f in match:
            shared |= {match[f]["random_hash"], match[f]["bts_hash"]}
        unshared_rows = np.isin(code, [idx[h] for h in union if h not in shared])
        report["user_features"][f] = {
            "categories_random": len(vocab["random"][f]),
            "categories_bts": len(vocab["bts"][f]),
            "shared_hashes": len(set(vocab["random"][f]) & set(vocab["bts"][f])),
            "placeholder_matched": f in match, "union_columns": len(union),
            "rows_in_unshared_categories": int(unshared_rows.sum()),
            "share_rows_unshared": float(unshared_rows.mean())}
        names += [f"{f}={i}" for i in range(len(union))]
        offset += len(union)
    active[:, -1] = offset + (s["pos"].astype(np.int32) - 1)
    names += [f"position={q}" for q in C.POSITIONS]
    d = offset + len(C.POSITIONS)

    # ---- item features ----------------------------------------------------
    ic_r = pd.read_csv(C.item_context_path("random")).sort_values("item_id")
    ic_b = pd.read_csv(C.item_context_path("bts")).sort_values("item_id")
    assert (ic_r["item_id"].to_numpy() == np.arange(C.K)).all()
    same_partition = {}
    for c in C.ITEM_CAT_COLS:
        ct = pd.crosstab(ic_r[c], ic_b[c])
        same_partition[c] = bool(((ct > 0).sum(axis=1) == 1).all() and ((ct > 0).sum(axis=0) == 1).all())
    same_numeric = bool(np.allclose(ic_r["item_feature_0"], ic_b["item_feature_0"]))
    if not (all(same_partition.values()) and same_numeric):
        raise SystemExit("item_context.csv differs between logs beyond hashing")
    f0 = ic_r["item_feature_0"].to_numpy(float)
    blocks = [((f0 - f0.mean()) / f0.std())[:, None]]
    item_names = ["item_feature_0 (z)"]
    for c in C.ITEM_CAT_COLS:
        cats = list(dict.fromkeys(ic_r[c]))
        blocks.append(pd.get_dummies(pd.Categorical(ic_r[c], categories=cats)).to_numpy(float))
        item_names += [f"{c}={i}" for i in range(len(cats))]
    S = np.hstack(blocks)                         # (K, k-1)
    k = S.shape[1] + 1
    item_names.append("log1p(user-item affinity)")
    report["item_features"] = {
        "partition_identical_across_logs": same_partition,
        "item_feature_0_identical": same_numeric,
        "item_categories": {c: int(ic_r[c].nunique()) for c in C.ITEM_CAT_COLS}}

    np.savez_compressed(C.CACHE_DIR / "context.npz", active=active, d=d,
                        item_static=S, k=k)
    report["dimensions"] = {
        "d_user_context": int(d), "k_shared_item_context": int(k),
        "user_context_columns": names, "item_context_columns": item_names,
        "affinity_transform": "log1p",
        "affinity_nonzero_rows": int(np.unique(s["aff_row"]).size)}
    C.write_json(report, C.OUT_DIR / "fix4_context_report.json")

    L = ["# Fix 4: user + item + affinity context", "",
         f"User context x (dimension d = {d}): bias, one-hot user features, one-hot position.",
         f"Shared item context z (dimension k = {k}): standardised item_feature_0, "
         "one-hot item_feature_1 to 3, log1p(user-item affinity).", "",
         "| User feature | Categories random | Categories bts | Identical hashes | Placeholder matched | One-hot columns | Rows still unmatched |",
         "|---|---|---|---|---|---|---|"]
    for f, r in report["user_features"].items():
        L.append(f"| {f} | {r['categories_random']} | {r['categories_bts']} | "
                 f"{r['shared_hashes']} | {'yes' if r['placeholder_matched'] else 'no'} | {r['union_columns']} | "
                 f"{r['rows_in_unshared_categories']:,} ({r['share_rows_unshared']:.3%}) |")
    pm = report["placeholder_matching"]
    L += ["", "Placeholder category: rows carrying a log-specific hash on all four user "
          f"features: random {pm['log_specific_rows']['random']['all']:.3%}, "
          f"bts {pm['log_specific_rows']['bts']['all']:.3%} "
          f"(same rows on all four features: {pm['rule_c_same_rows_all_features']}). "
          + ("These two hashes are matched as one category under rules (a) to (c)."
             if pm["matched"] else "Rules (a) to (c) not met: no placeholder matching applied.")]
    L += ["", "Item features: category partition identical across the two "
          f"item_context files: {all(same_partition.values())}; "
          f"item_feature_0 identical: {same_numeric}.", ""]
    (C.OUT_DIR / "fix4_context_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
