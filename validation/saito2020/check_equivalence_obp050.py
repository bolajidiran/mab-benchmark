"""
check_equivalence_obp050.py: does ope_native.py return what the official
Saito et al. code returns?

Runs the official obp 0.5.0 and pyieoe 0.1.1 classes, exactly as
benchmark/ope/benchmark_ope_estimators.py calls them, side by side with
ope_native.py on identical inputs, and compares:
  A. the evaluation-policy action distribution of BernoulliTS with the
     ZOZOTOWN priors (all three campaigns);
  B. the per-seed policy values of the seven estimators (bootstrap, cross-
     fitted reward model, tuning) for a few seeds of selected configurations;
  C. (--loader) the data obp's OpenBanditDataset loader produces from a raw
     CSV against the arrays in cache/ (row order, action, position, reward,
     propensity, one-hot context).

Environment: the official code needs Python 3.9 and the locked versions of
the benchmark (scikit-learn 0.24.2, numpy 1.21.2, pandas 1.3.3, scipy
1.7.1). obp 0.5.0 imports PyTorch at module level but never uses it in OPE;
a stand-in module (folder torch_stub/) is enough. Example:
  uv venv -p 3.9 .venv39
  uv pip install --python .venv39 numpy==1.21.2 pandas==1.3.3 scipy==1.7.1 scikit-learn==0.24.2 tqdm matplotlib==3.4.3 seaborn==0.11.2 pyyaml
  set PYTHONPATH=validation/saito2020/torch_stub;<obp-0.5.0 source>;<pyieoe-0.1.1 source>
  .venv39/bin/python validation/saito2020/check_equivalence_obp050.py
obp 0.5.0 source: https://pypi.org/project/obp/0.5.0/ (sdist); pyieoe 0.1.1:
https://pypi.org/project/pyieoe/0.1.1/ (sdist).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ope_native as O          # noqa: E402
import saito_common as C        # noqa: E402

import obp                                                   # noqa: E402
from obp.ope import (DirectMethod, DoublyRobust, DoublyRobustWithShrinkageTuning,  # noqa: E402
                     InverseProbabilityWeighting, SelfNormalizedDoublyRobust,
                     SelfNormalizedInverseProbabilityWeighting, SwitchDoublyRobustTuning)
from obp.policy import BernoulliTS, Random                   # noqa: E402
from pyieoe.evaluator import InterpretableOPEEvaluator       # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402


def check_action_dist():
    out = {}
    for c in C.CAMPAIGNS:
        K = C.N_ITEMS[c]
        t0 = time.time()
        off = BernoulliTS(n_actions=K, len_list=3, random_state=12345, is_zozotown_prior=True,
                          campaign=c).compute_batch_action_dist(n_rounds=1)[0]
        nat = O.bts_action_dist(C.PRIORS[c]["alpha"], C.PRIORS[c]["beta"], 3, 100_000, 12345)
        out[c] = {"max_abs_diff": float(np.abs(off - nat).max()), "identical": bool(np.array_equal(off, nat)),
                  "seconds_official": round(time.time() - t0, 1)}
        print("A", c, out[c], flush=True)
    return out


def bandit_feedback(beh, camp, n):
    d = C.load_cache(beh, camp)
    rows = np.arange(n)
    return {"n_rounds": n, "n_actions": C.N_ITEMS[camp],
            "action": d["item"][rows].astype(int), "position": d["pos"][rows].astype(int),
            "reward": d["click"][rows].astype(int), "pscore": d["pscore"][rows],
            "context": C.one_hot_context(d, rows), "action_context": np.eye(C.N_ITEMS[camp], dtype=int)}, d


def check_pipeline(beh, camp, n, seeds):
    bf, d = bandit_feedback(beh, camp, n)
    ev = "random" if beh == "bts" else "bts"
    e = C.load_cache(ev, camp)
    gt = float(e["click"].mean())
    K = C.N_ITEMS[camp]
    if ev == "bts":
        pol = BernoulliTS(n_actions=K, len_list=3, random_state=12345, is_zozotown_prior=True, campaign=camp)
    else:
        pol = Random(n_actions=K, len_list=3, random_state=12345)
    action_dist = pol.compute_batch_action_dist(n_rounds=n)   # official uses 10^6 rows; only the first n are indexed
    lambdas = O.LAMBDAS
    ests = [InverseProbabilityWeighting(estimator_name="IPW"),
            SelfNormalizedInverseProbabilityWeighting(estimator_name="SNIPW"),
            DirectMethod(estimator_name="DM"), DoublyRobust(estimator_name="DR"),
            SelfNormalizedDoublyRobust(estimator_name="SNDR"),
            SwitchDoublyRobustTuning(lambdas=lambdas, estimator_name="Switch-DR"),
            DoublyRobustWithShrinkageTuning(lambdas=lambdas, estimator_name="DRos")]
    np.random.seed(12345)
    evaluator = InterpretableOPEEvaluator(
        random_states=np.array(seeds), bandit_feedbacks=[bf], evaluation_policies=[(gt, action_dist)],
        ope_estimators=ests, regression_models=[HistGradientBoostingClassifier(**O.HGB_PARAMS, random_state=12345)])
    t0 = time.time()
    off = evaluator.estimate_policy_value(sample_size=n, n_folds_=2)
    t_off = time.time() - t0
    cached = {"Switch-DR": getattr(ests[5], "best_hyperparam", None), "DRos": getattr(ests[6], "best_hyperparam", None)}
    data = {"ctx_codes": d["user_codes"][:n], "Xctx": bf["context"], "action": bf["action"],
            "reward": bf["reward"], "position": bf["position"], "pscore": bf["pscore"]}
    pi_e = action_dist[0]
    t0 = time.time()
    nat = {k: [] for k in C.ESTIMATORS}
    lam = None
    for s in seeds:
        r, iw, dm, qf = O.run_seed(int(s), data, pi_e, n)
        if lam is None:
            lam = (O.tune(r, iw, dm, qf, "Switch-DR")[0], O.tune(r, iw, dm, qf, "DRos")[0])
        for k, v in O.estimate_all(r, iw, dm, qf, *lam).items():
            nat[k].append(v)
    t_nat = time.time() - t0
    res = {"config": f"{beh}->{ev} {camp} n={n}", "seeds": list(map(int, seeds)),
           "lambda_official_cached_after_run": cached, "lambda_native_first_seed": list(lam),
           "seconds_official": round(t_off, 1), "seconds_native": round(t_nat, 1), "estimators": {}}
    for k in C.ESTIMATORS:
        a, b = np.asarray(off[k]), np.asarray(nat[k])
        res["estimators"][k] = {"official": a.tolist(), "native": b.tolist(),
                                "max_abs_diff": float(np.abs(a - b).max()),
                                "max_rel_diff": float((np.abs(a - b) / np.abs(a)).max())}
    print("B", res["config"], {k: v["max_abs_diff"] for k, v in res["estimators"].items()},
          cached, lam, f"official {t_off:.0f}s native {t_nat:.0f}s", flush=True)
    return res


def check_loader(beh, camp, csv_dir):
    from obp.dataset import OpenBanditDataset
    ds = OpenBanditDataset(behavior_policy=beh, campaign=camp, data_path=Path(csv_dir))
    bf = ds.obtain_batch_bandit_feedback()
    d = C.load_cache(beh, camp)
    X = C.one_hot_context(d, slice(None))
    # obp sorts by timestamp with an unstable sort: compare as multisets within equal timestamps
    same_order = {k: bool(np.array_equal(bf[k], v)) for k, v in
                  (("action", d["item"]), ("position", d["pos"]), ("reward", d["click"]), ("pscore", d["pscore"]))}
    same_order["context"] = bool(np.array_equal(bf["context"], X))
    n_rows_differ = int((np.any(bf["context"] != X, axis=1) | (bf["action"] != d["item"])).sum())
    # obp's row order as a permutation of the file rows (index_col=0 holds the file row number)
    perm = ds.data.index.to_numpy()
    is_perm = bool(np.array_equal(np.sort(perm), np.arange(d["ts"].size)))
    ties_only = bool(np.array_equal(d["ts"][perm], d["ts"]))
    moved = perm != np.arange(perm.size)
    reordered = {k: bool(np.array_equal(bf[k], v[perm])) for k, v in
                 (("action", d["item"]), ("position", d["pos"]), ("reward", d["click"]), ("pscore", d["pscore"]))}
    reordered["context"] = bool(np.array_equal(bf["context"], X[perm]))
    res = {"log": f"{beh}/{camp}",
           "obp_order_is_a_permutation_of_file_order": is_perm,
           "rows_moved_by_obp_sort": int(moved.sum()),
           "every_moved_row_keeps_its_timestamp": ties_only,
           "identical_after_applying_obp_order": reordered,
           "first_10000_rows_same_set": bool(np.array_equal(np.sort(perm[:10000]), np.arange(10000))),
           "first_300000_rows_same_set": bool(np.array_equal(np.sort(perm[:300000]), np.arange(300000)))
           if perm.size >= 300000 else None, "n_rounds_obp": int(bf["n_rounds"]), "n_rows_cache": int(d["ts"].size),
           "context_dim_obp": int(bf["context"].shape[1]), "context_dim_cache": int(X.shape[1]),
           "identical_arrays_in_same_order": same_order, "rows_differing": n_rows_differ,
           "on_policy_value_obp": float(bf["reward"].mean()), "on_policy_value_cache": float(d["click"].mean())}
    print("C", res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", nargs=3, metavar=("POLICY", "CAMPAIGN", "CSV_ROOT"),
                    help="CSV_ROOT/<policy>/<campaign>/<campaign>.csv must be a plain CSV")
    ap.add_argument("--quick", action="store_true", help="n = 10,000 configurations only")
    ap.add_argument("--resume", action="store_true", help="keep results already in outputs/equivalence_obp050.json")
    a = ap.parse_args()
    out = {"environment": {**C.environment_info(), "obp": obp.__version__}}
    if a.loader:
        out["C_loader"] = check_loader(*a.loader)
        C.write_json(out, C.OUT_DIR / f"equivalence_loader_{a.loader[0]}_{a.loader[1]}.json")
        return
    prev = {}
    path = C.OUT_DIR / "equivalence_obp050.json"
    if a.resume and path.exists():
        import json
        prev = json.loads(path.read_text())
    out["A_action_dist"] = prev.get("A_action_dist") or check_action_dist()
    cfgs = [("bts", "all", 10_000, [0, 1, 2]), ("random", "all", 10_000, [0, 1, 2]),
            ("bts", "men", 10_000, [0, 1, 2]), ("random", "women", 10_000, [0, 1, 2])]
    if not a.quick:
        cfgs += [("bts", "all", 300_000, [0]), ("random", "all", 300_000, [0])]
    out["B_pipeline"] = list(prev.get("B_pipeline", []))
    done = {b["config"] for b in out["B_pipeline"]}
    for beh, camp, n, seeds in cfgs:
        if f"{beh}->{'random' if beh == 'bts' else 'bts'} {camp} n={n}" in done:
            continue
        try:
            C.load_cache(beh, camp); C.load_cache("random" if beh == "bts" else "bts", camp)
        except SystemExit as e:
            print("skip", beh, camp, e); continue
        out["B_pipeline"].append(check_pipeline(beh, camp, n, seeds))
        C.write_json(out, C.OUT_DIR / "equivalence_obp050.json")
    C.write_json(out, C.OUT_DIR / "equivalence_obp050.json")


if __name__ == "__main__":
    main()
