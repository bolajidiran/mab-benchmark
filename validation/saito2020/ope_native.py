"""
ope_native.py: a dependency-light reimplementation of the Saito et al. OPE
benchmark (zr-obp benchmark/ope, obp 0.5.0 + pyieoe 0.1.1).

Why reimplement: obp 0.5.0 requires Python < 3.10 and PyTorch, and pyieoe
does not build on current Python. This module needs only numpy, pandas and
scikit-learn. check_equivalence_obp050.py runs the official obp 0.5.0 and
pyieoe 0.1.1 code on identical inputs and confirms that this module returns
the same numbers.

What is reproduced, line by line from the official code:
  evaluation policy  BernoulliTS(is_zozotown_prior=True, len_list=3,
                     random_state=12345).compute_batch_action_dist(), i.e.
                     100,000 Monte Carlo slates from the production Beta
                     priors; or the uniform Random policy.
  bootstrap          seed s: np.random.seed(s); idx = np.random.choice(
                     np.arange(n), size=n, replace=True) on the first n rows.
  reward model       one HistGradientBoostingClassifier(max_iter=100,
                     learning_rate=0.01, max_depth=5, min_samples_leaf=10,
                     random_state=s) per position, features = user one-hot
                     context + one-hot item; 2-fold cross-fitting with
                     KFold(2, shuffle=True, random_state=s).
  estimators         IPW, SNIPW, DM, DR, SNDR, Switch-DR, DRos (obp 0.5.0).
  tuning             Switch-DR and DRos choose lambda from
                     [1, 5, 10, 50, 100, 500, 1000, 5000, 10000] by the
                     estimated MSE (sample variance / n + squared high-
                     probability bias upper bound, delta = 0.05). obp 0.5.0
                     stores the chosen value on the estimator object and
                     pyieoe reuses the same object for every seed, so the
                     value tuned on the first seed is used for all 200
                     seeds. tuning="first_seed" reproduces that; the value
                     each seed would choose on its own is also recorded.

One shortcut, exact by construction: the reward model is evaluated once per
distinct user context instead of once per row (OBD has few distinct
contexts), and its predictions are mapped back to the rows.
"""

from __future__ import annotations

import numpy as np
try:                                    # scikit-learn < 1.0 (Saito et al. used 0.24.2)
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

LAMBDAS = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
HGB_PARAMS = dict(max_iter=100, learning_rate=0.01, max_depth=5, min_samples_leaf=10)
DELTA = 0.05


# ---------------------------------------------------------------------------
# Evaluation policies
# ---------------------------------------------------------------------------
def bts_action_dist(alpha, beta, len_list=3, n_sim=100_000, random_state=12345) -> np.ndarray:
    """obp.policy.BernoulliTS.compute_batch_action_dist with zero counts:
    n_sim slates, each the top len_list of one Beta draw per item; returns
    pi[a, position] = share of slates with item a at that position."""
    rs = np.random.RandomState(random_state)
    alpha = np.asarray(alpha, float); beta = np.asarray(beta, float)
    K = alpha.size
    counts = np.zeros((K, len_list))
    done = 0
    while done < n_sim:                     # blocks keep memory small; same stream
        b = min(20_000, n_sim - done)
        draws = rs.beta(a=alpha, b=beta, size=(b, K))
        top = np.argsort(draws, axis=1)[:, ::-1][:, :len_list]
        for p in range(len_list):
            counts[:, p] += np.bincount(top[:, p], minlength=K)
        done += b
    return counts / n_sim


def random_action_dist(K, len_list=3) -> np.ndarray:
    return np.full((K, len_list), 1.0 / K)


# ---------------------------------------------------------------------------
# Reward model with cross-fitting (obp RegressionModel, fitting_method="normal")
# ---------------------------------------------------------------------------
def _fit_position_models(Xctx, action, reward, position, K, len_list, seed):
    models = []
    eye = np.eye(K, dtype=int)
    for p in range(len_list):
        m = position == p
        X = np.c_[Xctx[m], eye[action[m]]]
        clf = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=seed)
        clf.fit(X, reward[m])
        models.append(clf)
    return models


def _predict_unique(models, Xu, K, len_list):
    """q[u, a, p] for every distinct context u, item a and position p."""
    nu = Xu.shape[0]
    eye = np.eye(K, dtype=int)
    Xall = np.c_[np.repeat(Xu, K, axis=0), np.tile(eye, (nu, 1))]   # row u*K + a
    q = np.zeros((nu, K, len_list))
    for p in range(len_list):
        q[:, :, p] = models[p].predict_proba(Xall)[:, 1].reshape(nu, K)
    return q


def cross_fit_q(ctx_codes, Xctx, action, reward, position, pi_e, seed, n_folds=2):
    """Returns, per row, the DM term sum_a pi_e(a|pos) q(x, a, pos) (np.average,
    as obp) and the factual prediction q(x, a_i, pos_i)."""
    n = action.size
    K, L = pi_e.shape
    dm = np.zeros(n); qf = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(Xctx):
        models = _fit_position_models(Xctx[tr], action[tr], reward[tr], position[tr], K, L, seed)
        uniq, inv = np.unique(ctx_codes[te], axis=0, return_inverse=True)
        inv = inv.ravel()
        first = np.zeros(uniq.shape[0], np.int64)
        first[inv[::-1]] = np.arange(te.size)[::-1]          # a row index for each context
        q = _predict_unique(models, Xctx[te][first], K, L)    # (nu, K, L)
        pt = position[te]
        # DM term per (context, position), exactly np.average over items
        dm_u = np.stack([np.average(q[:, :, p], weights=np.broadcast_to(pi_e[:, p], q[:, :, p].shape), axis=1)
                         for p in range(L)], axis=1)          # (nu, L)
        dm[te] = dm_u[inv, pt]
        qf[te] = q[inv, action[te], pt]
    return dm, qf


# ---------------------------------------------------------------------------
# Estimators (obp 0.5.0 formulas)
# ---------------------------------------------------------------------------
def _bias_upper_bound(r, iw, iw_hat, qf, delta=DELTA):
    n = r.size
    b = np.abs(((iw - iw_hat) * (r - qf)).mean())
    b += np.sqrt((2 * (iw ** 2).mean() * np.log(2 / delta)) / n)
    b += (2 * iw.max() * np.log(2 / delta)) / (3 * n)
    return b


def _switch_rounds(r, iw, dm, qf, lam):
    return dm + np.array(iw <= lam, dtype=int) * iw * (r - qf)


def _dros_iw_hat(iw, lam):
    return (lam * iw) / (iw ** 2 + lam) if lam < np.inf else iw


def _dros_rounds(r, iw, dm, qf, lam):
    return dm + _dros_iw_hat(iw, lam) * (r - qf)


def tune(r, iw, dm, qf, which):
    """Estimated-MSE tuning of obp 0.5.0 (_tune_hyperparam). Ties go to the
    first lambda in LAMBDAS order (Python min over the dict items)."""
    n = r.size
    scores = {}
    for lam in LAMBDAS:
        if which == "Switch-DR":
            rr = _switch_rounds(r, iw, dm, qf, lam)
            iw_hat = iw * np.array(iw <= lam, dtype=int)
        else:
            rr = _dros_rounds(r, iw, dm, qf, lam)
            iw_hat = _dros_iw_hat(iw, lam)
        scores[lam] = np.var(rr) / n + _bias_upper_bound(r, iw, iw_hat, qf) ** 2
    return min(scores.items(), key=lambda x: x[1])[0], scores


def estimate_all(r, iw, dm, qf, lam_switch, lam_dros) -> dict:
    return {"IPW": float((r * iw).mean()),
            "SNIPW": float((r * iw / iw.mean()).mean()),
            "DM": float(dm.mean()),
            "DR": float((dm + iw * (r - qf)).mean()),
            "SNDR": float((dm + iw * (r - qf) / iw.mean()).mean()),
            "Switch-DR": float(_switch_rounds(r, iw, dm, qf, lam_switch).mean()),
            "DRos": float(_dros_rounds(r, iw, dm, qf, lam_dros).mean())}


# ---------------------------------------------------------------------------
# One bootstrap seed of the pyieoe loop
# ---------------------------------------------------------------------------
def bootstrap_index(seed: int, n: int) -> np.ndarray:
    np.random.seed(seed)
    return np.random.choice(np.arange(n), size=n, replace=True)


def run_seed(seed, data, pi_e, n, n_folds=2):
    """data: dict with first-n-rows arrays ctx_codes, Xctx, action, reward,
    position, pscore. Returns the per-row quantities needed by the estimators."""
    idx = bootstrap_index(seed, n)
    a = data["action"][idx]; ri = data["reward"][idx]
    p = data["position"][idx]; ps = data["pscore"][idx]
    dm, qf = cross_fit_q(data["ctx_codes"][idx], data["Xctx"][idx], a, ri, p, pi_e, seed, n_folds)
    r = ri.astype(float)
    iw = pi_e[a, p] / ps
    return r, iw, dm, qf
