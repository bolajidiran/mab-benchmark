"""
common.py: shared paths, constants, loaders and algorithms for the OBD
preprocessing pipeline (Fixes 1 to 5) of the S4 setting.

PhD thesis : Developing a Performance Model of Multi-Armed-Bandit
             Recommender Systems (LASU 2026)
Author     : Ojediran Alaba Bolaji
Dataset    : Open Bandit Dataset (Saito et al., 2020), CC BY 4.0
             https://research.zozo.com/data_release/open_bandit_dataset.zip

Every step script imports this module, so constants are defined once.
Run the steps in order from the repository root (see README.md):

    python obd_preprocessing/step1_fix5_integrity.py
    python obd_preprocessing/step2_fix3_positions.py
    python obd_preprocessing/step3_fix4_context.py
    python obd_preprocessing/step4_fix1_fix2_slices.py
    python obd_preprocessing/step5_run_s4_replay.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to this file, so the pipeline runs on any machine)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW_DIR = REPO / "open_bandit_dataset"
CACHE_DIR = HERE / "cache"        # intermediate arrays (not for the thesis)
OUT_DIR = HERE / "outputs"        # reports and results (for the thesis)
VARIANT = "main"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Variants keep sensitivity runs apart from the main results:
#   main             -> cache/ and outputs/
#   keep_duplicates  -> cache/keep_duplicates/ and outputs/keep_duplicates/
VARIANTS = ("main", "keep_duplicates")


def set_variant(name: str) -> None:
    """Point CACHE_DIR / OUT_DIR at the folders of one analysis variant."""
    global CACHE_DIR, OUT_DIR, VARIANT
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; choose from {VARIANTS}")
    VARIANT = name
    CACHE_DIR = HERE / "cache" if name == "main" else HERE / "cache" / name
    OUT_DIR = HERE / "outputs" if name == "main" else HERE / "outputs" / name
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_variant_arg(parser) -> None:
    parser.add_argument("--variant", default=None, choices=VARIANTS,
                        help="analysis variant (default: main)")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMPAIGN = "all"
POLICIES = ("random", "bts")      # logging (behaviour) policies in OBD
K = 80                            # items (arms) in the "all" campaign
POSITIONS = (1, 2, 3)             # left, centre, right slot
N_SLICES = 30                     # Requirement 1: n >= 30 independent runs
ACCEPT_FLOOR = 1.0 / K            # rejection-sampling floor c (uniform rate)
USER_COLS = [f"user_feature_{i}" for i in range(4)]
AFF_COLS = [f"user-item_affinity_{i}" for i in range(K)]
ITEM_CAT_COLS = ["item_feature_1", "item_feature_2", "item_feature_3"]
CHUNK_ROWS = 1_000_000            # chunked reading keeps memory below ~2 GB

# ZOZOTOWN production priors of the Bernoulli TS logging policy, copied
# from obp 0.4.1 (obp/policy/conf/prior_bts.yaml, campaign "all"; Apache 2.0).
ZOZOTOWN_PRIOR_FILE = HERE / "zozotown_prior_bts_all.json"

# Published live CTRs of the two logging policies (Saito et al., 2020/2021,
# "all" campaign) used only as an external sanity check.
PUBLISHED_CTR = {"random": 0.0035, "bts": 0.0050}


def seed_table() -> list[int]:
    """Pre-registered seeds, taken from the package so S4 matches S1 to S5."""
    from mab_benchmark.runner import SEED_TABLE
    return list(SEED_TABLE[:N_SLICES])


def raw_log_path(policy: str) -> Path:
    """Return all.csv, or all.csv.gz if only the compressed copy exists."""
    base = RAW_DIR / policy / CAMPAIGN
    for name in ("all.csv", "all.csv.gz"):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(f"No OBD log found in {base}")


def item_context_path(policy: str = "random") -> Path:
    return RAW_DIR / policy / CAMPAIGN / "item_context.csv"


def md5_of(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def environment_info() -> dict:
    import pandas, scipy
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def write_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fast algorithm implementations used in the replay
# ---------------------------------------------------------------------------
# B0 to B3 are imported unchanged from mab_benchmark.baselines. The two
# LinUCB variants below are re-implemented for speed: the package LinUCB
# inverts 80 matrices on every round, which is infeasible over 13.7M rows.
# LinUCBFast keeps A^{-1} up to date with the Sherman-Morrison identity and
# is tested for identical decisions against the package class in
# step5 (--self-test). FeelGoodTS mirrors the vectorised class used in
# s4_obd_experiment.py (lambda = 0.5, matching fgts_submission.json).

from mab_benchmark.core import BanditAlgorithm  # noqa: E402


class FeelGoodTS(BanditAlgorithm):
    """Feel-Good Thompson Sampling (Zhang, 2022); lambda = 0.5."""

    def __init__(self, n_arms, lambda_=0.5, alpha_0=1.0, beta_0=1.0):
        self.lambda_, self.alpha_0, self.beta_0 = lambda_, alpha_0, beta_0
        super().__init__(n_arms, context_dim=0)

    def reset(self):
        self.alpha = np.full(self.n_arms, self.alpha_0)
        self.beta = np.full(self.n_arms, self.beta_0)
        self.counts = np.zeros(self.n_arms, dtype=int)

    def choose_arm(self, t, context=None):
        theta = np.random.beta(self.alpha, self.beta)
        bonus = self.lambda_ * np.sqrt(np.log(t + 1) / (self.counts + 1))
        return int(np.argmax(theta + bonus))

    def update(self, arm, reward, t, context=None):
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r
        self.counts[arm] += 1


class BernoulliTSZozotown(BanditAlgorithm):
    """
    Bernoulli Thompson Sampling initialised with the ZOZOTOWN production
    priors (the policy that generated the bts log), the same priors that
    obp.policy.BernoulliTS(is_zozotown_prior=True, campaign="all") reads.
    """

    def __init__(self, n_arms, **kwargs):
        pri = json.loads(ZOZOTOWN_PRIOR_FILE.read_text())
        self.alpha_0 = np.array(pri["alpha"], dtype=float)
        self.beta_0 = np.array(pri["beta"], dtype=float)
        super().__init__(n_arms, context_dim=0)

    def reset(self):
        self.alpha = self.alpha_0.copy()
        self.beta = self.beta_0.copy()
        self.counts = np.zeros(self.n_arms, dtype=int)

    def choose_arm(self, t, context=None):
        return int(np.argmax(np.random.beta(self.alpha, self.beta)))

    def update(self, arm, reward, t, context=None):
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r
        self.counts[arm] += 1


class LinUCBFast(BanditAlgorithm):
    """
    B4: disjoint LinUCB (Li et al., 2010), alpha = 1.0, A_a = I_d initially.
    Decisions are identical to mab_benchmark.baselines.LinUCB; only the
    matrix inverse is maintained incrementally (Sherman-Morrison).
    Context = one-hot user features with a bias term (shared by all arms).
    """

    def __init__(self, n_arms, context_dim, alpha=1.0):
        self.alpha = alpha
        super().__init__(n_arms, context_dim)

    def reset(self):
        d = self.context_dim
        self.Ainv = np.stack([np.eye(d) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, d))
        self.theta = np.zeros((self.n_arms, d))

    def choose_arm(self, t, context=None):
        x = context
        quad = np.einsum("i,aij,j->a", x, self.Ainv, x)
        ucb = self.theta @ x + self.alpha * np.sqrt(np.maximum(quad, 0.0))
        return int(np.argmax(ucb))

    def update(self, arm, reward, t, context=None):
        x = context
        Ai = self.Ainv[arm]
        Ax = Ai @ x
        self.Ainv[arm] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[arm] += reward * x
        self.theta[arm] = self.Ainv[arm] @ self.b[arm]


class HybridLinUCB(BanditAlgorithm):
    """
    B4h: hybrid LinUCB (Li et al., 2010, Algorithm 2), alpha = 1.0.

    Per-arm (disjoint) features x_t : one-hot user features + bias (d).
    Shared features z_{t,a}         : item features of arm a plus the
                                      user-item affinity of (user, a) (k).
    This is the form that uses user + item + affinity information
    together, as chosen for Fix 4.
    """

    def __init__(self, n_arms, context_dim, item_static, alpha=1.0):
        self.alpha = alpha
        self.S = np.asarray(item_static, dtype=float)   # (K, k-1)
        self.k = self.S.shape[1] + 1                     # + affinity column
        super().__init__(n_arms, context_dim)

    def reset(self):
        d, k, K_ = self.context_dim, self.k, self.n_arms
        self.A0 = np.eye(k)
        self.b0 = np.zeros(k)
        self.A = np.stack([np.eye(d) for _ in range(K_)])
        self.B = np.zeros((K_, d, k))
        self.bb = np.zeros((K_, d))
        self._refresh()

    def _z(self, aff):
        return np.hstack([self.S, aff[:, None]])          # (K, k)

    def _refresh(self):
        self.A0inv = np.linalg.inv(self.A0)
        self.Ainv = np.linalg.inv(self.A)                  # (K, d, d)
        self.beta = self.A0inv @ self.b0
        self.C = np.einsum("aij,ajk->aik", self.Ainv, self.B)   # A^-1 B
        rhs = self.bb - np.einsum("aik,k->ai", self.B, self.beta)
        self.theta = np.einsum("aij,aj->ai", self.Ainv, rhs)

    def choose_arm(self, t, context=None):
        x, aff = context                                    # x: (d,), aff: (K,)
        Z = self._z(aff)
        P = self.A0inv
        ZP = Z @ P
        s1 = np.einsum("ak,ak->a", ZP, Z)
        u = np.einsum("aik,i->ak", self.C, x)               # B^T A^-1 x
        s2 = -2.0 * np.einsum("ak,ak->a", ZP, u)
        s3 = np.einsum("i,aij,j->a", x, self.Ainv, x)
        s4 = np.einsum("ak,ak->a", u @ P, u)
        s = np.maximum(s1 + s2 + s3 + s4, 0.0)
        p = Z @ self.beta + self.theta @ x + self.alpha * np.sqrt(s)
        return int(np.argmax(p))

    def update(self, arm, reward, t, context=None):
        x, aff = context
        z = self._z(aff)[arm]
        a = arm
        Ainv_a = self.Ainv[a]
        BtAinv = self.B[a].T @ Ainv_a
        self.A0 += BtAinv @ self.B[a]
        self.b0 += BtAinv @ self.bb[a]
        self.A[a] += np.outer(x, x)
        self.B[a] += np.outer(x, z)
        self.bb[a] += reward * x
        Ainv_new = np.linalg.inv(self.A[a])
        BtAinv = self.B[a].T @ Ainv_new
        self.A0 += np.outer(z, z) - BtAinv @ self.B[a]
        self.b0 += reward * z - BtAinv @ self.bb[a]
        # Only arm a's A, B, b and the shared A0, b0 changed.
        self.A0inv = np.linalg.inv(self.A0)
        self.beta = self.A0inv @ self.b0
        self.Ainv[a] = Ainv_new
        self.C[a] = Ainv_new @ self.B[a]
        rhs = self.bb - np.einsum("aik,k->ai", self.B, self.beta)
        self.theta = np.einsum("aij,aj->ai", self.Ainv, rhs)
