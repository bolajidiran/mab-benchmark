"""
S4-KR: a bias-free real-data environment built from the fully observed
KuaiRec small matrix (Gao et al., 2022; CC BY-SA 4.0).

KuaiRec's small matrix records the watch ratio of 1,411 users on 3,327 short
videos with almost every user-item pair observed. Because the reward of every
item for every user is known, an algorithm can interact with it online, with
no logging policy and no replay: the arm it chooses is the arm whose reward it
receives. That makes it the reference against which replay bias is measured.

Bandit formulation (context-free, as the S4 bridge algorithms)
  arms    K items drawn once, without replacement, from the fully observed
          items (item_seed fixes the draw; K = 80 matches OBD)
  rounds  a user drawn uniformly at random arrives
  reward  1[watch_ratio(u, a) >= threshold]; threshold = 2.0 by default
  regret  mu* - mu_a, mu_a = mean over users of the reward of item a
Missing user-item cells (fewer than 0.4 per cent) are filled with 0 and
reported.

Data: kuairec_dataset/small_matrix_compact.npz (written by
experiments/kuairec_compact.py), else kuairec_dataset/**/small_matrix.csv, else
KuaiRec.zip read in place. The data are not redistributed with mab_benchmark.
"""

from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "kuairec_dataset"


@lru_cache(maxsize=4)
def load_matrix(root: str = str(DEFAULT_ROOT), threshold: float = 2.0):
    """Users x items binary reward matrix and the share of missing cells."""
    import pandas as pd
    root = Path(root)
    npz = root / "small_matrix_compact.npz"
    if npz.exists():
        d = np.load(npz)
        df = pd.DataFrame({k: d[k] for k in ("user_id", "video_id", "watch_ratio")})
        return _to_matrix(df, threshold)
    csv = next(iter(sorted(root.rglob("small_matrix.csv"))), None)
    if csv is not None:
        df = pd.read_csv(csv, usecols=["user_id", "video_id", "watch_ratio"])
    else:
        z = next(iter(sorted(root.glob("*.zip"))), None)
        if z is None:
            raise FileNotFoundError(f"KuaiRec not found under {root} (small_matrix.csv or KuaiRec.zip)")
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith("small_matrix.csv"))
            df = pd.read_csv(io.BytesIO(zf.read(name)), usecols=["user_id", "video_id", "watch_ratio"])
    return _to_matrix(df, threshold)


def _to_matrix(df, threshold):
    users = np.sort(df.user_id.unique()); items = np.sort(df.video_id.unique())
    ui = np.searchsorted(users, df.user_id.to_numpy()); ii = np.searchsorted(items, df.video_id.to_numpy())
    R = np.zeros((users.size, items.size), dtype=np.int8)
    seen = np.zeros_like(R, dtype=bool)
    R[ui, ii] = (df.watch_ratio.to_numpy() >= threshold).astype(np.int8)
    seen[ui, ii] = True
    return R, seen, items


class KuaiRecBandit:
    """Fully observed real-data bandit; drop-in for the stationary runner interface."""

    def __init__(self, K: int = 80, T: int = 14_315, seed: int = 0, root: str | None = None,
                 threshold: float = 2.0, item_seed: int = 2026, min_coverage: float = 0.99) -> None:
        R, seen, items = load_matrix(str(root or DEFAULT_ROOT), threshold)
        cover = seen.mean(axis=0)
        pool = np.flatnonzero(cover >= min_coverage)
        if pool.size < K:
            raise ValueError(f"only {pool.size} items with coverage >= {min_coverage}")
        pick = np.sort(np.random.default_rng(item_seed).choice(pool, K, replace=False))
        self.R = R[:, pick].astype(float)                # users x K
        self.missing_share = float(1.0 - seen[:, pick].mean())
        self.item_ids = items[pick]
        self.K, self.T, self.seed = K, T, seed
        self.n_users = self.R.shape[0]
        self.means = self.R.mean(axis=0)
        self.mu_star = float(self.means.max()); self.a_star = int(self.means.argmax())

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        u = int(rng.integers(0, self.n_users))
        return float(self.R[u, arm])

    def regret_increment(self, arm: int) -> float:
        return self.mu_star - self.means[arm]

    def expected_random_regret(self) -> float:
        return float(self.T * (self.mu_star - self.means.mean()))

    def __repr__(self) -> str:
        return f"KuaiRecBandit(K={self.K}, T={self.T}, seed={self.seed}, mu*={self.mu_star:.4f})"
