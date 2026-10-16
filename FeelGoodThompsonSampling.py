"""Compatibility shim. FG-TS now lives in mab_benchmark.algorithms.fg_ts
(identical code, moved in 1.3.0 so that it ships with the package)."""
from mab_benchmark.algorithms.fg_ts import FeelGoodThompsonSampling  # noqa: F401

__all__ = ["FeelGoodThompsonSampling"]
