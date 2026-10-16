"""
mab_benchmark.study
===================

Declarative studies: a YAML (or JSON) file names the algorithms, the
settings and the metrics, and run_study() executes it under the benchmark's
rules (pre-registered seeds, 30 runs minimum, the four mandatory baselines,
Wilcoxon with Holm correction and effect sizes) and writes the results.

Example (examples/study_s1.yaml):

    name: s1-primary
    n_runs: 30                      # >= 30; seeds are SEED_TABLE[0:n_runs]
    algorithms:
      - name: FG-TS                 # registry name
        params: {lam: 0.5}          # optional, overrides the registered defaults
      - name: B2_UCB1
    settings:
      - environment: S1_bernoulli
        params: {K: 10, T: 10000}
        label: S1_K10_T10000        # optional; default "<environment>_K<K>_T<T>"
    metrics: [mean_regret, ncr]     # registry names
    include_mandatory_baselines: true   # default: B0 to B3 are added if missing
    output: results/s1-primary      # directory; created if absent

Outputs in the output directory
  runs.csv       one row per (setting, algorithm, run): seed, regret, random_regret
  summary.json   per setting: metrics per algorithm, statistical report, ranking,
                 the study configuration and the provenance block
"""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np

from mab_benchmark import registry as R
from mab_benchmark.analysis.statistics import StatisticalAnalyser
from mab_benchmark.runner import SEED_TABLE, STREAM_LAYOUT, BenchmarkRunner

MANDATORY = ("B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS")
TOP_KEYS = {"name", "n_runs", "algorithms", "settings", "metrics", "include_mandatory_baselines",
            "output", "seeds"}


class StudyConfigError(ValueError):
    pass


def load_config(path) -> dict:
    """Read a YAML or JSON study file. YAML needs PyYAML (pip install mab_benchmark[yaml])."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as e:                                    # pragma: no cover
            raise StudyConfigError("YAML configs need PyYAML: pip install 'mab_benchmark[yaml]'") from e
        cfg = yaml.safe_load(text)
    return validate_config(cfg)


def _norm_alg(a):
    if isinstance(a, str):
        return {"name": a, "params": {}}
    if not isinstance(a, dict) or "name" not in a:
        raise StudyConfigError(f"an algorithm entry needs a 'name': {a!r}")
    return {"name": a["name"], "params": dict(a.get("params") or {}), **({"label": a["label"]} if "label" in a else {})}


def validate_config(cfg) -> dict:
    """Check a parsed configuration and fill defaults. Raises StudyConfigError."""
    if not isinstance(cfg, dict):
        raise StudyConfigError("the study file must contain a mapping")
    extra = set(cfg) - TOP_KEYS
    if extra:
        raise StudyConfigError(f"unknown keys: {sorted(extra)}")
    for k in ("algorithms", "settings"):
        if not cfg.get(k):
            raise StudyConfigError(f"'{k}' must be a non-empty list")
    out = {"name": str(cfg.get("name", "study")),
           "n_runs": int(cfg.get("n_runs", 30)),
           "algorithms": [_norm_alg(a) for a in cfg["algorithms"]],
           "settings": [],
           "metrics": list(cfg.get("metrics", ["mean_regret", "ncr"])),
           "include_mandatory_baselines": bool(cfg.get("include_mandatory_baselines", True)),
           "output": cfg.get("output")}
    if "seeds" in cfg:
        raise StudyConfigError("seeds are fixed by the pre-registered SEED_TABLE and cannot be set in a study")
    if out["n_runs"] < 30:
        raise StudyConfigError(f"n_runs must be >= 30 (Requirement 1), got {out['n_runs']}")
    for s in cfg["settings"]:
        if not isinstance(s, dict) or "environment" not in s:
            raise StudyConfigError(f"a setting entry needs an 'environment': {s!r}")
        spec = R.get_environment(s["environment"])
        params = {**spec.defaults, **dict(s.get("params") or {})}
        label = s.get("label") or f"{s['environment']}_K{params.get('K')}_T{params.get('T')}"
        try:                                                          # catch bad parameters before running
            spec.load()(**params, seed=SEED_TABLE[0])
        except (ValueError, TypeError) as e:
            raise StudyConfigError(f"setting {label}: {e}") from None
        out["settings"].append({"environment": s["environment"], "params": params, "label": label})
    labels = [s["label"] for s in out["settings"]]
    if len(set(labels)) != len(labels):
        raise StudyConfigError(f"setting labels must be unique: {labels}")
    names = [a.get("label", a["name"]) for a in out["algorithms"]]
    for a in out["algorithms"]:
        R.get_algorithm(a["name"])                                   # raises on unknown names
    if out["include_mandatory_baselines"]:
        for b in MANDATORY:
            if b not in names:
                out["algorithms"].append({"name": b, "params": {}})
    if len(set(a.get("label", a["name"]) for a in out["algorithms"])) != len(out["algorithms"]):
        raise StudyConfigError("algorithm names (or labels) must be unique")
    for m in out["metrics"]:
        R.get_metric(m)
    return out


def run_study(cfg: dict, verbose: bool = False) -> dict:
    """Execute a validated configuration; returns the summary and writes it if 'output' is set."""
    cfg = validate_config(cfg)                                     # idempotent on a validated config
    runner = BenchmarkRunner(n_runs=cfg["n_runs"])
    seeds = runner.seeds[: cfg["n_runs"]]
    rows, summary = [], {"name": cfg["name"], "config": cfg, "settings": {},
                         "provenance": {"mab_benchmark": _version(), "stream_layout": STREAM_LAYOUT,
                                        "seeds": seeds, "python": platform.python_version(),
                                        "numpy": np.__version__, "platform": platform.platform()}}
    names = [a.get("label", a["name"]) for a in cfg["algorithms"]]
    complete = all(b in names for b in MANDATORY)
    for st in cfg["settings"]:
        spec = R.get_environment(st["environment"]); Env = spec.load()
        stationary = spec.kind == "stationary"; contextual = spec.kind == "contextual"
        try:
            rr = np.array([Env(**st["params"], seed=int(s)).expected_random_regret() for s in seeds], float)
        except (AttributeError, NotImplementedError):
            rr = None
        info = {"K": st["params"].get("K"), "T": st["params"].get("T"), "random_regret": rr, "n_runs": cfg["n_runs"]}
        res = {}
        for a in cfg["algorithms"]:
            aspec = R.get_algorithm(a["name"]); Alg = aspec.load()
            label = a.get("label", a["name"])
            if verbose:
                print(f"  {st['label']}: {label}", flush=True)
            reg = runner.run_setting(Alg, {**aspec.defaults, **a["params"]}, Env, st["params"],
                                     is_stationary=stationary or contextual, use_context=contextual)
            res[label] = reg
            for i, (s, r) in enumerate(zip(seeds, reg)):
                rows.append({"setting": st["label"], "algorithm": label, "run": i + 1, "seed": s,
                             "regret": float(r), "random_regret": None if rr is None else float(rr[i])})
        metrics = {}
        for m in cfg["metrics"]:
            f = R.get_metric(m)
            try:
                metrics[m] = {a: f(v, info) for a, v in res.items()}
            except ValueError as e:
                metrics[m] = {"error": str(e)}
        report = StatisticalAnalyser().analyse(res)
        summary["settings"][st["label"]] = {
            "environment": st["environment"], "params": st["params"], "metrics": metrics,
            "ranking": report["ranking"], "statistics": _jsonable(report),
            "mandatory_baselines_complete": complete}
    summary["mandatory_baselines_complete"] = complete
    if cfg.get("output"):
        out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)
        with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1), encoding="utf-8")
    return summary


def _version():
    import mab_benchmark
    return mab_benchmark.__version__


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k) if not isinstance(k, tuple) else " vs ".join(map(str, k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x
