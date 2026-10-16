"""
mab_benchmark.analysis.leaderboard
====================================

LeaderboardSubmitter and Leaderboard: the public results table of the MAB
Unified Benchmark Suite (architecture Layer 7).

Hosting
-------
The leaderboard lives in the repository itself, in the folder `Leaderboard/`:

    Leaderboard/submissions/<algorithm>.json   one validated submission each
    Leaderboard/leaderboard.json               the ranked table (generated)
    Leaderboard/LEADERBOARD.md                 the same table for GitHub

A new entry is proposed by pull request; `Leaderboard.add()` validates it and
`Leaderboard.build()` regenerates the table, so every listed number can be
checked against the JSON that produced it.

Workflow
--------
>>> from mab_benchmark import BenchmarkRunner
>>> from mab_benchmark.analysis.certification import Certifier
>>> runner = BenchmarkRunner(n_runs=30)
>>> suite  = runner.baseline_sweep()                       # B0 to B3
>>> suite["MyAlg"] = runner.full_suite(MyAlg, {})
>>> cert   = Certifier().certify("MyAlg", suite, bcs=bcs_result)
>>> sub    = LeaderboardSubmitter("MyAlg", paper_url, code_url).build(suite, cert)
>>> Leaderboard("Leaderboard").add(sub); Leaderboard("Leaderboard").build()

Ranking criterion: mean cumulative regret on S1, K = 10, T = 10,000 (lower is
better), as fixed in MAB_BenchmarkSuite.pdf Section 5.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

BENCHMARK_VERSION = "1.3.0"
SCHEMA_VERSION = 2
PRIMARY_CELL = "S1_K10_T10000"
LEADERBOARD_URL = "https://github.com/bolajidiran/mab-benchmark/tree/main/Leaderboard"


def _cell_stats(arr) -> dict:
    a = np.asarray(arr, float); n = a.size
    m, s = float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0
    h = float(scipy_stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean_regret": round(m, 4), "std": round(s, 4), "ci95_t_low": round(m - h, 4),
            "ci95_t_high": round(m + h, 4), "n_runs": int(n)}


class LeaderboardSubmitter:
    """Build a schema-2 leaderboard submission for one algorithm."""

    def __init__(self, algorithm_name: str, paper_url: str, code_url: str,
                 institution: str = "", notes: str = "") -> None:
        self.algorithm_name = algorithm_name
        self.paper_url = paper_url
        self.code_url = code_url
        self.institution = institution
        self.notes = notes

    # ------------------------------------------------------------------
    def build(self, suite: dict, certificate: dict) -> dict:
        """suite: {algorithm: {cell: regrets}}; certificate: Certifier.certify() output."""
        from mab_benchmark import __version__
        from mab_benchmark.runner import STREAM_LAYOUT
        if certificate.get("algorithm") != self.algorithm_name:
            raise ValueError("certificate belongs to a different algorithm")
        mine = suite[self.algorithm_name]
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark": f"MAB Unified Benchmark Suite v{BENCHMARK_VERSION}",
            "mab_benchmark_version": __version__,
            "stream_layout": STREAM_LAYOUT,
            "leaderboard_url": LEADERBOARD_URL,
            "submission_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "algorithm": self.algorithm_name, "paper_url": self.paper_url,
            "code_url": self.code_url, "institution": self.institution, "notes": self.notes,
            "primary_cell": PRIMARY_CELL,
            "settings": {c: _cell_stats(v) for c, v in mine.items()},
            "baselines": {b: {c: _cell_stats(v) for c, v in suite[b].items()}
                          for b in ("B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS") if b in suite},
            "certification": certificate,
        }

    @staticmethod
    def to_json(submission: dict) -> str:
        return json.dumps(submission, indent=2, default=_json_default)

    @staticmethod
    def save(submission, path: str) -> None:
        text = submission if isinstance(submission, str) else LeaderboardSubmitter.to_json(submission)
        Path(path).write_text(text, encoding="utf-8")

    @staticmethod
    def validate(sub: dict) -> list[str]:
        """Return a list of problems (empty = valid)."""
        p = []
        for k in ("schema_version", "algorithm", "code_url", "settings", "certification", "primary_cell"):
            if k not in sub:
                p.append(f"missing field: {k}")
        if p:
            return p
        if sub["schema_version"] != SCHEMA_VERSION:
            p.append(f"schema_version {sub['schema_version']} != {SCHEMA_VERSION}")
        if sub["primary_cell"] not in sub["settings"]:
            p.append(f"primary cell {sub['primary_cell']} missing from settings")
        for c, v in sub["settings"].items():
            if v.get("n_runs", 0) < 30:
                p.append(f"{c}: n_runs < 30")
        cert = sub["certification"]
        if cert.get("level") not in ("UNCERTIFIED", "C1", "C3", "C4"):
            p.append(f"unknown certification level {cert.get('level')!r}")
        if cert.get("algorithm") != sub["algorithm"]:
            p.append("certificate algorithm does not match submission")
        if cert.get("level") in ("C1", "C3", "C4") and not all(cert.get("checks", {}).values()):
            p.append("certificate claims a level but one of its C1 checks is false")
        return p

    @staticmethod
    def summary(sub) -> str:
        data = json.loads(sub) if isinstance(sub, str) else sub
        cert = data["certification"]
        L = ["=" * 65, "LEADERBOARD SUBMISSION SUMMARY", "=" * 65,
             f"Algorithm   : {data['algorithm']}", f"Code        : {data['code_url']}",
             f"Submitted   : {data['submission_date']}",
             f"Certificate : {cert['label']}"]
        for r in cert.get("reasons_not_higher", []):
            L.append(f"   - {r}")
        pc = data["settings"].get(data["primary_cell"])
        if pc:
            L.append(f"Primary ({data['primary_cell']}): mean regret {pc['mean_regret']:.2f} "
                     f"(95% CI {pc['ci95_t_low']:.2f} to {pc['ci95_t_high']:.2f})")
        if cert.get("bcs") and "BCS" in cert["bcs"]:
            L.append(f"BCS         : {cert['bcs']['BCS']:.4f} [{cert['bcs'].get('band', '')}]")
        L.append("=" * 65)
        return "\n".join(L)


class Leaderboard:
    """The repository-hosted leaderboard (folder with submissions/)."""

    def __init__(self, folder: str | Path = "Leaderboard") -> None:
        self.folder = Path(folder)
        self.sub_dir = self.folder / "submissions"

    @staticmethod
    def _slug(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "submission"

    def add(self, submission: dict | str | Path) -> Path:
        if isinstance(submission, (str, Path)) and Path(submission).exists():
            submission = json.loads(Path(submission).read_text(encoding="utf-8"))
        problems = LeaderboardSubmitter.validate(submission)
        if problems:
            raise ValueError("invalid submission: " + "; ".join(problems))
        self.sub_dir.mkdir(parents=True, exist_ok=True)
        path = self.sub_dir / f"{self._slug(submission['algorithm'])}.json"
        LeaderboardSubmitter.save(submission, str(path))
        return path

    def entries(self) -> list[dict]:
        rows = []
        for p in sorted(self.sub_dir.glob("*.json")):
            s = json.loads(p.read_text(encoding="utf-8"))
            if LeaderboardSubmitter.validate(s):
                continue
            pc = s["settings"][s["primary_cell"]]
            cert = s["certification"]
            rows.append({"algorithm": s["algorithm"], "mean_regret": pc["mean_regret"],
                         "ci95": [pc["ci95_t_low"], pc["ci95_t_high"]], "n_runs": pc["n_runs"],
                         "level": cert["level"],
                         "bcs": (cert.get("bcs") or {}).get("BCS"),
                         "beats": {b: v["better"] for b, v in cert.get("tier1_summary", {}).items()},
                         "cells": next(iter(cert.get("tier1_summary", {}).values()), {}).get("cells"),
                         "code_url": s["code_url"], "date": s["submission_date"][:10],
                         "file": f"submissions/{p.name}"})
        return sorted(rows, key=lambda r: r["mean_regret"])

    def build(self) -> list[dict]:
        rows = self.entries()
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "leaderboard.json").write_text(
            json.dumps({"benchmark": f"MAB Unified Benchmark Suite v{BENCHMARK_VERSION}",
                        "ranking_criterion": f"mean cumulative regret, {PRIMARY_CELL} (lower is better)",
                        "entries": rows}, indent=2, default=_json_default), encoding="utf-8")
        L = [f"# MAB Unified Benchmark Suite leaderboard (v{BENCHMARK_VERSION})", "",
             f"Ranked by mean cumulative regret on {PRIMARY_CELL.replace('_', ', ')} (lower is better), "
             "30 runs, pre-registered seeds. Generated by `Leaderboard.build()`; each row links to the "
             "submission JSON it comes from.", "",
             "| Rank | Algorithm | Mean regret | 95% CI (t) | Certification | BCS | Cells won vs B0 / B1 / B2 / B3 | Date | Submission |",
             "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            beats = " / ".join(str(r["beats"].get(b, "-")) for b in ("B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS"))
            bcs = "-" if r["bcs"] is None else f"{r['bcs']:.3f}"
            L.append(f"| {i} | {r['algorithm']} | {r['mean_regret']:,.2f} | {r['ci95'][0]:,.2f} to "
                     f"{r['ci95'][1]:,.2f} | {r['level']} | {bcs} | {beats} (of {r['cells']}) | {r['date']} | "
                     f"[json]({r['file']}) |")
        L += ["", "Levels: C1 Algorithmic Efficiency (Tier 1, BCS >= 0.9); C3 General Superiority "
              "(Tier 1 and Tier 2); C4 Deployment Validated (live A/B test). See "
              "`mab_benchmark/analysis/certification.py`."]
        (self.folder / "LEADERBOARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        return rows


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
