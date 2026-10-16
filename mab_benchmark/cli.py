"""
mab-benchmark command-line interface.

  mab-benchmark version                 package, stream layout and seed table
  mab-benchmark list                    registered environments, algorithms, metrics
  mab-benchmark validate STUDY.yaml     check a study file without running it
  mab-benchmark run STUDY.yaml [-o DIR] run a study and write runs.csv and summary.json
  mab-benchmark bcs --rho-s R --p-indep P -K K -T T [--rho-shuffle S --ratio ALG=M ...]
                                        Bridge Compliance Score of a setting, with the
                                        replay-integrity condition when --rho-shuffle is given
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_version(a):
    import mab_benchmark
    from mab_benchmark.runner import SEED_TABLE, STREAM_LAYOUT
    print(f"mab_benchmark {mab_benchmark.__version__}; stream layout {STREAM_LAYOUT}; "
          f"SEED_TABLE {SEED_TABLE[0]}..{SEED_TABLE[-1]} ({len(SEED_TABLE)} seeds)")
    return 0


def _cmd_list(a):
    from mab_benchmark.registry import list_components
    c = list_components()
    print("Environments:")
    for k, (kind, d) in c["environments"].items():
        print(f"  {k:<16} {kind:<14} {d}")
    print("Algorithms:")
    for k, d in c["algorithms"].items():
        print(f"  {k:<16} {d}")
    print("Metrics:\n  " + ", ".join(c["metrics"]))
    return 0


def _cmd_validate(a):
    from mab_benchmark.study import StudyConfigError, load_config
    try:
        cfg = load_config(a.study)
    except (StudyConfigError, KeyError) as e:
        print(f"invalid: {e}", file=sys.stderr)
        return 2
    print(f"valid: {cfg['name']}: {len(cfg['algorithms'])} algorithms x {len(cfg['settings'])} settings "
          f"x {cfg['n_runs']} runs; metrics {cfg['metrics']}")
    return 0


def _cmd_run(a):
    from mab_benchmark.study import StudyConfigError, load_config, run_study
    try:
        cfg = load_config(a.study)
    except (StudyConfigError, KeyError) as e:
        print(f"invalid: {e}", file=sys.stderr)
        return 2
    if a.output:
        cfg["output"] = a.output
    if not cfg.get("output"):
        cfg["output"] = f"results/{cfg['name']}"
    s = run_study(cfg, verbose=not a.quiet)
    for lab, d in s["settings"].items():
        print(f"{lab}: ranking (best first) {', '.join(d['ranking'])}")
    print(f"written: {cfg['output']}/runs.csv, {cfg['output']}/summary.json")
    if not s["mandatory_baselines_complete"]:
        print("warning: mandatory baselines B0 to B3 incomplete; results are not certifiable")
    return 0


def _cmd_bcs(a):
    from mab_benchmark.analysis.bcs import BCSCalculator
    if a.rho_shuffle is None:
        r = BCSCalculator.compute(a.rho_s, a.p_indep, a.K, a.T)
    else:
        ratios = {}
        for x in a.ratio or []:
            k, _, v = x.partition("=")
            ratios[k] = float(v)
        r = BCSCalculator.compute_replay(a.rho_s, a.p_indep, a.K, a.T, a.rho_shuffle, ratios,
                                         tier1_bias_free=not a.tier1_is_replay)
    print(json.dumps(r, indent=1, default=str))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="mab-benchmark", description="MAB Unified Benchmark Suite")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version").set_defaults(fn=_cmd_version)
    sub.add_parser("list").set_defaults(fn=_cmd_list)
    v = sub.add_parser("validate"); v.add_argument("study"); v.set_defaults(fn=_cmd_validate)
    r = sub.add_parser("run"); r.add_argument("study"); r.add_argument("-o", "--output")
    r.add_argument("-q", "--quiet", action="store_true"); r.set_defaults(fn=_cmd_run)
    b = sub.add_parser("bcs")
    b.add_argument("--rho-s", type=float, required=True); b.add_argument("--p-indep", type=float, required=True)
    b.add_argument("-K", type=int, required=True); b.add_argument("-T", type=int, required=True)
    b.add_argument("--rho-shuffle", type=float); b.add_argument("--ratio", action="append",
                                                                help="ALG=matched/random, repeatable")
    b.add_argument("--tier1-is-replay", action="store_true"); b.set_defaults(fn=_cmd_bcs)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
