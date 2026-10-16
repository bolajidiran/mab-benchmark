"""Item 4: plug-in registries, YAML studies and the command-line interface."""
import json
from importlib.metadata import EntryPoint

import numpy as np
import pytest

from mab_benchmark import registry as R
from mab_benchmark.baselines import UCB1
from mab_benchmark.cli import main
from mab_benchmark.core import BanditAlgorithm
from mab_benchmark.environments import BernoulliBandit
from mab_benchmark.runner import BenchmarkRunner
from mab_benchmark.study import StudyConfigError, load_config, run_study, validate_config

CELL = {"environment": "S1_bernoulli", "params": {"K": 10, "T": 500}}


class AlwaysFirst(BanditAlgorithm):
    def choose_arm(self, t, context=None):
        return 0

    def update(self, arm, reward, t, context=None):
        pass

    def reset(self):
        pass


def test_builtins_are_registered_and_unknown_names_fail():
    c = R.list_components()
    assert {"S1_bernoulli", "S3_piecewise", "S4_synthetic", "S4_kuairec", "S5_coldstart"} <= set(c["environments"])
    assert {"B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS", "FG-TS"} <= set(c["algorithms"])
    assert {"mean_regret", "ncr"} <= set(c["metrics"])
    with pytest.raises(KeyError):
        R.get_algorithm("no-such-algorithm")
    with pytest.raises(ValueError):
        R.register_algorithm("B2_UCB1", UCB1)                    # built-ins are never silently replaced


def test_validation_enforces_the_protocol():
    ok = validate_config({"algorithms": ["FG-TS"], "settings": [CELL]})
    assert [a["name"] for a in ok["algorithms"]] == ["FG-TS", "B0_Random", "B1_EpsGreedy", "B2_UCB1", "B3_TS"]
    assert ok["settings"][0]["label"] == "S1_bernoulli_K10_T500"
    assert validate_config(ok) == ok                                # idempotent
    for bad in ({"algorithms": ["FG-TS"], "settings": [CELL], "n_runs": 10},
                {"algorithms": ["FG-TS"], "settings": [CELL], "seeds": [1, 2]},
                {"algorithms": ["FG-TS"], "settings": [CELL], "colour": "red"},
                {"algorithms": ["FG-TS"], "settings": [CELL, CELL]},
                {"algorithms": [], "settings": [CELL]},
                {"algorithms": ["FG-TS"], "settings": [{"environment": "S1_bernoulli", "params": {"T": 123}}]}):
        with pytest.raises(StudyConfigError):
            validate_config(bad)


def test_study_reproduces_the_runner_and_ncr_of_random_is_one():
    s = run_study({"algorithms": ["B2_UCB1"], "settings": [CELL], "metrics": ["mean_regret", "ncr"],
                   "include_mandatory_baselines": False})
    direct = BenchmarkRunner(n_runs=30).run_setting(UCB1, {}, BernoulliBandit, {"K": 10, "T": 500})
    cell = s["settings"]["S1_bernoulli_K10_T500"]
    assert cell["metrics"]["mean_regret"]["B2_UCB1"] == pytest.approx(direct.mean(), rel=1e-12)
    assert not s["mandatory_baselines_complete"]
    full = run_study({"algorithms": ["B0_Random"], "settings": [CELL], "metrics": ["ncr"]})
    assert full["mandatory_baselines_complete"]
    assert 0.8 < full["settings"]["S1_bernoulli_K10_T500"]["metrics"]["ncr"]["B0_Random"] < 1.2


def test_runtime_plugins_and_entry_points(monkeypatch, tmp_path):
    R.register_algorithm("test-always-first", AlwaysFirst, overwrite=True)
    R.register_metric("test-max", lambda regrets, info: float(np.max(regrets)), overwrite=True)
    out = tmp_path / "study"
    s = run_study({"algorithms": ["test-always-first"], "settings": [CELL], "metrics": ["test-max"],
                   "include_mandatory_baselines": False, "output": str(out)})
    assert s["settings"]["S1_bernoulli_K10_T500"]["metrics"]["test-max"]["test-always-first"] > 0
    assert (out / "runs.csv").read_text().count("\n") == 31 and (out / "summary.json").exists()

    fake = {"mab_benchmark.algorithms": [EntryPoint("ep-ucb", "mab_benchmark.baselines.b2_ucb1:UCB1",
                                                    "mab_benchmark.algorithms")]}
    monkeypatch.setattr(R, "entry_points", lambda group: fake.get(group, []))
    monkeypatch.setattr(R, "_DISCOVERED", False)
    assert R.get_algorithm("ep-ucb").load() is UCB1


def test_cli_commands(tmp_path, capsys):
    assert main(["version"]) == 0 and "SEED_TABLE 1000..1199" in capsys.readouterr().out
    assert main(["list"]) == 0 and "S4_kuairec" in capsys.readouterr().out
    study = tmp_path / "s.json"
    study.write_text(json.dumps({"name": "t", "algorithms": ["B2_UCB1"], "settings": [CELL],
                                 "include_mandatory_baselines": False}))
    assert main(["validate", str(study)]) == 0
    assert main(["run", str(study), "-o", str(tmp_path / "o"), "-q"]) == 0
    assert (tmp_path / "o" / "summary.json").exists()
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"algorithms": ["nope"], "settings": [CELL]}))
    assert main(["validate", str(bad)]) == 2
    capsys.readouterr()
    assert main(["bcs", "--rho-s", "0.94", "--p-indep", "1.0", "-K", "80", "-T", "14315",
                 "--rho-shuffle", "0.2", "--ratio", "B2_UCB1=0.18", "--ratio", "B0_Random=1"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["starved_algorithms"] == ["B2_UCB1"] and r["band_R"] != "PASS"


def test_yaml_example_loads():
    cfg = load_config("examples/study_quick.yaml")
    assert len(cfg["settings"]) == 3 and cfg["n_runs"] == 30
