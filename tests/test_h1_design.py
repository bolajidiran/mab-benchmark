"""Regression test for the H1 comparability study recorder (experiments/h1_comparability.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import h1_comparability as H  # noqa: E402


def test_verdicts_recorded_whatever_the_study_order():
    # a fragmented study lists FG-TS first, so its verdict keys are reversed
    v = {("FG-TS", "B2_UCB1"): "B2_UCB1", ("FG-TS", "B0_Random"): "FG-TS", ("B2_UCB1", "B0_Random"): "none"}
    row = H.verdict_columns(v)
    assert row["B2_UCB1~FG-TS"] == "B2_UCB1"
    assert row["B0_Random~FG-TS"] == "FG-TS"
    assert row["B0_Random~B2_UCB1"] == "none"
    assert row["B1_EpsGreedy~B3_TS"] == ""          # not compared
    assert sum(x != "" for x in row.values()) == 3


def test_accuracy_reads_the_T_column_not_the_transpose():
    import pandas as pd
    df = pd.DataFrame([{"K": 10.0, "T": 500.0, **H.verdict_columns({("B0_Random", "B3_TS"): "B3_TS"})}])
    truth = {(10, 500, a, b): "B3_TS" if (a, b) == ("B0_Random", "B3_TS") else "none" for a, b in H.PAIRS}
    assert H.accuracy(df, truth) == 1.0
