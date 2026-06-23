import pandas as pd

import syneva
from syneva import FairnessSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _df():
    y = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
    return pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "y": y})


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_fairness_off_by_default():
    rep = syneva.evaluate(_df(), _df(), _meta(), tiers=("core", "extended"))
    assert "statistical_parity" not in {r.spec.name for r in rep.results}


def test_fairness_runs_when_enabled():
    rep = syneva.evaluate(
        _df(),
        _df(),
        _meta(),
        tiers=("core", "extended"),
        run_fairness=True,
        fairness_specs=[FairnessSpec("g", "y")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert "statistical_parity" in by
    assert by["statistical_parity"].error is None
    assert by["statistical_parity"].scalars["parity_drift"] < 0.05
