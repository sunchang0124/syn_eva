import pandas as pd

from syneva import FairnessSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.fairness.statistical_parity import StatisticalParity


def _meta_gy():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def _biased(seed_a_pos, seed_b_pos):
    y = [1] * seed_a_pos + [0] * (50 - seed_a_pos) + [1] * seed_b_pos + [0] * (50 - seed_b_pos)
    return pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "y": y})


def test_preserved_bias_low_drift():
    real = _biased(40, 10)
    syn = _biased(40, 10)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars["parity_drift"] < 0.05
    assert r.scalars["score"] > 0.95


def test_destroyed_bias_high_drift():
    real = _biased(40, 10)
    syn = _biased(25, 25)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars["parity_drift"] > 0.5
    assert r.scalars["score"] < 0.5


def test_no_specs_score_one():
    df = _biased(40, 10)
    r = StatisticalParity(specs=[]).compute(df, df, _meta_gy())
    assert r.scalars["score"] == 1.0
