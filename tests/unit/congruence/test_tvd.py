# tests/unit/congruence/test_tvd.py
import pandas as pd

from syneva.congruence.tvd import TotalVariationDistance
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta() -> Metadata:
    return Metadata(
        columns={
            "sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL),
            "age": ColumnMetadata(name="age", dtype=ColumnType.NUMERIC),
        }
    )


def test_identical_categorical_distribution_score_perfect():
    df = pd.DataFrame({"sex": ["M", "F", "M", "F"] * 25, "age": [1] * 100})
    r = TotalVariationDistance().compute(df, df, _meta())
    assert r.scalars["score"] >= 0.99
    assert r.per_column["sex"]["tvd"] == 0.0


def test_mode_collapse_lowers_score():
    real = pd.DataFrame({"sex": ["M"] * 50 + ["F"] * 50, "age": [1] * 100})
    syn = pd.DataFrame({"sex": ["M"] * 100, "age": [1] * 100})
    r = TotalVariationDistance().compute(real, syn, _meta())
    assert r.per_column["sex"]["tvd"] == 0.5
    assert r.scalars["score"] <= 0.5


def test_unseen_synthetic_category_counts_against():
    real = pd.DataFrame({"sex": ["M"] * 50 + ["F"] * 50, "age": [1] * 100})
    syn = pd.DataFrame({"sex": ["X"] * 100, "age": [1] * 100})
    r = TotalVariationDistance().compute(real, syn, _meta())
    assert r.per_column["sex"]["tvd"] == 1.0


def test_skips_numeric_columns():
    df = pd.DataFrame({"sex": ["M"], "age": [1]})
    r = TotalVariationDistance().compute(df, df, _meta())
    assert "age" not in r.per_column
