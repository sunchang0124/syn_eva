import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.minority_class_density import MinorityClassDensity


def _meta():
    return Metadata(columns={"cat": ColumnMetadata(name="cat", dtype=ColumnType.CATEGORICAL)})


def _real():
    # minority class "B" at 10%
    return pd.DataFrame({"cat": ["A"] * 90 + ["B"] * 10})


def test_identical_scores_one():
    res = MinorityClassDensity().compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] == 1.0


def test_halved_minority_scores_about_half():
    syn = pd.DataFrame({"cat": ["A"] * 95 + ["B"] * 5})
    res = MinorityClassDensity().compute(_real(), syn, _meta())
    assert abs(res.scalars["score"] - 0.5) < 0.02


def test_doubled_minority_also_penalized():
    syn = pd.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
    res = MinorityClassDensity().compute(_real(), syn, _meta())
    assert abs(res.scalars["score"] - 0.5) < 0.02


def test_vanished_minority_scores_zero():
    syn = pd.DataFrame({"cat": ["A"] * 100})
    res = MinorityClassDensity().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 0.0
    assert res.scalars["worst_density_ratio"] == 0.0


def test_minority_tie_is_deterministic():
    real = pd.DataFrame({"cat": ["A"] * 40 + ["B"] * 30 + ["C"] * 30})
    syn = pd.DataFrame({"cat": ["A"] * 40 + ["B"] * 15 + ["C"] * 45})
    # tie between B and C -> sorted label "B" wins; B halved -> ratio 0.5
    res = MinorityClassDensity().compute(real, syn, _meta())
    assert abs(res.per_column["cat"]["density_ratio"] - 0.5) < 0.02


def test_no_categorical_columns_scores_one_with_note():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    df = pd.DataFrame({"x": [1.0, 2.0]})
    res = MinorityClassDensity().compute(df, df.copy(), meta)
    assert res.scalars["score"] == 1.0
    assert res.notes
