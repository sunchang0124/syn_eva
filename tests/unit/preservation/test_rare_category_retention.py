import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.rare_category_retention import RareCategoryRetention


def _meta():
    return Metadata(columns={"cat": ColumnMetadata(name="cat", dtype=ColumnType.CATEGORICAL)})


def _real():
    # "B" has frequency 4% < 5% threshold -> rare
    return pd.DataFrame({"cat": ["A"] * 96 + ["B"] * 4})


def test_rare_category_kept_scores_one():
    res = RareCategoryRetention().compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert res.scalars["n_rare_categories"] == 1.0
    assert res.scalars["pct_rare_missing"] == 0.0


def test_rare_category_dropped_scores_zero():
    syn = pd.DataFrame({"cat": ["A"] * 100})
    res = RareCategoryRetention().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 0.0
    assert res.scalars["pct_rare_missing"] == 1.0


def test_rare_category_halved_scores_half():
    syn = pd.DataFrame({"cat": ["A"] * 98 + ["B"] * 2})
    res = RareCategoryRetention().compute(_real(), syn, _meta())
    assert abs(res.scalars["score"] - 0.5) < 1e-9


def test_oversampled_rare_category_capped_at_one():
    syn = pd.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
    res = RareCategoryRetention().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 1.0


def test_no_rare_categories_is_skipped():
    real = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 50})
    res = RareCategoryRetention().compute(real, real.copy(), _meta())
    assert res.scalars is None
    assert "no rare categories" in res.skip_reason


def test_unused_category_level_excluded():
    real = pd.DataFrame(
        {"cat": pd.Categorical(["A"] * 90 + ["B"] * 10, categories=["A", "B", "C"])}
    )
    res = RareCategoryRetention().compute(real, real.copy(), _meta())
    # the unused level C is not a lost rare category, so nothing is rare
    assert res.scalars is None
    assert res.skip_reason


def test_no_categorical_columns_is_skipped():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    res = RareCategoryRetention().compute(df, df.copy(), meta)
    assert res.scalars is None
    assert res.skip_reason
