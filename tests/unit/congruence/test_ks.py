# tests/unit/congruence/test_ks.py
import pandas as pd

from syneva.congruence.ks import KSStatistic
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta() -> Metadata:
    return Metadata(
        columns={
            "age": ColumnMetadata(name="age", dtype=ColumnType.NUMERIC),
            "sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_identical_distributions_score_perfect():
    df = pd.DataFrame({"age": [1.0, 2.0, 3.0, 4.0, 5.0], "sex": ["M"] * 5})
    r = KSStatistic().compute(df, df, _meta())
    assert r.scalars is not None
    assert r.scalars["score"] >= 0.99
    assert "age" in r.per_column
    assert r.per_column["age"]["ks_statistic"] == 0.0


def test_shifted_distribution_lowers_score():
    real = pd.DataFrame({"age": list(range(100)), "sex": ["M"] * 100})
    syn = pd.DataFrame({"age": [v + 50 for v in range(100)], "sex": ["M"] * 100})
    r = KSStatistic().compute(real, syn, _meta())
    assert r.scalars["score"] <= 0.5


def test_skips_non_numeric_columns():
    real = pd.DataFrame({"age": [1, 2, 3], "sex": ["M", "F", "M"]})
    syn = pd.DataFrame({"age": [1, 2, 3], "sex": ["M", "F", "F"]})
    r = KSStatistic().compute(real, syn, _meta())
    assert "sex" not in r.per_column
    assert "age" in r.per_column


def test_handles_all_nan_column(real_df, syn_good_df):
    syn = syn_good_df.copy()
    syn["age"] = float("nan")
    r = KSStatistic().compute(real_df, syn, Metadata.infer(real_df))
    assert any("all NaN" in n for n in r.notes)
