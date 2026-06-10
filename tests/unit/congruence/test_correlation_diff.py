# tests/unit/congruence/test_correlation_diff.py
import numpy as np
import pandas as pd

from syneva.congruence.correlation_diff import CorrelationDifference
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(
        columns={
            "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
            "b": ColumnMetadata(name="b", dtype=ColumnType.NUMERIC),
            "c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_identical_corr_matrix_score_perfect():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"a": a, "b": b, "c": ["X", "Y"] * 100})
    r = CorrelationDifference().compute(df, df, _meta())
    assert r.scalars["score"] >= 0.99


def test_destroyed_correlation_lowers_score():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.05, size=200)
    real = pd.DataFrame({"a": a, "b": b, "c": ["X"] * 200})
    syn = pd.DataFrame({"a": a, "b": rng.permutation(b), "c": ["X"] * 200})
    r = CorrelationDifference().compute(real, syn, _meta())
    assert r.scalars["score"] < 0.7
