import numpy as np
import pandas as pd
import pytest

from syneva.core.errors import MetricError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.tail_coverage import TailCoverage


def _meta():
    return Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})


def _real():
    return pd.DataFrame({"x": np.linspace(0.0, 1.0, 1001)})


def test_identical_data_covers_tails():
    res = TailCoverage().compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] > 0.9


def test_truncated_synthetic_scores_zero():
    syn = pd.DataFrame({"x": np.linspace(0.1, 0.9, 1001)})
    res = TailCoverage().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 0.0
    assert res.scalars["lower_tail_coverage"] == 0.0
    assert res.scalars["upper_tail_coverage"] == 0.0


def test_one_sided_truncation():
    syn = pd.DataFrame({"x": np.linspace(0.0, 0.9, 1001)})
    res = TailCoverage().compute(_real(), syn, _meta())
    assert res.scalars["upper_tail_coverage"] == 0.0
    assert res.scalars["lower_tail_coverage"] > 0.9


def test_constant_column_skipped_with_note():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    df = pd.DataFrame({"x": [1.0] * 100})
    res = TailCoverage().compute(df, df.copy(), meta)
    assert res.scalars["score"] == 1.0
    assert any("degenerate" in n for n in res.notes)


def test_no_numeric_columns_scores_one_with_note():
    meta = Metadata(columns={"c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL)})
    df = pd.DataFrame({"c": ["a", "b"]})
    res = TailCoverage().compute(df, df.copy(), meta)
    assert res.scalars["score"] == 1.0
    assert res.notes


def test_tail_quantile_zero_raises():
    with pytest.raises(MetricError):
        TailCoverage(tail_quantile=0.0).compute(_real(), _real().copy(), _meta())


def test_tail_quantile_half_raises():
    with pytest.raises(MetricError):
        TailCoverage(tail_quantile=0.5).compute(_real(), _real().copy(), _meta())
