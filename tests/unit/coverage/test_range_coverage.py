# tests/unit/coverage/test_range_coverage.py
import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.coverage.range_coverage import RangeCoverage


def _meta():
    return Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})


def test_identical_range_score_one():
    df = pd.DataFrame({"x": list(range(100))})
    r = RangeCoverage().compute(df, df, _meta())
    assert r.per_column["x"]["coverage"] == 1.0


def test_narrow_synthetic_range_lowers_coverage():
    real = pd.DataFrame({"x": list(range(100))})
    syn = pd.DataFrame({"x": list(range(20, 50))})
    r = RangeCoverage().compute(real, syn, _meta())
    assert r.per_column["x"]["coverage"] < 0.5
