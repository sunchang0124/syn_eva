# tests/unit/coverage/test_category_coverage.py
import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.coverage.category_coverage import CategoryCoverage


def _meta():
    return Metadata(columns={"sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL)})


def test_full_coverage_score_one():
    real = pd.DataFrame({"sex": ["M", "F"] * 50})
    syn = pd.DataFrame({"sex": ["M", "F"] * 50})
    r = CategoryCoverage().compute(real, syn, _meta())
    assert r.per_column["sex"]["coverage"] == 1.0
    assert r.scalars["score"] == 1.0


def test_mode_collapse_lowers_coverage():
    real = pd.DataFrame({"sex": ["M", "F", "X"] * 30})
    syn = pd.DataFrame({"sex": ["M"] * 90})
    r = CategoryCoverage().compute(real, syn, _meta())
    assert r.per_column["sex"]["coverage"] == 1 / 3
