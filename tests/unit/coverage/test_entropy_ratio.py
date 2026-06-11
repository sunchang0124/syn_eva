import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.coverage.entropy_ratio import EntropyRatio


def _meta():
    return Metadata(columns={"sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL)})


def test_identical_distribution_ratio_one():
    df = pd.DataFrame({"sex": ["M", "F"] * 50})
    r = EntropyRatio().compute(df, df, _meta())
    assert abs(r.per_column["sex"]["ratio"] - 1.0) < 1e-9


def test_mode_collapse_low_ratio():
    real = pd.DataFrame({"sex": ["M", "F"] * 50})
    syn = pd.DataFrame({"sex": ["M"] * 100})
    r = EntropyRatio().compute(real, syn, _meta())
    assert r.per_column["sex"]["ratio"] < 0.1
