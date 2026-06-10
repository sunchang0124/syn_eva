import pandas as pd

from syneva.congruence.wasserstein import Wasserstein1
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta_num():
    return Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})


def test_identical_distribution_zero_distance():
    df = pd.DataFrame({"x": list(range(100))})
    r = Wasserstein1().compute(df, df, _meta_num())
    assert r.per_column["x"]["wasserstein"] == 0.0
    assert r.scalars["score"] >= 0.99


def test_shift_increases_distance():
    real = pd.DataFrame({"x": list(range(100))})
    syn = pd.DataFrame({"x": [v + 10 for v in range(100)]})
    r = Wasserstein1().compute(real, syn, _meta_num())
    assert abs(r.per_column["x"]["wasserstein"] - 10.0) < 0.5
