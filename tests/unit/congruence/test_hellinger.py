import pandas as pd

from syneva.congruence.hellinger import Hellinger
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta(cols):
    return Metadata(columns={n: ColumnMetadata(name=n, dtype=t) for n, t in cols.items()})


def test_identical_zero_distance():
    df = pd.DataFrame({"y": ["a", "b", "a", "b"] * 25, "x": list(range(100))})
    meta = _meta({"y": ColumnType.CATEGORICAL, "x": ColumnType.NUMERIC})
    r = Hellinger().compute(df, df, meta)
    assert r.scalars["mean_hellinger"] < 1e-9
    assert r.scalars["score"] > 0.99


def test_disjoint_categories_high_distance():
    real = pd.DataFrame({"y": ["a"] * 100})
    syn = pd.DataFrame({"y": ["b"] * 100})
    meta = _meta({"y": ColumnType.CATEGORICAL})
    r = Hellinger().compute(real, syn, meta)
    assert r.per_column["y"]["hellinger"] > 0.9
    assert r.scalars["score"] < 0.1
