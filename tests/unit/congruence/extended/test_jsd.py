import pandas as pd

from syneva.congruence.extended.jsd import JensenShannon
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(columns={"y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL)})


def test_identical_zero_jsd():
    df = pd.DataFrame({"y": ["a", "b"] * 50})
    r = JensenShannon().compute(df, df, _meta())
    assert r.per_column["y"]["jsd"] == 0.0


def test_diverged_positive_jsd():
    real = pd.DataFrame({"y": ["a"] * 100})
    syn = pd.DataFrame({"y": ["b"] * 100})
    r = JensenShannon().compute(real, syn, _meta())
    assert r.per_column["y"]["jsd"] > 0.5
