import pandas as pd

from syneva.compliance.k_anonymity import KAnonymity
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(
        columns={
            "sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL, sensitive=True),
            "age": ColumnMetadata(name="age", dtype=ColumnType.NUMERIC, sensitive=True),
            "other": ColumnMetadata(name="other", dtype=ColumnType.NUMERIC),
        }
    )


def test_uniform_synthetic_high_k():
    syn = pd.DataFrame(
        {
            "sex": ["M"] * 100,
            "age": [30] * 100,
            "other": list(range(100)),
        }
    )
    r = KAnonymity().compute(None, syn, _meta())
    assert r.scalars["min_k"] == 100


def test_per_row_uniqueness_low_k():
    syn = pd.DataFrame(
        {
            "sex": ["M", "F"] * 50,
            "age": list(range(100)),
            "other": [0] * 100,
        }
    )
    r = KAnonymity().compute(None, syn, _meta())
    assert r.scalars["min_k"] == 1
