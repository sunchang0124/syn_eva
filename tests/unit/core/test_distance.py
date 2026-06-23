import numpy as np
import pandas as pd

from syneva.core.distance import gower_matrix
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(
        columns={
            "num": ColumnMetadata(name="num", dtype=ColumnType.NUMERIC),
            "cat": ColumnMetadata(name="cat", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_identical_rows_zero_distance():
    df = pd.DataFrame({"num": [0.0, 5.0, 10.0], "cat": ["x", "y", "x"]})
    d = gower_matrix(df, df, _meta())
    assert np.allclose(np.diag(d), 0.0)


def test_mixed_example_values():
    a = pd.DataFrame({"num": [0.0], "cat": ["x"]})
    b = pd.DataFrame({"num": [10.0, 5.0], "cat": ["y", "x"]})
    d = gower_matrix(a, b, _meta())
    assert d.shape == (1, 2)
    assert abs(d[0, 0] - 1.0) < 1e-9
    assert abs(d[0, 1] - 0.25) < 1e-9


def test_no_usable_columns_zero():
    df = pd.DataFrame({"id": ["a", "b"]})
    meta = Metadata(columns={"id": ColumnMetadata(name="id", dtype=ColumnType.ID)})
    d = gower_matrix(df, df, meta)
    assert d.shape == (2, 2)
    assert np.allclose(d, 0.0)
