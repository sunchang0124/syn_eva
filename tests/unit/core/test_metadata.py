# tests/unit/core/test_metadata.py
import pandas as pd
import pytest

from syneva.core.errors import MetadataError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def test_infer_basic_types():
    df = pd.DataFrame(
        {
            "age": [1, 2, 3],
            "sex": ["M", "F", "M"],
            "ts": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "active": [True, False, True],
            "patient_id": ["a", "b", "c"],
        }
    )
    m = Metadata.infer(df)
    assert m.columns["age"].dtype is ColumnType.NUMERIC
    assert m.columns["sex"].dtype is ColumnType.CATEGORICAL
    assert m.columns["ts"].dtype is ColumnType.DATETIME
    assert m.columns["active"].dtype is ColumnType.BOOLEAN
    assert m.columns["patient_id"].dtype is ColumnType.ID


def test_override_changes_dtype():
    df = pd.DataFrame({"x": [1, 2, 3]})
    m = Metadata.infer(df).override(x={"dtype": "categorical"})
    assert m.columns["x"].dtype is ColumnType.CATEGORICAL


def test_validate_rejects_dtype_mismatch():
    df = pd.DataFrame({"x": ["a", "b"]})
    m = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    with pytest.raises(MetadataError):
        m.validate_against(df)
