import pandas as pd
import pytest

from syneva.ui import core

FIX = "tests/fixtures"


def test_load_table_parquet():
    df = core.load_table(f"{FIX}/adult_income_real_500.parquet")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500


def test_load_table_csv(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
    df = core.load_table(str(p))
    assert list(df.columns) == ["a", "b"]


def test_load_table_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("nope")
    with pytest.raises(ValueError, match="unsupported"):
        core.load_table(str(p))
