# tests/conftest.py
import json
from pathlib import Path

import pandas as pd
import pytest

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

FIXTURES = Path(__file__).parent / "fixtures"


def _load_metadata() -> Metadata:
    raw = json.loads((FIXTURES / "metadata.json").read_text())
    cols = {
        n: ColumnMetadata(
            name=n,
            dtype=ColumnType(c["dtype"]),
            sensitive=c.get("sensitive", False),
        )
        for n, c in raw["columns"].items()
    }
    return Metadata(columns=cols, primary_key=raw.get("primary_key"))


@pytest.fixture
def real_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "adult_income_real_500.parquet")


@pytest.fixture
def syn_good_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "adult_income_syn_good_500.parquet")


@pytest.fixture
def syn_shifted_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "adult_income_syn_shifted_500.parquet")


@pytest.fixture
def syn_leaky_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "adult_income_syn_leaky_100.parquet")


@pytest.fixture
def metadata() -> Metadata:
    return _load_metadata()


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="overwrite golden JSON snapshots with current output",
    )


@pytest.fixture
def update_golden(request) -> bool:
    return request.config.getoption("--update-golden")
