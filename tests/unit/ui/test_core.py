import pandas as pd
import pytest

import syneva  # noqa: F401  populates the global registry on import
from syneva.core.metadata import ColumnType, Metadata
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


def test_metadata_rows_from_inferred():
    df = pd.DataFrame({"age": [1, 2], "sex": ["M", "F"]})
    rows = core.metadata_rows(Metadata.infer(df))
    by_col = {r["column"]: r for r in rows}
    assert by_col["age"]["dtype"] == "numeric"
    assert by_col["sex"]["dtype"] == "categorical"
    assert by_col["age"]["sensitive"] is False


def test_metadata_from_editor_roundtrip():
    rows = [
        {"column": "age", "dtype": "numeric", "sensitive": False},
        {"column": "sex", "dtype": "categorical", "sensitive": True},
    ]
    meta = core.metadata_from_editor(rows)
    assert meta.columns["age"].dtype is ColumnType.NUMERIC
    assert meta.columns["sex"].dtype is ColumnType.CATEGORICAL
    assert meta.columns["sex"].sensitive is True


def test_metric_catalog_groups_and_tiers():
    cat = core.metric_catalog()
    names = {m["name"] for m in cat}
    assert "ks_statistic" in names
    assert "sliced_wasserstein" in names
    ks = next(m for m in cat if m["name"] == "ks_statistic")
    assert ks["c"] == "congruence"
    assert ks["tier"] == "core"


def test_build_selection_registry_contains_only_selected():
    reg = core.build_selection_registry(["ks_statistic", "dcr"])
    selected = {c.spec.name for c in reg.metrics()}
    assert selected == {"ks_statistic", "dcr"}


def test_build_selection_registry_unknown_name_raises():
    with pytest.raises(KeyError):
        core.build_selection_registry(["does_not_exist"])
