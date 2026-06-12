import io
import json as _json

import pandas as pd
import pytest

import syneva  # noqa: F401  populates the global registry on import
from syneva.core.metadata import ColumnType, Metadata
from syneva.core.report import Report
from syneva.ui import core
from syneva.utility.task import UtilityTask

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


def test_load_table_file_like_with_name(tmp_path):
    p = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(p, index=False)

    class FakeUploadedFile(io.BytesIO):
        name = "data.csv"

    obj = FakeUploadedFile(p.read_bytes())
    df = core.load_table(obj)
    assert list(df.columns) == ["x"]
    assert len(df) == 2


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


def test_run_report_runs_only_selected_metrics():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    meta = Metadata.infer(real)
    rep = core.run_report(real, syn, meta, ["ks_statistic", "tvd"], utility_tasks=None)
    assert isinstance(rep, Report)
    assert {r.spec.name for r in rep.results} == {"ks_statistic", "tvd"}


def test_run_report_runs_utility_when_selected():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    meta = Metadata.infer(real)
    rep = core.run_report(
        real,
        syn,
        meta,
        ["tstr_suite"],
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert "tstr_suite" in by
    assert by["tstr_suite"].scalars["score"] > 0.0


def _small_report():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    return core.run_report(real, syn, Metadata.infer(real), ["ks_statistic"])


def test_report_html_str_returns_html():
    html = core.report_html_str(_small_report())
    assert "<html" in html.lower()
    assert "syneva" in html.lower()


def test_report_json_str_is_valid_json():
    data = _json.loads(core.report_json_str(_small_report()))
    assert data["results"]


def test_report_pdf_bytes_guarded():
    try:
        import weasyprint  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("weasyprint unavailable")
    pdf = core.report_pdf_bytes(_small_report())
    assert pdf[:4] == b"%PDF"
