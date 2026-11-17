import io
import json as _json

import pandas as pd
import pytest

import syneva  # noqa: F401  populates the global registry on import
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
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
        {"column": "age", "dtype": "numeric", "quasi_identifier": True, "sensitive": False},
        {"column": "sex", "dtype": "categorical", "quasi_identifier": False, "sensitive": True},
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


def test_run_report_runs_fairness_when_selected():
    import pandas as pd

    from syneva import FairnessSpec
    from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

    y = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
    df = pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "y": y})
    meta = Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )
    rep = core.run_report(
        df, df, meta, ["statistical_parity"], fairness_specs=[FairnessSpec("g", "y")]
    )
    by = {r.spec.name: r for r in rep.results}
    assert "statistical_parity" in by
    assert by["statistical_parity"].error is None


def test_run_report_accepts_holdout():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    hold = real.sample(frac=0.3, random_state=1)
    rep = core.run_report(real, syn, Metadata.infer(real), ["dcr"], holdout=hold)
    by = {r.spec.name: r for r in rep.results}
    assert "p05_dcr_holdout" in by["dcr"].scalars


def test_run_report_preset_uses_default_registry():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    # preset drives selection even though selected_names is empty
    rep = core.run_report(real, syn, Metadata.infer(real), [], preset="fast")
    assert len(rep.results) > 0
    assert all(r.spec.tier == "core" for r in rep.results)


def test_run_benchmark_ranks_good_first():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    good = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    shifted = pd.read_parquet(f"{FIX}/adult_income_syn_shifted_500.parquet")
    res = core.run_benchmark(real, {"good": good, "shifted": shifted}, Metadata.infer(real))
    ranked = [name for _, name, _ in res.ranking()]
    assert ranked[0] == "good"


def test_run_benchmark_runs_every_candidate_at_the_given_random_state():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    good = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    res = core.run_benchmark(
        real, {"good": good}, Metadata.infer(real), preset="fast", random_state=7
    )
    assert res.random_state == 7
    assert res.reports["good"].run_info.random_state == 7


def test_leaderboard_html_str_matches_renderer():
    from syneva.render.html.benchmark_renderer import render_leaderboard

    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    good = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    res = core.run_benchmark(real, {"good": good}, Metadata.infer(real))
    html = core.leaderboard_html_str(res, normalization="linear")
    assert html == render_leaderboard(res, normalization="linear")
    assert "good" in html


def test_benchmark_json_str_roundtrips():
    from syneva.benchmark.engine import BenchmarkResult

    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    good = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    res = core.run_benchmark(real, {"good": good}, Metadata.infer(real))
    s = core.benchmark_json_str(res)
    import json

    restored = BenchmarkResult.from_dict(json.loads(s))
    assert restored.ranking() == res.ranking()


def test_metadata_editor_carries_quasi_identifier():
    rows = core.metadata_rows(
        Metadata(
            columns={
                "zip": ColumnMetadata(
                    name="zip", dtype=ColumnType.CATEGORICAL, quasi_identifier=True
                ),
                "income": ColumnMetadata(name="income", dtype=ColumnType.NUMERIC, sensitive=True),
            }
        )
    )
    by_col = {r["column"]: r for r in rows}
    assert by_col["zip"]["quasi_identifier"] is True
    assert by_col["income"]["quasi_identifier"] is False
    cols = core.metadata_from_editor(rows).columns
    assert (cols["zip"].quasi_identifier, cols["zip"].sensitive) == (True, False)
    assert (cols["income"].quasi_identifier, cols["income"].sensitive) == (False, True)


@pytest.mark.parametrize("dtype", list(ColumnType))
def test_subgroup_condition_kind_builds_conditions_the_spec_accepts(dtype):
    from syneva import SubgroupSpec

    kind = core.subgroup_condition_kind(dtype)
    condition = (0.0, 1.0) if kind == "range" else ["a"]
    meta = Metadata(columns={"c": ColumnMetadata(name="c", dtype=dtype)})
    assert SubgroupSpec(name="g", conditions={"c": condition}).validate(meta) is None


def test_edited_column_types_fall_back_to_the_inferred_type():
    df = pd.DataFrame({"age": [30, 40], "sex": ["M", "F"], "ok": [True, False]})
    inferred = core.metadata_rows(Metadata.infer(df))
    rows = [dict(r) for r in inferred]
    rows[0]["dtype"] = "categorical"  # an edit: taken
    rows[1]["dtype"] = None  # a cleared cell: ignored
    rows[2]["column"] = "renamed"  # a renamed row: ignored
    types = core.edited_column_types(rows, inferred)
    assert types == {
        "age": ColumnType.CATEGORICAL,
        "sex": ColumnType.CATEGORICAL,
        "ok": ColumnType.BOOLEAN,
    }


def _report_with_nan_secondary_scalar() -> Report:
    from syneva.core.metric import MetricResult, MetricSpec
    from syneva.core.run_info import RunInfo

    spec = MetricSpec(
        name="m",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    result = MetricResult(spec=spec, scalars={"score": 0.5, "p": float("nan")})
    return Report(metadata=meta, results=[result], run_info=RunInfo.capture(random_state=42))


def _strict_loads(text: str):
    """Parse JSON the way strict parsers do: bare NaN / Infinity tokens are errors."""

    def reject(token):
        raise ValueError(f"invalid JSON token {token}")

    return _json.loads(text, parse_constant=reject)


# Issue #109: the JSON downloads used bare NaN tokens, which strict parsers reject.
def test_report_json_str_is_strict_json_with_null_for_nan():
    data = _strict_loads(core.report_json_str(_report_with_nan_secondary_scalar()))
    assert data["results"][0]["scalars"]["p"] is None


def test_benchmark_json_str_is_strict_json_with_null_for_nan():
    from syneva.benchmark.engine import BenchmarkResult

    res = BenchmarkResult(reports={"A": _report_with_nan_secondary_scalar()})
    data = _strict_loads(core.benchmark_json_str(res))
    assert data["reports"]["A"]["results"][0]["scalars"]["p"] is None


def test_metric_option_catalog_lists_known_options():
    by_key = {(o["metric"], o["option"]): o for o in core.metric_option_catalog()}
    use_real_se = by_key[("ci_overlap", "use_real_se")]
    assert use_real_se["kind"] == "bool"
    assert use_real_se["default"] is False
    assert use_real_se["display_name"] == "Confidence-interval overlap"
    assert "standard error" in use_real_se["label"]
    assert "0.8" in use_real_se["help"]
    n_permutations = by_key[("pmse", "n_permutations")]
    assert n_permutations["kind"] == "int"
    assert n_permutations["default"] == 20
    assert n_permutations["label"] and n_permutations["help"]


_USE_REAL_SE = {"ci_overlap": {"use_real_se": True}}


def _small_frames():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet").head(100)
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet").head(100)
    return real, syn, Metadata.infer(real)


def test_run_report_passes_metric_options_for_a_manual_selection():
    real, syn, meta = _small_frames()
    rep = core.run_report(real, syn, meta, ["ci_overlap"], metric_options=_USE_REAL_SE)
    assert rep.run_info.metric_options == _USE_REAL_SE


def test_run_report_passes_metric_options_with_a_preset():
    real, syn, meta = _small_frames()
    rep = core.run_report(real, syn, meta, [], preset="fast", metric_options=_USE_REAL_SE)
    assert rep.run_info.metric_options == _USE_REAL_SE


def test_run_benchmark_passes_metric_options_to_every_candidate():
    real, syn, meta = _small_frames()
    res = core.run_benchmark(
        real, {"a": syn, "b": syn}, meta, preset="fast", metric_options=_USE_REAL_SE
    )
    assert all(r.run_info.metric_options == _USE_REAL_SE for r in res.reports.values())


def test_run_report_ignores_options_for_metrics_not_selected():
    # The sidebar shows every option whatever is ticked, and says an option is
    # used only when its metric runs; the run must not fail over an unticked one.
    real, syn, meta = _small_frames()
    rep = core.run_report(real, syn, meta, ["ks_statistic"], metric_options=_USE_REAL_SE)
    assert [r.spec.name for r in rep.results] == ["ks_statistic"]
    assert rep.run_info.metric_options == {}


def test_describe_metric_options_names_only_metrics_that_ran():
    # A preset can leave out a metric whose option was set; the results must not
    # claim that option was used.
    options = {"ci_overlap": {"use_real_se": True}, "pmse": {"n_permutations": 50}}
    assert core.describe_metric_options(options, ran={"pmse", "dcr"}) == "pmse.n_permutations = 50"
    assert core.describe_metric_options(options, ran={"dcr"}) == ""


def test_metric_option_catalog_keeps_n_permutations_at_one_or_more():
    by_key = {(o["metric"], o["option"]): o for o in core.metric_option_catalog()}
    assert by_key[("pmse", "n_permutations")]["min_value"] == 1
    assert by_key[("ci_overlap", "use_real_se")]["min_value"] is None
