# tests/unit/core/test_runner.py
import math

import numpy as np
import pandas as pd
import pytest

from syneva.core.errors import MetadataError, SchemaError, SynevaError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.presets import Preset
from syneva.core.registry import MetricRegistry
from syneva.core.runner import evaluate_with


class _OkMetric:
    spec = MetricSpec(
        name="ok",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta):
        return MetricResult(spec=self.spec, scalars={"score": 0.9})


class _BadMetric:
    spec = MetricSpec(
        name="bad",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta):
        raise ValueError("boom")


class _SynOnlyMetric:
    spec = MetricSpec(
        name="syn_only",
        c="constraint",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=False,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta):
        return MetricResult(spec=self.spec, scalars={"score": 1.0})


def _df():
    return pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})


def test_returns_report_with_results():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    rep = evaluate_with(reg, real=_df(), synthetic=_df(), tiers=("core",), data_type="static")
    assert len(rep.results) == 1
    assert rep.results[0].scalars == {"score": 0.9}


def test_metric_failure_is_isolated():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    reg.register(_BadMetric)
    rep = evaluate_with(reg, real=_df(), synthetic=_df(), tiers=("core",), data_type="static")
    by_name = {r.spec.name: r for r in rep.results}
    assert by_name["ok"].scalars == {"score": 0.9}
    assert by_name["bad"].error is not None
    assert "boom" in str(by_name["bad"].error)


def _metric(name: str, scalars: dict[str, float], *, scoring: bool = True):
    class _M:
        spec = MetricSpec(
            name=name,
            c="congruence",
            tier="core",
            data_types=frozenset({"static"}),
            requires_real=True,
            scope="table-level",
            scoring=scoring,
        )

        def compute(self, real, synthetic, meta):
            return MetricResult(spec=self.spec, scalars=dict(scalars))

    return _M


def _run(*metrics):
    reg = MetricRegistry()
    for m in metrics:
        reg.register(m)
    return evaluate_with(reg, real=_df(), synthetic=_df(), tiers=("core",), data_type="static")


@pytest.mark.parametrize(
    "scalars",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        # no "score" key: aggregation reads the first scalar as a distance, and
        # 1 - min(1, max(0, nan)) used to come out as a perfect 1.0
        {"distance": float("nan")},
    ],
)
def test_non_finite_score_is_recorded_as_error(scalars):
    rep = _run(_OkMetric, _metric("nan_metric", scalars))
    by_name = {r.spec.name: r for r in rep.results}
    assert by_name["nan_metric"].scalars is None
    assert "nan_metric returned a non-finite score" in str(by_name["nan_metric"].error)
    # before: NaN was clamped into the mean as 0 (or 1), so this was 0.45 or 0.95
    assert rep.aggregated == {"congruence": 0.9}


def test_non_finite_secondary_scalar_is_kept():
    rep = _run(_metric("p_nan", {"score": 0.5, "p_value": float("nan")}))
    r = rep.results[0]
    assert r.error is None
    assert math.isnan(r.scalars["p_value"])
    assert rep.aggregated == {"congruence": 0.5}


def test_non_scoring_metric_with_nan_is_kept():
    rep = _run(_metric("info", {"epsilon": float("nan")}, scoring=False))
    assert rep.results[0].error is None


def test_schema_mismatch_raises():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    a = _df()
    b = a.drop(columns=["y"])
    with pytest.raises(SchemaError):
        evaluate_with(reg, real=a, synthetic=b, tiers=("core",), data_type="static")


def test_real_none_with_no_syn_only_metric_raises():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    with pytest.raises(SynevaError):
        evaluate_with(reg, real=None, synthetic=_df(), tiers=("core",), data_type="static")


def test_real_none_runs_syn_only_metrics():
    reg = MetricRegistry()
    reg.register(_SynOnlyMetric)
    rep = evaluate_with(reg, real=None, synthetic=_df(), tiers=("core",), data_type="static")
    assert len(rep.results) == 1
    assert rep.results[0].spec.name == "syn_only"


def test_extended_utility_metrics_do_not_crash_on_instantiation():
    import pandas as pd

    import syneva  # noqa: F401  populate global registry
    from syneva.core.registry import MetricRegistry
    from syneva.core.registry import registry as g

    by = {c.spec.name: c for c in g.metrics()}
    reg = MetricRegistry()
    for n in ["discriminative_score", "feature_importance_spearman", "multi_target_utility"]:
        reg.register(by[n])
    df = pd.read_parquet("tests/fixtures/adult_income_real_500.parquet")
    syn = pd.read_parquet("tests/fixtures/adult_income_syn_good_500.parquet")
    rep = evaluate_with(reg, real=df, synthetic=syn, tiers=("core", "extended"), run_utility=True)
    # None of these should have errored during instantiation
    for r in rep.results:
        assert r.error is None, f"{r.spec.name} errored: {r.error}"


def test_feature_importance_receives_declared_utility_tasks(real_df, syn_good_df, metadata):
    # Regression for #16: the runner never passed a target, so the metric always skipped.
    import syneva  # noqa: F401  populate global registry
    from syneva.core.registry import registry as g
    from syneva.utility.task import UtilityTask

    by = {c.spec.name: c for c in g.metrics()}
    reg = MetricRegistry()
    reg.register(by["feature_importance_spearman"])
    rep = evaluate_with(
        reg,
        real=real_df,
        synthetic=syn_good_df,
        metadata=metadata,
        tiers=("extended",),
        run_utility=True,
        utility_tasks=[UtilityTask("high_income", "classification")],
    )
    (r,) = rep.results
    assert r.error is None
    assert r.skip_reason is None
    assert r.scalars is not None
    assert -1.0 <= r.scalars["spearman_rho"] <= 1.0
    assert set(r.per_column) == {"high_income"}


@pytest.mark.parametrize("empty", ["real", "synthetic"])
def test_empty_dataset_raises(empty):
    # zero rows would otherwise give a plausible-looking scorecard: some metrics
    # score 1.0 (nothing to leak), others skip and drop out of the aggregate
    reg = MetricRegistry()
    reg.register(_OkMetric)
    real = _df().iloc[:0] if empty == "real" else _df()
    syn = _df().iloc[:0] if empty == "synthetic" else _df()
    with pytest.raises(SynevaError, match=f"{empty} data has no rows"):
        evaluate_with(reg, real=real, synthetic=syn, tiers=("core",), data_type="static")


def test_empty_synthetic_raises_without_real():
    reg = MetricRegistry()
    reg.register(_SynOnlyMetric)
    with pytest.raises(SynevaError, match="synthetic data has no rows"):
        evaluate_with(reg, real=None, synthetic=_df().iloc[:0], tiers=("core",), data_type="static")


def test_empty_holdout_raises():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    with pytest.raises(SynevaError, match="holdout has no rows"):
        evaluate_with(
            reg,
            real=_df(),
            synthetic=_df(),
            holdout=_df().iloc[:0],
            tiers=("core",),
            data_type="static",
        )


def _mistyped_holdout():
    # numeric x read as object because one value is a placeholder string
    return pd.DataFrame({"x": [1, "unknown", 3], "y": ["a", "b", "c"]})


def test_mistyped_holdout_raises():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    with pytest.raises(MetadataError, match=r"holdout: column 'x' declared numeric"):
        evaluate_with(
            reg,
            real=_df(),
            synthetic=_df(),
            holdout=_mistyped_holdout(),
            tiers=("core",),
            data_type="static",
        )


def _metadata_with_x_as(dtype):
    return Metadata(
        columns={
            "x": ColumnMetadata(name="x", dtype=dtype),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_mistyped_holdout_raises_with_explicit_metadata():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    with pytest.raises(MetadataError, match=r"holdout: column 'x'"):
        evaluate_with(
            reg,
            real=_df(),
            synthetic=_df(),
            metadata=_metadata_with_x_as(ColumnType.NUMERIC),
            holdout=_mistyped_holdout(),
            tiers=("core",),
            data_type="static",
        )


def test_numeric_holdout_column_declared_categorical_is_accepted():
    reg = MetricRegistry()
    reg.register(_OkMetric)
    rep = evaluate_with(
        reg,
        real=_df(),
        synthetic=_df(),
        metadata=_metadata_with_x_as(ColumnType.CATEGORICAL),
        holdout=_df(),
        tiers=("core",),
        data_type="static",
    )
    assert len(rep.results) == 1


class _ScaledMetric:
    """Reports its `factor` option as the score, so a test can see what it was given."""

    spec = MetricSpec(
        name="scaled",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def __init__(self, factor: float = 0.5, random_state: int = 42) -> None:
        if not 0 <= factor <= 1:
            raise ValueError(f"factor must be in [0, 1], got {factor}")
        self.factor = factor

    def compute(self, real, synthetic, meta):
        return MetricResult(spec=self.spec, scalars={"score": self.factor})


def _run_with_options(*metrics, **kwargs):
    reg = MetricRegistry()
    for m in metrics:
        reg.register(m)
    if "preset" not in kwargs:
        kwargs.setdefault("tiers", ("core",))
    return evaluate_with(reg, real=_df(), synthetic=_df(), data_type="static", **kwargs)


def test_metric_options_reach_the_metric():
    rep = _run_with_options(_ScaledMetric, metric_options={"scaled": {"factor": 0.25}})
    assert rep.results[0].scalars == {"score": 0.25}


def test_metric_options_default_leaves_constructor_defaults():
    rep = _run_with_options(_ScaledMetric)
    assert rep.results[0].scalars == {"score": 0.5}


class _RecordingMetric:
    """Records that it ran, so a test can check validation happened before any metric."""

    ran = False
    spec = MetricSpec(
        name="recording",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta):
        type(self).ran = True
        return MetricResult(spec=self.spec, scalars={"score": 1.0})


def test_metric_options_unknown_metric_raises_before_running():
    _RecordingMetric.ran = False
    with pytest.raises(SynevaError, match=r"unknown metric 'scaledd'.*recording.*scaled"):
        _run_with_options(
            _RecordingMetric, _ScaledMetric, metric_options={"scaledd": {"factor": 0.25}}
        )
    assert _RecordingMetric.ran is False


def test_metric_options_unknown_option_raises_before_running():
    _RecordingMetric.ran = False
    with pytest.raises(SynevaError, match=r"'scaled' has no option 'facter'.*\['factor'\]"):
        _run_with_options(
            _RecordingMetric, _ScaledMetric, metric_options={"scaled": {"facter": 0.25}}
        )
    assert _RecordingMetric.ran is False


def test_metric_options_for_metric_without_options_lists_none():
    with pytest.raises(SynevaError, match=r"'recording' has no option 'x'.*accepts none"):
        _run_with_options(_RecordingMetric, metric_options={"recording": {"x": 1}})


def test_metric_options_runner_controlled_option_points_to_evaluate_argument():
    with pytest.raises(SynevaError, match=r"'random_state'.*evaluate\(\)'s own 'random_state'"):
        _run_with_options(_ScaledMetric, metric_options={"scaled": {"random_state": 1}})


def test_metric_options_fairness_specs_option_names_its_evaluate_argument():
    class _SpecMetric(_ScaledMetric):
        spec = MetricSpec(
            name="spec_metric",
            c="fairness",
            tier="core",
            data_types=frozenset({"static"}),
            requires_real=True,
            scope="table-level",
        )

        def __init__(self, specs=None) -> None:
            self.factor = 0.5

    with pytest.raises(SynevaError, match=r"'specs'.*evaluate\(\)'s own 'fairness_specs'"):
        _run_with_options(_SpecMetric, _ScaledMetric, metric_options={"spec_metric": {"specs": []}})


def test_metric_options_for_unselected_metric_are_allowed():
    class _ExtendedScaled(_ScaledMetric):
        spec = MetricSpec(
            name="extended_scaled",
            c="congruence",
            tier="extended",
            data_types=frozenset({"static"}),
            requires_real=True,
            scope="table-level",
        )

    rep = _run_with_options(
        _OkMetric, _ExtendedScaled, metric_options={"extended_scaled": {"factor": 0.25}}
    )
    assert [r.spec.name for r in rep.results] == ["ok"]


def test_metric_options_rejected_value_fails_only_that_metric():
    rep = _run_with_options(_OkMetric, _ScaledMetric, metric_options={"scaled": {"factor": 2.0}})
    by_name = {r.spec.name: r for r in rep.results}
    assert by_name["ok"].scalars == {"score": 0.9}
    assert by_name["scaled"].error is not None
    assert "factor must be in [0, 1]" in str(by_name["scaled"].error)


def test_metric_options_are_recorded_in_run_info():
    rep = _run_with_options(_ScaledMetric, metric_options={"scaled": {"factor": 0.25}})
    assert rep.run_info.metric_options == {"scaled": {"factor": 0.25}}


def test_run_info_metric_options_empty_by_default():
    rep = _run_with_options(_ScaledMetric)
    assert rep.run_info.metric_options == {}


class _TwoOptionMetric:
    """Scores factor + offset, so a test can see both options it was given."""

    spec = MetricSpec(
        name="two_option",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def __init__(self, factor: float = 0.5, offset: float = 0.0) -> None:
        self.factor = factor
        self.offset = offset

    def compute(self, real, synthetic, meta):
        return MetricResult(spec=self.spec, scalars={"score": self.factor + self.offset})


@pytest.fixture
def preset_with_options(monkeypatch):
    from syneva.core import presets

    preset = Preset(
        tiers=("core",),
        cs=None,
        run_utility=False,
        run_fairness=False,
        distance="euclidean",
        metric_options={"two_option": {"factor": 0.25, "offset": 0.125}},
    )
    monkeypatch.setitem(presets.PRESETS, "with_options", preset)
    return "with_options"


def test_preset_metric_options_are_used(preset_with_options):
    rep = _run_with_options(_TwoOptionMetric, preset=preset_with_options)
    assert rep.results[0].scalars == {"score": 0.375}
    assert rep.run_info.metric_options == {"two_option": {"factor": 0.25, "offset": 0.125}}


def test_explicit_metric_options_override_preset_per_option(preset_with_options):
    rep = _run_with_options(
        _TwoOptionMetric,
        preset=preset_with_options,
        metric_options={"two_option": {"offset": 0.5}},
    )
    assert rep.results[0].scalars == {"score": 0.75}
    assert rep.run_info.metric_options == {"two_option": {"factor": 0.25, "offset": 0.5}}


@pytest.mark.parametrize("value", [None, "factor", ["factor"]])
def test_metric_options_value_must_be_a_mapping(value):
    with pytest.raises(SynevaError, match=r"'scaled' must map option names to values"):
        _run_with_options(_ScaledMetric, metric_options={"scaled": value})


def test_metric_options_runner_controlled_message_lists_accepted_options():
    with pytest.raises(SynevaError, match=r"valid: \['factor'\]"):
        _run_with_options(_ScaledMetric, metric_options={"scaled": {"random_state": 1}})


def test_metric_options_numpy_scalars_are_recorded_as_plain_values():
    rep = _run_with_options(_ScaledMetric, metric_options={"scaled": {"factor": np.float64(0.25)}})
    recorded = rep.run_info.metric_options["scaled"]["factor"]
    assert recorded == 0.25
    assert type(recorded) is float
    assert rep.results[0].scalars == {"score": 0.25}


@pytest.mark.parametrize("value", [(0.25,), object(), float("nan"), float("inf")])
def test_metric_options_value_that_json_cannot_keep_raises(value):
    with pytest.raises(SynevaError, match=r"'scaled' option 'factor'.*JSON"):
        _run_with_options(_ScaledMetric, metric_options={"scaled": {"factor": value}})


def test_metric_options_accepts_any_name_for_kwargs_constructor():
    class _KwargsMetric(_ScaledMetric):
        spec = MetricSpec(
            name="kwargs_metric",
            c="congruence",
            tier="core",
            data_types=frozenset({"static"}),
            requires_real=True,
            scope="table-level",
        )

        def __init__(self, factor: float = 0.5, **extra) -> None:
            self.factor = extra.get("override", factor)

    rep = _run_with_options(_KwargsMetric, metric_options={"kwargs_metric": {"override": 0.75}})
    assert rep.results[0].scalars == {"score": 0.75}
