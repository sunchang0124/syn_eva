# tests/unit/core/test_runner.py
import pandas as pd
import pytest

from syneva.core.errors import SchemaError, SynevaError
from syneva.core.metric import MetricResult, MetricSpec
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
