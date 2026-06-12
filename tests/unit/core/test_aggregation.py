from syneva.core.errors import MetricError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.report import Report
from syneva.core.run_info import RunInfo


def _spec(name, c):
    return MetricSpec(
        name=name,
        c=c,
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )


def test_aggregated_uses_score_key():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    results = [
        MetricResult(spec=_spec("a", "congruence"), scalars={"score": 0.8}),
        MetricResult(spec=_spec("b", "congruence"), scalars={"score": 0.6}),
        MetricResult(spec=_spec("c", "coverage"), scalars={"score": 0.9}),
    ]
    rep = Report(metadata=meta, results=results, run_info=RunInfo.capture(random_state=0))
    agg = rep.aggregated
    assert abs(agg["congruence"] - 0.7) < 1e-9
    assert abs(agg["coverage"] - 0.9) < 1e-9


def test_failed_metric_excluded():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    results = [
        MetricResult(spec=_spec("a", "congruence"), scalars={"score": 0.8}),
        MetricResult(spec=_spec("b", "congruence"), error=MetricError("boom")),
    ]
    rep = Report(metadata=meta, results=results, run_info=RunInfo.capture(random_state=0))
    assert abs(rep.aggregated["congruence"] - 0.8) < 1e-9
