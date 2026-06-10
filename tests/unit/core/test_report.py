import json
import tempfile
from pathlib import Path

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.report import Report
from syneva.core.run_info import RunInfo


def _example_report() -> Report:
    spec = MetricSpec(
        name="ks",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    result = MetricResult(spec=spec, scalars={"ks": 0.1, "p": 0.5})
    return Report(metadata=meta, results=[result], run_info=RunInfo.capture(random_state=42))


def test_by_c_groups_results():
    r = _example_report()
    assert "congruence" in r.by_c
    assert len(r.by_c["congruence"]) == 1


def test_aggregated_returns_per_c_score():
    r = _example_report()
    agg = r.aggregated
    assert "congruence" in agg
    assert 0.0 <= agg["congruence"] <= 1.0


def test_to_dict_from_dict_round_trip():
    r = _example_report()
    d = r.to_dict()
    r2 = Report.from_dict(d)
    assert r2.results[0].spec.name == "ks"
    assert r2.results[0].scalars == {"ks": 0.1, "p": 0.5}


def test_to_json_writes_valid_file():
    r = _example_report()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        r.to_json(path)
        loaded = json.loads(path.read_text())
    assert loaded["results"][0]["spec"]["name"] == "ks"
