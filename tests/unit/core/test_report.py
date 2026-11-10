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
    assert r2.results[0].skip_reason is None


def test_skip_reason_round_trips_and_is_not_aggregated():
    r = _example_report()
    spec = r.results[0].spec
    r.results.append(MetricResult(spec=spec, skip_reason="needs a holdout"))
    r2 = Report.from_dict(r.to_dict())
    assert r2.results[1].skip_reason == "needs a holdout"
    assert r2.results[1].scalars is None
    assert set(r2.aggregated) == {"congruence"}


def test_from_dict_accepts_reports_without_skip_reason():
    d = _example_report().to_dict()
    del d["results"][0]["skip_reason"]
    assert Report.from_dict(d).results[0].skip_reason is None


def test_to_json_writes_valid_file():
    r = _example_report()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        r.to_json(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["results"][0]["spec"]["name"] == "ks"


def test_to_html_writes_utf8(tmp_path, cp1252_locale):
    from syneva.render.html.renderer import render_html

    r = _example_report()
    r.results[0].notes.append("年龄 ≥50K")
    path = tmp_path / "out.html"
    r.to_html(path)
    assert path.read_text(encoding="utf-8") == render_html(r)


def test_privacy_roles_round_trip():
    r = _example_report()
    r.metadata.columns["zip"] = ColumnMetadata(
        name="zip", dtype=ColumnType.CATEGORICAL, quasi_identifier=True
    )
    r.metadata.columns["income"] = ColumnMetadata(
        name="income", dtype=ColumnType.NUMERIC, sensitive=True
    )
    cols = Report.from_dict(json.loads(json.dumps(r.to_dict()))).metadata.columns
    assert (cols["zip"].quasi_identifier, cols["zip"].sensitive) == (True, False)
    assert (cols["income"].quasi_identifier, cols["income"].sensitive) == (False, True)


def test_from_dict_defaults_quasi_identifier_for_older_reports():
    d = _example_report().to_dict()
    for c in d["metadata"]["columns"].values():
        c.pop("quasi_identifier", None)
    assert Report.from_dict(d).metadata.columns["x"].quasi_identifier is False
