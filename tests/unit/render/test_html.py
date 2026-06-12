from bs4 import BeautifulSoup

from syneva.core.errors import MetricError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.report import Report
from syneva.core.run_info import RunInfo
from syneva.render.html.renderer import render_html


def _report():
    spec = MetricSpec(
        name="ks",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    return Report(
        metadata=meta,
        results=[MetricResult(spec=spec, scalars={"score": 0.9, "mean_ks_statistic": 0.05})],
        run_info=RunInfo.capture(random_state=42),
    )


def test_renders_h1_with_title():
    html = render_html(_report(), interactive=False)
    soup = BeautifulSoup(html, "html.parser")
    assert soup.find("h1") is not None
    assert "syneva" in soup.find("h1").text.lower()


def test_renders_each_c_section():
    html = render_html(_report(), interactive=False)
    assert "congruence" in html.lower()
    assert "score" in html.lower()


def test_metric_info_tooltip_rendered():
    from syneva.core.metric_info import describe_metric

    spec = MetricSpec(
        name="dcr",
        c="compliance",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    rep = Report(
        metadata=meta,
        results=[MetricResult(spec=spec, scalars={"score": 0.5})],
        run_info=RunInfo.capture(random_state=0),
    )
    html = render_html(rep, interactive=False)
    # the dcr description text is embedded as a hover tooltip (title attribute)
    assert describe_metric("dcr") in html
    assert 'class="info"' in html


def test_failed_metric_shown_with_error():
    spec = MetricSpec(
        name="bad",
        c="compliance",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    rep = _report()
    rep.results.append(MetricResult(spec=spec, error=MetricError("kaboom")))
    html = render_html(rep, interactive=False)
    assert "kaboom" in html
    assert "metric-failed" in html
