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


def test_metric_full_name_and_visible_description_rendered():
    from syneva.core.metric_info import describe_metric, display_name

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
    # full human-readable name is shown, the code appears as a secondary tag,
    # and the description is visible body text (not a hover-only tooltip).
    assert display_name("dcr") == "Distance to closest record"
    assert "Distance to closest record" in html
    assert "dcr" in html
    assert describe_metric("dcr") in html
    assert 'class="metric-desc"' in html


def test_actual_mode_shows_raw_value_not_verdict():
    spec = MetricSpec(
        name="ks_statistic",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    rep = Report(
        metadata=meta,
        results=[MetricResult(spec=spec, scalars={"score": 0.98, "mean_ks_statistic": 0.0170})],
        run_info=RunInfo.capture(random_state=0),
    )
    normalized = render_html(rep, score_mode="normalized")
    actual = render_html(rep, score_mode="actual")

    # normalized: 0-1 score headline (/ 1.00) with raw values under a collapsible toggle
    assert "/ 1.00" in normalized
    assert "Technical details" in normalized
    assert "normalized scores" in normalized.lower()
    # actual: the raw statistic is the visible headline; no /1.00 units, no collapse
    assert "0.0170" in actual
    assert "actual measured statistics" in actual.lower()
    assert "/ 1.00" not in actual
    assert "Technical details" not in actual
    # the raw-direction hint is correct for a KS distance: lower is better
    assert "How to read it" in actual
    assert "Lower is better" in actual


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


def _report_with_skipped():
    spec = MetricSpec(
        name="mia_auc",
        c="compliance",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    rep = _report()
    rep.results.append(MetricResult(spec=spec, skip_reason="needs a holdout"))
    return rep


def test_skipped_metric_shows_badge_and_reason():
    soup = BeautifulSoup(render_html(_report_with_skipped()), "html.parser")
    card = soup.find("code", string="mia_auc").find_parent("div", class_="metric")
    assert "metric-skipped" in card["class"]
    assert card.find(class_="v-skipped").get_text(strip=True) == "Skipped"
    assert "needs a holdout" in card.find(class_="skip-reason").get_text()


def test_skipped_metric_renders_in_actual_mode():
    soup = BeautifulSoup(render_html(_report_with_skipped(), score_mode="actual"), "html.parser")
    card = soup.find("code", string="mia_auc").find_parent("div", class_="metric")
    assert card.find(class_="v-skipped") is not None
    assert "needs a holdout" in card.find(class_="skip-reason").get_text()
