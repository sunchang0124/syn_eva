import json
import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, Tag

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


def _golden_report():
    path = Path(__file__).parents[2] / "golden" / "adult_income_good.scorecard.json"
    return Report.from_dict(json.loads(path.read_text()))


def _soup(rep, **render_kwargs):
    return BeautifulSoup(render_html(rep, **render_kwargs), "html.parser")


def _card(soup, name=None) -> Tag:
    """The metric card for `name`, or the first card when no name is given."""
    codes = soup.select("div.metric > code")
    code = next(c for c in codes if name is None or c.get_text() == name)
    card = code.find_parent("div", class_="metric")
    assert card is not None
    return card


def _breakdown_section(card: Tag) -> Tag | None:
    section = card.find("details", class_="breakdown")
    return section if isinstance(section, Tag) else None


def _only_breakdown(rep, **render_kwargs) -> Tag | None:
    return _breakdown_section(_card(_soup(rep, **render_kwargs)))


def _cells(section: Tag, selector: str) -> list[str]:
    return [el.get_text(strip=True) for el in section.select(selector)]


@pytest.mark.parametrize("mode", ["normalized", "actual"])
def test_breakdown_lists_every_per_column_key_in_order(mode):
    rep = _golden_report()
    soup = _soup(rep, score_mode=mode)
    with_breakdown = [r for r in rep.results if r.per_column]
    assert with_breakdown
    for r in with_breakdown:
        assert r.per_column is not None
        section = _breakdown_section(_card(soup, r.spec.name))
        assert section is not None, r.spec.name
        assert not section.has_attr("open")
        assert _cells(section, "tbody th") == list(r.per_column)


def test_breakdown_headers_use_scalar_labels():
    section = _breakdown_section(_card(_soup(_golden_report()), "ks_statistic"))
    assert section is not None
    assert _cells(section, "thead th") == ["", "KS statistic", "p-value"]


def _one_metric_report(**result_kwargs):
    spec = MetricSpec(
        name="subgroup_fidelity",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    return Report(
        metadata=meta,
        results=[MetricResult(spec=spec, **result_kwargs)],
        run_info=RunInfo.capture(random_state=0),
    )


def test_breakdown_unions_value_names_and_renders_non_finite():
    rep = _one_metric_report(
        scalars={"score": 0.5},
        per_column={
            "a": {"first": 0.25, "second": float("nan")},
            "b": {"second": float("inf"), "third": 0.5},
        },
    )
    section = _only_breakdown(rep)
    assert section is not None
    assert _cells(section, "thead th") == ["", "First", "Second", "Third"]
    rows = [_cells(tr, "td") for tr in section.select("tbody tr")]
    assert rows == [["0.2500", "nan", ""], ["", "inf", "0.5000"]]


@pytest.mark.parametrize(
    ("result_kwargs", "shown"),
    [
        ({"scalars": {"score": 0.5}, "per_column": None}, False),
        ({"scalars": {"score": 0.5}, "per_column": {}}, False),
        ({"skip_reason": "too few rows", "per_column": {"a": {"first": 0.1}}}, True),
        ({"error": MetricError("kaboom"), "per_column": {"a": {"first": 0.1}}}, False),
    ],
    ids=["none", "empty", "skipped", "errored"],
)
def test_breakdown_shown_only_for_non_errored_metrics_with_values(result_kwargs, shown):
    assert (_only_breakdown(_one_metric_report(**result_kwargs)) is not None) is shown


def test_breakdown_escapes_keys():
    rep = _one_metric_report(scalars={"score": 0.5}, per_column={"<b>x</b>": {"first": 0.1}})
    section = _only_breakdown(rep)
    assert section is not None
    assert section.find("b") is None
    assert _cells(section, "tbody th") == ["<b>x</b>"]


def test_breakdown_scrolls_on_screen_but_prints_in_full():
    soup = _soup(_one_metric_report(scalars={"score": 0.5}, per_column={"a": {"first": 0.1}}))
    section = _breakdown_section(_card(soup))
    assert section is not None
    assert section.select_one(".breakdown-scroll > table") is not None
    css = " ".join(style.get_text() for style in soup.select("style"))
    assert re.search(r"\.breakdown-scroll\s*\{[^}]*max-height:\s*\d", css)
    print_rules = re.search(r"@media print\s*\{(.*?)\}\s*\}", css, re.S)
    assert print_rules
    assert re.search(r"\.breakdown-scroll\s*\{[^}]*max-height:\s*none", print_rules[1])


def test_user_derived_text_is_escaped():
    rep = _one_metric_report(
        skip_reason="column '<script>alert(1)</script>' has no values",
        notes=["subgroup '<i>x</i>' skipped"],
    )
    card = _card(_soup(rep))
    assert card.find("script") is None
    assert card.find("i") is None
    assert "<script>alert(1)</script>" in card.get_text()
