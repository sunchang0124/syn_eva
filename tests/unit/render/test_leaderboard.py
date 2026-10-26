import pandas as pd

import syneva
from syneva.render.html.benchmark_renderer import render_leaderboard


def _result():
    real = pd.read_parquet("tests/fixtures/adult_income_real_500.parquet")
    good = pd.read_parquet("tests/fixtures/adult_income_syn_good_500.parquet")
    shifted = pd.read_parquet("tests/fixtures/adult_income_syn_shifted_500.parquet")
    meta = syneva.Metadata.infer(real)
    return syneva.benchmark(real, {"good": good, "shifted": shifted}, meta)


def test_leaderboard_contains_candidates_and_sections():
    html = render_leaderboard(_result())
    assert html.lstrip().startswith("<!doctype html")
    assert "good" in html and "shifted" in html
    assert "leaderboard" in html.lower()
    assert "score-matrix" in html.lower()


def test_leaderboard_rank_order_matches_ranking():
    res = _result()
    html = render_leaderboard(res, normalization="linear")
    ranked = [name for _, name, _ in res.ranking(normalization="linear")]
    assert html.index(ranked[0]) < html.index(ranked[1])
    assert "linear" in html.lower()


def test_leaderboard_embeds_candidate_scorecards():
    html = render_leaderboard(_result())
    assert "<details" in html.lower()
    assert "summary" in html.lower()


def test_to_html_writes_file(tmp_path):
    res = _result()
    p = tmp_path / "lb.html"
    res.to_html(p)
    assert p.read_text(encoding="utf-8") == render_leaderboard(res)


def test_leaderboard_styles_embedded_scorecards():
    # the leaderboard <head> must define the scorecard card classes so embedded
    # drill-down scorecards are styled, not bare HTML
    html = render_leaderboard(_result())
    head = html.split("</head>")[0]
    for cls in (".metric", ".metric-header", ".metric-desc", ".breakdown-scroll"):
        assert cls in head, f"{cls} missing from leaderboard <head> styles"


def test_leaderboard_escapes_candidate_names_but_embeds_scorecards():
    from bs4 import BeautifulSoup

    real = pd.read_parquet("tests/fixtures/adult_income_real_500.parquet")
    good = pd.read_parquet("tests/fixtures/adult_income_syn_good_500.parquet")
    meta = syneva.Metadata.infer(real)
    res = syneva.benchmark(real, {"<i>evil</i>": good}, meta, preset="fast")
    soup = BeautifulSoup(render_leaderboard(res), "html.parser")
    assert soup.find("i") is None
    assert "<i>evil</i>" in soup.get_text()
    # the embedded per-candidate scorecard is trusted HTML and must still render
    assert soup.select_one("details.drill div.metric") is not None
