import pandas as pd

import syneva
from syneva.render.html.renderer import render_html


def _report():
    real = pd.read_parquet("tests/fixtures/adult_income_real_500.parquet")
    syn = pd.read_parquet("tests/fixtures/adult_income_syn_good_500.parquet")
    return syneva.evaluate(real, syn, syneva.Metadata.infer(real))


def test_default_is_full_document():
    html = render_html(_report())
    assert html.lstrip().startswith("<!doctype html")
    assert "</html>" in html


def test_fragment_omits_doctype_and_head():
    html = render_html(_report(), fragment=True)
    assert "<!doctype html" not in html.lower()
    assert "<head>" not in html.lower()
    assert "</html>" not in html.lower()
    assert "syneva scorecard" in html or "summary" in html
