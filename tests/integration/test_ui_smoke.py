import json

import pandas as pd

from syneva.core.metadata import Metadata
from syneva.ui import core


def test_ui_pipeline_end_to_end():
    real = core.load_table("tests/fixtures/adult_income_real_500.parquet")
    syn = core.load_table("tests/fixtures/adult_income_syn_good_500.parquet")
    assert isinstance(real, pd.DataFrame)

    rows = core.metadata_rows(Metadata.infer(real))
    rows[[r["column"] for r in rows].index("income")]["sensitive"] = True
    meta = core.metadata_from_editor(rows)

    names = ["ks_statistic", "tvd", "dcr", "k_anonymity"]
    report = core.run_report(real, syn, meta, names)
    assert {r.spec.name for r in report.results} == set(names)

    html = core.report_html_str(report)
    assert "<html" in html.lower()
    json.loads(core.report_json_str(report))
