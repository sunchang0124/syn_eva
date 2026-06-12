import pandas as pd

from syneva.compliance.extended.dp_ledger import DPLedger
from syneva.core.metadata import Metadata


def test_no_ledger_reports_unknown():
    df = pd.DataFrame({"x": [1, 2]})
    r = DPLedger().compute(None, df, Metadata.infer(df))
    assert r.scalars["score"] == 1.0  # neutral when ledger absent
    assert any("no DP ledger" in n for n in r.notes)


def test_ledger_passes_through_epsilon():
    df = pd.DataFrame({"x": [1, 2]})
    df.attrs["dp_ledger"] = {"epsilon": 1.5, "delta": 1e-5}
    r = DPLedger().compute(None, df, Metadata.infer(df))
    assert r.scalars["epsilon"] == 1.5
    assert r.scalars["delta"] == 1e-5
