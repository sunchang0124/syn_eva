import pandas as pd

from syneva.compliance.hitting_rate import HittingRate
from syneva.core.metadata import Metadata


def test_identical_synthetic_full_hit():
    df = pd.DataFrame({"x": list(range(100))})
    r = HittingRate().compute(df, df, Metadata.infer(df))
    assert r.scalars["hit_rate"] == 1.0
    assert r.scalars["score"] == 0.0


def test_disjoint_no_hits():
    real = pd.DataFrame({"x": list(range(100))})
    syn = pd.DataFrame({"x": list(range(1000, 1100))})
    r = HittingRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["hit_rate"] == 0.0
    assert r.scalars["score"] == 1.0
