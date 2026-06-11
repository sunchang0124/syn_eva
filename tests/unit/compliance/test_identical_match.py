import pandas as pd

from syneva.compliance.identical_match import IdenticalMatchRate
from syneva.core.metadata import Metadata


def test_no_overlap_score_one():
    real = pd.DataFrame({"x": list(range(100))})
    syn = pd.DataFrame({"x": list(range(100, 200))})
    r = IdenticalMatchRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["identical_match_rate"] == 0.0
    assert r.scalars["score"] == 1.0


def test_leaky_synthetic_fires(real_df, syn_leaky_df, metadata):
    r = IdenticalMatchRate().compute(real_df, syn_leaky_df, metadata)
    assert r.scalars["identical_match_rate"] >= 0.6
    assert r.scalars["score"] < 0.5
