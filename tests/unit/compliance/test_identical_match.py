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


def test_reordered_columns_still_match():
    real = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})
    syn = real[["y", "x"]]
    r = IdenticalMatchRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["identical_match_rate"] == 1.0
    assert r.scalars["score"] == 0.0


def test_nan_rows_match():
    real = pd.DataFrame({"x": [1.0, float("nan")], "y": ["a", None]})
    r = IdenticalMatchRate().compute(real, real.copy(), Metadata.infer(real))
    assert r.scalars["identical_match_rate"] == 1.0


def test_id_columns_ignored():
    real = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    syn = real.assign(id=[101, 102, 103])
    r = IdenticalMatchRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["identical_match_rate"] == 1.0
