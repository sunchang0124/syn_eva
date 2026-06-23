import numpy as np
import pandas as pd

from syneva.congruence.ci_overlap import CIOverlap
from syneva.core.metadata import Metadata


def test_identical_data_full_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300)})
    r = CIOverlap().compute(df, df, Metadata.infer(df))
    assert r.scalars["mean_ci_overlap"] > 0.95
    assert r.scalars["score"] > 0.95


def test_disjoint_means_no_overlap():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(loc=0, scale=1, size=300)})
    syn = pd.DataFrame({"x": rng.normal(loc=50, scale=1, size=300)})
    r = CIOverlap().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_ci_overlap"] < 0.05
    assert r.scalars["score"] < 0.05
