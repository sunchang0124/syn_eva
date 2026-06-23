import numpy as np
import pandas as pd

from syneva.congruence.extended.mmd import MMD
from syneva.core.metadata import Metadata


def test_identical_distribution_small_mmd():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    r = MMD().compute(df, df, Metadata.infer(df))
    assert r.scalars["mmd"] < 0.1
    assert r.scalars["score"] > 0.9


def test_shifted_distribution_larger_mmd():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    syn = pd.DataFrame({"x": rng.normal(loc=5, size=300), "y": rng.normal(loc=5, size=300)})
    r = MMD().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mmd"] > 0.3
    assert r.scalars["score"] < 0.7
