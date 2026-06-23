import numpy as np
import pandas as pd

from syneva.compliance.extended.epsilon_identifiability import EpsilonIdentifiability
from syneva.core.metadata import Metadata


def test_synthetic_equals_real_high_risk():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    r = EpsilonIdentifiability().compute(df, df, Metadata.infer(df))
    assert r.scalars["identifiability_risk"] > 0.9
    assert r.scalars["score"] < 0.1


def test_far_synthetic_low_risk():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    syn = pd.DataFrame({"x": rng.normal(loc=50, size=200), "y": rng.normal(loc=50, size=200)})
    r = EpsilonIdentifiability().compute(real, syn, Metadata.infer(real))
    assert r.scalars["identifiability_risk"] < 0.1
    assert r.scalars["score"] > 0.9
