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


def test_duplicate_real_rows_still_flagged_when_copied():
    import numpy as np

    rng = np.random.default_rng(1)
    base = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    # real has every row duplicated (d_self == 0 for all); synthetic copies real
    real = pd.concat([base, base], ignore_index=True)
    r = EpsilonIdentifiability().compute(real, real, Metadata.infer(real))
    assert r.scalars["identifiability_risk"] > 0.9  # exact copies must be flagged
