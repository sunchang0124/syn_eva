import numpy as np
import pandas as pd

from syneva.congruence.extended.sliced_wasserstein import SlicedWasserstein
from syneva.core.metadata import Metadata


def test_identical_data_small_distance():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    r = SlicedWasserstein().compute(df, df, Metadata.infer(df))
    assert r.scalars["sliced_wasserstein"] < 0.1
    assert r.scalars["score"] > 0.9


def test_shifted_data_larger_distance():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    syn = pd.DataFrame({"x": rng.normal(loc=5, size=200), "y": rng.normal(loc=5, size=200)})
    r = SlicedWasserstein().compute(real, syn, Metadata.infer(real))
    assert r.scalars["sliced_wasserstein"] > 1.0
