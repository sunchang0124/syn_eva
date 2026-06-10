import numpy as np
import pandas as pd

from syneva.congruence.pmse import PMSE
from syneva.core.metadata import Metadata


def test_indistinguishable_data_low_pmse():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    syn = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars["pmse"] < 0.02
    assert r.scalars["score"] > 0.9


def test_separable_data_high_pmse():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    syn = pd.DataFrame({"x": rng.normal(5, 1, 500), "y": rng.normal(5, 1, 500)})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars["pmse"] > 0.05
    assert r.scalars["score"] < 0.5
