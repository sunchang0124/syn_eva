import numpy as np
import pandas as pd
import pytest

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


def test_separable_unequal_sizes_scores_zero():
    # Issue #21: with c = 100/1100 the maximum pMSE is c(1-c) ~= 0.083, which a
    # fixed x8 rescale maps to a score of ~0.35 instead of 0.
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    syn = pd.DataFrame({"x": rng.normal(100, 1, 100)})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    c = 100 / 1100
    assert r.scalars["pmse"] > 0.95 * c * (1 - c)
    assert r.scalars["score"] < 0.05


def test_score_normalizes_by_maximum_pmse():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    syn = pd.DataFrame({"x": rng.normal(1, 1, 500)})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    # Equal sizes: c = 0.5, so the maximum pMSE is 0.25.
    assert r.scalars["score"] == pytest.approx(1 - r.scalars["pmse"] / 0.25)
