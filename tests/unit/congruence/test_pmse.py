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
    # Equal sizes: the maximum pMSE is 0.25, and pMSE 0.2 maps to a score of 0.2.
    assert r.scalars["pmse"] > 0.2
    assert r.scalars["score"] < 0.2


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


# The maximum pMSE c(1-c) is worked out by hand. The unequal case (c = 0.25)
# tells it apart from wrong denominators that coincide at c = 0.5, such as c**2
# (0.0625) or (1-c)**2 (0.5625).
@pytest.mark.parametrize(
    ("n_real", "n_syn", "max_pmse"),
    [(500, 500, 0.25), (300, 100, 0.1875)],
)
def test_score_normalizes_by_maximum_pmse(n_real, n_syn, max_pmse):
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, n_real)})
    syn = pd.DataFrame({"x": rng.normal(1, 1, n_syn)})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    assert 0 < r.scalars["score"] < 1
    assert r.scalars["score"] == pytest.approx(1 - r.scalars["pmse"] / max_pmse)


@pytest.mark.parametrize("empty", ["real", "synthetic"])
def test_empty_dataset_is_skipped(empty):
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    real = data.iloc[:0] if empty == "real" else data
    syn = data.iloc[:0] if empty == "synthetic" else data
    r = PMSE().compute(real, syn, Metadata.infer(data))
    assert r.skip_reason is not None
    assert r.scalars is None
