import numpy as np
import pandas as pd
import pytest

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


def _ci_width(x: pd.Series) -> float:
    return 2 * 1.96 * float(x.std()) / np.sqrt(len(x))


def test_nested_interval_follows_karr_formula():
    # Same mean, 100x more synthetic rows: the synthetic CI sits inside the real one.
    # Karr et al. (2006): J = 0.5 * (overlap / w_real + overlap / w_syn), and overlap = w_syn.
    real = pd.DataFrame({"x": [-1.0, 1.0] * 100})
    syn = pd.DataFrame({"x": [-1.0, 1.0] * 10_000})
    w_r, w_s = _ci_width(real["x"]), _ci_width(syn["x"])
    expected = 0.5 * (w_s / w_r + 1.0)
    r = CIOverlap().compute(real, syn, Metadata.infer(real))
    assert r.per_column["x"]["ci_overlap"] == pytest.approx(expected)
    assert r.scalars["score"] == pytest.approx(expected)


def test_nested_interval_is_symmetric_in_real_and_synthetic():
    small = pd.DataFrame({"x": [-1.0, 1.0] * 100})
    large = pd.DataFrame({"x": [-1.0, 1.0] * 10_000})
    a = CIOverlap().compute(small, large, Metadata.infer(small)).scalars["score"]
    b = CIOverlap().compute(large, small, Metadata.infer(large)).scalars["score"]
    assert a == pytest.approx(b)


def test_partial_overlap_follows_karr_formula():
    real = pd.DataFrame({"x": [-1.0, 1.0] * 100})
    syn = pd.DataFrame({"x": [-0.85, 1.15] * 400})
    w_r, w_s = _ci_width(real["x"]), _ci_width(syn["x"])
    lo = max(-w_r / 2, 0.15 - w_s / 2)
    hi = min(w_r / 2, 0.15 + w_s / 2)
    assert lo < hi < 0.15 + w_s / 2  # genuinely partial, not nested
    overlap = hi - lo
    expected = 0.5 * (overlap / w_r + overlap / w_s)
    r = CIOverlap().compute(real, syn, Metadata.infer(real))
    assert r.per_column["x"]["ci_overlap"] == pytest.approx(expected)


def test_identical_constant_column_scores_one():
    df = pd.DataFrame({"x": [5.0] * 100})
    r = CIOverlap().compute(df, df, Metadata.infer(df))
    assert r.scalars["score"] == 1.0
    assert r.per_column["x"]["ci_overlap"] == 1.0


def test_constant_synthetic_against_varying_real_scores_zero():
    real = pd.DataFrame({"x": [-1.0, 1.0] * 100})
    syn = pd.DataFrame({"x": [0.0] * 200})
    r = CIOverlap().compute(real, syn, Metadata.infer(real))
    assert r.per_column["x"]["ci_overlap"] == 0.0
