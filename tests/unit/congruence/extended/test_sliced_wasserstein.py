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


def test_large_scale_identical_distributions_score_high():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(50000, 10000, 1000)})
    syn = pd.DataFrame({"x": rng.normal(50000, 10000, 1000)})
    r = SlicedWasserstein().compute(real, syn, Metadata.infer(real))
    assert r.scalars["sliced_wasserstein"] < 0.1
    assert r.scalars["score"] > 0.9


def test_score_is_invariant_to_column_units():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    syn = pd.DataFrame({"x": rng.normal(0.5, size=500), "y": rng.normal(size=500)})
    base = SlicedWasserstein().compute(real, syn, Metadata.infer(real)).scalars
    scaled = SlicedWasserstein().compute(real * 1e4, syn * 1e4, Metadata.infer(real)).scalars
    assert np.isclose(base["sliced_wasserstein"], scaled["sliced_wasserstein"])
    assert np.isclose(base["score"], scaled["score"])


def test_large_scale_column_does_not_hide_shift_in_small_one():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"inc": rng.normal(50000, 10000, 1000), "age": rng.normal(40, 10, 1000)})
    same = pd.DataFrame({"inc": rng.normal(50000, 10000, 1000), "age": rng.normal(40, 10, 1000)})
    shifted = pd.DataFrame({"inc": rng.normal(50000, 10000, 1000), "age": rng.normal(80, 10, 1000)})
    meta = Metadata.infer(real)
    good = SlicedWasserstein().compute(real, same, meta).scalars["score"]
    bad = SlicedWasserstein().compute(real, shifted, meta).scalars["score"]
    assert good > 0.9
    assert bad < good - 0.3


def test_nan_rows_are_dropped_not_scored_zero():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    syn = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    real.loc[3, "y"] = np.nan
    syn.loc[[0, 7], "x"] = np.nan
    r = SlicedWasserstein().compute(real, syn, Metadata.infer(real))
    assert np.isfinite(r.scalars["sliced_wasserstein"])
    assert r.scalars["score"] > 0.9
    assert any("1 real" in n and "2 synthetic" in n for n in r.notes)


def test_nullable_int_with_pd_na():
    real = pd.DataFrame({"x": pd.array([1, 2, 3, None, 5] * 40, dtype="Int64")})
    r = SlicedWasserstein().compute(real, real.copy(), Metadata.infer(real))
    assert r.skip_reason is None
    assert r.scalars["sliced_wasserstein"] < 1e-9
    assert r.scalars["score"] == 1.0


def test_skips_when_no_complete_rows():
    real = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, 2.0, np.nan]})
    r = SlicedWasserstein().compute(real, real.copy(), Metadata.infer(real))
    assert r.skip_reason is not None
