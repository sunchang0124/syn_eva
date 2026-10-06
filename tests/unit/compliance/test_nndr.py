import numpy as np
import pandas as pd
import pytest

from syneva.compliance.nndr import NNDR
from syneva.core.metadata import Metadata


def test_far_synthetic_balanced_ratio(real_df, syn_good_df, metadata):
    r = NNDR().compute(real_df, syn_good_df, metadata)
    assert r.scalars["score"] > 0.3


def test_leaky_synthetic_low_ratio(real_df, syn_leaky_df, metadata):
    r = NNDR().compute(real_df, syn_leaky_df, metadata)
    assert r.scalars["nndr_median"] < 0.3
    assert r.scalars["score"] < 0.5


def test_nndr_gower_backend_runs(real_df, syn_good_df, metadata):
    r = NNDR(distance="gower").compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["score"] <= 1.0


def test_nndr_holdout_baseline_runs(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.4, random_state=5)
    r = NNDR(holdout=holdout).compute(real_df, syn_good_df, metadata)
    assert "nndr_median_holdout" in r.scalars
    assert 0.0 <= r.scalars["score"] <= 1.0


def _numeric_frames(seed=0, n=200):
    rng = np.random.default_rng(seed)
    real = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    near = real + rng.normal(0, 0.01, real.shape)
    return real, near, Metadata.infer(real)


@pytest.mark.parametrize("distance", ["euclidean", "gower"])
def test_duplicating_synthetic_rows_does_not_improve_score(distance):
    real, near, meta = _numeric_frames()
    once = NNDR(distance=distance).compute(real, near, meta)
    twice = NNDR(distance=distance).compute(real, pd.concat([near, near]), meta)
    assert once.scalars["score"] < 0.3
    # euclidean scaling is fit on real+synthetic, so duplication shifts it slightly
    assert twice.scalars["score"] == pytest.approx(once.scalars["score"], rel=1e-3)


def test_ratio_is_first_over_second_real_neighbour():
    real = pd.DataFrame({"x": [0.0, 1.0, 3.0, 10.0]})
    syn = pd.DataFrame({"x": [0.25, 0.25, 0.25]})  # d1 = 0.25, d2 = 0.75
    r = NNDR().compute(real, syn, Metadata.infer(real))
    assert r.scalars["nndr_median"] == pytest.approx(1 / 3)


def test_copy_of_duplicated_real_row_scores_zero():
    real = pd.DataFrame({"x": [1.0, 1.0, 5.0, 9.0]})
    syn = pd.DataFrame({"x": [1.0, 1.0, 1.0]})  # d1 = d2 = 0
    r = NNDR().compute(real, syn, Metadata.infer(real))
    assert r.scalars["nndr_median"] == 0.0
    assert r.scalars["score"] == 0.0
