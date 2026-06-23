import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metadata import Metadata
from syneva.core.neighbors import Neighbors


def _data():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)})
    syn = pd.DataFrame({"x": rng.normal(size=40), "y": rng.normal(size=40)})
    return real, syn, Metadata.infer(real)


def test_euclidean_matches_direct_sklearn():
    real, syn, meta = _data()
    nb = Neighbors(real, syn, meta, distance="euclidean")
    x_real, x_syn = encode_pair(real, syn, meta)
    exp_self = NearestNeighbors(n_neighbors=2).fit(x_real).kneighbors(x_real)[0][:, 1]
    assert np.allclose(nb.real_self(1), exp_self)
    exp_rs = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real)[0][:, 0]
    assert np.allclose(nb.real_to_syn(1), exp_rs)
    exp_sr = NearestNeighbors(n_neighbors=1).fit(x_real).kneighbors(x_syn)[0][:, 0]
    assert np.allclose(nb.syn_to_real(1), exp_sr)
    exp_idx = (
        NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real, return_distance=False)[:, 0]
    )
    assert np.array_equal(nb.real_to_syn_index(), exp_idx)
    assert nb.n_features == x_real.shape[1]


def test_gower_self_excludes_self_and_orders():
    real = pd.DataFrame({"n": [0.0, 1.0, 100.0], "c": ["a", "a", "b"]})
    syn = pd.DataFrame({"n": [0.0], "c": ["a"]})
    meta = Metadata.infer(real)
    nb = Neighbors(real, syn, meta, distance="gower")
    ds = nb.real_self(1)
    assert len(ds) == 3
    assert (ds >= 0).all()
    assert ds[2] > ds[0]


def test_invalid_distance_raises():
    real, syn, meta = _data()
    with pytest.raises(ValueError, match="distance"):
        Neighbors(real, syn, meta, distance="manhattan")


def test_euclidean_k2_matches_direct():
    real, syn, meta = _data()
    nb = Neighbors(real, syn, meta, distance="euclidean")
    x_real, _ = encode_pair(real, syn, meta)
    exp = NearestNeighbors(n_neighbors=3).fit(x_real).kneighbors(x_real)[0][:, 2]
    assert np.allclose(nb.real_self(2), exp)


def test_gower_capping_truncates_and_flags():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=50), "c": ["a", "b"] * 25})
    syn = pd.DataFrame({"x": rng.normal(size=50), "c": ["a", "b"] * 25})
    nb = Neighbors(real, syn, Metadata.infer(real), distance="gower", cap=10)
    assert nb.capped is True
    assert nb._nr == 10 and nb._ns == 10


def test_gower_k_too_large_raises():
    real = pd.DataFrame({"n": [0.0, 1.0, 2.0], "c": ["a", "b", "a"]})
    syn = pd.DataFrame({"n": [0.0], "c": ["a"]})
    nb = Neighbors(real, syn, Metadata.infer(real), distance="gower")
    with pytest.raises(ValueError, match="too large"):
        nb.real_self(3)  # only 2 other rows available


def test_holdout_to_real_euclidean():
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    syn = pd.DataFrame({"x": rng.normal(size=40), "y": rng.normal(size=40)})
    hold = pd.DataFrame({"x": rng.normal(size=20), "y": rng.normal(size=20)})
    meta = Metadata.infer(real)
    nb = Neighbors(real, syn, meta, distance="euclidean", holdout=hold)
    d = nb.holdout_to_real(1)
    assert d.shape == (20,)
    assert (d >= 0).all()
    ds = nb.holdout_self(1)
    assert ds.shape == (20,)


def test_holdout_methods_raise_without_holdout():
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"x": rng.normal(size=10)})
    syn = pd.DataFrame({"x": rng.normal(size=10)})
    nb = Neighbors(real, syn, Metadata.infer(real), distance="euclidean")
    with pytest.raises(ValueError, match="holdout"):
        nb.holdout_to_real(1)


def test_holdout_gower_runs():
    real = pd.DataFrame({"n": [0.0, 1.0, 2.0, 3.0], "c": ["a", "b", "a", "b"]})
    syn = pd.DataFrame({"n": [0.5, 1.5], "c": ["a", "b"]})
    hold = pd.DataFrame({"n": [0.2, 2.2], "c": ["a", "a"]})
    nb = Neighbors(real, syn, Metadata.infer(real), distance="gower", holdout=hold)
    assert nb.holdout_to_real(1).shape == (2,)
    assert nb.holdout_self(1).shape == (2,)


def test_holdout_to_real_euclidean_k_too_large_raises():
    real = pd.DataFrame({"x": [0.0, 1.0]})  # only 2 real rows
    syn = pd.DataFrame({"x": [0.5, 1.5, 2.5]})
    hold = pd.DataFrame({"x": [0.2, 0.8]})
    nb = Neighbors(real, syn, Metadata.infer(real), distance="euclidean", holdout=hold)
    with pytest.raises(ValueError, match="too large"):
        nb.holdout_to_real(3)  # k=3 > 2 real rows


def test_holdout_self_euclidean_k_too_large_raises():
    real = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    syn = pd.DataFrame({"x": [0.5, 1.5]})
    hold = pd.DataFrame({"x": [0.2, 0.8]})  # only 2 holdout rows
    nb = Neighbors(real, syn, Metadata.infer(real), distance="euclidean", holdout=hold)
    with pytest.raises(ValueError, match="too large"):
        nb.holdout_self(2)  # k=2 >= 2 holdout rows (needs a DIFFERENT other row)
