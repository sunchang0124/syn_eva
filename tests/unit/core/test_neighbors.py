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
