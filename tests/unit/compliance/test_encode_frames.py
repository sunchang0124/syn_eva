import numpy as np
import pandas as pd

from syneva.compliance._encode import encode_frames, encode_pair
from syneva.core.metadata import Metadata


def test_two_frames_match_encode_pair():
    real = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    syn = pd.DataFrame({"x": [2.0, 3.0], "c": ["b", "a"]})
    meta = Metadata.infer(real)
    xr, xs = encode_pair(real, syn, meta)
    fr, fs = encode_frames([real, syn], meta)
    assert np.allclose(fr, xr)
    assert np.allclose(fs, xs)


def test_three_frames_share_space_identical_row_same_vector():
    real = pd.DataFrame({"x": [0.0, 10.0], "c": ["a", "b"]})
    syn = pd.DataFrame({"x": [5.0], "c": ["a"]})
    hold = pd.DataFrame({"x": [0.0], "c": ["a"]})  # identical to real row 0
    meta = Metadata.infer(real)
    xr, xs, xh = encode_frames([real, syn, hold], meta)
    assert xr.shape[1] == xs.shape[1] == xh.shape[1]
    assert np.allclose(xr[0], xh[0])  # same raw row -> same encoded vector
