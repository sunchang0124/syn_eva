import numpy as np
import pandas as pd

from syneva.congruence.extended.c2st import C2ST
from syneva.core.metadata import Metadata


def test_indistinguishable_data_auc_near_half():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    syn = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    r = C2ST().compute(real, syn, Metadata.infer(real))
    assert 0.45 <= r.scalars["c2st_auc"] <= 0.65


def test_separable_data_auc_near_one():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    syn = pd.DataFrame({"x": rng.normal(loc=5, size=500), "y": rng.normal(loc=5, size=500)})
    r = C2ST().compute(real, syn, Metadata.infer(real))
    assert r.scalars["c2st_auc"] > 0.95


def test_missing_rare_category_stays_near_half():
    # Real has a 1% category the synthetic data lacks; otherwise identical.
    rng = np.random.default_rng(0)
    n = 2000
    real = pd.DataFrame(
        {"c": rng.choice(list("ABCD"), n, p=[0.30, 0.01, 0.30, 0.39]), "x": rng.normal(size=n)}
    )
    p = np.array([0.30, 0.30, 0.39]) / 0.99
    syn = pd.DataFrame({"c": rng.choice(list("ACD"), n, p=p), "x": rng.normal(size=n)})
    r = C2ST().compute(real, syn, Metadata.infer(real))
    assert r.scalars["c2st_auc"] <= 0.56


def test_different_labels_with_same_count_are_separable():
    # {A, B} vs {B, C}: positional one-hot columns would make these look identical.
    rng = np.random.default_rng(0)
    n = 2000
    real = pd.DataFrame({"c": rng.choice(list("AB"), n), "x": rng.normal(size=n)})
    syn = pd.DataFrame({"c": rng.choice(list("BC"), n), "x": rng.normal(size=n)})
    r = C2ST().compute(real, syn, Metadata.infer(real))
    assert r.scalars["c2st_auc"] > 0.7
