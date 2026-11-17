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


# The maximum pMSE c(1-c), where the score reaches 0, is worked out by hand. The
# unequal case (c = 0.25) tells it apart from wrong denominators that coincide at
# c = 0.5, such as c**2 (0.0625) or (1-c)**2 (0.5625).
@pytest.mark.parametrize(
    ("n_real", "n_syn", "max_pmse"),
    [(500, 500, 0.25), (300, 100, 0.1875)],
)
def test_score_normalizes_by_maximum_pmse(n_real, n_syn, max_pmse):
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, n_real)})
    syn = pd.DataFrame({"x": rng.normal(1, 1, n_syn)})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    s = r.scalars
    assert s is not None
    assert 0 < s["score"] < 1
    # rescaled between the permutation null (score 1) and the maximum (score 0)
    assert s["score"] == pytest.approx(
        1 - (s["pmse"] - s["pmse_null"]) / (max_pmse - s["pmse_null"])
    )


@pytest.mark.parametrize("empty", ["real", "synthetic"])
def test_empty_dataset_is_skipped(empty):
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    real = data.iloc[:0] if empty == "real" else data
    syn = data.iloc[:0] if empty == "synthetic" else data
    r = PMSE().compute(real, syn, Metadata.infer(data))
    assert r.skip_reason is not None
    assert r.scalars is None


def test_categories_align_when_category_sets_differ():
    # synthetic lacks the rare category "c": the one-hot columns must still line up
    # by category, not by position, so matching frequencies stay indistinguishable
    real = pd.DataFrame({"g": ["a", "b"] * 245 + ["c"] * 10})
    syn = pd.DataFrame({"g": ["b", "a"] * 250})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars["score"] > 0.9


def test_separable_categories_score_zero():
    real = pd.DataFrame({"g": ["a", "b"] * 250})
    syn = pd.DataFrame({"g": ["c", "d"] * 250})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars["score"] < 0.05


def test_no_encodable_columns_is_skipped():
    data = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=20)})
    r = PMSE().compute(data, data.copy(), Metadata.infer(data))
    assert r.scalars is None
    assert "numeric or categorical" in r.skip_reason


def _same_distribution(n, cats, seed=0):
    """Real and synthetic drawn from one distribution: x ~ N(0, 1) plus `cats`
    categorical columns with 30 equally likely levels."""
    rng = np.random.default_rng(seed)

    def make():
        d = {"x": rng.normal(0, 1, n)}
        for j in range(cats):
            d[f"c{j}"] = rng.integers(0, 30, n).astype(str)
        return pd.DataFrame(d)

    return make(), make()


# Issue #61: an in-sample model fits noise, so the uncorrected score of a perfect
# generator fell as the model grew (0.80 for 200 rows and 88 predictors).
@pytest.mark.parametrize(("n", "cats"), [(1000, 0), (200, 0), (1000, 3), (200, 3)])
def test_identically_distributed_data_scores_near_one(n, cats):
    real, syn = _same_distribution(n, cats)
    r = PMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars is not None
    assert r.scalars["score"] >= 0.95
    if cats:  # with one predictor the null pMSE is too variable for a tight band
        assert 0.8 <= r.scalars["pmse_ratio"] <= 1.25


def test_constant_features_score_one():
    # Every pooled feature is constant, so the model can only fit the intercept:
    # pMSE and its permutation null are both 0, and the data can't be told apart.
    data = pd.DataFrame({"x": [1.0] * 50, "g": ["a"] * 50})
    r = PMSE().compute(data, data.copy(), Metadata.infer(data))
    assert r.scalars is not None
    assert r.scalars["score"] == 1.0
    assert r.scalars["pmse_ratio"] == 1.0


def test_more_parameters_than_rows_is_skipped():
    # One category level per row: 39 independent dummies plus the intercept for 40
    # pooled rows. The model can fit any labelling, permuted ones included, so pMSE
    # matches its null even though real and synthetic share no category at all.
    real = pd.DataFrame({"g": [f"r{i}" for i in range(20)]})
    syn = pd.DataFrame({"g": [f"s{i}" for i in range(20)]})
    r = PMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars is None
    assert r.skip_reason is not None and "parameters" in r.skip_reason


def test_shift_still_lowers_the_score():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    same = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    shifted = same.assign(x=same["x"] + 1)
    meta = Metadata.infer(real)
    r_same = PMSE().compute(real, same, meta).scalars
    r_shift = PMSE().compute(real, shifted, meta).scalars
    assert r_same is not None and r_shift is not None
    # a 1-SD mean shift in one column: the classifier finds it well above the null
    assert r_shift["score"] < 0.9
    assert r_shift["score"] < r_same["score"]
    assert r_shift["pmse_ratio"] > 10


def test_same_random_state_is_reproducible():
    real, syn = _same_distribution(200, 1)
    meta = Metadata.infer(real)
    assert PMSE(random_state=7).compute(real, syn, meta).scalars == (
        PMSE(random_state=7).compute(real, syn, meta).scalars
    )


def test_zero_permutations_is_rejected():
    with pytest.raises(ValueError, match="n_permutations"):
        PMSE(n_permutations=0)
