import math

import pytest

from syneva.benchmark.normalize import normalize_across
from syneva.core.errors import SynevaError


def test_absolute_is_identity():
    assert normalize_across([0.2, 0.8, 0.5], "absolute") == [0.2, 0.8, 0.5]


def test_linear_min_max():
    assert normalize_across([0.2, 0.8, 0.5], "linear") == [0.0, 1.0, 0.5]


def test_linear_constant_all_ones():
    assert normalize_across([0.4, 0.4, 0.4], "linear") == [1.0, 1.0, 1.0]


def test_quantile_ranks():
    assert normalize_across([0.2, 0.8, 0.5], "quantile") == [0.0, 1.0, 0.5]


def test_quantile_handles_ties():
    # two tied lowest -> average rank 0.5 each, top -> rank 2
    out = normalize_across([0.3, 0.3, 0.9], "quantile")
    assert out == [0.25, 0.25, 1.0]


def test_quantile_single_is_half():
    assert normalize_across([0.7], "quantile") == [0.5]


def test_normal_zscores_sum_to_zero():
    out = normalize_across([0.2, 0.8, 0.5], "normal")
    assert math.isclose(sum(out), 0.0, abs_tol=1e-9)
    assert out[1] > out[2] > out[0]


def test_normal_constant_all_zero():
    assert normalize_across([0.4, 0.4, 0.4], "normal") == [0.0, 0.0, 0.0]


def test_unknown_mode_raises():
    with pytest.raises(SynevaError, match="normalization"):
        normalize_across([0.1, 0.2], "bogus")
