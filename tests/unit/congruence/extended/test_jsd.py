import math

import pandas as pd
import pytest

from syneva.congruence.extended.jsd import JensenShannon
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(columns={"y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL)})


def test_identical_zero_jsd():
    df = pd.DataFrame({"y": ["a", "b"] * 50})
    r = JensenShannon().compute(df, df, _meta())
    assert r.per_column["y"]["jsd"] == 0.0


def test_disjoint_distributions_reach_maximum_jsd():
    real = pd.DataFrame({"y": ["a"] * 100})
    syn = pd.DataFrame({"y": ["b"] * 100})
    r = JensenShannon().compute(real, syn, _meta())
    assert r.per_column["y"]["jsd"] == pytest.approx(1.0)
    assert r.scalars["score"] == pytest.approx(0.0)


def test_jsd_uses_base_2():
    # P = (1, 0), Q = (1/2, 1/2), M = (3/4, 1/4).
    # JSD = 1/2 * log2(4/3) + 1/4 * log2(2/3) + 1/4 * log2(2) = 3/2 - 3/4 * log2(3)
    real = pd.DataFrame({"y": ["a"] * 100})
    syn = pd.DataFrame({"y": ["a", "b"] * 50})
    r = JensenShannon().compute(real, syn, _meta())
    assert r.per_column["y"]["jsd"] == pytest.approx(1.5 - 0.75 * math.log2(3))
