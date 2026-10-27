import pandas as pd
import pytest

from syneva.compliance.extended.attribute_disclosure import AttributeDisclosure
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta_sens():
    return Metadata(
        columns={
            "qi": ColumnMetadata(name="qi", dtype=ColumnType.NUMERIC),
            "secret": ColumnMetadata(name="secret", dtype=ColumnType.CATEGORICAL, sensitive=True),
        }
    )


def test_copied_synthetic_high_disclosure():
    qi = list(range(100))
    secret = ["A" if v < 50 else "B" for v in qi]
    df = pd.DataFrame({"qi": qi, "secret": secret})
    r = AttributeDisclosure().compute(df, df, _meta_sens())
    assert r.scalars["disclosure_rate"] > 0.9
    assert r.scalars["score"] < 0.1


def test_no_sensitive_columns_skips():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    r = AttributeDisclosure().compute(df, df, Metadata.infer(df))
    assert r.scalars is None
    assert "sensitive" in r.skip_reason


def test_attribute_disclosure_gower_backend_runs():
    import pandas as pd

    from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

    qi = list(range(60))
    secret = ["A" if v < 30 else "B" for v in qi]
    df = pd.DataFrame({"qi": qi, "secret": secret})
    meta = Metadata(
        columns={
            "qi": ColumnMetadata(name="qi", dtype=ColumnType.NUMERIC),
            "secret": ColumnMetadata(name="secret", dtype=ColumnType.CATEGORICAL, sensitive=True),
        }
    )
    r = AttributeDisclosure(distance="gower").compute(df, df, meta)
    assert r.scalars["disclosure_rate"] > 0.9


def _mixed_frames():
    import numpy as np

    rng = np.random.default_rng(7)

    def one(n):
        return pd.DataFrame(
            {
                "age": rng.integers(18, 90, n).astype(float),
                "job": rng.choice(list("abcde"), n),
                "income": rng.choice(["lo", "hi"], n),
                "score": rng.normal(50, 10, n),
            }
        )

    meta = Metadata(
        columns={
            "age": ColumnMetadata(name="age", dtype=ColumnType.NUMERIC),
            "job": ColumnMetadata(name="job", dtype=ColumnType.CATEGORICAL),
            "income": ColumnMetadata(name="income", dtype=ColumnType.CATEGORICAL, sensitive=True),
            "score": ColumnMetadata(name="score", dtype=ColumnType.NUMERIC, sensitive=True),
        }
    )
    return one(600), one(500), meta


def test_gower_compares_real_blocks_with_synthetic_only(monkeypatch):
    import syneva.core.distance as distance
    import syneva.core.neighbors as neighbors

    calls = []
    original = distance.gower_matrix

    def spy(a, b, *args, **kwargs):
        calls.append((len(a), len(b)))
        return original(a, b, *args, **kwargs)

    monkeypatch.setattr(distance, "gower_matrix", spy)
    monkeypatch.setattr(neighbors, "gower_matrix", spy)
    real, syn, meta = _mixed_frames()
    AttributeDisclosure(distance="gower").compute(real, syn, meta)
    assert calls
    # Never real x real or syn x syn, and never the whole real table at once.
    assert all(n_b == len(syn) and n_a < len(real) for n_a, n_b in calls)
    assert sum(n_a for n_a, _ in calls) == len(real)


def test_gower_disclosure_rate_on_mixed_data():
    real, syn, meta = _mixed_frames()
    r = AttributeDisclosure(distance="gower").compute(real, syn, meta)
    assert r.scalars is not None
    # Value from the full-matrix implementation before #31.
    assert r.scalars["disclosure_rate"] == pytest.approx(34 / 600)
    assert r.scalars["score"] == pytest.approx(1 - 34 / 600)
    assert not r.notes
