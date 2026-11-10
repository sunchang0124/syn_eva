import pandas as pd
import pytest

from syneva.compliance.extended.attribute_disclosure import AttributeDisclosure
from syneva.core.distance import GOWER_BLOCK_ROWS
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

    def make_frame(n):
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
    return make_frame(600), make_frame(500), meta


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
    # Every call is one block of real rows against all synthetic rows; 600 real and 500
    # synthetic rows both exceed a block, so a real x real or syn x syn call would fail.
    assert len(real) > GOWER_BLOCK_ROWS and len(syn) > GOWER_BLOCK_ROWS
    assert all(n_a <= GOWER_BLOCK_ROWS and n_b == len(syn) for n_a, n_b in calls)
    assert sum(n_a for n_a, _ in calls) == len(real)


def test_gower_disclosure_rate_on_mixed_data():
    real, syn, meta = _mixed_frames()
    r = AttributeDisclosure(distance="gower").compute(real, syn, meta)
    assert r.scalars is not None
    # Value from the full-matrix implementation before #31.
    assert r.scalars["disclosure_rate"] == pytest.approx(34 / 600)
    assert r.scalars["score"] == pytest.approx(1 - 34 / 600)
    assert not r.notes


def test_unknown_distance_raises_even_with_empty_synthetic():
    df = pd.DataFrame({"qi": [1, 2, 3], "secret": ["A", "B", "A"]})
    with pytest.raises(ValueError, match="unknown distance"):
        AttributeDisclosure(distance="bogus").compute(df, df.iloc[:0], _meta_sens())


_NOISE = [f"noise{i}" for i in range(6)]


def _linkage_frames():
    # `zip` fully determines `diagnosis`. The six noise columns are shuffled in the
    # synthetic copy, so they pull the nearest match towards a random row: only `zip`
    # links a real row to a synthetic row with the same diagnosis.
    import numpy as np

    rng = np.random.default_rng(3)
    n = 200
    zip_ = rng.choice(["a", "b", "c", "d"], n)
    real = pd.DataFrame({"zip": zip_, "diagnosis": np.where(np.isin(zip_, ["a", "b"]), "x", "y")})
    for c in _NOISE:
        real[c] = rng.normal(0, 1, n)
    syn = real.copy()
    for c in _NOISE:
        syn[c] = rng.permutation(syn[c].to_numpy())
    return real, syn


def _linkage_meta(zip_is_qi: bool, diagnosis_is_qi: bool = False) -> Metadata:
    cols = {
        "zip": ColumnMetadata(name="zip", dtype=ColumnType.CATEGORICAL, quasi_identifier=zip_is_qi),
        "diagnosis": ColumnMetadata(
            name="diagnosis",
            dtype=ColumnType.CATEGORICAL,
            sensitive=True,
            quasi_identifier=diagnosis_is_qi,
        ),
    }
    for c in _NOISE:
        cols[c] = ColumnMetadata(name=c, dtype=ColumnType.NUMERIC)
    return Metadata(columns=cols)


def test_infers_only_from_declared_quasi_identifiers():
    real, syn = _linkage_frames()
    scalars = AttributeDisclosure().compute(real, syn, _linkage_meta(zip_is_qi=True)).scalars
    assert scalars is not None
    # Linking on zip alone always lands on a row with the same diagnosis.
    assert scalars["disclosure_rate"] == 1.0


def test_without_declared_quasi_identifiers_uses_all_non_sensitive_columns():
    real, syn = _linkage_frames()
    fallback = AttributeDisclosure().compute(real, syn, _linkage_meta(zip_is_qi=False)).scalars
    # Same table with every non-sensitive column declared a quasi-identifier explicitly.
    explicit_meta = _linkage_meta(zip_is_qi=True)
    for c in _NOISE:
        explicit_meta.columns[c].quasi_identifier = True
    explicit = AttributeDisclosure().compute(real, syn, explicit_meta).scalars
    assert fallback is not None
    assert explicit is not None
    assert fallback == explicit
    # The shuffled noise drags some matches to other zips: short of zip-only linkage.
    assert fallback["disclosure_rate"] < 1.0


def test_quasi_identifier_only_on_the_target_still_falls_back():
    # The target can never be linked on, so declaring only it leaves nothing declared to
    # link on: the metric falls back instead of skipping.
    real, syn = _linkage_frames()
    target_only = _linkage_meta(zip_is_qi=False, diagnosis_is_qi=True)
    r = AttributeDisclosure().compute(real, syn, target_only)
    fallback = AttributeDisclosure().compute(real, syn, _linkage_meta(zip_is_qi=False))
    assert r.skip_reason is None
    assert r.scalars == fallback.scalars


def test_column_with_both_roles_is_a_target_not_a_quasi_identifier():
    # Unique per-row diagnoses, and every synthetic zip collapsed to "a": linking on zip
    # gives no information, so the rate is near zero. If diagnosis were also linked on,
    # each real row would find its own synthetic copy and the rate would be 1.0.
    real, shuffled = _linkage_frames()
    real = real.assign(diagnosis=[f"d{i}" for i in range(len(real))])
    syn = real.assign(zip="a", **{c: shuffled[c] for c in _NOISE})
    meta = _linkage_meta(zip_is_qi=True, diagnosis_is_qi=True)
    scalars = AttributeDisclosure().compute(real, syn, meta).scalars
    assert scalars is not None
    assert scalars["disclosure_rate"] < 0.1
