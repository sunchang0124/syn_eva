import pandas as pd

from syneva.compliance.k_anonymity import KAnonymity
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(
        columns={
            "sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL, quasi_identifier=True),
            "age": ColumnMetadata(name="age", dtype=ColumnType.NUMERIC, quasi_identifier=True),
            "other": ColumnMetadata(name="other", dtype=ColumnType.NUMERIC),
        }
    )


def test_uniform_synthetic_high_k():
    syn = pd.DataFrame(
        {
            "sex": ["M"] * 100,
            "age": [30] * 100,
            "other": list(range(100)),
        }
    )
    r = KAnonymity().compute(None, syn, _meta())
    assert r.scalars["min_k"] == 100


def test_per_row_uniqueness_low_k():
    syn = pd.DataFrame(
        {
            "sex": ["M", "F"] * 50,
            "age": list(range(100)),
            "other": [0] * 100,
        }
    )
    r = KAnonymity().compute(None, syn, _meta())
    assert r.scalars["min_k"] == 1


def test_groups_on_quasi_identifiers_not_sensitive_columns():
    # Sweeney (2002): equivalence classes are formed over the quasi-identifiers; the
    # sensitive attribute (here a unique-per-row income) must not split them.
    syn = pd.DataFrame(
        {
            "zip": ["a"] * 50 + ["b"] * 50,
            "income": [float(i) for i in range(100)],
        }
    )
    meta = Metadata(
        columns={
            "zip": ColumnMetadata(name="zip", dtype=ColumnType.CATEGORICAL, quasi_identifier=True),
            "income": ColumnMetadata(name="income", dtype=ColumnType.NUMERIC, sensitive=True),
        }
    )
    scalars = KAnonymity().compute(None, syn, meta).scalars
    assert scalars is not None
    assert scalars["min_k"] == 50
    assert scalars["unique_groups"] == 2


def test_column_with_both_roles_is_grouped_on():
    syn = pd.DataFrame({"zip": ["a"] * 6 + ["b"] * 4, "diagnosis": ["x"] * 3 + ["y"] * 7})
    meta = Metadata(
        columns={
            "zip": ColumnMetadata(name="zip", dtype=ColumnType.CATEGORICAL, quasi_identifier=True),
            "diagnosis": ColumnMetadata(
                name="diagnosis",
                dtype=ColumnType.CATEGORICAL,
                sensitive=True,
                quasi_identifier=True,
            ),
        }
    )
    scalars = KAnonymity().compute(None, syn, meta).scalars
    assert scalars is not None
    # Groups (a, x)=3, (a, y)=3, (b, y)=4: diagnosis splits the zip groups.
    assert scalars["min_k"] == 3
    assert scalars["unique_groups"] == 3


def test_sensitive_only_skips_naming_quasi_identifier():
    syn = pd.DataFrame({"income": [float(i) for i in range(100)]})
    meta = Metadata(
        columns={
            "income": ColumnMetadata(name="income", dtype=ColumnType.NUMERIC, sensitive=True),
        }
    )
    r = KAnonymity().compute(None, syn, meta)
    assert r.scalars is None
    assert r.skip_reason is not None
    assert "quasi_identifier" in r.skip_reason
