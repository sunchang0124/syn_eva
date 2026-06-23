import pandas as pd

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
    assert r.scalars["score"] == 1.0
    assert any("sensitive" in n for n in r.notes)
