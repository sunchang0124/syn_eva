import numpy as np
import pandas as pd
import pytest

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.utility._models import (
    build_pipeline,
    encoded_features,
    select_features,
    source_columns,
)


def _fit(df: pd.DataFrame, meta: Metadata, target, features=None):
    features = features or select_features(meta, target, None)
    pipe = build_pipeline("classification", meta, features, 0)
    return pipe.fit(df[features], df[target])


def test_source_columns_maps_one_hot_outputs_to_features(real_df, metadata):
    pipe = _fit(real_df, metadata, "high_income")
    src = source_columns(pipe.named_steps["pre"])
    assert len(src) == len(pipe.named_steps["model"].feature_importances_)
    assert src.count("education") == real_df["education"].nunique()
    assert src.count("sex") == real_df["sex"].nunique()
    assert set(src) == set(select_features(metadata, "high_income", None))


def test_source_columns_omits_all_missing_feature(real_df, metadata):
    df = real_df.assign(age=np.nan)
    pipe = _fit(df, metadata, "high_income")
    src = source_columns(pipe.named_steps["pre"])
    assert "age" not in src
    assert len(src) == len(pipe.named_steps["model"].feature_importances_)


def test_source_columns_matches_by_position_not_name():
    # Non-string labels make sklearn report placeholder names (x0, x1, ...)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({0: rng.normal(size=50), 1: rng.choice(["a", "b", "c"], 50)})
    df[2] = (df[0] > 0).astype(int)
    meta = Metadata(
        columns={
            0: ColumnMetadata(name=0, dtype=ColumnType.NUMERIC),  # pyright: ignore[reportArgumentType]
            1: ColumnMetadata(name=1, dtype=ColumnType.CATEGORICAL),  # pyright: ignore[reportArgumentType]
            2: ColumnMetadata(name=2, dtype=ColumnType.CATEGORICAL),  # pyright: ignore[reportArgumentType]
        }
    )
    pipe = _fit(df, meta, 2, features=[0, 1])
    assert source_columns(pipe.named_steps["pre"]) == [0, 1, 1, 1]


def test_source_columns_rejects_unexpected_output_width(real_df, metadata):
    features = select_features(metadata, "high_income", None)
    pipe = build_pipeline("classification", metadata, features, 0)
    pipe.set_params(pre__cat__oh__drop="if_binary")  # sex now encodes to 1 column, not 2
    pipe.fit(real_df[features], real_df["high_income"])
    with pytest.raises(RuntimeError, match="cannot map 'cat'"):
        source_columns(pipe.named_steps["pre"])


def test_encoded_features_drops_unencoded_dtypes_and_duplicates():
    meta = Metadata(
        columns={
            "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
            "b": ColumnMetadata(name="b", dtype=ColumnType.CATEGORICAL),
            "c": ColumnMetadata(name="c", dtype=ColumnType.BOOLEAN),
            "d": ColumnMetadata(name="d", dtype=ColumnType.DATETIME),
            "e": ColumnMetadata(name="e", dtype=ColumnType.ID),
        }
    )
    assert encoded_features(meta, ["b", "a", "d", "b", "e", "c"]) == ["b", "a", "c"]
