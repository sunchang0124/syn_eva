import numpy as np
import pandas as pd
import pytest

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.utility._models import _as_object, build_pipeline, select_features, source_columns


def _fit(df: pd.DataFrame, meta: Metadata, target, features=None):
    features = features or select_features(meta, target, None)
    pipe = build_pipeline("classification", meta, features, 0)
    return pipe.fit(df[features], df[target])


def _assert_maps_every_feature(pipe, features) -> list:
    src = source_columns(pipe.named_steps["pre"])
    assert len(src) == len(pipe.named_steps["model"].feature_importances_)
    assert set(src) == set(features)
    return src


def test_source_columns_maps_one_hot_outputs_to_features(real_df, metadata):
    pipe = _fit(real_df, metadata, "high_income")
    src = _assert_maps_every_feature(pipe, select_features(metadata, "high_income", None))
    assert src.count("education") == real_df["education"].nunique()
    assert src.count("sex") == real_df["sex"].nunique()


@pytest.mark.parametrize("col", ["age", "education"])
def test_source_columns_keeps_all_missing_feature(real_df, metadata, col):
    df = real_df.assign(**{col: np.nan})
    pipe = _fit(df, metadata, "high_income")
    src = _assert_maps_every_feature(pipe, select_features(metadata, "high_income", None))
    assert src.count(col) == 1


def test_pipeline_fits_when_every_numeric_feature_is_missing(real_df, metadata):
    # The imputer used to drop all of them, leaving StandardScaler with 0 columns
    df = real_df.assign(age=np.nan, hours_per_week=np.nan, income=np.nan)
    pipe = _fit(df, metadata, "high_income")
    _assert_maps_every_feature(pipe, select_features(metadata, "high_income", None))


def test_none_is_imputed_like_nan(real_df, metadata):
    # read_parquet returns None for nulls in object columns. None used to be kept
    # as its own category, and as the mode it broke the one-hot mapping.
    features = select_features(metadata, "high_income", None)
    mask = np.arange(len(real_df)) % 3 != 0
    with_none = real_df.copy()
    with_none["education"] = with_none["education"].astype(object).mask(mask, None)
    with_nan = real_df.assign(education=real_df["education"].mask(mask, np.nan))
    pipe = _fit(with_none, metadata, "high_income")
    _assert_maps_every_feature(pipe, features)
    np.testing.assert_array_equal(
        pipe.named_steps["pre"].transform(with_none[features]),
        _fit(with_nan, metadata, "high_income").named_steps["pre"].transform(with_nan[features]),
    )


def test_pipeline_accepts_nullable_dtypes(real_df, metadata):
    df = real_df.assign(
        education=real_df["education"].astype("string").mask(real_df.index % 4 == 0),
        age=real_df["age"].astype("Int64").mask(real_df.index % 5 == 0),
    )
    pipe = _fit(df, metadata, "high_income")
    _assert_maps_every_feature(pipe, select_features(metadata, "high_income", None))


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


_MIXED = Metadata(
    columns={
        "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
        "b": ColumnMetadata(name="b", dtype=ColumnType.CATEGORICAL),
        "c": ColumnMetadata(name="c", dtype=ColumnType.BOOLEAN),
        "d": ColumnMetadata(name="d", dtype=ColumnType.DATETIME),
        "e": ColumnMetadata(name="e", dtype=ColumnType.ID),
        "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
    }
)


def test_select_features_keeps_only_encoded_dtypes():
    assert select_features(_MIXED, "y", None) == ["a", "b", "c"]


def test_select_features_filters_and_dedupes_override():
    assert select_features(_MIXED, "y", ["b", "a", "d", "b", "e", "c"]) == ["b", "a", "c"]


def test_select_features_notes_only_what_an_override_drops():
    notes: list[str] = []
    select_features(_MIXED, "y", None, notes)
    select_features(_MIXED, "y", ["a", "a", "b"], notes)
    assert notes == []  # the default path and duplicates add nothing


def test_select_features_names_each_cause_separately():
    notes: list[str] = []
    assert select_features(_MIXED, "y", ["a", "d", "y", "e"], notes) == ["a"]
    assert len(notes) == 2
    assert "target" in notes[0]
    assert "features 'd', 'e' ignored" in notes[1]  # the target is not listed again


def test_nullable_boolean_and_category_are_imputed_like_object(real_df):
    # Both reach _as_object; NA must be imputed, not kept as a category
    meta = Metadata(
        columns={
            "flag": ColumnMetadata(name="flag", dtype=ColumnType.BOOLEAN),
            "level": ColumnMetadata(name="level", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )
    mask = real_df.index % 4 == 0
    flag = (real_df["age"] > 40).mask(mask)
    level = real_df["education"].mask(mask)
    typed = pd.DataFrame(
        {
            "flag": flag.astype("boolean"),
            "level": level.astype("category"),
            "y": real_df["high_income"],
        }
    )
    plain = pd.DataFrame({"flag": flag.astype(object), "level": level, "y": typed["y"]})
    features = ["flag", "level"]
    pipe = _fit(typed, meta, "y", features=features)
    _assert_maps_every_feature(pipe, features)
    np.testing.assert_array_equal(
        pipe.named_steps["pre"].transform(typed[features]),
        _fit(plain, meta, "y", features=features).named_steps["pre"].transform(plain[features]),
    )


def test_as_object_accepts_ndarray():
    # ColumnTransformer passes arrays through when selecting by column position
    out = _as_object(np.array([["a", None], ["b", "c"]], dtype=object))
    assert out.isna().to_numpy().tolist() == [[False, True], [False, False]]
