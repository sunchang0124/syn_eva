import numpy as np
import pandas as pd
import pytest

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


@pytest.fixture
def with_datetime(real_df, syn_good_df, metadata) -> tuple[pd.DataFrame, pd.DataFrame, Metadata]:
    """(real, synthetic, metadata) with an extra datetime column 'signup'."""
    signup = pd.Timestamp("2020-01-01")
    cols = {**metadata.columns, "signup": ColumnMetadata("signup", ColumnType.DATETIME)}
    return (
        real_df.assign(signup=signup),
        syn_good_df.assign(signup=signup),
        Metadata(columns=cols, primary_key=metadata.primary_key),
    )


@pytest.fixture
def null_variants(syn_good_df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The same synthetic nulls three ways: NaN, None, and pd.NA in nullable dtypes."""
    mask = syn_good_df.index % 3 != 0
    nan = syn_good_df.assign(education=syn_good_df["education"].mask(mask, np.nan))
    none = syn_good_df.copy()
    none["education"] = none["education"].astype(object).mask(mask, None)
    nullable = syn_good_df.assign(
        education=syn_good_df["education"].astype("string").mask(mask),
        age=syn_good_df["age"].astype("Int64"),
    )
    return nan, none, nullable
