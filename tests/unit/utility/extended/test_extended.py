import numpy as np
import pandas as pd
import pytest

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.utility._models import build_pipeline, select_features
from syneva.utility.extended.discriminative import DiscriminativeScore
from syneva.utility.extended.feature_importance import (
    FeatureImportanceCorrelation,
    _per_feature_importance,
)
from syneva.utility.extended.multi_target import MultiTargetUtility
from syneva.utility.task import UtilityTask


def test_multi_target_runs_all_columns(real_df, syn_good_df, metadata):
    r = MultiTargetUtility().compute(real_df, syn_good_df, metadata)
    assert r.scalars["score"] >= 0.0
    assert r.per_column  # at least one auto-suggested task


def test_multi_target_survives_rare_category(real_df, syn_good_df, metadata):
    # One rare category in any auto-suggested target used to crash the stratified split
    real = real_df.copy()
    real.loc[real.index[0], "education"] = "Preschool"
    r = MultiTargetUtility().compute(real, syn_good_df, metadata)
    assert "education" in r.per_column


def test_feature_importance_correlation_returns_value(real_df, syn_good_df, metadata):
    r = FeatureImportanceCorrelation(tasks=[UtilityTask("high_income")]).compute(
        real_df, syn_good_df, metadata
    )
    assert -1.0 <= r.scalars["spearman_rho"] <= 1.0


def test_feature_importance_averages_over_tasks(real_df, syn_good_df, metadata):
    tasks = [UtilityTask("high_income", "classification"), UtilityTask("age", "regression")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn_good_df, metadata)
    assert set(r.per_column) == {"high_income", "age"}
    rhos = [r.per_column[t]["spearman_rho"] for t in ("high_income", "age")]
    assert r.scalars["spearman_rho"] == pytest.approx(sum(rhos) / 2)
    assert r.scalars["score"] == pytest.approx((r.scalars["spearman_rho"] + 1) / 2)


def test_feature_importance_honours_task_features(real_df, syn_good_df, metadata):
    task = UtilityTask("high_income", "classification", features=["age", "hours_per_week"])
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real_df, syn_good_df, metadata)
    assert r.per_column["high_income"]["n_features"] == 2


def test_feature_importance_notes_missing_target(real_df, syn_good_df, metadata):
    tasks = [UtilityTask("nope", "classification"), UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn_good_df, metadata)
    assert set(r.per_column) == {"high_income"}
    assert any("nope" in n for n in r.notes)


def test_feature_importance_without_tasks_is_skipped(real_df, syn_good_df, metadata):
    r = FeatureImportanceCorrelation().compute(real_df, syn_good_df, metadata)
    assert r.scalars is None
    assert r.skip_reason


def test_discriminative_score_returns_auc(real_df, syn_good_df, metadata):
    r = DiscriminativeScore().compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["auc"] <= 1.0


def test_feature_importance_undefined_rho_falls_back_to_zero(real_df, syn_shifted_df, metadata):
    # The shifted fixture's synthetic high_income is constant, so its model has
    # all-zero importances and Spearman rho is undefined (NaN).
    tasks = [UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn_shifted_df, metadata)
    assert r.scalars["spearman_rho"] == 0.0
    assert r.scalars["score"] == 0.5
    assert any("undefined" in n for n in r.notes)


def test_feature_importance_aligns_features_when_category_sets_differ(real_df, metadata):
    # Synthetic = real without one education level, so the encoders emit different
    # one-hot widths. Importances used to be compared by encoder column index,
    # pairing unrelated features and pulling rho down to ~0.5 for near-identical data.
    syn = real_df[real_df["education"] != "MSc"]
    tasks = [UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn, metadata)
    assert r.scalars is not None
    assert r.scalars["spearman_rho"] >= 0.8


def test_per_feature_importance_sums_one_hot_columns_per_feature(real_df, metadata):
    features = select_features(metadata, "high_income", None)
    pipe = build_pipeline("classification", metadata, features, 42)
    pipe.fit(real_df[features], real_df["high_income"])
    raw = pipe.named_steps["model"].feature_importances_
    names = pipe.named_steps["pre"].get_feature_names_out()
    fi = _per_feature_importance(pipe, features)
    assert fi.sum() == pytest.approx(1.0)
    for col in ("sex", "education"):  # two categoricals, so a cross-wired mapping shows up
        expected = raw[[n.startswith(f"cat__{col}_") for n in names]].sum()
        assert fi[features.index(col)] == pytest.approx(expected)


def test_per_feature_importance_scores_all_missing_feature_zero(real_df, metadata):
    features = select_features(metadata, "high_income", None)
    df = real_df.assign(age=np.nan)
    pipe = build_pipeline("classification", metadata, features, 42)
    pipe.fit(df[features], df["high_income"])
    fi = _per_feature_importance(pipe, features)
    assert fi[features.index("age")] == 0.0
    assert fi.sum() == pytest.approx(1.0)


def _with_datetime(df: pd.DataFrame, metadata: Metadata) -> tuple[pd.DataFrame, Metadata]:
    df = df.assign(signup=pd.Timestamp("2020-01-01"))
    cols = {**metadata.columns, "signup": ColumnMetadata("signup", ColumnType.DATETIME)}
    return df, Metadata(columns=cols, primary_key=metadata.primary_key)


def test_feature_importance_ignores_unencoded_features(real_df, syn_good_df, metadata):
    # Datetime columns are dropped by the preprocessor; counting them added tied
    # zeros to both vectors and let a one-feature task score rho = 1.
    real, meta = _with_datetime(real_df, metadata)
    syn, _ = _with_datetime(syn_good_df, metadata)
    tasks = [
        UtilityTask("high_income", "classification", features=["age", "signup"]),
        UtilityTask("age", "regression"),
    ]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real, syn, meta)
    assert r.per_column is not None
    assert set(r.per_column) == {"age"}
    assert r.per_column["age"]["n_features"] == 5  # signup is not counted
    assert any("high_income" in n and "usable" in n for n in r.notes)


def test_feature_importance_dedupes_task_features(real_df, syn_good_df, metadata):
    task = UtilityTask("high_income", "classification", features=["age", "age", "education"])
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real_df, syn_good_df, metadata)
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 2


def test_feature_importance_notes_all_missing_synthetic_feature(real_df, metadata):
    # Only age and income are ranked: income dominates on both sides, and age is
    # all-missing in synthetic, so it scores 0 there and keeps second place in real.
    syn = real_df.assign(age=np.nan)
    task = UtilityTask("high_income", "classification", features=["age", "income"])
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real_df, syn, metadata)
    assert r.scalars is not None
    assert r.per_column is not None
    assert r.scalars["spearman_rho"] == pytest.approx(1.0)
    assert r.per_column["high_income"]["n_features"] == 2
    assert any("'age'" in n and "synthetic" in n and "missing" in n for n in r.notes)


def test_feature_importance_survives_every_numeric_feature_missing(real_df, metadata):
    syn = real_df.assign(age=np.nan, hours_per_week=np.nan, income=np.nan)
    tasks = [UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn, metadata)
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 5


def test_feature_importance_handles_none_as_most_common_category(real_df, metadata):
    syn = real_df.copy()
    syn["education"] = syn["education"].astype(object).mask(syn.index % 3 != 0, None)
    tasks = [UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn, metadata)
    assert r.per_column is not None
    assert "high_income" in r.per_column
