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


def test_discriminative_score_without_encodable_columns_is_skipped():
    data = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=20)})
    r = DiscriminativeScore().compute(data, data.copy(), Metadata.infer(data))
    assert r.scalars is None
    assert "numeric or categorical" in r.skip_reason


# a stratified train/test split needs at least 2 rows on each side
@pytest.mark.parametrize(("n_real", "n_syn"), [(40, 0), (0, 40), (40, 1), (1, 1)])
def test_discriminative_score_with_too_few_rows_is_skipped(n_real, n_syn):
    data = pd.DataFrame({"x": np.arange(40.0)})
    r = DiscriminativeScore().compute(data.iloc[:n_real], data.iloc[:n_syn], Metadata.infer(data))
    assert r.scalars is None
    assert "at least 2 rows" in r.skip_reason


def test_discriminative_score_with_two_rows_each_is_scored():
    data = pd.DataFrame({"x": np.arange(40.0)})
    r = DiscriminativeScore().compute(data, data.iloc[:2], Metadata.infer(data))
    assert r.skip_reason is None
    assert 0.0 <= r.scalars["score"] <= 1.0


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


def test_feature_importance_ignores_unencoded_features(with_datetime):
    # Datetime columns are dropped by the preprocessor; counting them added tied
    # zeros to both vectors and let a one-feature task score rho = 1.
    real, syn, meta = with_datetime
    tasks = [
        UtilityTask("high_income", "classification", features=["age", "signup"]),
        UtilityTask("age", "regression"),
    ]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real, syn, meta)
    assert r.per_column is not None
    assert set(r.per_column) == {"age"}
    assert r.per_column["age"]["n_features"] == 5  # signup is not counted
    assert any("high_income" in n and "usable" in n for n in r.notes)
    assert any("'signup'" in n and "ignored" in n for n in r.notes)


def test_feature_importance_dedupes_task_features(real_df, syn_good_df, metadata):
    task = UtilityTask("high_income", "classification", features=["age", "age", "education"])
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real_df, syn_good_df, metadata)
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 2


def test_feature_importance_notes_all_missing_synthetic_feature(real_df, metadata):
    # Real ranks income > age > hours_per_week. With age all-missing in synthetic it
    # drops to 0 there, so synthetic ranks income > hours_per_week > age: rho = 0.5.
    syn = real_df.assign(age=np.nan)
    features = ["age", "hours_per_week", "income"]
    task = UtilityTask("high_income", "classification", features=features)
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real_df, syn, metadata)
    assert r.scalars is not None
    assert r.per_column is not None
    assert r.scalars["spearman_rho"] == pytest.approx(0.5)
    assert r.per_column["high_income"]["n_features"] == 3
    assert any("'age'" in n and "synthetic" in n and "missing" in n for n in r.notes)


def test_feature_importance_survives_every_numeric_feature_missing(real_df, metadata):
    syn = real_df.assign(age=np.nan, hours_per_week=np.nan, income=np.nan)
    tasks = [UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn, metadata)
    assert r.scalars is not None
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 5
    # income dominates real; synthetic can only use sex and education
    assert r.scalars["spearman_rho"] < 0.5
    for col in ("age", "hours_per_week", "income"):
        assert any(f"'{col}'" in n and "synthetic" in n for n in r.notes)


def test_feature_importance_treats_none_like_nan(real_df, metadata):
    # None as the most common category used to crash the one-hot mapping
    mask = real_df.index % 3 != 0
    with_none = real_df.copy()
    with_none["education"] = with_none["education"].astype(object).mask(mask, None)
    with_nan = real_df.assign(education=real_df["education"].mask(mask, np.nan))
    tasks = [UtilityTask("high_income", "classification")]
    r_none = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, with_none, metadata)
    r_nan = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, with_nan, metadata)
    assert r_none.per_column is not None
    assert r_none.per_column == r_nan.per_column


def test_feature_importance_drops_features_uninformative_on_both_sides(real_df, metadata):
    # A feature with no variation on either side scores 0 in both vectors; that
    # tied pair would pull rho towards 1 without saying anything about the data.
    cols = {
        **metadata.columns,
        "blank": ColumnMetadata("blank", ColumnType.NUMERIC),
        "const": ColumnMetadata("const", ColumnType.CATEGORICAL),
    }
    meta = Metadata(columns=cols, primary_key=metadata.primary_key)
    real = real_df.assign(blank=np.nan, const="x")
    task = UtilityTask(
        "high_income", "classification", features=["blank", "const", "age", "income"]
    )
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real, real.copy(), meta)
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 2
    for col in ("blank", "const"):
        assert any(f"'{col}'" in n and "both" in n for n in r.notes)


def test_feature_importance_excludes_target_from_override(real_df, syn_good_df, metadata):
    task = UtilityTask("high_income", "classification", features=["age", "income", "high_income"])
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real_df, syn_good_df, metadata)
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 2
    assert any("'high_income'" in n and "ignored" in n for n in r.notes)


def test_feature_importance_keeps_feature_constant_on_one_side(real_df, metadata):
    # A constant synthetic column is a real difference: it scores 0 there but stays ranked
    syn = real_df.assign(sex="Male")
    tasks = [UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn, metadata)
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 5
    assert any("'sex'" in n and "missing in synthetic" in n for n in r.notes)
    assert not any("'sex'" in n and "real" in n for n in r.notes)


def test_feature_importance_notes_different_constants(real_df, metadata):
    # Constant on both sides is uninformative for ranking even when the values
    # differ, but the note should not hide that mismatch.
    cols = {**metadata.columns, "region": ColumnMetadata("region", ColumnType.CATEGORICAL)}
    meta = Metadata(columns=cols, primary_key=metadata.primary_key)
    real = real_df.assign(region="EU")
    syn = real_df.assign(region="US")
    task = UtilityTask("high_income", "classification", features=["region", "age", "income"])
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real, syn, meta)
    assert r.per_column is not None
    assert r.per_column["high_income"]["n_features"] == 2
    assert any("'region'" in n and "different value" in n for n in r.notes)
