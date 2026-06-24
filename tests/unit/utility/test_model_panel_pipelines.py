import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from syneva.core.metadata import Metadata
from syneva.utility._models import build_panel_pipelines, select_features

_DF = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"], "y": [0, 1, 0, 1]})


def _setup():
    meta = Metadata.infer(_DF)
    features = select_features(meta, "y", None)
    return meta, features


def test_panel_has_three_named_models_classification():
    meta, features = _setup()
    pipes = build_panel_pipelines("classification", meta, features, 42)
    assert set(pipes) == {"linear", "random_forest", "hist_gbdt"}
    for p in pipes.values():
        assert isinstance(p, Pipeline)
        assert "pre" in p.named_steps and "model" in p.named_steps


def test_panel_classification_estimator_types():
    meta, features = _setup()
    pipes = build_panel_pipelines("classification", meta, features, 42)
    assert isinstance(pipes["linear"].named_steps["model"], LogisticRegression)
    assert isinstance(pipes["random_forest"].named_steps["model"], RandomForestClassifier)
    assert isinstance(pipes["hist_gbdt"].named_steps["model"], HistGradientBoostingClassifier)


def test_panel_regression_estimator_types():
    meta, features = _setup()
    pipes = build_panel_pipelines("regression", meta, features, 42)
    assert isinstance(pipes["linear"].named_steps["model"], Ridge)
    assert isinstance(pipes["hist_gbdt"].named_steps["model"], HistGradientBoostingRegressor)
