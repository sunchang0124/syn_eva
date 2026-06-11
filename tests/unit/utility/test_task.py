import pandas as pd

from syneva.core.metadata import Metadata
from syneva.utility.task import UtilityTask, suggest_tasks


def test_suggest_classification_for_categorical():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "a"]})
    tasks = suggest_tasks(Metadata.infer(df))
    by_target = {t.target: t for t in tasks}
    assert by_target["y"].task_type == "classification"


def test_suggest_regression_for_numeric():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "a"]})
    tasks = suggest_tasks(Metadata.infer(df))
    by_target = {t.target: t for t in tasks}
    assert by_target["x"].task_type == "regression"


def test_id_columns_not_suggested():
    df = pd.DataFrame({"patient_id": ["a", "b"], "x": [1, 2]})
    tasks = suggest_tasks(Metadata.infer(df))
    assert all(t.target != "patient_id" for t in tasks)
    assert UtilityTask(target="x").target == "x"
