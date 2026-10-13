import pandas as pd

from syneva.core.metadata import Metadata
from syneva.utility.task import UtilityTask, resolve_task_type, suggest_tasks


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


def test_resolve_task_type_keeps_explicit_type():
    meta = Metadata.infer(pd.DataFrame({"y": ["a", "b"]}))
    assert resolve_task_type(UtilityTask(target="y", task_type="regression"), meta) == "regression"


def test_resolve_task_type_infers_from_metadata():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"], "flag": [True, False]})
    meta = Metadata.infer(df)
    assert resolve_task_type(UtilityTask(target="num"), meta) == "regression"
    assert resolve_task_type(UtilityTask(target="cat"), meta) == "classification"
    assert resolve_task_type(UtilityTask(target="flag"), meta) == "classification"
