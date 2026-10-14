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
    df = pd.DataFrame({"y": ["a", "b"]})
    task = UtilityTask(target="y", task_type="regression")
    assert resolve_task_type(task, Metadata.infer(df), df) == "regression"


def test_resolve_task_type_infers_from_metadata():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"], "flag": [True, False]})
    meta = Metadata.infer(df)
    assert resolve_task_type(UtilityTask(target="num"), meta, df) == "regression"
    assert resolve_task_type(UtilityTask(target="cat"), meta, df) == "classification"
    assert resolve_task_type(UtilityTask(target="flag"), meta, df) == "classification"


def test_resolve_task_type_infers_from_data_when_target_lacks_metadata():
    df = pd.DataFrame({"x": [1.0, 2.0], "num": [1.0, 2.0], "cat": ["a", "b"]})
    meta = Metadata.infer(df[["x"]])
    assert resolve_task_type(UtilityTask(target="num"), meta, df) == "regression"
    assert resolve_task_type(UtilityTask(target="cat"), meta, df) == "classification"
