import pandas as pd

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric_info import describe_c
from syneva.preservation._util import categorical_columns, numeric_columns, runnable_specs


def _meta():
    return Metadata(
        columns={
            "sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL),
            "age": ColumnMetadata(name="age", dtype=ColumnType.NUMERIC),
            "flag": ColumnMetadata(name="flag", dtype=ColumnType.BOOLEAN),
            "row_id": ColumnMetadata(name="row_id", dtype=ColumnType.ID),
        }
    )


def _df():
    return pd.DataFrame(
        {
            "sex": ["F", "M", "F", "M", "F"],
            "age": [70, 30, 66, 20, 64],
            "flag": [True, False, True, True, False],
            "row_id": [1, 2, 3, 4, 5],
        }
    )


def test_subgroup_spec_fields():
    sp = SubgroupSpec(name="elderly women", conditions={"sex": ["F"], "age": (65, None)})
    assert sp.name == "elderly women"
    assert sp.conditions["age"] == (65, None)


def test_matches_values_and_range():
    sp = SubgroupSpec(name="ew", conditions={"sex": ["F"], "age": (65, None)})
    mask = sp.matches(_df())
    assert mask.tolist() == [True, False, True, False, False]


def test_matches_range_bounds_inclusive():
    sp = SubgroupSpec(name="band", conditions={"age": (30, 66)})
    assert sp.matches(_df()).tolist() == [False, True, True, False, True]


def test_matches_open_lower_bound():
    sp = SubgroupSpec(name="young", conditions={"age": (None, 30)})
    assert sp.matches(_df()).tolist() == [False, True, False, True, False]


def test_validate_ok():
    sp = SubgroupSpec(name="ok", conditions={"sex": ["F"], "age": (None, 30)})
    assert sp.validate(_meta()) is None


def test_validate_unknown_column():
    sp = SubgroupSpec(name="bad", conditions={"nope": ["x"]})
    assert "unknown column" in sp.validate(_meta())


def test_validate_range_on_categorical():
    sp = SubgroupSpec(name="bad", conditions={"sex": (0, 1)})
    assert "range" in sp.validate(_meta())


def test_validate_values_on_numeric():
    sp = SubgroupSpec(name="bad", conditions={"age": [65]})
    assert "values list" in sp.validate(_meta())


def test_util_column_selectors():
    meta = _meta()
    assert categorical_columns(meta) == ["sex", "flag"]
    assert numeric_columns(meta) == ["age"]


def test_runnable_specs_partitions():
    good = SubgroupSpec(name="g", conditions={"sex": ["F"]})
    bad = SubgroupSpec(name="b", conditions={"nope": ["x"]})
    ok, notes = runnable_specs([good, bad], _meta())
    assert ok == [good]
    assert len(notes) == 1 and "b" in notes[0]


def test_preservation_dimension_described():
    assert describe_c("preservation") != ""
