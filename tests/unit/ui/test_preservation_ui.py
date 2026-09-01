import numpy as np
import pandas as pd

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.ui import core


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"g": ["A"] * 160 + ["B"] * 40, "x": rng.normal(0, 1, 200)})


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def test_run_report_forwards_subgroup_specs():
    rep = core.run_report(
        _df(),
        _df(),
        _meta(),
        selected_names=["subgroup_fidelity", "rare_category_retention"],
        subgroup_specs=[SubgroupSpec(name="b", conditions={"g": ["B"]})],
    )
    names = {r.spec.name for r in rep.results}
    assert {"subgroup_fidelity", "rare_category_retention"} <= names


def test_run_report_without_specs_drops_subgroup_metrics():
    rep = core.run_report(
        _df(),
        _df(),
        _meta(),
        selected_names=["subgroup_fidelity", "rare_category_retention"],
    )
    names = {r.spec.name for r in rep.results}
    assert "rare_category_retention" in names
    assert "subgroup_fidelity" not in names
