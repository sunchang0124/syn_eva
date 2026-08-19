import numpy as np
import pandas as pd

from syneva import SubgroupSpec, UtilityTask
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.minority_utility_gap import MinorityUtilityGap


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def _real():
    # y = 1[x > 0] for group A, y = 1[x > 5] for group B: group-shifted
    # thresholds a linear model WITH group dummies can represent exactly.
    # (Do NOT test with subgroup-flipped labels: a no-interaction logistic
    # model cannot learn a flip, so the distortion never reaches the eval.)
    n_a, n_b = 280, 120
    xa = np.tile([-1.0, 1.0], n_a // 2)
    xb = np.tile([4.0, 6.0], n_b // 2)
    x = np.concatenate([xa, xb])
    y = np.concatenate([(xa > 0), (xb > 5)]).astype(int).astype(str)
    g = ["A"] * n_a + ["B"] * n_b
    return pd.DataFrame({"g": g, "x": x, "y": y})


_SPEC = SubgroupSpec(name="b-group", conditions={"g": ["B"]})
_TASK = UtilityTask(target="y", task_type="classification")


def test_faithful_synthetic_scores_one():
    real = _real()
    m = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[_TASK])
    res = m.compute(real, real.copy(), _meta())
    assert res.scalars["score"] > 0.95


def test_distorted_subgroup_relationship_scores_low():
    real = _real()
    syn = real.copy()
    b = syn.g == "B"
    # shift B's x by -4: B's learned threshold moves from 5 to 1, so the
    # synthetic-trained model misclassifies real B rows at x=4
    syn.loc[b, "x"] = syn.loc[b, "x"] - 4.0
    m = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[_TASK])
    res = m.compute(real, syn, _meta())
    assert res.scalars["score"] < 0.85
    assert res.per_column["b-group|y"]["excess_gap"] > 0.1


def test_no_tasks_scores_one_with_note():
    real = _real()
    res = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=None).compute(real, real.copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert any("task" in n for n in res.notes)


def test_regression_tasks_skipped():
    real = _real()
    task = UtilityTask(target="x", task_type="regression")
    res = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[task]).compute(
        real, real.copy(), _meta()
    )
    assert res.scalars["score"] == 1.0
    assert any("regression" in n for n in res.notes)


def test_tiny_subgroup_pair_skipped():
    real = _real()
    spec = SubgroupSpec(name="tiny", conditions={"x": (100.0, None)})
    res = MinorityUtilityGap(subgroup_specs=[spec], tasks=[_TASK]).compute(
        real, real.copy(), _meta()
    )
    assert res.scalars["score"] == 1.0
    assert any("tiny" in n for n in res.notes)


def test_holdout_used_as_eval_frame():
    real = _real()
    holdout = _real()
    m = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[_TASK], holdout=holdout)
    res = m.compute(real, real.copy(), _meta())
    assert res.scalars["score"] > 0.95
